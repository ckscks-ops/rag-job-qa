"""
export_recruitops.py —— 从 RecruitOps 的 PostgreSQL 备份里导出真实岗位 JD

背景
====
RecruitOps 秋招工作台把岗位数据存在它自带的 PostgreSQL 里：

    D:\\RecruitOps\\.data\\desktop-runtime-tests\\shell-runtime-<id>\\pgdata

这个库**连不上**：
  - 服务没在跑（2026-10-07 23:47 因 startup process failure 关闭）
  - `pg_hba.conf` 全部是 `scram-sha-256`
  - 密码在 `instance.json` 里，是 **DPAPI 加密**的（`AQAAANCMnd8BFdERjHoAwE...`），无法解密

**但是**它每次 schema 迁移前都会 `pg_dump` 一份备份到 `backups/`：

    pre-migration-<run_id>-<hash>.dump

这是标准 pg_dump 自定义格式，可以用 RecruitOps **自带的 pg_restore** 解出
`COPY ... FROM stdin` 文本。**不需要起数据库、不需要密码**，纯离线读取。

用法
----
    # 先看有什么数据（不写文件）
    python src/export_recruitops.py --stats-only

    # 正式导出
    python src/export_recruitops.py --out data/prod_jobs.json

    # 小样本试跑
    python src/export_recruitops.py --out data/prod_jobs.small.json --limit 200

输出格式与 `src/ingest.py` 的输入一致（JSON 数组）：

    [{ "doc_key": "...", "company": "...", "title": "...",
       "city": "...", "job_type": "...", "source": "...",
       "raw_text": "JD 原文", ... }]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

# ─────────────── 默认路径 ───────────────

DEFAULT_PG_RESTORE = r'D:\RecruitOps\resources\desktop-runtime\postgres\bin\pg_restore.exe'
DEFAULT_BACKUP_GLOB = (
    r'D:\RecruitOps\.data\.desktop-runtime-tests'
    r'\shell-runtime-*\backups\*.dump'
)

# 需要的两张表：岗位 + 公司
TABLES = ('job_snapshots', 'company_snapshots')

# COPY public.<table> (<cols>) FROM stdin;
_COPY_RE = re.compile(
    r'^COPY\s+public\.(?P<table>[A-Za-z0-9_]+)\s*\((?P<cols>[^)]*)\)\s*FROM\s+stdin;\s*$'
)

_OCTAL_RE = re.compile(r'\\([0-7]{1,3})')
_HEX_RE = re.compile(r'\\x([0-9A-Fa-f]{1,2})')

_SIMPLE_ESCAPES = {
    'n': '\n', 't': '\t', 'r': '\r', 'b': '\b',
    'f': '\f', 'v': '\v', '\\': '\\',
}


# ─────────────── COPY 文本解析 ───────────────

def _unescape(field: str):
    """
    还原 COPY 文本格式的转义。

    规则（PostgreSQL 文档 COPY 一节）：
      \\N      -> NULL
      \\n \\t \\r \\b \\f \\v \\\\  -> 对应控制字符
      \\<octal> / \\x<hex>        -> 一个字节
    其余反斜杠后被转义掉的字符，去掉反斜杠取原字符。
    """
    if field == r'\N':
        return None

    out = []
    i = 0
    n = len(field)
    while i < n:
        c = field[i]
        if c != '\\' or i + 1 >= n:
            out.append(c)
            i += 1
            continue

        nxt = field[i + 1]
        if nxt in _SIMPLE_ESCAPES:
            out.append(_SIMPLE_ESCAPES[nxt])
            i += 2
            continue

        m = _OCTAL_RE.match(field, i)
        if m:
            out.append(chr(int(m.group(1), 8)))
            i = m.end()
            continue

        m = _HEX_RE.match(field, i)
        if m:
            out.append(chr(int(m.group(1), 16)))
            i = m.end()
            continue

        # pg_dump 会转义 \. 之类，直接取原字符
        out.append(nxt)
        i += 2

    return ''.join(out)


def parse_copy_blocks(sql_text: str) -> dict[str, tuple[list[str], list[list]]]:
    """
    从 pg_restore 输出的 SQL 文本里取出所有 COPY 块。

    返回 {表名: (列名列表, 行列表)}，行是字段列表（已反转义）。
    """
    result: dict[str, tuple[list[str], list[list]]] = {}
    table = None
    cols: list[str] = []
    rows: list[list] = []

    for line in sql_text.split('\n'):
        if table is None:
            m = _COPY_RE.match(line)
            if m:
                table = m.group('table')
                cols = [c.strip() for c in m.group('cols').split(',')]
                rows = []
            continue

        # 正在读一个 COPY 块
        if line == r'\.':
            result[table] = (cols, rows)
            table = None
        elif line == '' and not rows:
            # COPY 块之间的空行，忽略
            continue
        else:
            rows.append([_unescape(f) for f in line.split('\t')])

    return result


# ─────────────── 调用 pg_restore ───────────────

def extract_table(pg_restore: str, dump: Path, table: str,
                  tmpdir: Path) -> tuple[list[str], list[list]]:
    """把一张表的 data-only SQL 解到临时文件，再解析。"""
    out_file = tmpdir / ('%s.sql' % table)
    cmd = [
        pg_restore,
        '--data-only',
        '-t', table,
        '--no-owner',
        '--no-privileges',
        '-f', str(out_file),
        str(dump),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 and not out_file.is_file():
        raise RuntimeError(
            'pg_restore 失败（%s）:\n%s'
            % (proc.returncode, proc.stderr.decode('utf-8', 'replace')[:800])
        )

    text = out_file.read_text(encoding='utf-8', errors='replace')
    blocks = parse_copy_blocks(text)
    if table not in blocks:
        return [], []
    return blocks[table]


# ─────────────── 选择备份文件 ───────────────

def pick_dump(pattern: str, explicit: str | None) -> Path:
    """
    选一个备份文件。

    没显式指定时，在匹配到的 dump 里挑 job_snapshots 数据量最大的那个
    （`pg_restore --list` 看不出行数，所以用文件体积近似：job_snapshots 是
    整个库里最大的表，dump 越大通常岗位越多）。
    """
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit('备份文件不存在: %s' % p)
        return p

    if not glob.glob(pattern):
        raise SystemExit('没有找到任何备份文件，检查路径: %s' % pattern)

    candidates = []
    for f in glob.glob(pattern):
        p = Path(f)
        candidates.append((p.stat().st_size, p))
    candidates.sort(reverse=True)
    if len(candidates) > 1:
        print('  找到 %d 个备份，按体积降序：' % len(candidates))
        for size, p in candidates:
            print('    %10.2f MB  %s' % (size / 1024 / 1024, p.name))
    return candidates[0][1]


# ─────────────── 组装岗位记录 ───────────────

def build_jobs(rows: list[list], cols: list[str],
               companies: dict[str, str]) -> list[dict]:
    """把 job_snapshots 的行转成项目要的 JSON 结构。"""
    idx = {c: i for i, c in enumerate(cols)}
    jobs = []
    seen = set()

    for r in rows:
        def get(name, default=None):
            i = idx.get(name)
            if i is None or i >= len(r):
                return default
            v = r[i]
            return default if v is None else v

        raw = (get('jd_raw') or '').strip()
        title = (get('title') or '').strip()
        company_id = get('company_id') or ''
        company = companies.get(company_id) or get('source_tenant') or '未知公司'
        city = (get('city') or '').strip()
        batch = (get('batch') or '').strip()
        key = (get('business_key') or '').strip()
        if not key:
            key = '|'.join(x for x in (company, title, city) if x)

        if not raw or not title:
            continue
        if key in seen:
            continue
        seen.add(key)

        jobs.append({
            'doc_key': key,
            'company': company,
            'title': title,
            'city': city,
            'job_type': batch or (get('cohort_status') or ''),
            'source': get('source_platform') or get('source') or 'RecruitOps',
            'raw_text': raw,
            # 以下作为附加元数据入库（ingest.py 会放进 metadata）
            'detail_url': get('detail_url'),
            'match_score': get('match_score'),
            'availability_status': get('availability_status'),
            'first_seen_at': get('first_seen_at'),
        })

    return jobs


# ─────────────── 统计输出 ───────────────

def dedup_jobs(jobs: list[dict]) -> list[dict]:
    """
    按 JD 正文去重。

    为什么必须做：同一个 JD 会挂在多个公司别名下。实测「数据服务后端开发工程师」
    这条正文重复了 99 次（字节跳动 9 个别名 × 11 个岗位），
    不去重的话向量库里 74% 是重复内容，评测和演示都会被污染。

    保留第一次出现的记录，把重复次数和别名写进元数据，信息不丢。
    """
    seen: dict[str, dict] = {}
    order: list[str] = []
    for j in jobs:
        key = re.sub(r'\s+', '', j['raw_text'])
        if key in seen:
            seen[key]['dup_count'] += 1
            alias = seen[key]['dup_companies']
            if j['company'] not in alias:
                alias.append(j['company'])
        else:
            j = dict(j)
            j['dup_count'] = 1
            j['dup_companies'] = [j['company']]
            seen[key] = j
            order.append(key)
    return [seen[k] for k in order]


def print_stats(jobs: list[dict]) -> None:
    print()
    print('=' * 70)
    print('导出数据统计')
    print('=' * 70)
    print('  岗位总数        : %d' % len(jobs))

    lens = sorted(len(j['raw_text']) for j in jobs)
    if lens:
        def pct(p):
            return lens[min(len(lens) - 1, int(len(lens) * p))]
        print('  JD 字符数       : min=%d  p25=%d  中位=%d  p75=%d  p95=%d  max=%d'
              % (lens[0], pct(.25), statistics.median(lens), pct(.75), pct(.95), lens[-1]))
        print('  JD >= 200 字符  : %d (%.1f%%)'
              % (sum(1 for x in lens if x >= 200),
                 100.0 * sum(1 for x in lens if x >= 200) / len(lens)))

    def top(field, n=12):
        cnt: dict[str, int] = {}
        for j in jobs:
            v = (j.get(field) or '').strip() or '(空)'
            cnt[v] = cnt.get(v, 0) + 1
        return sorted(cnt.items(), key=lambda kv: -kv[1])[:n]

    print()
    print('  公司数(去重)    : %d' % len({j['company'] for j in jobs}))
    print('  岗位名(去重)    : %d' % len({j['title'] for j in jobs}))
    print()
    print('  城市 Top:')
    for k, v in top('city'):
        print('    %-14s %5d' % (k, v))
    print()
    print('  批次 Top:')
    for k, v in top('job_type', 8):
        print('    %-14s %5d' % (k, v))
    print()
    print('  来源 Top:')
    for k, v in top('source', 8):
        print('    %-14s %5d' % (k, v))
    print()
    print('  公司 Top:')
    for k, v in top('company', 12):
        print('    %-24s %5d' % (k[:24], v))


# ─────────────── 主流程 ───────────────

def main() -> int:
    ap = argparse.ArgumentParser(description='从 RecruitOps 备份导出岗位 JD')
    ap.add_argument('--dump', help='指定 .dump 备份文件（默认自动挑选最大的）')
    ap.add_argument('--glob', default=DEFAULT_BACKUP_GLOB, help='备份文件搜索模式')
    ap.add_argument('--pg-restore', default=DEFAULT_PG_RESTORE, help='pg_restore 路径')
    ap.add_argument('--out', default='data/prod_jobs.json', help='输出 JSON 路径')
    ap.add_argument('--limit', type=int, default=0, help='只导出前 N 条（调试用）')
    ap.add_argument('--min-chars', type=int, default=80,
                    help='JD 少于这个字符数就丢弃（默认 80）')
    ap.add_argument('--dedup', action='store_true',
                    help='按 JD 正文去重（强烈建议：原始数据里同一条正文最多重复 99 次）')
    ap.add_argument('--stats-only', action='store_true', help='只统计，不写文件')
    args = ap.parse_args()

    if not Path(args.pg_restore).is_file():
        print('  [错误] 找不到 pg_restore: %s' % args.pg_restore)
        return 1

    print('=' * 70)
    print('RecruitOps 岗位数据导出')
    print('=' * 70)

    dump = pick_dump(args.glob, args.dump)
    print('  使用备份: %s' % dump)
    print('  体积    : %.2f MB' % (dump.stat().st_size / 1024 / 1024))

    tmpdir = Path(tempfile.mkdtemp(prefix='ro_export_'))
    try:
        # 1. 公司表
        ccols, crows = extract_table(args.pg_restore, dump, 'company_snapshots', tmpdir)
        companies: dict[str, str] = {}
        if 'id' in ccols and 'name' in ccols:
            ci, ni = ccols.index('id'), ccols.index('name')
            for r in crows:
                if len(r) > max(ci, ni) and r[ci] and r[ni]:
                    companies[r[ci]] = r[ni]
        print('  公司表  : %d 行（解析出 %d 个公司名）' % (len(crows), len(companies)))

        # 2. 岗位表
        jcols, jrows = extract_table(args.pg_restore, dump, 'job_snapshots', tmpdir)
        print('  岗位表  : %d 行' % len(jrows))
        if not jrows:
            print('  [错误] job_snapshots 没有数据段，换一个 dump 试试')
            return 1

        jobs = build_jobs(jrows, jcols, companies)
        print('  可用岗位: %d （丢弃了没有 JD 正文或标题为空的 %d 条）'
              % (len(jobs), len(jrows) - len(jobs)))

        before = len(jobs)
        jobs = [j for j in jobs if len(j['raw_text']) >= args.min_chars]
        if before != len(jobs):
            print('  短 JD 过滤: 再去掉 %d 条（< %d 字符）' % (before - len(jobs), args.min_chars))

        if args.limit:
            jobs = jobs[:args.limit]
            print('  --limit 生效，只保留前 %d 条' % len(jobs))

        if args.dedup:
            before = len(jobs)
            jobs = dedup_jobs(jobs)
            print('  --dedup 去重: %d -> %d（合并掉 %d 条重复正文）'
                  % (before, len(jobs), before - len(jobs)))

        print_stats(jobs)

        if args.stats_only:
            print()
            print('  （--stats-only，未写文件）')
            return 0

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(jobs, ensure_ascii=False, indent=1),
            encoding='utf-8',
        )
        print()
        print('  已写出: %s  (%.2f MB)'
              % (out.resolve(), out.stat().st_size / 1024 / 1024))
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
