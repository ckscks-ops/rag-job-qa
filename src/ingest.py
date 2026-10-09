"""
ingest.py —— 把岗位数据入库（切块 + 建立索引 + 可选向量化）

用法
----
    # 用示例数据（零依赖，立即可跑）
    python src/ingest.py --input data/sample_jobs.json

    # 接入真实岗位数据（RecruitOps 导出的 JSON，字段同构即可）
    python src/ingest.py --input data/prod_jobs.json --db data/rag.db

    # 同时计算向量（用占位 embedder；换成真实模型见 embeddings.py）
    python src/ingest.py --input data/sample_jobs.json --embed hash --dim 256

输入 JSON 格式（数组，每项一个岗位）
    [{ "doc_key": "...", "company": "...", "title": "...",
       "city": "...", "job_type": "...", "source": "...",
       "raw_text": "JD 原文" }, ...]

其中 doc_key 是业务唯一键，用于幂等重跑（同 key 覆盖更新）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# 允许从项目根或 src 目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunking import chunk_jd                    # noqa: E402
from embeddings import embedder_info, get_embedder  # noqa: E402
from schema import connect, init_db, insert_chunk, insert_document, stats  # noqa: E402


def load_jobs(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, dict):
        data = data.get('jobs') or data.get('data') or []
    if not isinstance(data, list):
        raise ValueError('输入必须是 JSON 数组，或含 jobs/data 字段的对象')
    return data


def make_doc_key(job: dict) -> str:
    """没有 doc_key 时，用 公司|岗位|城市|类型 合成。"""
    if job.get('doc_key'):
        return str(job['doc_key'])
    parts = [job.get('company'), job.get('title'), job.get('city'), job.get('job_type')]
    return '|'.join(str(p) for p in parts if p)


def main() -> int:
    ap = argparse.ArgumentParser(description='岗位数据入库')
    ap.add_argument('--input', required=True, help='岗位 JSON 文件')
    ap.add_argument('--db', default='data/rag.db', help='SQLite 数据库路径')
    ap.add_argument('--embed',
                    choices=['none', 'hash', 'fastembed', 'fe', 'onnx', 'st', 'local'],
                    default='none',
                    help='向量化方式：fastembed=ONNX 中文模型（推荐）；hash=占位实现')
    ap.add_argument('--model', default='BAAI/bge-small-zh-v1.5',
                    help='embedding 模型名（fastembed / st 用）')
    ap.add_argument('--dim', type=int, default=256, help='占位 embedding 维度')
    ap.add_argument('--batch-size', type=int, default=64, help='批量向量化批大小')
    ap.add_argument('--reset', action='store_true', help='入库前清空现有数据')
    ap.add_argument('--allow-dup-keys', action='store_true',
                    help='允许 doc_key 重复（默认直接报错退出，因为会导致岗位被静默合并）')
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        print('  [错误] 输入文件不存在: %s' % inp)
        return 1

    jobs = load_jobs(inp)

    # ⚠️ doc_key 唯一性必须在入库前拦住（2026-10-09 踩坑）
    #    documents.doc_key 是 UNIQUE，insert_document 用 ON CONFLICT DO UPDATE，
    #    所以重复的 doc_key 会让不同岗位**静默合并成一个**；
    #    合并后同一 (doc_id, chunk_index) 会被插两次，进而触发 FTS 表的删除报错。
    #    与其等它在中途崩掉，不如在开头就说清楚。
    keys = [make_doc_key(j) for j in jobs
            if (j.get('raw_text') or j.get('text') or '').strip()]
    dupes = sorted(k for k, n in Counter(keys).items() if n > 1)
    if dupes:
        print('  [错误] 输入里有 %d 个重复的 doc_key，岗位会被静默合并：' % len(dupes))
        for k in dupes[:5]:
            print('        %s' % k[:100])
        if not args.allow_dup_keys:
            print()
            print('  修数据，或加 --allow-dup-keys 强制继续（不推荐）。')
            return 1
        print('  （--allow-dup-keys 已指定，继续）')

    print('=' * 70)
    print('岗位数据入库')
    print('=' * 70)
    print('  输入: %s  (%d 个岗位)' % (inp, len(jobs)))
    print('  数据库: %s' % args.db)

    con = connect(args.db)
    init_db(con)

    if args.reset:
        from schema import reset_db
        reset_db(con)
        print('  已清空现有数据（--reset）')

    # 准备 embedder
    embed_fn = None
    einfo = {}
    if args.embed != 'none':
        embed_fn = get_embedder(args.embed, dim=args.dim, model_name=args.model)
        einfo = embedder_info(embed_fn)
        print('  向量: provider=%s model=%s dim=%d'
              % (args.embed, einfo['model'], einfo['dim']))
        if args.embed == 'hash':
            print('        ⚠️ 占位实现，无真正语义能力，仅用于打通链路')

    t0 = time.perf_counter()
    n_docs = n_chunks = n_vec = 0
    skipped = 0
    pending: list[tuple[int, str]] = []      # (chunk_id, text) 待向量化

    # ── 阶段一：文档 + 切块入库（向量先留空，便于批量计算）──
    for job in jobs:
        key = make_doc_key(job)
        raw = (job.get('raw_text') or job.get('text') or '').strip()
        if not key or not raw:
            skipped += 1
            continue

        doc_id = insert_document(
            con, doc_key=key, raw_text=raw,
            company=job.get('company'), title=job.get('title'),
            city=job.get('city'), job_type=job.get('job_type'),
            source=job.get('source'),
            metadata={k: v for k, v in job.items()
                      if k not in ('raw_text', 'text')},
            commit=False,
        )
        n_docs += 1

        pieces = chunk_jd(raw, company=job.get('company'), title=job.get('title'))
        for c in pieces:
            cid = insert_chunk(con, doc_id=doc_id, chunk_index=c['index'],
                               text=c['text'], section=c['section'],
                               embedding=None, embedding_model=None,
                               commit=False)
            pending.append((cid, c['text']))
            n_chunks += 1
        if n_docs % 200 == 0:
            con.commit()
            print('  已入库 %d / %d 岗位 …' % (n_docs, len(jobs)))
    con.commit()
    t_insert = time.perf_counter() - t0

    # ── 阶段二：批量向量化并回写 ──
    t_vec = 0.0
    if embed_fn is not None and pending:
        from schema import update_chunk_embeddings_bulk
        tv0 = time.perf_counter()
        batch_fn = getattr(embed_fn, 'embed_batch', None)
        model_name = einfo.get('model')
        bs = max(1, args.batch_size)
        print()
        print('  开始向量化 %d 个切块（batch=%d）…' % (len(pending), bs))
        for i in range(0, len(pending), bs):
            part = pending[i:i + bs]
            texts = [t for _, t in part]
            if batch_fn is not None:
                mat = batch_fn(texts)
            else:
                import numpy as _np
                mat = _np.stack([embed_fn(t) for t in texts])
            rows = [(mat[j].astype('float32').tobytes(), model_name, part[j][0])
                    for j in range(len(part))]
            update_chunk_embeddings_bulk(con, rows)
            n_vec += len(rows)
            if (i // bs) % 20 == 0:
                done = min(i + bs, len(pending))
                print('    向量化 %d / %d  (%.1f 秒)'
                      % (done, len(pending), time.perf_counter() - tv0))
        t_vec = time.perf_counter() - tv0

    dt = time.perf_counter() - t0
    s = stats(con)

    print()
    print('=== 入库结果 ===')
    print('  处理岗位: %d  (跳过 %d)' % (n_docs, skipped))
    print('  切块总数: %d  平均每岗位 %.1f 块'
          % (n_chunks, n_chunks / n_docs if n_docs else 0))
    print('  计算向量: %d' % n_vec)
    print('  耗时    : %.2f 秒  (入库 %.2f 秒 + 向量化 %.2f 秒)'
          % (dt, t_insert, t_vec))
    print()
    print('=== 库内统计 ===')
    for k, v in s.items():
        print('  %-16s %s' % (k, v))

    con.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
