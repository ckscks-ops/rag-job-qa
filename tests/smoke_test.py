"""
smoke_test.py —— 全链路自测

验证：分词 / 切块 / 关键词检索 / 向量检索 / RRF 融合 / 结果补全
每一步都断言，失败就非零退出，便于以后改代码后回归。

⚠️ 2026-10-08 修正：本测试原先直接用 `data/rag.db`，于是**依赖入库状态**——
   一旦这个库换成真实语料 + 512 维 fastembed 向量，用 256 维 hash 查询就
   维度不匹配，"向量检索有结果"必然失败（实测踩到）。
   现在改为**自建隔离临时库**（示例数据 + 256 维 hash 向量），
   与 data/rag.db 的实际内容完全解耦。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from chunking import chunk_jd                                        # noqa: E402
from embeddings import get_embedder                                   # noqa: E402
from retrieval import fetch_chunks, hybrid_search, keyword_search, rrf_fuse, vector_search  # noqa: E402
from schema import connect, init_db, insert_chunk, insert_document, stats, tokenize  # noqa: E402

PASS, FAIL = [], []

TMPDIR: Path | None = None


def check(name: str, ok: bool, detail: str = ''):
    (PASS if ok else FAIL).append(name)
    print('  [%s] %s%s' % ('OK  ' if ok else 'FAIL', name,
                           ('   (%s)' % detail) if detail else ''))


def build_isolated_db() -> tuple[Path, object]:
    """用示例数据建一个隔离的临时库（256 维 hash 向量）。"""
    global TMPDIR
    TMPDIR = Path(tempfile.mkdtemp(prefix='rag_smoke_'))
    db = TMPDIR / 'rag.db'
    con = connect(db)
    init_db(con)

    jobs = json.loads((ROOT / 'data' / 'sample_jobs.json').read_text(encoding='utf-8'))
    embed = get_embedder('hash', dim=256)
    for job in jobs:
        raw = (job.get('raw_text') or '').strip()
        if not raw:
            continue
        key = job.get('doc_key') or '|'.join(
            str(job.get(k)) for k in ('company', 'title', 'city', 'job_type') if job.get(k))
        did = insert_document(con, doc_key=key, raw_text=raw,
                              company=job.get('company'), title=job.get('title'),
                              city=job.get('city'), job_type=job.get('job_type'),
                              source=job.get('source'), metadata={})
        for c in chunk_jd(raw, company=job.get('company'), title=job.get('title')):
            insert_chunk(con, doc_id=did, chunk_index=c['index'], text=c['text'],
                         section=c['section'],
                         embedding=embed(c['text']).tobytes(),
                         embedding_model='hash-placeholder')
    con.commit()
    return db, con


def main() -> int:
    print('=' * 72)
    print('RAG 全链路自测')
    print('=' * 72)
    db, con = build_isolated_db()
    print('  隔离临时库: %s' % db)
    s = stats(con)
    print('  统计: %s' % s)
    print()

    # ── 1. 分词 ──
    print('--- 1. 分词 ---')
    toks = tokenize('数据分析师，负责用户分层与 A/B 实验')
    check('分词产生多个词', len(toks.split()) >= 4, '%d 个词' % len(toks.split()))
    check('领域词未被切碎', '数据分析' in toks or '数据分析师' in toks, toks[:40])

    # ── 2. 切块 ──
    print()
    print('--- 2. 切块 ---')
    jd = ('岗位职责：\n1. 负责数据清洗与建模；\n2. 输出分析报告。\n'
          '任职要求：\n1. 熟练使用 SQL；\n2. 掌握 Python 数据分析。\n'
          '加分项：\n有大数据平台经验者优先。')
    cs = chunk_jd(jd, company='测试公司', title='数据分析师')
    check('切块非空', len(cs) >= 2, '%d 块' % len(cs))
    check('识别出资历段名', any(c['section'] in ('岗位职责', '任职要求', '加分项') for c in cs),
          str([c['section'] for c in cs]))
    check('块文本带公司·岗位前缀', all(c['text'].startswith('测试公司 · 数据分析师') for c in cs))

    # ── 3. 关键词检索 ──
    print()
    print('--- 3. 关键词检索（BM25）---')
    hits = keyword_search(con, 'SQL 数据分析', top_k=5, debug=True)
    check('关键词检索有结果', len(hits) > 0, '%d 条' % len(hits))
    if hits:
        det = fetch_chunks(con, [h['chunk_id'] for h in hits])
        titles = [d['title'] for d in det]
        check('命中数据分析相关岗位',
              any('数据' in (t or '') for t in titles), str(titles[:3]))

    hits2 = keyword_search(con, '嵌入式 单片机 STM32', top_k=5)
    check('能检索到嵌入式岗位（对照）', len(hits2) > 0, '%d 条' % len(hits2))

    # ── 4. 向量检索 ──
    print()
    print('--- 4. 向量检索 ---')
    embed = get_embedder('hash', dim=256)
    qv = embed('数据分析 SQL')
    vhits = vector_search(con, qv, top_k=5, debug=True)
    check('向量检索有结果', len(vhits) > 0, '%d 条' % len(vhits))
    check('向量维度匹配', qv.shape[0] == 256, str(qv.shape))

    # 维度不匹配时应安全返回空
    bad = vector_search(con, get_embedder('hash', dim=64)('x'), top_k=3)
    check('维度不匹配时安全退化', bad == [], '%d 条' % len(bad))

    # ── 5. RRF 融合 ──
    print()
    print('--- 5. RRF 融合 ---')
    a = [{'chunk_id': 1}, {'chunk_id': 2}, {'chunk_id': 3}]
    b = [{'chunk_id': 3}, {'chunk_id': 1}, {'chunk_id': 4}]
    fused = rrf_fuse([a, b], k=60, top_k=4)
    ids = [f['chunk_id'] for f in fused]
    check('融合结果非空', len(fused) == 4)
    check('两路都靠前的排最前', ids[0] in (1, 3), str(ids))
    check('记录了来源通道', all('from' in f for f in fused))

    # ── 6. 混合检索端到端 ──
    print()
    print('--- 6. 混合检索（端到端）---')
    hh = hybrid_search(con, 'SQL 数据分析', top_k=5, query_vec=qv, debug=True)
    check('混合检索有结果', len(hh) > 0, '%d 条' % len(hh))
    det = fetch_chunks(con, [h['chunk_id'] for h in hh])
    check('结果含岗位与公司字段',
          all(d.get('company') and d.get('title') for d in det) if det else False)
    if det:
        print('       Top-3 命中:')
        for d in det[:3]:
            print('         %-16s %-20s %s'
                  % (d['company'], d['title'], (d['section'] or '-')))

    # ── 7. 无向量时退化 ──
    print()
    print('--- 7. 无向量时自动退化为关键词 ---')
    h3 = hybrid_search(con, 'SQL', top_k=3, query_vec=None)
    check('无向量也能返回结果', len(h3) > 0, '%d 条' % len(h3))

    con.close()
    if TMPDIR is not None:
        shutil.rmtree(TMPDIR, ignore_errors=True)

    print()
    print('=' * 72)
    print('通过 %d / %d' % (len(PASS), len(PASS) + len(FAIL)))
    if FAIL:
        print('❌ 失败项: %s' % ', '.join(FAIL))
    else:
        print('✅ 全部通过')
    print('=' * 72)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
