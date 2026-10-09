# -*- coding: utf-8 -*-
"""
rerank_budget.py —— 精排延迟预算实验

问题
====
cross-encoder 精排把 Precision@5 从 0.465 拉到 0.627，但平均延迟 13 秒，
交互式搜索完全不能用。要压到 1~2 秒，能动的杠杆有三个：

    1. **候选池大小**（rerank_pool）：线性影响耗时
    2. **送入精排的文本长度**（max_chars）：截断后耗时近似线性下降
    3. 换更小的重排模型：要重新下 1 GB，且换模型后评测基线全变，先不动

本脚本把 1 和 2 的网格跑一遍，给出「精度 vs 延迟」曲线，
用数据决定取哪组参数，而不是拍脑袋。

**另外一个被证伪的思路**（记录在这里免得以后重复踩）：
    曾经想「两路召回一致时就跳过精排」。实测 199 条查询里，
    岗位级 Jaccard 一致度**中位数是 0.000**（最高 0.429），
    也就是绝大多数查询下 BM25 和向量的 top-5 **没有任何重合岗位**。
    信号压根不存在，阈值定多少都跳不过去。详见 eval/calibrate_adaptive.py。

用法
----
    python eval/rerank_budget.py --sample 60
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'eval'))

from embeddings import get_embedder                        # noqa: E402
from evaluate import (RECALL_K, load_eval_set, materialize_eval_set,   # noqa: E402
                      metrics_for)
from retrieval import fetch_chunks, keyword_search, rrf_fuse, vector_search  # noqa: E402
from schema import connect                                 # noqa: E402

# (候选池, 截断字符数)  None = 不截断
CONFIGS = [(30, None), (30, 200), (20, 200), (10, 200), (10, 300)]


def doc_keys_of(con, hits) -> list[str]:
    det = fetch_chunks(con, [h['chunk_id'] for h in hits])
    seen, out = set(), []
    for d in det:
        k = d.get('doc_key') or d['chunk_id']
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='精排延迟预算实验')
    ap.add_argument('--db', default=str(ROOT / 'data' / 'rag.db'))
    ap.add_argument('--eval-set', default=str(ROOT / 'eval' / 'eval_set.json'))
    ap.add_argument('--sample', type=int, default=60, help='抽样条数（0=全部）')
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--out', default=str(ROOT / 'eval' / 'rerank_budget.json'))
    args = ap.parse_args()

    con = connect(args.db)
    ev = materialize_eval_set(
        con, load_eval_set(con, Path(args.eval_set)))
    if args.sample:
        step = max(1, len(ev) // args.sample)
        ev = ev[::step][:args.sample]

    embed_fn = get_embedder('fastembed', model_name='BAAI/bge-small-zh-v1.5')
    from rerank import get_reranker, rerank_hits

    print('=' * 76)
    print('精排延迟预算实验')
    print('=' * 76)
    print('  抽样: %d 条查询' % len(ev))
    print('  正在加载重排模型…')
    rr = get_reranker()
    print('  就绪')
    print()

    # 召回只算一次，后面所有配置复用（省时间，也保证比较公平）
    print('  预计算召回池（recall_k=%d）…' % RECALL_K)
    t0 = time.perf_counter()
    pools: list[tuple[dict, list[dict], list[str]]] = []
    for case in ev:
        q = case['query']
        kw = keyword_search(con, q, top_k=RECALL_K)
        vec = vector_search(con, embed_fn(q), top_k=RECALL_K)
        fused = rrf_fuse([kw, vec], k=60, top_k=RECALL_K)
        pools.append((case, fused, doc_keys_of(con, fused)))
    t_recall = time.perf_counter() - t0
    print('  完成，%.1f 秒（平均 %.0f ms/条）' % (t_recall, t_recall / len(pools) * 1000))
    print()
    print('  %-14s %-8s %-9s %-8s %-8s %-10s' %
          ('配置(池,截断)', 'Hit@5', 'Prec@5', 'MRR', '精排ms/条', '总ms/条'))
    print('  ' + '-' * 62)

    # ⚠️ 逐条查询**交错**跑完所有配置，而不是一个配置跑完所有查询。
    #
    # 为什么（2026-10-09 踩坑）：
    #   第一版是"外层循环配置"，于是最慢的 (30, 不截断) 先把 CPU 打热/打完缓存，
    #   后面几组配置是在**完全不同的机器状态**下测的，
    #   结果算出"截断到 200 字省 85% 时间"——比单独直接测量（2.04x）夸张了一倍多。
    #   交错测能让四组配置面对大致相同的机器状态，代价是召回结果要缓存（已经缓存了）。
    acc = {i: {'hit': [], 'prec': [], 'mrr': [], 'lat': []}
           for i in range(len(CONFIGS))}
    for j, (case, fused, _) in enumerate(pools, 1):
        rel = set(case['relevant_keys'])
        for i, (pool_k, max_chars) in enumerate(CONFIGS):
            sub = fused[:pool_k]
            t1 = time.perf_counter()
            hits = rerank_hits(con, case['query'], sub, rr,
                               top_k=args.top_k, max_chars=max_chars)
            dt = (time.perf_counter() - t1) * 1000
            m = metrics_for(doc_keys_of(con, hits), rel, args.top_k)
            acc[i]['hit'].append(m['hit'])
            acc[i]['prec'].append(m['precision'])
            acc[i]['mrr'].append(m['rr'])
            acc[i]['lat'].append(dt)
        if j % 10 == 0:
            print('    … %d / %d' % (j, len(pools)), flush=True)

    results = []
    for i, (pool_k, max_chars) in enumerate(CONFIGS):
        a = acc[i]
        row = {
            'pool': pool_k,
            'max_chars': max_chars,
            'hit': statistics.mean(a['hit']),
            'precision': statistics.mean(a['prec']),
            'mrr': statistics.mean(a['mrr']),
            'rerank_ms': statistics.mean(a['lat']),
            'total_ms': statistics.mean(a['lat']) + 700,   # + 召回约 700ms
        }
        results.append(row)
    for r in results:
        print('  %-14s %-8.3f %-9.3f %-8.3f %-10.0f %.0f'
              % ('(%d, %s)' % (r['pool'], r['max_chars'] or '不截断'),
                 r['hit'], r['precision'], r['mrr'],
                 r['rerank_ms'], r['total_ms']))

    base = results[0]
    print()
    print('=' * 76)
    print('结论')
    print('=' * 76)
    for r in results[1:]:
        print('  池 %2d + 截断 %-6s : Precision@5 %+.3f（%.1f%%），延迟降到 %.0f%%（%.1f s → %.1f s）'
              % (r['pool'], r['max_chars'] or '无',
                 r['precision'] - base['precision'],
                 100.0 * (r['precision'] - base['precision']) / base['precision'],
                 100.0 * r['rerank_ms'] / base['rerank_ms'],
                 base['total_ms'] / 1000, r['total_ms'] / 1000))

    best = max(results, key=lambda r: r['precision'] - 0.0000)
    print()
    print('  精度最高 : 池 %d + 截断 %s（P@5 %.3f）'
          % (best['pool'], best['max_chars'] or '无', best['precision']))
    # 在"精度不低于最高值 0.03"的前提下取最快的
    ok = [r for r in results if r['precision'] >= best['precision'] - 0.03]
    fast = min(ok, key=lambda r: r['rerank_ms'])
    print('  推荐     : 池 %d + 截断 %s（P@5 %.3f，延迟 %.1f s）'
          % (fast['pool'], fast['max_chars'] or '无', fast['precision'],
             fast['total_ms'] / 1000))
    print('             —— 在"精度损失不超过 0.03"的前提下取最快的这组')

    Path(args.out).write_text(json.dumps({
        'sample': len(ev), 'top_k': args.top_k, 'recall_k': RECALL_K,
        'results': results, 'recommended': fast,
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    print()
    print('  已写出: %s' % args.out)
    con.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
