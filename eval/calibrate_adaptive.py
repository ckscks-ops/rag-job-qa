# -*- coding: utf-8 -*-
"""
calibrate_adaptive.py —— 给"自适应精排"的阈值定标

背景
====
自适应精排的想法：两路召回高度一致时跳过 cross-encoder，省掉那 8~13 秒。
但阈值不能拍脑袋——第一版用 chunk 级 Jaccard + 阈值 0.5，
跑了 199 条查询**一次都没触发**，等于白做还多花时间。

本脚本做两件事：
  1. 统计「两路一致度」的分布（岗位级），看阈值定在哪合理
  2. **验证这个信号到底有没有预测力**：一致度低的查询，精排带来的提升是否真的更大？
     —— 如果没有相关性，那这个启发式根本不成立，应该换思路或直接放弃。

用法
----
    python eval/calibrate_adaptive.py --sample 60
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
from retrieval import (channel_agreement, fetch_chunks, hybrid_search,  # noqa: E402
                       keyword_search, rrf_fuse, vector_search)
from schema import connect                                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description='自适应精排阈值定标')
    ap.add_argument('--db', default=str(ROOT / 'data' / 'rag.db'))
    ap.add_argument('--sample', type=int, default=60, help='抽样条数（0=全部）')
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--out', default=str(ROOT / 'eval' / 'adaptive_calibration.json'))
    args = ap.parse_args()

    con = connect(args.db)
    ev = materialize_eval_set(con, load_eval_set(con, ROOT / 'eval' / 'eval_set.json'))
    if args.sample:
        step = max(1, len(ev) // args.sample)
        ev = ev[::step][:args.sample]

    embed_fn = get_embedder('fastembed', model_name='BAAI/bge-small-zh-v1.5')
    from rerank import get_reranker, rerank_hits

    print('=' * 76)
    print('自适应精排阈值定标')
    print('=' * 76)
    print('  抽样: %d 条查询' % len(ev))
    print('  正在加载重排模型…')
    rr = get_reranker()
    print('  就绪')
    print()
    print('  逐条计算：一致度 / 混合指标 / 精排指标 / 精排耗时')
    print('  %-4s %-8s %-8s %-8s %-8s %s' % ('#', '一致度', 'hyb P@5', 'rr P@5', 'ΔP@5', '精排ms'))
    print('  ' + '-' * 62)

    rows = []
    t_all = time.perf_counter()
    for i, case in enumerate(ev, 1):
        q = case['query']
        rel = set(case['relevant_keys'])
        qvec = embed_fn(q)

        kw = keyword_search(con, q, top_k=RECALL_K)
        vec = vector_search(con, qvec, top_k=RECALL_K)
        pool = rrf_fuse([kw, vec], k=60, top_k=RECALL_K)
        agree = channel_agreement(con, kw, vec, k=5)

        def doc_keys(hits):
            det = fetch_chunks(con, [h['chunk_id'] for h in hits])
            seen, out = set(), []
            for d in det:
                k2 = d.get('doc_key') or d['chunk_id']
                if k2 not in seen:
                    seen.add(k2)
                    out.append(k2)
            return out

        m_hyb = metrics_for(doc_keys(pool), rel, args.top_k)

        t0 = time.perf_counter()
        rr_hits = rerank_hits(con, q, pool, rr, top_k=RECALL_K)
        dt = (time.perf_counter() - t0) * 1000
        m_rr = metrics_for(doc_keys(rr_hits), rel, args.top_k)

        rows.append({'query': q, 'category': case.get('category', '?'),
                     'agree': round(agree, 4),
                     'p_hybrid': m_hyb['precision'], 'p_rerank': m_rr['precision'],
                     'mrr_hybrid': m_hyb['rr'], 'mrr_rerank': m_rr['rr'],
                     'rerank_ms': round(dt, 1), 'n_relevant': len(rel)})
        print('  %-4d %-8.3f %-8.3f %-8.3f %+-8.3f %.0f'
              % (i, agree, m_hyb['precision'], m_rr['precision'],
                 m_rr['precision'] - m_hyb['precision'], dt))

    print()
    print('  总耗时 %.1f 分钟' % ((time.perf_counter() - t_all) / 60))

    ags = sorted(r['agree'] for r in rows)
    print()
    print('=' * 76)
    print('① 一致度分布（岗位级 Jaccard）')
    print('=' * 76)
    print('  min=%.3f  p25=%.3f  中位=%.3f  p75=%.3f  p90=%.3f  max=%.3f'
          % (ags[0], ags[len(ags) // 4], statistics.median(ags),
             ags[len(ags) * 3 // 4], ags[int(len(ags) * .9)], ags[-1]))
    print()
    print('  阈值 -> 会跳过多少条')
    for th in [0.0, 0.1, 0.2, 0.33, 0.5, 0.67]:
        n = sum(1 for a in ags if a >= th)
        print('    >= %.2f : 跳过 %3d / %d  (%.0f%%)' % (th, n, len(ags), 100.0 * n / len(ags)))

    print()
    print('=' * 76)
    print('② 这个信号有预测力吗？（一致度 vs 精排带来的提升）')
    print('=' * 76)
    # 按一致度分两半，比较精排的平均增益
    mid = statistics.median(ags)
    low = [r for r in rows if r['agree'] <= mid]
    high = [r for r in rows if r['agree'] > mid]
    for name, grp in (('一致度低的一半', low), ('一致度高的一半', high)):
        if not grp:
            continue
        d_p = statistics.mean(r['p_rerank'] - r['p_hybrid'] for r in grp)
        d_m = statistics.mean(r['mrr_rerank'] - r['mrr_hybrid'] for r in grp)
        print('  %-14s n=%-3d  精排让 P@5 %+.3f，MRR %+.3f'
              % (name, len(grp), d_p, d_m))

    # 相关系数（皮尔逊）
    xs = [r['agree'] for r in rows]
    ys = [r['p_rerank'] - r['p_hybrid'] for r in rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    corr = num / den if den else 0.0
    print()
    print('  相关系数 corr(一致度, ΔP@5) = %+.3f' % corr)
    if abs(corr) < 0.15:
        print('  → 相关性很弱：**一致度这个信号预测不了"精排有没有用"**，')
        print('    自适应精排这个思路不成立，应该换判据或直接放弃。')
    elif corr < 0:
        print('  → 负相关：一致度越低，精排提升越大 —— 假设成立，可以按阈值跳过。')
    else:
        print('  → 正相关：与假设相反（一致度高时精排反而提升更多），思路不成立。')

    out = Path(args.out)
    out.write_text(json.dumps({
        'sample': len(rows), 'top_k': args.top_k, 'recall_k': RECALL_K,
        'agree_dist': {'min': ags[0], 'median': statistics.median(ags), 'max': ags[-1]},
        'corr_agree_delta_p': round(corr, 4),
        'rows': rows,
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    print()
    print('  已写出: %s' % out)
    con.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
