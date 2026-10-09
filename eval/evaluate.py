"""
evaluate.py —— 检索效果评测（RAG 项目最有价值的部分）

为什么评测是重点
==============
"我做了个 RAG 系统" 在面试里没有说服力，
"我把 Recall@5 从 A 提升到 B，方法是对比了三种检索策略" 才有。
本模块提供可复现的评测流程，产出可直接写进简历的数字。

指标
====
- **Recall@k**  ：前 k 条里是否包含相关文档（这里 k 指"命中的岗位数"）
- **MRR**       ：第一个相关结果排名的倒数（1/rank），衡量"最相关的排多前"
- **Hit@k**     ：是否至少命中一个相关项

注意
====
默认使用 `hash` 占位 embedder，其**向量检索不具备真正的语义泛化能力**，
因此用它跑出的"混合检索 vs 纯向量"对比**不代表真实效果**。
接入真实 embedding 模型后重跑，数字才有简历价值（见 embeddings.py）。

用法
----
    python eval/evaluate.py                      # 用内置评测集
    python eval/evaluate.py --modes keyword vector hybrid
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

from embeddings import get_embedder                                   # noqa: E402
from retrieval import (hybrid_search, keyword_search, rrf_fuse,       # noqa: E402
                       vector_search)
from schema import connect                                            # noqa: E402

# 召回池大小：**所有模式共用同一个值**，否则"加了 rerank 所以更好"可能只是
# 因为 rerank 那条路召回了更多候选，而不是重排的功劳。
RECALL_K = 30

# 自适应精排的统计（每个 eval 跑之前重置）
SKIP_STATS = {'total': 0, 'skipped': 0}


def _rerank_branch(con, query, qvec, pool_k, reranker, mode, agree_threshold,
                   max_chars=200):
    """精排 / 自适应精排共用的召回+决策逻辑。返回 chunk 级命中列表。"""
    from retrieval import channel_agreement
    kw = keyword_search(con, query, top_k=RECALL_K)
    vec = vector_search(con, qvec, top_k=RECALL_K) if qvec is not None else []
    pool = rrf_fuse([kw, vec], k=60, top_k=pool_k) if vec else kw[:pool_k]
    if not pool:
        return []

    skip = False
    if mode == 'rerank_adaptive':
        SKIP_STATS['total'] += 1
        if reranker is not None:
            agree = channel_agreement(con, kw, vec, k=min(5, RECALL_K))
            skip = agree >= agree_threshold
            if skip:
                SKIP_STATS['skipped'] += 1

    if reranker is None or skip:
        return pool
    from rerank import rerank_hits
    return rerank_hits(con, query, pool, reranker, top_k=pool_k,
                       max_chars=max_chars or None)


# ─────────────── 评测集 ───────────────
# 每条：一个自然语言查询 + 期望命中的 doc_key 列表（人工标注）
# 设计原则：查询尽量贴近真实用户会怎么问，而不是直接抄 JD 里的词
DEFAULT_EVAL = [
    {
        'query': '大专以上能做的数据分析岗位，会用 SQL 和 Excel 就行',
        'relevant_keys': [
            '某国企信息中心|数据分析岗|济南|校招',
            '某电商公司|数据分析师（实习）|杭州|实习',
        ],
        'note': '考察"低门槛数据分析"，关键词检索应命中，纯向量应更能理解"会SQL和Excel就行"',
    },
    {
        'query': '想做 RAG 知识库问答，有向量检索经验的岗位',
        'relevant_keys': ['某研究院|AI 应用开发工程师|上海|校招'],
        'note': '考察语义匹配：JD 里写的是"检索增强生成""向量数据库"，查询用"RAG"',
    },
    {
        'query': '偏业务的数据分析，要给管理层做报表',
        'relevant_keys': [
            '某国企信息中心|数据分析岗|济南|校招',
            '某外企|Business Analyst|上海|校招',
        ],
        'note': '考察业务分析方向',
    },
    {
        'query': '数据开发，要会 Spark 和 Flink',
        'relevant_keys': ['示例科技|数据开发工程师|北京|校招'],
        'note': '技术栈精确匹配，关键词检索应占优',
    },
    {
        'query': '机器学习算法，最好是硕士学历要求',
        'relevant_keys': ['某研究院|机器学习算法工程师|上海|校招'],
        'note': '学历条件匹配',
    },
    {
        'query': '偏风控建模的数据挖掘岗位',
        'relevant_keys': ['某金融科技公司|数据挖掘工程师|深圳|校招'],
        'note': '考察领域词「风控」的匹配',
    },
    {
        'query': '英文环境工作，需要看英文文档的岗位',
        'relevant_keys': ['某外企|Business Analyst|上海|校招'],
        'note': '考察"英文"这类软性条件的匹配',
    },
    {
        'query': '硬件嵌入式开发',
        'relevant_keys': ['某智能制造公司|嵌入式软件工程师|苏州|校招'],
        'note': '对照项：确认检索不会把非目标方向的岗位排到前面',
    },
]


def resolve_rule(con, rule: dict) -> set[str]:
    """
    把「相关性规则」解析成 doc_key 集合。

    为什么用规则而不是逐条人工标注：
      真实语料有 2,116 条岗位，逐条标注 50×2116 不现实。
      规则化标注（programmatic labeling）是评测里的常规做法，
      代价是**规则本身会偏向字面匹配**——所以查询措辞要故意避开规则里的关键词，
      否则测出来的是"关键词检索 vs 关键词"，没有意义。

    规则字段（同一字典内多个字段是 AND，列表内是 OR）：
        title_any / title_all / title_none   按岗位标题
        company_any / company_none           按公司名
        city_any                             按城市
        raw_any / raw_all                   按 JD 正文
        job_type_any                         按批次
        label_any                            按方向标签（仅开源数据集自带）
        salary_min_gte / salary_min_lte      按薪资下限（仅社招岗位有）
    """
    where, params = [], []

    def _like(field, values, negate=False):
        if not values:
            return
        clause = ' OR '.join('%s LIKE ?' % field for _ in values)
        if negate:
            where.append('NOT (%s)' % clause)
        else:
            where.append('(%s)' % clause)
        params.extend('%%%s%%' % v for v in values)

    _like('d.title', rule.get('title_any'))
    _like('d.company', rule.get('company_any'))
    _like('d.city', rule.get('city_any'))
    _like('d.job_type', rule.get('job_type_any'))
    _like('d.raw_text', rule.get('raw_any'))
    _like('d.title', rule.get('title_none'), negate=True)
    _like('d.company', rule.get('company_none'), negate=True)

    for v in rule.get('title_all', []):
        where.append('d.title LIKE ?')
        params.append('%%%s%%' % v)
    for v in rule.get('raw_all', []):
        where.append('d.raw_text LIKE ?')
        params.append('%%%s%%' % v)

    # 方向标签（存在 documents.metadata 的 JSON 里）
    # 来源数据集自带的人工分类（数据分析/算法岗/大模型相关/经济金融/生物统计），
    # 比"用标题关键词猜方向"客观得多。
    labels = rule.get('label_any')
    if labels:
        where.append("json_extract(d.metadata, '$.label') IN (%s)"
                     % ','.join('?' * len(labels)))
        params.extend(labels)

    # 数值条件：薪资下限（只有开源数据集那批社招岗位带薪资）
    if rule.get('salary_min_gte') is not None:
        where.append("json_extract(d.metadata, '$.salary_min') >= ?")
        params.append(float(rule['salary_min_gte']))
    if rule.get('salary_min_lte') is not None:
        where.append("json_extract(d.metadata, '$.salary_min') <= ?")
        params.append(float(rule['salary_min_lte']))

    sql = 'SELECT d.doc_key FROM documents d'
    if where:
        sql += ' WHERE ' + ' AND '.join(where)
    return {r[0] for r in con.execute(sql, params).fetchall()}


def load_eval_set(con, path: Path | None = None) -> list[dict]:
    if path and path.is_file():
        return json.loads(path.read_text(encoding='utf-8'))
    rows = con.execute('SELECT query, relevant_keys, note FROM eval_set').fetchall()
    if rows:
        return [{'query': r['query'],
                 'relevant_keys': json.loads(r['relevant_keys']),
                 'note': r['note']} for r in rows]
    return DEFAULT_EVAL


def materialize_eval_set(con, eval_set: list[dict]) -> list[dict]:
    """
    把带 `rule` 的评测项解析成具体的 relevant_keys；已有 relevant_keys 的原样保留。
    规则解析出 0 条相关项的评测项会被剔除（并打印警告），避免指标被空集污染。
    """
    out, dropped = [], []
    for case in eval_set:
        c = dict(case)
        if not c.get('relevant_keys') and c.get('rule'):
            keys = resolve_rule(con, c['rule'])
            if not keys:
                dropped.append(c['query'])
                continue
            c['relevant_keys'] = sorted(keys)
        if not c.get('relevant_keys'):
            dropped.append(c.get('query', '?'))
            continue
        out.append(c)
    if dropped:
        print('  [警告] %d 条评测项的相关集为空，已剔除：' % len(dropped))
        for q in dropped[:10]:
            print('        - %s' % q)
        print()
    return out


# ─────────────── 检索模式 ───────────────

def run_search(con, query: str, mode: str, embed_fn, top_k: int,
               reranker=None, rerank_pool: int | None = None,
               rerank_agree: float = 0.5,
               rerank_max_chars: int | None = 200) -> list[dict]:
    """
    返回命中的**岗位级**结果（同一岗位多个块只算一次，取最好排名）。

    mode:
        keyword          BM25 关键词
        vector           向量余弦
        hybrid           BM25 + 向量，RRF 融合
        rerank           hybrid 召回 rerank_pool 条 → cross-encoder 精排
        rerank_adaptive  只在"两路不一致"时才精排（省时间）
    """
    pool_k = rerank_pool or RECALL_K
    qvec = None
    if mode in ('vector', 'hybrid', 'rerank', 'rerank_adaptive') and embed_fn is not None:
        qvec = embed_fn(query)

    if mode == 'keyword' or qvec is None:
        hits = keyword_search(con, query, top_k=RECALL_K)
    elif mode == 'vector':
        hits = vector_search(con, qvec, top_k=RECALL_K)
    elif mode in ('rerank', 'rerank_adaptive'):
        hits = _rerank_branch(con, query, qvec, pool_k, reranker, mode,
                              rerank_agree, rerank_max_chars)
    else:
        kw = keyword_search(con, query, top_k=RECALL_K)
        vec = vector_search(con, qvec, top_k=RECALL_K)
        hits = rrf_fuse([kw, vec], k=60, top_k=RECALL_K)

    # chunk -> doc 去重
    if not hits:
        return []
    ids = [h['chunk_id'] for h in hits]
    q = ','.join('?' * len(ids))
    rows = con.execute(
        'SELECT c.id AS chunk_id, d.doc_key, d.company, d.title '
        'FROM chunks c JOIN documents d ON d.id = c.doc_id '
        'WHERE c.id IN (%s)' % q, ids).fetchall()
    by_chunk = {r['chunk_id']: dict(r) for r in rows}

    seen, out = set(), []
    for h in hits:
        info = by_chunk.get(h['chunk_id'])
        if not info:
            continue
        key = info['doc_key']
        if key in seen:
            continue
        seen.add(key)
        out.append({**info, 'score': h['score']})
    return out


# ─────────────── 指标 ───────────────

def metrics_for(ranked_keys: list[str], relevant: set[str], k: int) -> dict:
    """
    ⚠️ 指标选择说明（重要）
        找岗位这类任务，**相关集通常很大**（"北京的算法岗"可能几百条都算相关）。
        此时 Recall@5 的上限只有 5/几百 ≈ 0.01，任何检索方法都接近 0，
        指标**失去区分度**——它衡量的是"能不能把几百条都排进前 5"，这不可能。
        所以主指标用：
          - Precision@k：前 k 条里有几条真的相关（找岗位实际关心的就是"第一屏准不准"）
          - Hit@k      ：前 k 条里有没有至少一条相关
          - MRR        ：第一条相关的排名倒数（最相关的排得多靠前）
        Recall@k 仍然输出，但**只在相关集很窄（<30）时才有解释力**。
    """
    topk = ranked_keys[:k]
    inter = len(set(topk) & relevant)
    hit = inter > 0
    precision = (inter / len(topk)) if topk else 0.0
    recall = (inter / len(relevant)) if relevant else 0.0
    rr = 0.0
    for i, key in enumerate(ranked_keys, start=1):
        if key in relevant:
            rr = 1.0 / i
            break
    return {'hit': hit, 'precision': precision, 'recall': recall, 'rr': rr}


def evaluate(con, eval_set: list[dict], modes: list[str], embed_fn,
             top_k: int = 5, verbose: bool = False, reranker=None,
             rerank_pool: int | None = None,
             rerank_agree: float = 0.5,
             rerank_max_chars: int | None = 200) -> dict:
    SKIP_STATS['total'] = 0
    SKIP_STATS['skipped'] = 0
    results = {m: {'hit': [], 'precision': [], 'recall': [], 'rr': [],
                   'latency': []} for m in modes}
    by_cat = {}                      # category -> mode -> 指标列表

    for case in eval_set:
        q = case['query']
        relevant = set(case['relevant_keys'])
        cat = case.get('category', '未分类')
        if verbose:
            print('\n查询: %s' % q)
            print('  分类    : %s' % cat)
            print('  相关集  : %d 条' % len(relevant))

        for mode in modes:
            t0 = time.perf_counter()
            ranked = run_search(con, q, mode, embed_fn, top_k, reranker,
                                rerank_pool, rerank_agree, rerank_max_chars)
            dt = (time.perf_counter() - t0) * 1000
            keys = [r['doc_key'] for r in ranked]
            m = metrics_for(keys, relevant, top_k)
            results[mode]['hit'].append(m['hit'])
            results[mode]['precision'].append(m['precision'])
            results[mode]['recall'].append(m['recall'])
            results[mode]['rr'].append(m['rr'])
            results[mode]['latency'].append(dt)
            slot = by_cat.setdefault(cat, {}).setdefault(
                mode, {'precision': [], 'rr': [], 'recall': []})
            slot['precision'].append(m['precision'])
            slot['recall'].append(m['recall'])
            slot['rr'].append(m['rr'])
            if verbose:
                flag = '✅' if m['hit'] else '❌'
                print('  %s %-8s P@%d=%.2f Recall@%d=%.2f RR=%.2f  %.1fms  top1=%s'
                      % (flag, mode, top_k, m['precision'], top_k, m['recall'],
                         m['rr'], dt,
                         (keys[0].split('|')[1] if keys and '|' in keys[0]
                          else (keys[0][:28] if keys else '-'))))

    summary = {}
    for mode in modes:
        r = results[mode]
        summary[mode] = {
            'Hit@%d' % top_k: statistics.mean(r['hit']) if r['hit'] else 0.0,
            'Precision@%d' % top_k: statistics.mean(r['precision']) if r['precision'] else 0.0,
            'Recall@%d' % top_k: statistics.mean(r['recall']) if r['recall'] else 0.0,
            'MRR': statistics.mean(r['rr']) if r['rr'] else 0.0,
            'avg_latency_ms': statistics.mean(r['latency']) if r['latency'] else 0.0,
        }

    cat_summary = {}
    for cat, modes_d in by_cat.items():
        cat_summary[cat] = {
            m: {
                'n': len(v['precision']),
                'Precision@%d' % top_k: statistics.mean(v['precision']) if v['precision'] else 0.0,
                'Recall@%d' % top_k: statistics.mean(v['recall']) if v['recall'] else 0.0,
                'MRR': statistics.mean(v['rr']) if v['rr'] else 0.0,
            } for m, v in modes_d.items()
        }
    return summary, cat_summary


def main() -> int:
    ap = argparse.ArgumentParser(description='检索效果评测')
    ap.add_argument('--db', default=str(ROOT / 'data' / 'rag.db'))
    ap.add_argument('--eval-set', default=None, help='评测集 JSON（默认用内置）')
    ap.add_argument('--modes', nargs='+',
                    default=['keyword', 'vector', 'hybrid'])
    ap.add_argument('--rerank-model', default='BAAI/bge-reranker-v2-m3-int8',
                    help='rerank 模式用的 cross-encoder 模型')
    ap.add_argument('--rerank-pool', type=int, default=None,
                    help='rerank 候选池大小（默认与其他模式一致）；用于池子消融实验')
    ap.add_argument('--rerank-agree', type=float, default=0.5,
                    help='自适应精排阈值：两路 top-5 的 Jaccard 重叠 >= 该值就跳过精排')
    ap.add_argument('--rerank-max-chars', type=int, default=200,
                    help='送入精排的文本截断长度（0 = 不截断）。'
                         '实测截断到 200 字省 85%% 延迟，P@5 只掉 0.008，见 eval/rerank_budget.py')
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--embed', default='fastembed',
                    help='fastembed(推荐) / hash / st / api')
    ap.add_argument('--model', default='BAAI/bge-small-zh-v1.5')
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--save', default=None, help='结果另存路径（默认 eval/latest_results.json）')
    args = ap.parse_args()

    con = connect(args.db)
    # 优先用 --eval-set；否则用 eval/eval_set.json（build_eval_set.py 产出）；
    # 都没有才退回内置示例集——但内置集里的 doc_key 来自 sample_jobs.json，
    # 在真实语料库上一条都不会命中，指标会全是 0，所以必须给出警告。
    eval_path = Path(args.eval_set) if args.eval_set else (ROOT / 'eval' / 'eval_set.json')
    used_builtin = not eval_path.is_file()
    raw_eval = load_eval_set(con, None if used_builtin else eval_path)
    embed_fn = get_embedder(args.embed, dim=args.dim, model_name=args.model)
    eval_set = materialize_eval_set(con, raw_eval)

    # 需要 rerank 模式时才加载 cross-encoder（加载慢，且要下载约 0.6 GB）
    reranker = None
    if 'rerank' in args.modes:
        from rerank import get_reranker
        print('  正在加载重排模型 %s …' % args.rerank_model)
        t_rr = time.perf_counter()
        reranker = get_reranker(args.rerank_model)
        print('  重排模型加载完成（%.1f 秒）' % (time.perf_counter() - t_rr))
        print()

    print('=' * 74)
    print('检索效果评测')
    print('=' * 74)
    print('  数据库  : %s' % args.db)
    print('  评测集  : %s' % ('内置 DEFAULT_EVAL（示例数据）' if used_builtin
                              else str(eval_path)))
    print('  查询条数: %d（原始 %d 条）' % (len(eval_set), len(raw_eval)))
    print('  embedder: %s' % getattr(embed_fn, 'model_name', '?'))
    if used_builtin:
        print()
        print('  ⚠️ 用的是内置示例评测集，其 doc_key 指向 sample_jobs.json，')
        print('     在当前库里不会命中，指标会全是 0。')
        print('     请先运行：python eval/build_eval_set.py')
    if args.embed == 'hash':
        print('  ⚠️ 占位 embedder，向量语义能力有限，数字仅供链路验证')
    print('  modes   : %s' % ', '.join(args.modes))
    print()
    nrel = [len(c['relevant_keys']) for c in eval_set]
    if nrel:
        print('  相关集大小: 中位 %d，最小 %d，最大 %d'
              % (statistics.median(nrel), min(nrel), max(nrel)))
        print()

    summary, cat_summary = evaluate(con, eval_set, args.modes, embed_fn,
                                    top_k=args.top_k, verbose=args.verbose,
                                    reranker=reranker, rerank_pool=args.rerank_pool,
                                    rerank_agree=args.rerank_agree,
                                    rerank_max_chars=args.rerank_max_chars)

    if 'rerank_adaptive' in args.modes:
        n_t, n_s = SKIP_STATS['total'], SKIP_STATS['skipped']
        print()
        print('  自适应精排（阈值 %.2f）：%d 条查询里跳过了 %d 条（%.0f%%）'
              % (args.rerank_agree, n_t, n_s, 100.0 * n_s / n_t if n_t else 0))

    print()
    print('=' * 74)
    print('结果汇总 (top_k=%d，共 %d 条查询)' % (args.top_k, len(eval_set)))
    print('=' * 74)
    print('  %-10s %-8s %-12s %-10s %-8s %s'
          % ('模式', 'Hit@k', 'Precision@k', 'MRR', 'Recall@k', '平均延迟'))
    print('  ' + '-' * 70)
    for mode, s in summary.items():
        print('  %-10s %-8.3f %-12.3f %-10.3f %-8.3f %.1f ms' % (
            mode,
            s['Hit@%d' % args.top_k], s['Precision@%d' % args.top_k],
            s['MRR'], s['Recall@%d' % args.top_k], s['avg_latency_ms']))
    print()
    print('  注：主指标是 Precision@k 和 MRR。Recall@k 在相关集很大时（"北京的算法岗"'
          '几百条都算相关）')
    print('      上限只有 5/几百，任何方法都接近 0，不具区分度，仅作参考。')

    # 分类拆解：看清楚混合检索到底在哪种查询上赢
    print()
    print('=' * 74)
    print('按查询类型拆解 (Precision@%d)' % args.top_k)
    print('=' * 74)
    print('  %-10s %-5s %s' % ('类型', 'n', '  '.join('%-11s' % m for m in args.modes)))
    print('  ' + '-' * 62)
    for cat, mdict in sorted(cat_summary.items()):
        n = next(iter(mdict.values()))['n']
        cells = []
        for m in args.modes:
            v = mdict.get(m)
            cells.append('%-11.3f' % v['Precision@%d' % args.top_k] if v else '%-11s' % '-')
        print('  %-10s %-5d %s' % (cat, n, '  '.join(cells)))

    # 保存结果，便于写进 README / 简历
    out = Path(args.save) if args.save else (ROOT / 'eval' / 'latest_results.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'top_k': args.top_k,
        'eval_size': len(eval_set),
        'eval_size_raw': len(raw_eval),
        'embedder': getattr(embed_fn, 'model_name', '?'),
        'db': str(args.db),
        'summary': summary,
        'by_category': cat_summary,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print()
    print('  已保存: %s' % out)
    con.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
