"""
retrieval.py —— 检索层：关键词（BM25） + 向量 + RRF 融合

三种检索模式
============
1. `keyword_search`  基于 SQLite FTS5 + bm25()，对 **jieba 分词后的词串**检索。
                     优点：零依赖、精确、JD 里关键词密集时非常有效。
2. `vector_search`   基于 numpy 的余弦相似度，向量存在 `chunks.embedding`。
                     ⚠️ 当前默认用的是 `embeddings.hash_embed`（占位实现），
                     它**不具备真正的语义泛化能力**，仅用于打通链路与验证评测框架。
                     换成真实模型后，本函数无需改动。
3. `hybrid_search`   两路召回后用 **RRF（Reciprocal Rank Fusion）** 融合。
                     RRF 只依赖排名不依赖分数，因此不需要对两路分数做归一化。

用法
----
    from retrieval import keyword_search, vector_search, hybrid_search
    hits = hybrid_search(con, '数据分析 校招', top_k=5)
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Iterable

import numpy as np

from schema import tokenize

# ─────────────── 关键词检索 ───────────────

def _build_match_expr(query: str, *, mode: str = 'OR') -> str:
    """
    把自然语言查询转成 FTS5 的 MATCH 表达式。

    做法：jieba 分词 -> 转义双引号 -> 用 OR 连接。
    用 OR 而非 AND 的原因：查「数据分析 校招」时，
    只命中"数据分析"的岗位也应该排上来，靠 bm25 排序区分相关度。
    """
    toks = tokenize(query).split()
    if not toks:
        return ''
    # FTS5 里用双引号包住每个词，避免被当成语法
    quoted = ['"%s"' % t.replace('"', '""') for t in toks]
    return (' %s ' % mode).join(quoted)


def keyword_search(con: sqlite3.Connection, query: str, top_k: int = 10,
                   *, debug: bool = False) -> list[dict]:
    """BM25 关键词检索。返回 [{chunk_id, doc_id, score, ...}]，score 越大越相关。"""
    expr = _build_match_expr(query)
    if not expr:
        return []
    t0 = time.perf_counter()
    try:
        rows = con.execute(
            """SELECT f.rowid AS chunk_id,
                      bm25(chunks_fts) AS bm25_score
               FROM chunks_fts f
               WHERE chunks_fts MATCH ?
               ORDER BY bm25_score
               LIMIT ?""",
            (expr, top_k * 4),          # 多取一些，融合时再截断
        ).fetchall()
    except sqlite3.OperationalError as e:
        if debug:
            print('  [keyword_search] MATCH 失败: %s  expr=%r' % (e, expr))
        return []

    # SQLite 的 bm25() 返回**负值**，越小越相关 -> 取负号变成"越大越相关"
    out = []
    for r in rows:
        out.append({
            'chunk_id': r['chunk_id'],
            'score': -float(r['bm25_score']),
        })
    out.sort(key=lambda x: -x['score'])
    out = out[:top_k]
    if debug:
        print('  [keyword_search] %d 命中，用时 %.1f ms' % (len(out), (time.perf_counter() - t0) * 1000))
    return out


# ─────────────── 向量检索 ───────────────

def _load_vectors(con: sqlite3.Connection, model: str | None = None):
    """载入所有已计算向量的块。返回 (ids, matrix) 或 (None, None)。"""
    sql = ('SELECT id, embedding, embedding_model FROM chunks '
           'WHERE embedding IS NOT NULL')
    args: tuple = ()
    if model:
        sql += ' AND embedding_model = ?'
        args = (model,)
    rows = con.execute(sql, args).fetchall()
    if not rows:
        return None, None
    ids, vecs = [], []
    for r in rows:
        v = np.frombuffer(r['embedding'], dtype=np.float32)
        if v.size == 0:
            continue
        ids.append(r['id'])
        vecs.append(v)
    if not vecs:
        return None, None
    return ids, np.vstack(vecs)


def vector_search(con: sqlite3.Connection, query_vec: np.ndarray,
                  top_k: int = 10, *, model: str | None = None,
                  debug: bool = False) -> list[dict]:
    """余弦相似度检索。query_vec 必须与库中向量同维度。"""
    t0 = time.perf_counter()
    ids, mat = _load_vectors(con, model)
    if ids is None:
        if debug:
            print('  [vector_search] 库中没有向量（需先跑 embed 步骤）')
        return []
    q = np.asarray(query_vec, dtype=np.float32).ravel()
    if q.size != mat.shape[1]:
        if debug:
            print('  [vector_search] 维度不匹配：查询 %d vs 库 %d' % (q.size, mat.shape[1]))
        return []

    # 归一化后点积 = 余弦相似度
    def norm(a):
        n = np.linalg.norm(a, axis=-1, keepdims=True)
        n[n == 0] = 1.0
        return a / n

    sims = norm(mat) @ norm(q)
    order = np.argsort(-sims)[:top_k]
    out = [{'chunk_id': ids[i], 'score': float(sims[i])} for i in order]
    if debug:
        print('  [vector_search] 扫描 %d 向量，用时 %.1f ms' % (len(ids), (time.perf_counter() - t0) * 1000))
    return out


# ─────────────── RRF 融合 ───────────────

def rrf_fuse(rankings: Iterable[list[dict]], *, k: int = 60,
             top_k: int = 10, weights: list[float] | None = None) -> list[dict]:
    """
    Reciprocal Rank Fusion。

    公式：score(d) = Σ_i  w_i / (k + rank_i(d))
    其中 rank 从 1 开始。k 默认 60（原论文推荐值）。

    优点：只用排名，无需归一化两路分数——BM25 与余弦相似度量纲完全不同，
    直接加权求和需要调参，RRF 通常更稳。
    """
    rankings = list(rankings)
    if weights is None:
        weights = [1.0] * len(rankings)

    agg: dict[int, float] = {}
    hits_by_rank: dict[int, list[int]] = {}
    for ri, hits in enumerate(rankings):
        w = weights[ri]
        for rank, h in enumerate(hits, start=1):
            cid = h['chunk_id']
            agg[cid] = agg.get(cid, 0.0) + w / (k + rank)
            hits_by_rank.setdefault(cid, []).append(ri)

    fused = [{'chunk_id': cid, 'score': s, 'from': hits_by_rank[cid]}
             for cid, s in agg.items()]
    fused.sort(key=lambda x: -x['score'])
    return fused[:top_k]


def hybrid_search(con: sqlite3.Connection, query: str, top_k: int = 10,
                  *, query_vec: np.ndarray | None = None,
                  recall_k: int = 30, rrf_k: int = 60,
                  weights: tuple[float, float] = (1.0, 1.0),
                  debug: bool = False) -> list[dict]:
    """混合检索：关键词 + 向量，RRF 融合。无向量时自动退化为纯关键词。"""
    kw = keyword_search(con, query, top_k=recall_k, debug=debug)
    vec = []
    if query_vec is not None:
        vec = vector_search(con, query_vec, top_k=recall_k, debug=debug)

    if not vec:
        if debug:
            print('  [hybrid_search] 无向量结果，退化为纯关键词检索')
        return kw[:top_k]

    return rrf_fuse([kw, vec], k=rrf_k, top_k=top_k,
                    weights=[weights[0], weights[1]])


# ─────────────── 结果补全 ───────────────

def fetch_chunks(con: sqlite3.Connection, chunk_ids: list[int]) -> list[dict]:
    """按 chunk_id 取回块内容（含所属岗位信息），顺序与传入一致。"""
    if not chunk_ids:
        return []
    q = ','.join('?' * len(chunk_ids))
    rows = con.execute(
        """SELECT c.id AS chunk_id, c.doc_id, c.chunk_index, c.section, c.text,
                  d.doc_key, d.company, d.title, d.city, d.job_type, d.source
           FROM chunks c JOIN documents d ON d.id = c.doc_id
           WHERE c.id IN (%s)""" % q,
        chunk_ids,
    ).fetchall()
    by_id = {r['chunk_id']: dict(r) for r in rows}
    return [by_id[cid] for cid in chunk_ids if cid in by_id]


def search(con: sqlite3.Connection, query: str, top_k: int = 5, *,
           mode: str = 'hybrid', embed_fn=None, debug: bool = False) -> list[dict]:
    """
    统一入口。

    mode: 'keyword' | 'vector' | 'hybrid'
    embed_fn: 可调用对象，把文本映射为 np.ndarray；为 None 时不做向量检索（退化为关键词）
    """
    qvec = None
    if mode in ('vector', 'hybrid') and embed_fn is not None:
        qvec = embed_fn(query)

    if mode == 'keyword' or qvec is None:
        hits = keyword_search(con, query, top_k=top_k, debug=debug)
    elif mode == 'vector':
        hits = vector_search(con, qvec, top_k=top_k, debug=debug)
    else:
        hits = hybrid_search(con, query, top_k=top_k, query_vec=qvec, debug=debug)

    detailed = fetch_chunks(con, [h['chunk_id'] for h in hits])
    score_by_id = {h['chunk_id']: h['score'] for h in hits}
    for d in detailed:
        d['score'] = score_by_id.get(d['chunk_id'], 0.0)
    return detailed


# ─────────────── 两阶段：召回 + 重排 ───────────────

def rerank_search(con: sqlite3.Connection, query: str, top_k: int = 5, *,
                  query_vec: np.ndarray | None = None,
                  recall_k: int = 30, rerank_pool: int = 30,
                  reranker=None, debug: bool = False) -> list[dict]:
    """
    两阶段检索：hybrid 召回 `rerank_pool` 条 → cross-encoder 精排 → 取 top_k。

    为什么分两段（而不是直接让 cross-encoder 扫全库）：
        cross-encoder 必须把 (query, doc) 拼起来跑前向，**无法预计算**。
        5,209 个切块逐条打分约需 5,209 × 20ms ≈ 100 秒，不可接受。
        先用 bi-encoder 召回 30 条（追求不漏），再精排（追求排得准），
        这样只跑 30 次前向，约 1~2 秒。

    reranker 为 None 时自动退化为纯 hybrid，方便对照组。
    """
    if reranker is None:
        return hybrid_search(con, query, top_k=top_k, query_vec=query_vec,
                             recall_k=recall_k, debug=debug)

    t0 = time.perf_counter()
    pool = hybrid_search(con, query, top_k=rerank_pool, query_vec=query_vec,
                         recall_k=recall_k, debug=debug)
    t_recall = (time.perf_counter() - t0) * 1000
    if not pool:
        return []

    from rerank import rerank_hits          # 局部导入避免循环依赖
    t1 = time.perf_counter()
    out = rerank_hits(con, query, pool, reranker, top_k=top_k)
    t_rerank = (time.perf_counter() - t1) * 1000
    if debug:
        print('  [rerank_search] 召回 %d 条 (%.1f ms) → 精排取 %d 条 (%.1f ms)'
              % (len(pool), t_recall, len(out), t_rerank))
    return out
