"""
api.py —— 把检索 + 生成包成 HTTP 服务

为什么需要它
===========
命令行能跑说明功能对不对，HTTP 服务才能说明**能交付**：
面试官可以直接 curl 或打开 /docs 试，不用装环境。

接口
====
    GET  /health                     健康检查（库里有多少岗位/切块、用了什么模型）
    GET  /search?q=...&top_k=5       只检索，返回岗位列表（快，不花钱）
    POST /ask   {"query": "...", "top_k": 5}
                                     检索 + LLM 生成答案（慢，消耗 API）

启动
====
    set DEEPSEEK_API_KEY=sk-xxx
    python src/api.py                 # 等价于 uvicorn api:app --host 127.0.0.1 --port 8000
    然后打开 http://127.0.0.1:8000/docs 交互测试

依赖
====
    pip install fastapi uvicorn
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

DB = os.environ.get('RAG_DB', str(ROOT / 'data' / 'rag.db'))
EMBED_MODEL = os.environ.get('RAG_EMBED_MODEL', 'BAAI/bge-small-zh-v1.5')
RERANK_MODEL = os.environ.get('RAG_RERANK_MODEL', 'BAAI/bge-reranker-v2-m3-int8')

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError:                                     # pragma: no cover
    print('未安装 fastapi / uvicorn。请先：pip install fastapi uvicorn')
    raise

from embeddings import get_embedder                     # noqa: E402
from generate import build_context, call_llm            # noqa: E402
from retrieval import (fetch_chunks, hybrid_search, keyword_search,   # noqa: E402
                       rrf_fuse, vector_search)
from schema import connect, stats, tokenize            # noqa: E402

app = FastAPI(
    title='RAG 岗位智能问答',
    description='基于真实校招岗位 JD 的混合检索（BM25 + 向量 + RRF）与问答服务',
    version='0.1.0',
)

_embed_fn = None
_reranker = None
WEB_DIR = ROOT / 'web'


@app.get('/', include_in_schema=False)
def index():
    """检索对比演示台（静态页）。"""
    f = WEB_DIR / 'index.html'
    if not f.is_file():
        raise HTTPException(status_code=404, detail='web/index.html 不存在')
    return FileResponse(str(f), media_type='text/html; charset=utf-8')


if WEB_DIR.is_dir():
    app.mount('/static', StaticFiles(directory=str(WEB_DIR)), name='static')


def embed_fn():
    """进程内只加载一次 embedding 模型（加载要几十秒）。"""
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = get_embedder('fastembed', model_name=EMBED_MODEL)
    return _embed_fn


def reranker_fn():
    """cross-encoder 重排模型，进程内只加载一次。"""
    global _reranker
    if _reranker is None:
        from rerank import get_reranker
        _reranker = get_reranker(RERANK_MODEL)
    return _reranker


def _search(query: str, top_k: int) -> list[dict[str, Any]]:
    con = connect(DB)
    try:
        qvec = embed_fn()(query)
        hits = hybrid_search(con, query, top_k=top_k * 3, query_vec=qvec)
        details = fetch_chunks(con, [h['chunk_id'] for h in hits])
        by_chunk = {d['chunk_id']: d for d in details}

        seen, out = set(), []
        for h in hits:
            d = by_chunk.get(h['chunk_id'])
            if not d:
                continue
            key = d.get('doc_key') or d['id']
            if key in seen:
                continue
            seen.add(key)
            body = (d.get('text') or '').strip()
            if '\n' in body:
                first, rest = body.split('\n', 1)
                if '·' in first and len(first) < 60:
                    body = rest.strip()
            out.append({
                'rank': len(out) + 1,
                'doc_key': d.get('doc_key'),
                'company': d.get('company'),
                'title': d.get('title'),
                'city': d.get('city'),
                'section': d.get('section'),
                'score': round(float(h['score']), 6),
                'text': body[:600],
            })
            if len(out) >= top_k:
                break
        return out
    finally:
        con.close()


@app.get('/health')
def health():
    con = connect(DB)
    try:
        s = stats(con)
    finally:
        con.close()
    return {
        'status': 'ok',
        'db': DB,
        'embed_model': 'fastembed:' + EMBED_MODEL,
        'llm_configured': bool(os.environ.get('DEEPSEEK_API_KEY')),
        'stats': s,
    }


@app.get('/search')
def search(q: str = Query(..., min_length=1, description='自然语言查询'),
           top_k: int = Query(5, ge=1, le=20)):
    t0 = time.perf_counter()
    results = _search(q, top_k)
    return {
        'query': q,
        'top_k': top_k,
        'count': len(results),
        'latency_ms': round((time.perf_counter() - t0) * 1000, 1),
        'results': results,
    }


def _details(con, chunk_ids) -> dict:
    """chunk_id -> 岗位/段名/正文片段。"""
    if not chunk_ids:
        return {}
    q = ','.join('?' * len(chunk_ids))
    rows = con.execute(
        'SELECT c.id AS chunk_id, c.section, c.text, '
        '       d.doc_key, d.company, d.title, d.city '
        'FROM chunks c JOIN documents d ON d.id = c.doc_id '
        'WHERE c.id IN (%s)' % q, list(chunk_ids)).fetchall()
    return {r['chunk_id']: dict(r) for r in rows}


def _to_docs(hits, det, top_k, from_labels=None) -> list[dict]:
    """chunk 级命中 -> 岗位级结果（同岗位去重，保留排名最高的一条）。"""
    seen, out = set(), []
    for h in hits:
        d = det.get(h['chunk_id'])
        if not d:
            continue
        key = d.get('doc_key') or h['chunk_id']
        if key in seen:
            continue
        seen.add(key)
        body = (d.get('text') or '').strip()
        if '\n' in body:
            first, rest = body.split('\n', 1)
            if '·' in first and len(first) < 60:
                body = rest.strip()
        row = {
            'rank': len(out) + 1,
            'doc_key': d.get('doc_key'),
            'company': d.get('company'),
            'title': d.get('title'),
            'city': d.get('city'),
            'section': d.get('section'),
            'score': round(float(h['score']), 4),
            'text': body[:400],
        }
        if from_labels is not None:
            row['from'] = [from_labels[i] for i in h.get('from', [])
                           if i < len(from_labels)]
        out.append(row)
        if len(out) >= top_k:
            break
    return out


@app.get('/compare')
def compare(q: str = Query(..., min_length=1, description='自然语言查询'),
            top_k: int = Query(5, ge=1, le=10),
            recall_k: int = Query(30, ge=5, le=100),
            rerank: bool = Query(False, description='是否追加 cross-encoder 精排（慢）'),
            rerank_pool: int = Query(20, ge=3, le=50,
                                     description='精排候选池大小（消融实验最优值=20）')):
    """
    一次返回**三路检索的并排结果**，这是本项目的核心演示接口。

    为什么专门做这个接口：
      项目真正的价值不是"能问答"，而是**知道每种检索在什么时候失效**——
      纯关键词在口语化改写上塌方、纯向量在精确技术栈上塌方。
      把三路并排放出来，这件事一眼就看得出来。
    """
    con = connect(DB)
    try:
        t0 = time.perf_counter()
        kw = keyword_search(con, q, top_k=recall_k)
        t_kw = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        qvec = embed_fn()(q)
        t_emb = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        vec = vector_search(con, qvec, top_k=recall_k)
        t_vec = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fused = rrf_fuse([kw, vec], k=60, top_k=recall_k)
        t_rrf = (time.perf_counter() - t0) * 1000

        ids = ({h['chunk_id'] for h in kw} | {h['chunk_id'] for h in vec}
               | {h['chunk_id'] for h in fused})

        # 可选：cross-encoder 精排（两阶段检索的第二阶段）
        rr_pool_hits, t_rerank = None, None
        if rerank:
            from rerank import rerank_hits
            t0 = time.perf_counter()
            pool = fused[:rerank_pool]
            rr_pool_hits = rerank_hits(con, q, pool, reranker_fn(),
                                       top_k=rerank_pool)
            t_rerank = (time.perf_counter() - t0) * 1000
            ids |= {h['chunk_id'] for h in rr_pool_hits}

        det = _details(con, ids)

        modes = {
            'keyword': {
                'label': '纯关键词', 'sublabel': 'BM25 (FTS5 + jieba)',
                'latency_ms': round(t_kw, 1), 'pool': len(kw),
                'results': _to_docs(kw, det, top_k),
            },
            'vector': {
                'label': '纯向量', 'sublabel': '余弦相似度 (bge-small-zh 512维)',
                'latency_ms': round(t_emb + t_vec, 1), 'pool': len(vec),
                'results': _to_docs(vec, det, top_k),
            },
            'hybrid': {
                'label': '混合', 'sublabel': 'RRF 融合 (k=60)',
                'latency_ms': round(t_kw + t_emb + t_vec + t_rrf, 1), 'pool': len(fused),
                'results': _to_docs(fused, det, top_k,
                                    from_labels=['关键词', '向量']),
            },
        }
        if rr_pool_hits is not None:
            modes['rerank'] = {
                'label': '混合 + 精排',
                'sublabel': 'cross-encoder (bge-reranker-v2-m3-int8)',
                'latency_ms': round(t_kw + t_emb + t_vec + t_rrf + (t_rerank or 0), 1),
                'pool': rerank_pool,
                'results': _to_docs(rr_pool_hits, det, top_k),
            }

        return {
            'query': q, 'top_k': top_k, 'recall_k': recall_k,
            # 把 jieba 分词结果一并返回，前端用它高亮命中词——
            # 这样"关键词检索为什么命中/为什么没命中"就可见了
            'tokens': [t for t in tokenize(q).split() if len(t) >= 2][:24],
            'timing': {
                'embed_ms': round(t_emb, 1), 'keyword_ms': round(t_kw, 1),
                'vector_ms': round(t_vec, 1), 'rrf_ms': round(t_rrf, 1),
            },
            'modes': modes,
        }
    finally:
        con.close()


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post('/ask')
def ask(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail='query 不能为空')
    top_k = max(1, min(req.top_k, 10))

    t0 = time.perf_counter()
    sources = _search(req.query, top_k)
    t_ret = time.perf_counter() - t0
    if not sources:
        return {'query': req.query, 'answer': None,
                'message': '没有检索到相关岗位', 'sources': []}

    # 用原始切块文本拼上下文（_search 已裁到 600 字，够用）
    pseudo_docs = [{'company': s['company'], 'title': s['title'],
                    'city': s['city'], 'section': s['section'],
                    'text': (s['company'] or '') + ' · ' + (s['title'] or '') + '\n' + s['text']}
                   for s in sources]
    context = build_context(pseudo_docs)

    t1 = time.perf_counter()
    answer, info = call_llm(req.query, context)
    t_llm = time.perf_counter() - t1

    return {
        'query': req.query,
        'answer': answer,
        'answer_error': None if answer else info,
        'retrieval_latency_ms': round(t_ret * 1000, 1),
        'generation_latency_ms': round(t_llm * 1000, 1),
        'sources': sources,
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('RAG_PORT', '8000')))
