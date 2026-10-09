"""
generate.py —— 端到端问答：检索 Top-K → 拼提示词 → LLM 生成答案

流程
====
    用户提问
      → hybrid_search（BM25 + 向量，RRF 融合）取 Top-K 切块
      → 按「公司 · 岗位 · 段名」组织成上下文，要求模型**只依据上下文回答并标注来源**
      → 调 DeepSeek（OpenAI 兼容 /chat/completions）
      → 输出答案 + 引用到的岗位列表

为什么提示词要强制"只依据检索结果"
==============================
不做约束时模型会用预训练知识编答案，而岗位信息变化极快，
编出来的"要求"会误导求职者。这也是 RAG 相对纯 LLM 的核心价值。

⚠️ 模型选择
    `deepseek-flash` 是**推理型模型**，思考过程会吃掉 max_tokens，
    实测 max_tokens=20 时 reasoning_tokens 就用完了、答案是空的。
    生成答案推荐用 `deepseek-chat`（非推理），并把 max_tokens 给到 500+。

用法
----
    set DEEPSEEK_API_KEY=sk-xxx
    python src/generate.py --query "找偏业务的数据分析岗，要给管理层做报表"
    python src/generate.py --query "..." --no-llm      # 只看检索结果

环境变量
    DEEPSEEK_API_KEY   必填（无则自动退化为只返回检索结果）
    DEEPSEEK_BASE_URL  可选，默认 https://api.deepseek.com/v1
    DEEPSEEK_MODEL     可选，默认 deepseek-chat
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from embeddings import get_embedder                      # noqa: E402
from retrieval import fetch_chunks, hybrid_search        # noqa: E402
from schema import connect                               # noqa: E402

DEFAULT_BASE = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
DEFAULT_MODEL = os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat')

SYSTEM_PROMPT = (
    '你是求职岗位检索助手。你只能依据【检索到的岗位片段】回答，'
    '不得使用任何未在片段中出现的岗位信息，也不要推测。'
    '如果片段不足以回答，就直说"检索结果里没有相关信息"。'
    '回答用中文，先给结论，再列出依据的岗位与原文要点，并标注 [编号]。'
)


def retrieve(db: str, query: str, top_k: int = 5, embed_provider: str = 'fastembed',
             model_name: str = 'BAAI/bge-small-zh-v1.5'):
    """返回 (检索到的岗位级结果, 原始切块命中)。"""
    con = connect(db)
    embed_fn = get_embedder(embed_provider, model_name=model_name)
    qvec = embed_fn(query)
    hits = hybrid_search(con, query, top_k=top_k * 3, query_vec=qvec)

    details = fetch_chunks(con, [h['chunk_id'] for h in hits])
    by_chunk = {d['chunk_id']: d for d in details}

    seen, docs = set(), []
    for h in hits:
        d = by_chunk.get(h['chunk_id'])
        if not d:
            continue
        key = d.get('doc_key') or d['id']
        if key in seen:
            continue
        seen.add(key)
        docs.append(d)
        if len(docs) >= top_k:
            break
    con.close()
    return docs


def build_context(docs: list[dict]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        body = (d.get('text') or '').strip()
        # 去掉切块自带的「公司 · 岗位」前缀，避免重复
        if '\n' in body:
            first, rest = body.split('\n', 1)
            if '·' in first and len(first) < 60:
                body = rest.strip()
        lines.append(
            '[%d] 公司：%s | 岗位：%s | 城市：%s | 段名：%s\n%s'
            % (i, d.get('company') or '-', d.get('title') or '-',
               d.get('city') or '-', d.get('section') or '-', body[:600])
        )
    return '\n\n'.join(lines)


def call_llm(question: str, context: str, model: str = DEFAULT_MODEL,
             base_url: str = DEFAULT_BASE, max_tokens: int = 900,
             timeout: int = 180) -> tuple[str | None, str]:
    """返回 (答案, 错误信息)。答案 None 表示调用失败。"""
    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if not key:
        return None, '未设置 DEEPSEEK_API_KEY'

    try:
        import requests
    except ImportError:
        return None, '未安装 requests'

    user = ('【检索到的岗位片段】\n%s\n\n【用户问题】\n%s\n\n'
            '请依据上面的片段回答，并标注 [编号]。' % (context, question))
    try:
        r = requests.post(
            base_url.rstrip('/') + '/chat/completions',
            headers={'Authorization': 'Bearer ' + key,
                     'Content-Type': 'application/json'},
            json={
                'model': model,
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': user},
                ],
                'max_tokens': max_tokens,
                'temperature': 0.2,
                'stream': False,
            },
            timeout=timeout,
        )
        if r.status_code != 200:
            return None, 'HTTP %d: %s' % (r.status_code, r.text[:300])
        data = r.json()
        choice = data['choices'][0]
        content = (choice['message'].get('content') or '').strip()
        usage = data.get('usage') or {}
        info = ('tokens: prompt=%s completion=%s (reasoning=%s)'
                % (usage.get('prompt_tokens'), usage.get('completion_tokens'),
                   (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', '-')))
        if not content:
            return None, '模型返回空内容（推理型模型可能把预算用在思考上，' \
                         '把 max_tokens 调大或换 deepseek-chat）  %s' % info
        return content, info
    except Exception as e:                                   # noqa: BLE001
        return None, '%s: %s' % (type(e).__name__, e)


def main() -> int:
    ap = argparse.ArgumentParser(description='RAG 岗位问答（端到端）')
    ap.add_argument('--query', required=True)
    ap.add_argument('--db', default=str(ROOT / 'data' / 'rag.db'))
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--embed', default='fastembed')
    ap.add_argument('--model', default='BAAI/bge-small-zh-v1.5', help='embedding 模型')
    ap.add_argument('--llm-model', default=DEFAULT_MODEL)
    ap.add_argument('--no-llm', action='store_true', help='只做检索，不调 LLM')
    args = ap.parse_args()

    print('=' * 76)
    print('RAG 岗位问答')
    print('=' * 76)
    print('  问题: %s' % args.query)
    print()

    t0 = time.perf_counter()
    docs = retrieve(args.db, args.query, args.top_k, args.embed, args.model)
    t_ret = time.perf_counter() - t0

    print('  [检索] %.0f ms，命中 %d 个岗位' % (t_ret * 1000, len(docs)))
    for i, d in enumerate(docs, 1):
        print('    [%d] %-18s %-26s %-10s %s'
              % (i, (d.get('company') or '-')[:18], (d.get('title') or '-')[:26],
                 (d.get('city') or '-')[:10], d.get('section') or '-'))
    if not docs:
        print('  没有检索到任何岗位。')
        return 1

    if args.no_llm:
        return 0

    print()
    context = build_context(docs)
    print('  [生成] 调用 %s …' % args.llm_model)
    t1 = time.perf_counter()
    answer, info = call_llm(args.query, context, model=args.llm_model)
    t_llm = time.perf_counter() - t1

    print()
    print('=' * 76)
    print('答案')
    print('=' * 76)
    if answer is None:
        print('  [未生成] %s' % info)
        print()
        print('  提示：设置环境变量 DEEPSEEK_API_KEY 后重试。')
        print('       下面的上下文就是模型本该看到的内容：')
        print()
        print(context[:1500])
        return 2
    print(answer)
    print()
    print('  （生成耗时 %.1f 秒；%s）' % (t_llm, info))
    return 0


if __name__ == '__main__':
    sys.exit(main())
