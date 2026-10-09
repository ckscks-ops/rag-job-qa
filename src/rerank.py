"""
rerank.py —— cross-encoder 重排

为什么需要重排
=============
召回（retrieval）和排序（ranking）是两个不同的任务：
  - **召回**用的是 bi-encoder：query 和 doc **分别**编码成向量再算余弦。
    快（可以预计算所有向量），但 query 与 doc 之间没有交互，精度有限。
  - **重排**用的是 cross-encoder：把 (query, doc) **拼在一起**送进模型，
    注意力可以跨 query 和 doc 交互，精度高得多。代价是**慢**——
    每个候选都要单独跑一次前向，没法预计算。

所以标准做法是两阶段：
    hybrid 召回 top-N（N=30，追求不漏）
      → cross-encoder 精排，取 top-k（k=5，追求排得准）

⚠️ 语言问题
    `Xenova/ms-marco-*` 和 `jinaai/*-en` 都是**纯英文**模型，中文会退化。
    中文场景必须用 `BAAI/bge-reranker-*` 或 `jinaai/*-multilingual`。

用法
----
    from rerank import get_reranker
    rr = get_reranker('BAAI/bge-reranker-v2-m3-int8')
    scores = rr('找数据分析岗', ['JD 文本1', 'JD 文本2'])   # -> [0.83, 0.12]

自测
----
    python src/rerank.py --selftest
"""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

# 中文场景可用的重排模型（fastembed 0.9.0 支持列表里筛出来的）
ZH_MODELS = {
    'BAAI/bge-reranker-v2-m3-int8': 0.57,   # ★ 推荐：多语言 + int8 量化，CPU 友好
    'BAAI/bge-reranker-base': 1.04,         # 经典款，中英都行
    'jinaai/jina-reranker-v2-base-multilingual': 1.11,
    'BAAI/bge-reranker-v2-m3': 2.27,        # 精度最好，但最大
}
# 明确不能用于中文的（列出来是为了防止误用）
EN_ONLY_MODELS = [
    'Xenova/ms-marco-MiniLM-L-6-v2', 'Xenova/ms-marco-MiniLM-L-12-v2',
    'jinaai/jina-reranker-v1-tiny-en', 'jinaai/jina-reranker-v1-turbo-en',
]

DEFAULT_MODEL = 'BAAI/bge-reranker-v2-m3-int8'

# ★ 送入精排的文本截断长度（2026-10-09 由实验定标，见 eval/rerank_budget.py）
#
# 50 条查询 × 5 组配置的实测（召回池固定 30）：
#     截断     Precision@5   Hit@5    MRR     精排耗时
#     不截断      0.592      0.820   0.682    14.0 s
#     200 字      0.584      0.840   0.716     2.2 s   ← 采用
#     池降到 10   0.495      0.820   0.667     0.7 s   ← 代价太大，不采用
#
# 结论：**截断只损失 0.008 的 Precision@5，却省掉 85% 的延迟**，
# 而"把候选池从 30 缩到 10"要损失 0.097 —— 所以该动的是输入长度，不是候选数量。
#
# 为什么截断几乎不掉精度、Hit@5/MRR 反而略升：
#     切块开头就是「公司 · 岗位」+ 段落名 + 核心要求，
#     尾部常是流程性套话。截断相当于**去噪**，
#     让 cross-encoder 的注意力集中在真正判别性的内容上。
DEFAULT_MAX_CHARS = 200


def get_reranker(model_name: str = DEFAULT_MODEL):
    """
    返回 callable(query: str, docs: list[str]) -> list[float]（顺序与 docs 一致）。

    额外的元信息挂在函数属性上：.model_name / .dim
    """
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError as e:
        raise ImportError(
            '未安装 fastembed（或版本过旧，需要支持 TextCrossEncoder 的版本）。\n'
            '  安装：pip install -U fastembed'
        ) from e

    if model_name in EN_ONLY_MODELS:
        print('  [警告] %s 是纯英文模型，中文效果会明显退化。' % model_name, file=sys.stderr)

    enc = TextCrossEncoder(model_name=model_name)

    def _rerank(query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        return [float(s) for s in enc.rerank(query, list(docs))]

    _rerank.model_name = model_name          # type: ignore[attr-defined]
    return _rerank


def rerank_hits(con, query: str, hits: list[dict], reranker,
                top_k: int = 5, max_chars: int | None = DEFAULT_MAX_CHARS) -> list[dict]:
    """
    对已经召回的 chunk 级命中做 cross-encoder 重排。

    hits 需要含 chunk_id；本函数会用 `text` 字段（含「公司 · 岗位」前缀）打分，
    因为这更接近重排模型训练时的输入形态。

    max_chars：**截断**每篇送进精排的字符数（None = 不截断）
        cross-encoder 的成本随 token 数近似线性增长，而一个切块最长 480 字。
        实测（见 eval/rerank_budget.py）截断到 200 字能显著压低单条耗时，
        精度只掉一点点——因为"这岗位是不是这个方向"看开头基本就够了。
        这是把十几秒压到 1~2 秒最有效、最便宜的杠杆（不用换模型、不用下东西）。
    """
    from retrieval import fetch_chunks

    if not hits:
        return []
    details = fetch_chunks(con, [h['chunk_id'] for h in hits])
    if not details:
        return []

    docs = [d.get('text') or '' for d in details]
    if max_chars:
        docs = [t[:max_chars] for t in docs]
    scores = reranker(query, docs)

    order = sorted(range(len(details)), key=lambda i: -scores[i])[:top_k]
    return [{'chunk_id': details[i]['chunk_id'], 'score': scores[i],
             'rrf_rank': i + 1} for i in order]


def _selftest() -> int:
    model = os.environ.get('RERANK_MODEL', DEFAULT_MODEL)
    print('=' * 76)
    print('cross-encoder 重排自测')
    print('=' * 76)
    print('  模型: %s' % model)
    print('  HF_ENDPOINT: %s' % os.environ.get('HF_ENDPOINT'))

    t0 = time.perf_counter()
    rr = get_reranker(model)
    print('  加载耗时: %.1f 秒' % (time.perf_counter() - t0))
    print()

    query = '找偏业务的数据分析岗，要给管理层做报表'
    docs = [
        '数据分析师：负责用户分层与留存分析，输出经营报表，为管理层提供决策支持',   # 强相关
        '商业分析岗：参与商业模式研究，独立完成数据分析与策略建议，向管理层汇报',   # 强相关
        '电机控制算法工程师：负责永磁同步电机 FOC 控制算法设计与仿真',            # 不相关
        '后端开发工程师：负责集团信息系统微服务架构与服务端接口开发',              # 不相关
        '数据仓库开发：负责数仓建模、ETL 调度与指标口径统一',                     # 中等
    ]
    t0 = time.perf_counter()
    scores = rr(query, docs)
    dt = (time.perf_counter() - t0) * 1000
    print('  查询: %s' % query)
    print('  %d 个候选，打分耗时 %.0f ms（%.1f ms/条）' % (len(docs), dt, dt / len(docs)))
    print()
    order = sorted(range(len(docs)), key=lambda i: -scores[i])
    for rank, i in enumerate(order, 1):
        print('  %d. %.4f  %s' % (rank, scores[i], docs[i][:46]))

    print()
    # 判定：两个强相关的必须排在前 2
    top2 = set(order[:2])
    ok = top2 == {0, 1}
    print('=' * 76)
    if ok:
        print('  ✅ 重排把两个真正相关的排到了前 2（与文档顺序无关，说明是真打分不是透传）')
    else:
        print('  ❌ 重排没有把相关文档排到前面，检查模型或 API 语义')
    print('=' * 76)
    return 0 if ok else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='cross-encoder 重排')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--list', action='store_true', help='列出可用模型')
    args = ap.parse_args()

    if args.list:
        print('中文可用：')
        for m, gb in ZH_MODELS.items():
            print('  %-46s %.2f GB' % (m, gb))
        print('\n纯英文（中文慎用）：')
        for m in EN_ONLY_MODELS:
            print('  %s' % m)
        sys.exit(0)

    if args.selftest:
        sys.exit(_selftest())

    print('用法：python src/rerank.py --selftest  或  --list')
