"""
embeddings.py —— 向量化接口（可插拔）

⚠️ 重要说明
==========
本文件默认提供的 `hash_embed` 是 **占位实现（placeholder）**，
它**不具备真正的语义泛化能力**：
  - 它把文本映射为固定维度的向量，但语义相近的词（如"数据分析"与"数据挖掘"）
    在该空间里**并不必然接近**。
  - 它的作用是 **把整条链路（存储/检索/融合/评测）跑通并可测试**，
    让你在接入真实模型之前就能验证代码正确性。

因此：
  - ✅ 可以用它做**链路自测**与**评测脚本调试**
  - ❌ **不要**用它跑出的效果去填简历里的"召回率提升"数字

接入真实模型的方法
================
把 `get_embedder()` 里的 provider 换成下面任一实现即可，
**其余代码（retrieval / ingest / eval）无需改动**。

方案 A：本地模型（推荐，离线可用，免费）
    需要先安装：pip install sentence-transformers
    模型建议：BAAI/bge-small-zh-v1.5（约 95 MB，中文效果好，CPU 可跑）

        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        def embed(texts):
            return _model.encode(texts, normalize_embeddings=True)

方案 B：OpenAI 兼容的 embedding API
    DeepSeek 目前**不提供** embedding 接口，可换用其他兼容服务：

        def embed(texts):
            r = requests.post(f'{BASE}/embeddings',
                              headers={'Authorization': f'Bearer {KEY}'},
                              json={'model': MODEL, 'input': texts})
            return np.array([d['embedding'] for d in r.json()['data']], dtype=np.float32)

方案 C：迁移到 PostgreSQL + pgvector
    本项目的 `chunks.embedding` 已预留，
    迁移时改为 `vector(N)` 列并用 `<=>` 运算符检索即可。

用法
----
    from embeddings import get_embedder
    embed = get_embedder('hash', dim=256)      # 占位实现
    vec = embed('数据分析师 岗位职责')           # -> np.ndarray(256,)
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Callable, Sequence

import numpy as np

# 向量维度：占位实现用 256；接入 bge-small-zh 时应改为 512；
# bge-base-zh / text-embedding-3-small 为 768 / 1536
DEFAULT_DIM = 256


# ─────────────── 占位实现 ───────────────

def hash_embed(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """
    占位 embedding：字符 n-gram 哈希到固定维度。

    ⚠️ 仅用于打通链路与自测，**不具备真正的语义泛化能力**。

    原理：把文本拆成 1~3 字符的 n-gram，每个 n-gram 用 md5 映射到维度之一并累加，
    最后做 L2 归一化。这样**字面重叠越多**的文本向量越接近——
    它在"关键词重叠"这个意义上有效，但没有语义理解。
    """
    v = np.zeros(dim, dtype=np.float32)
    if not text:
        return v
    s = re.sub(r'\s+', '', str(text))
    for n in (1, 2, 3):
        for i in range(len(s) - n + 1):
            gram = s[i:i + n]
            h = hashlib.md5(gram.encode('utf-8')).digest()
            idx = int.from_bytes(h[:4], 'little') % dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            v[idx] += sign * (1.0 / n)      # 长 n-gram 权重略低
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def make_hash_embedder(dim: int = DEFAULT_DIM) -> Callable[[str], np.ndarray]:
    """返回绑定维度的占位 embedder。"""
    def _embed(text: str) -> np.ndarray:
        return hash_embed(text, dim=dim)
    _embed.dim = dim                      # type: ignore[attr-defined]
    _embed.model_name = 'hash-placeholder'  # type: ignore[attr-defined]
    return _embed


# ─────────────── 真实模型适配 ───────────────

def make_fastembed_embedder(model_name: str = 'BAAI/bge-small-zh-v1.5'
                            ) -> Callable[[str], np.ndarray]:
    """
    ONNX 版中文 embedding（**推荐**：不需要 PyTorch，模型约 95 MB）。

    依赖：pip install fastembed   （会带 onnxruntime，约 60 MB）

    ⚠️ 模型默认从 HuggingFace 下载，国内直连不稳。
    本函数会把 HF_ENDPOINT 默认指向 hf-mirror.com 镜像。

    返回的 embed 函数额外带：
        .embed_batch(texts) -> np.ndarray (n, dim)   批量，比逐条快一个量级
        .dim / .model_name
    """
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise ImportError(
            '未安装 fastembed。\n'
            '  安装：pip install fastembed\n'
            '  或用 get_embedder("hash") 先跑通链路。'
        ) from e

    model = TextEmbedding(model_name=model_name)

    def _one(text: str) -> np.ndarray:
        vec = next(iter(model.embed([text])))
        return np.asarray(vec, dtype=np.float32)

    def _batch(texts: Sequence[str]) -> np.ndarray:
        arr = np.asarray(list(model.embed(list(texts))), dtype=np.float32)
        return arr

    def _embed(text: str) -> np.ndarray:
        return _one(text)

    _embed.embed_batch = _batch                       # type: ignore[attr-defined]
    _embed.dim = int(_batch(['维度探测']).shape[1])    # type: ignore[attr-defined]
    _embed.model_name = 'fastembed:' + model_name     # type: ignore[attr-defined]
    return _embed


def make_sentence_transformer_embedder(model_name: str = 'BAAI/bge-small-zh-v1.5'
                                       ) -> Callable[[str], np.ndarray]:
    """
    本地 sentence-transformers 适配器。

    需要先 `pip install sentence-transformers`（会带 torch，约 2 GB）。
    未安装时抛 ImportError，并给出明确提示。
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            '未安装 sentence-transformers。\n'
            '  安装：pip install sentence-transformers\n'
            '  或用 get_embedder("hash") 先跑通链路。'
        ) from e

    model = SentenceTransformer(model_name)

    def _embed(text: str) -> np.ndarray:
        vec = model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)

    _embed.dim = int(model.get_sentence_embedding_dimension())  # type: ignore[attr-defined]
    _embed.model_name = model_name                              # type: ignore[attr-defined]
    return _embed


def make_api_embedder(base_url: str, api_key: str, model: str,
                      dim: int | None = None,
                      timeout: int = 60) -> Callable[[str], np.ndarray]:
    """
    OpenAI 兼容 embedding API 适配器（DeepSeek 目前不提供 embedding 接口，
    需要换用其他提供 /embeddings 的服务）。
    """
    import requests

    def _embed(text: str) -> np.ndarray:
        r = requests.post(
            base_url.rstrip('/') + '/embeddings',
            headers={'Authorization': 'Bearer ' + api_key,
                     'Content-Type': 'application/json'},
            json={'model': model, 'input': text},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()['data'][0]['embedding']
        return np.asarray(data, dtype=np.float32)

    _embed.dim = dim or 0                   # type: ignore[attr-defined]
    _embed.model_name = model               # type: ignore[attr-defined]
    return _embed


# ─────────────── 工厂 ───────────────

def get_embedder(provider: str = 'hash', **kwargs) -> Callable[[str], np.ndarray]:
    """
    按名字取 embedder。

    provider:
        'hash'                  占位实现（零依赖，无真正语义能力，仅链路自测）
        'fastembed' / 'onnx'    ONNX 中文模型（推荐，不需 PyTorch）
        'sentence-transformer'  本地 torch 模型（需装 torch，约 2 GB）
        'api'                   OpenAI 兼容接口
    """
    if provider == 'hash':
        return make_hash_embedder(kwargs.get('dim', DEFAULT_DIM))
    if provider in ('fastembed', 'onnx', 'fe'):
        return make_fastembed_embedder(
            kwargs.get('model_name', 'BAAI/bge-small-zh-v1.5'))
    if provider in ('sentence-transformer', 'st', 'local'):
        return make_sentence_transformer_embedder(
            kwargs.get('model_name', 'BAAI/bge-small-zh-v1.5'))
    if provider in ('api', 'openai'):
        return make_api_embedder(
            kwargs['base_url'], kwargs['api_key'], kwargs['model'],
            kwargs.get('dim'))
    raise ValueError('未知的 embedding provider: %r' % provider)


def embedder_info(embed_fn) -> dict:
    """返回 embedder 的元信息，用于写入数据库并区分不同模型算出的向量。"""
    return {
        'model': getattr(embed_fn, 'model_name', 'unknown'),
        'dim': int(getattr(embed_fn, 'dim', 0) or 0),
    }
