# -*- coding: utf-8 -*-
"""
验证真实中文 embedding 是否具备语义能力。

判据（hash_embed 占位实现做不到的）：
  1. 语义相近但**字面不同**的文本相似度要高
     例："数据分析" vs "数据挖掘" / "数据分析师" vs "商业分析师"
  2. 语义无关的文本相似度要低
     例："数据分析" vs "电机控制算法"
  3. 同一句话的不同说法相似度最高

用法：
    set HF_ENDPOINT=https://hf-mirror.com
    python tests/check_embedder.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

MODEL = os.environ.get('EMB_MODEL', 'BAAI/bge-small-zh-v1.5')


def main() -> int:
    try:
        from fastembed import TextEmbedding
    except ImportError:
        print('[错误] 未安装 fastembed。pip install fastembed')
        return 1

    print('=' * 72)
    print('真实中文 embedding 语义能力验证')
    print('=' * 72)
    print('  HF_ENDPOINT = %s' % os.environ.get('HF_ENDPOINT'))
    print('  模型        = %s' % MODEL)
    print()

    t0 = time.perf_counter()
    try:
        model = TextEmbedding(model_name=MODEL)
    except Exception as e:
        print('[失败] 加载模型出错: %s' % e)
        print()
        print('该版本 fastembed 支持的模型（前 60 个）：')
        try:
            names = sorted(TextEmbedding.list_supported_models())
            for n in names[:60]:
                print('   ', n)
        except Exception as e2:
            print('   无法列出:', e2)
        return 1
    print('  加载耗时    = %.1f 秒' % (time.perf_counter() - t0))

    import numpy as np

    def vec(t: str):
        v = np.asarray(list(model.embed([t]))[0], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v

    def cos(a: str, b: str) -> float:
        return float(np.dot(vec(a), vec(b)))

    dim = len(vec('测试'))
    print('  向量维度    = %d' % dim)
    print()

    # ---------- 语义相近但字面不同 ----------
    pairs_similar = [
        ('数据分析', '数据挖掘'),
        ('数据分析师', '商业分析师'),
        ('做报表给管理层看', '为业务方提供数据支持与可视化汇报'),
        ('检索增强生成', 'RAG'),
        ('向量数据库', 'pgvector'),
        ('熟练掌握 SQL 和 Excel', '精通数据库查询与表格处理'),
    ]
    # ---------- 语义无关 ----------
    pairs_unrelated = [
        ('数据分析', '电机控制算法'),
        ('数据分析', '光学镜头设计'),
        ('数据分析', '人力资源招聘'),
        ('数据分析', '结构工程师'),
    ]

    print('-' * 72)
    print('① 语义相近但字面不同（期望高，> 0.5）')
    print('-' * 72)
    s1 = []
    for a, b in pairs_similar:
        c = cos(a, b)
        s1.append(c)
        print('  %.4f   %-24s | %s' % (c, a[:24], b[:34]))

    print()
    print('-' * 72)
    print('② 语义无关（期望低，< 0.5）')
    print('-' * 72)
    s2 = []
    for a, b in pairs_unrelated:
        c = cos(a, b)
        s2.append(c)
        print('  %.4f   %-24s | %s' % (c, a[:24], b[:34]))

    print()
    print('-' * 72)
    print('③ 同义句（期望最高）')
    print('-' * 72)
    c3 = cos('负责用户分层与留存分析', '通过用户分群提升留存率')
    print('  %.4f   负责用户分层与留存分析 | 通过用户分群提升留存率' % c3)

    print()
    print('=' * 72)
    print('判定')
    print('=' * 72)
    print('  ① 相近均值 = %.4f' % (sum(s1) / len(s1)))
    print('  ② 无关均值 = %.4f' % (sum(s2) / len(s2)))
    gap = sum(s1) / len(s1) - sum(s2) / len(s2)
    print('  语义间隔   = %.4f' % gap)

    ok = gap > 0.10 and sum(s1) / len(s1) > 0.45
    print()
    if ok:
        print('  ✅ 有语义能力：相近远高于无关，可作为真实向量检索的底座')
    else:
        print('  ❌ 区分度不足，检查模型是否正确加载')
    print()
    print('  对比：hash_embed 占位实现靠字符重叠，"数据分析"与"数据挖掘"')
    print('        只共享"数据"二字，相似度会明显偏低，且与无关词拉不开差距。')

    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
