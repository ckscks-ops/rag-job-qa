# -*- coding: utf-8 -*-
"""
label_documents.py —— 给没有方向标签的岗位补上 label

背景
====
合并语料有两个来源：
  - **开源数据集**（9,218 条）：自带人工分类 label（数据分析/算法岗/大模型相关/经济金融/生物统计）
  - **本机 RecruitOps 导出**（2,116 条）：没有 label

如果不补齐，按 `label_any` 做评测标注时，那 2,116 条会全部被当成"不相关"，
**正确检索到的校招岗位反而被算成错误**，指标被系统性低估。

本脚本用标题关键词给缺 label 的文档打上同一个体系的标签，让字段在全库一致。

⚠️ 诚实说明
    这是**规则打标**，不是人工标注，和来源数据集自带的标签不是一回事。
    README 里要写清楚：全库 label = 「数据集自带（9,218）」+「规则推断（2,116）」。
    metadata 里会加 `label_source` 字段区分二者。

用法
----
    python src/label_documents.py --dry-run     # 只看会打成什么，不改库
    python src/label_documents.py               # 实际写入
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from schema import connect          # noqa: E402

# 与开源数据集同一套标签体系
RULES = [
    # 顺序有意义：先匹配更专的
    ('大模型相关', ['大模型', 'LLM', 'AIGC', '生成式', 'Agent', '多模态', '语言模型',
                    'NLP', '自然语言', '提示词', 'Prompt']),
    ('数据分析',   ['数据分析', '商业分析', '数据科学', '数据运营', '数据挖掘', 'BI',
                    '数据工程', '大数据', '报表', '数据产品', '算法工程师-数据']),
    ('算法岗',     ['算法', '机器学习', '深度学习', '计算机视觉', '视觉', '感知',
                    '推荐', '搜索', '语音', '自动驾驶', '具身智能', '机器人']),
    ('生物统计',   ['生物', '医药', '医学', '临床', '统计', '药物', '基因', '生命科学']),
    ('经济金融',   ['金融', '经济', '风控', '量化', '投资', '财务', '会计', '审计',
                    '银行', '证券', '保险', '基金']),
]
DEFAULT = '其他'


def classify(title: str, raw: str = '') -> str:
    t = (title or '')
    for label, kws in RULES:
        if any(k in t for k in kws):
            return label
    # 标题没命中就看正文开头（JD 里通常写明方向）
    head = (raw or '')[:200]
    for label, kws in RULES:
        if any(k in head for k in kws):
            return label
    return DEFAULT


def main() -> int:
    ap = argparse.ArgumentParser(description='给缺少方向标签的岗位补 label')
    ap.add_argument('--db', default=str(ROOT / 'data' / 'rag.db'))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    con = connect(args.db)
    rows = con.execute(
        "SELECT id, title, raw_text, metadata FROM documents "
        "WHERE json_extract(metadata, '$.label') IS NULL").fetchall()
    total = con.execute('SELECT COUNT(*) FROM documents').fetchone()[0]

    print('=' * 76)
    print('补方向标签')
    print('=' * 76)
    print('  数据库    : %s' % args.db)
    print('  岗位总数  : %d' % total)
    print('  缺 label  : %d' % len(rows))
    print()

    if not rows:
        print('  无需处理。')
        con.close()
        return 0

    updates, dist = [], Counter()
    for r in rows:
        label = classify(r['title'], r['raw_text'])
        dist[label] += 1
        try:
            meta = json.loads(r['metadata'] or '{}')
        except json.JSONDecodeError:
            meta = {}
        meta['label'] = label
        meta['label_source'] = 'rule'          # 与数据集自带的 'dataset' 区分
        updates.append((json.dumps(meta, ensure_ascii=False), r['id']))

    print('  将打标分布（共 %d 条）：' % len(updates))
    for k, v in dist.most_common():
        print('    %-12s %5d' % (k, v))
    print()

    # 抽样看看打得对不对
    print('  抽样 8 条（标题 -> 标签）：')
    for (meta, _did), r in list(zip(updates, rows))[:8]:
        print('    %-38s -> %s' % ((r['title'] or '')[:38],
                                   json.loads(meta)['label']))

    if args.dry_run:
        print()
        print('  [--dry-run] 未写入。')
        con.close()
        return 0

    con.executemany('UPDATE documents SET metadata=? WHERE id=?', updates)
    con.commit()

    after = con.execute(
        "SELECT json_extract(metadata,'$.label') AS l, COUNT(*) FROM documents "
        "GROUP BY l ORDER BY 2 DESC").fetchall()
    print()
    print('  写入完成。全库 label 分布：')
    for r in after:
        print('    %-12s %5d' % (r[0] or '(无)', r[1]))
    con.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
