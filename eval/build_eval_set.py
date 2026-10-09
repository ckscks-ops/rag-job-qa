# -*- coding: utf-8 -*-
"""
build_eval_set.py —— 生成 50 条查询的评测集（规则化标注）

设计要点
=======
1. **规则化标注**：真实语料 2,116 条，逐条人工标注不现实。
   每条查询配一条「相关性规则」（按岗位标题/公司/城市/JD 正文判定），
   由脚本解析成 doc_key 集合。规则是客观、可复现、可审计的。
2. **查询措辞刻意分类**：
   - `改写类`：查询里**不含**规则关键词 → 考语义，纯关键词检索会失手
   - `术语类`：查询就是技术术语 → 考精确匹配，关键词检索应占优
   - `公司类`/`城市类`/`条件类`：考结构化条件
   - `对照类`：窄方向查询，考精度（别把无关岗位排前面）
   这样「纯关键词 vs 纯向量 vs 混合」才有区分度，否则三者都是 1.000。

产出：eval/eval_set.json

用法：
    python eval/build_eval_set.py
    python eval/build_eval_set.py --check-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'eval'))

from evaluate import resolve_rule            # noqa: E402
from schema import connect                   # noqa: E402


QUERIES = [
    # ══════════ 改写类：查询里不含规则关键词，考语义泛化 ══════════
    dict(category='改写类', query='偏业务、经常要给管理层做汇报的岗位',
         rule=dict(title_all=['数据'], title_any=['分析', '科学', '运营']),
         note='规则词是"数据分析/数据科学"，查询里一个都没有'),
    dict(category='改写类', query='从杂乱的数据里找出规律、给决策做支撑',
         rule=dict(title_all=['数据'], title_any=['挖掘', '科学']),
         note='考"数据挖掘/数据科学"的语义近邻'),
    dict(category='改写类', query='负责把模型接进业务系统、调接口落地的岗位',
         rule=dict(title_all=['应用'], title_any=['AI', '人工智能', '大模型']),
         note='口语描述 AI 应用开发'),
    dict(category='改写类', query='做智能问答、知识库检索这类产品的',
         rule=dict(raw_any=['RAG', '检索增强', '知识库']),
         note='不出现 RAG 三个字，考同义理解'),
    dict(category='改写类', query='能从一堆图片里认出东西的算法',
         rule=dict(title_any=['视觉', '图像', '多模态']),
         note='口语描述计算机视觉'),
    dict(category='改写类', query='让机器自己开车的那些岗位',
         rule=dict(title_any=['自动驾驶', '智能驾驶', '具身智能']),
         note='极度口语化'),
    dict(category='改写类', query='写服务端接口、做调度和存储的',
         rule=dict(title_any=['后端']),
         note='不出现"后端"二字'),
    dict(category='改写类', query='管数据仓库、把散在各处的数据汇总起来的',
         rule=dict(title_any=['大数据', '数据开发', '数据工程']),
         note='口语描述数据开发'),
    dict(category='改写类', query='让机器听懂人说话的',
         rule=dict(title_any=['语音', '音频']),
         note='不出现"语音"'),
    dict(category='改写类', query='做推荐或者搜索结果排序的',
         rule=dict(title_any=['推荐', '搜索']),
         note='口语描述搜推'),
    dict(category='改写类', query='做平台底座、给其他团队提供能力的',
         rule=dict(title_any=['平台']),
         note='口语描述平台工程'),
    dict(category='改写类', query='搞安全防护的',
         rule=dict(title_any=['安全']),
         note='口语描述安全岗'),

    # ══════════ 术语类：精确技术栈，关键词检索应占优 ══════════
    dict(category='术语类', query='SQL 和 Excel 都要会',
         rule=dict(raw_all=['SQL', 'Excel']), note='精确技术栈'),
    dict(category='术语类', query='算法岗要会写 C++',
         rule=dict(title_any=['算法'], raw_any=['C++']), note='精确语言名'),
    dict(category='术语类', query='算法岗要会 PyTorch',
         rule=dict(title_any=['算法'], raw_any=['PyTorch']), note='精确框架名'),
    dict(category='术语类', query='后端岗要会 Java',
         rule=dict(title_any=['后端'], raw_any=['Java']), note='精确语言名'),
    dict(category='术语类', query='大数据方向，要会 Spark 或 Hadoop',
         rule=dict(raw_any=['Spark', 'Hadoop']), note='精确组件名'),
    dict(category='术语类', query='要懂 Docker 或 Kubernetes',
         rule=dict(raw_any=['Docker', 'Kubernetes']), note='精确组件名'),
    dict(category='术语类', query='做检索增强生成、向量检索的',
         rule=dict(raw_any=['RAG', '检索增强']), note='精确术语'),
    dict(category='术语类', query='要会 Linux 命令行的算法岗',
         rule=dict(title_any=['算法'], raw_any=['Linux']), note='精确系统名'),
    dict(category='术语类', query='用 MATLAB 或 Simulink 做仿真的',
         rule=dict(raw_any=['MATLAB', 'Simulink']), note='精确工具名'),
    dict(category='术语类', query='算法岗要会 Git 协作',
         rule=dict(title_any=['算法'], raw_any=['Git']), note='精确工具名'),
    dict(category='术语类', query='用 OpenCV 做图像处理的',
         rule=dict(raw_any=['OpenCV']), note='精确库名'),
    dict(category='术语类', query='FPGA 或 Verilog 数字设计',
         rule=dict(raw_any=['FPGA', 'Verilog']), note='精确术语'),

    # ══════════ 方向类：用数据集自带的人工分类做标注（最客观） ══════════
    # 开源数据集（Job-posting-data）自带 label：数据分析 / 算法岗 / 大模型相关 /
    # 经济金融 / 生物统计。这是**人工分类**，比"用标题关键词猜方向"可靠得多。
    # 本机 RecruitOps 那批没有 label，由 src/label_documents.py 用规则补齐
    # （metadata.label_source 区分 'dataset' 与 'rule'）。
    dict(category='方向类', query='我想找数据分析方向的岗位',
         rule=dict(label_any=['数据分析']), note='人工分类标签，最可靠的标注'),
    dict(category='方向类', query='大模型相关的岗位',
         rule=dict(label_any=['大模型相关']), note='人工分类标签'),
    dict(category='方向类', query='算法岗',
         rule=dict(label_any=['算法岗']), note='人工分类标签'),
    dict(category='方向类', query='经济金融方向的机会',
         rule=dict(label_any=['经济金融']), note='人工分类标签'),
    dict(category='方向类', query='生物统计 / 医药数据方向',
         rule=dict(label_any=['生物统计']), note='人工分类标签'),
    dict(category='方向类', query='数据分析方向，本科能投的',
         rule=dict(label_any=['数据分析'], raw_any=['本科']), note='方向 + 学历'),
    dict(category='方向类', query='数据分析方向，在北京的',
         rule=dict(label_any=['数据分析'], city_any=['北京']), note='方向 + 城市'),
    dict(category='方向类', query='大模型方向，要会 PyTorch',
         rule=dict(label_any=['大模型相关'], raw_any=['PyTorch']), note='方向 + 技术栈'),

    # ══════════ 条件类：软性/硬性条件 ══════════
    dict(category='条件类', query='本科就能投的算法岗',
         rule=dict(raw_any=['本科及以上', '本科以上'], title_any=['算法']),
         note='学历条件'),
    dict(category='条件类', query='要求硕士及以上的算法岗',
         rule=dict(raw_any=['硕士及以上', '硕士'], title_any=['算法']),
         note='学历条件'),
    dict(category='条件类', query='算法岗，有竞赛获奖要求',
         rule=dict(raw_any=['竞赛'], title_any=['算法']), note='加分条件'),
    dict(category='条件类', query='专业不限的岗位',
         rule=dict(raw_any=['专业不限', '不限专业']), note='专业条件'),
    dict(category='条件类', query='英语要求比较高的算法岗',
         rule=dict(raw_any=['CET-6', 'CET6', '英语六级'], title_any=['算法']),
         note='语言条件'),
    dict(category='条件类', query='算法岗，JD 里写了加分项',
         rule=dict(raw_any=['加分项'], title_any=['算法']), note='JD 结构条件'),

    # ══════════ 公司类：结构化条件 ══════════
    dict(category='公司类', query='军工/兵器系统的单位',
         rule=dict(company_any=['兵器', '研究所']), note='公司名匹配'),
    dict(category='公司类', query='汽车行业的机会',
         rule=dict(company_any=['汽车']), note='行业匹配'),
    dict(category='公司类', query='外企或英文工作环境',
         rule=dict(company_any=['TikTok', 'Shop', 'Insta360', '瑞声']),
         note='外企匹配'),
    dict(category='公司类', query='AI 创业公司或独角兽',
         rule=dict(company_any=['元戎', '影石', '科大讯飞']), note='创业公司'),

    # ══════════ 城市类 ══════════
    dict(category='城市类', query='想去西安发展',
         rule=dict(city_any=['西安', '陕西']), note='城市匹配'),
    dict(category='城市类', query='芜湖或者安徽的机会',
         rule=dict(city_any=['芜湖', '安徽']), note='城市匹配'),
    dict(category='城市类', query='北京或上海的数据岗',
         rule=dict(city_any=['北京', '上海'], title_any=['数据']), note='城市 + 方向'),
    dict(category='城市类', query='深圳的技术岗',
         rule=dict(city_any=['深圳'], title_any=['算法', '数据', '开发']), note='城市 + 方向'),

    # ══════════ 对照类：窄方向，考精度 ══════════
    dict(category='对照类', query='大模型算法工程师',
         rule=dict(title_all=['大模型'], title_any=['算法']), note='窄方向'),
    dict(category='对照类', query='多模态大模型',
         rule=dict(title_any=['多模态']), note='窄方向'),
    dict(category='对照类', query='具身智能或机器人算法',
         rule=dict(title_any=['具身智能', '机器人']), note='窄方向'),
    dict(category='对照类', query='数据科学/数据分析师',
         rule=dict(title_any=['数据科学', '数据分析师']), note='窄方向'),
]


def main() -> int:
    ap = argparse.ArgumentParser(description='生成/校验评测集')
    ap.add_argument('--db', default=str(ROOT / 'data' / 'rag.db'))
    ap.add_argument('--out', default=str(ROOT / 'eval' / 'eval_set.json'))
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()

    con = connect(args.db)
    total_docs = con.execute('SELECT COUNT(*) FROM documents').fetchone()[0]

    print('=' * 76)
    print('评测集构建')
    print('=' * 76)
    print('  数据库    : %s' % args.db)
    print('  岗位总数  : %d' % total_docs)
    print('  查询条数  : %d' % len(QUERIES))
    print()

    rows, empty, huge = [], [], []
    by_cat: dict[str, list[int]] = {}
    for q in QUERIES:
        keys = resolve_rule(con, q['rule'])
        n = len(keys)
        by_cat.setdefault(q['category'], []).append(n)
        if n == 0:
            empty.append(q['query'])
        # 相关集大本身不是问题（主指标是 Precision@k，不是 Recall@k）：
        # "北京的算法岗"几百条都算相关，Precision@5 照样有意义。
        # 这里只拦**真正退化**的规则（几乎匹配全库），那说明规则写错了。
        if n > max(1000, int(total_docs * 0.6)):
            huge.append((q['query'], n))
        rows.append({**q, 'n_relevant': n})

    print('  %-16s %-4s %-6s %s' % ('类型', 'n', '相关集', '查询'))
    print('  ' + '-' * 70)
    for r in rows:
        flag = '  ' if r['n_relevant'] else '❗'
        print('  %s%-14s %-4d %-6d %s'
              % (flag, r['category'], r['n_relevant'],
                 r['n_relevant'], r['query'][:34]))

    print()
    print('  各类型相关集大小（中位）：')
    for cat, ns in sorted(by_cat.items()):
        ns_sorted = sorted(ns)
        print('    %-16s 中位 %4d   最小 %4d   最大 %4d'
              % (cat, ns_sorted[len(ns_sorted) // 2], ns_sorted[0], ns_sorted[-1]))

    print()
    if empty:
        print('  ❗ %d 条查询解析出空相关集（必须修规则）：' % len(empty))
        for q in empty:
            print('        - %s' % q)
    if huge:
        print('  ❗ %d 条查询相关集过大（规则太宽）：' % len(huge))
        for q, n in huge:
            print('        - %s (%d)' % (q, n))
    if not empty and not huge:
        print('  ✅ 所有查询的相关集都非空且规模合理')

    if args.check_only:
        return 1 if (empty or huge) else 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        [{k: v for k, v in r.items() if k != 'n_relevant'} for r in rows],
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print()
    print('  已写出: %s' % out)
    return 1 if (empty or huge) else 0


if __name__ == '__main__':
    sys.exit(main())
