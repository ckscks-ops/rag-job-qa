# -*- coding: utf-8 -*-
"""去重后语料质量分析 —— 决定 RAG 语料怎么裁。"""
import json, re, collections, statistics, io, sys
from pathlib import Path

# 路径相对脚本，别写死绝对路径（别人克隆后路径完全不同）
ROOT = Path(__file__).resolve().parent.parent
P = ROOT / 'data' / 'prod_jobs.json'
if not P.is_file():
    P = ROOT / 'data' / 'corpus.json'
if not P.is_file():
    print('找不到 data/prod_jobs.json 或 data/corpus.json')
    print('先运行：python setup.py data')
    sys.exit(1)
jobs = json.loads(P.read_text(encoding='utf-8'))

# 优先用导出器产出的去重文件，保证和入库用的语料完全一致
DEDUP = P.parent / 'prod_jobs.dedup.json'
dedup_file = json.loads(DEDUP.read_text(encoding='utf-8')) if DEDUP.is_file() else None

out = io.StringIO()
W = lambda s='': out.write(s + '\n')

W('=' * 72)
W('去重分析：%d 条原始岗位' % len(jobs))
W('=' * 72)

if dedup_file is not None:
    dedup = dedup_file
    W('按 JD 正文去重后: %d 条（丢掉重复 %d 条）' % (len(dedup), len(jobs) - len(dedup)))
    W('  [取自 %s，与入库语料一致]' % DEDUP.name)
else:
    # 兜底：自己做一次（空白字符归一化后比较）
    by_raw = {}
    for j in jobs:
        by_raw.setdefault(re.sub(r'\s+', '', j['raw_text']), j)
    dedup = list(by_raw.values())
    W('按 JD 正文去重后: %d 条（丢掉重复 %d 条）' % (len(dedup), len(jobs) - len(dedup)))
W()

# 重复最多的正文
W('重复最多的 JD 正文:')
for j in sorted(dedup, key=lambda x: -x.get('dup_count', 1))[:8]:
    aliases = j.get('dup_companies') or [j['company']]
    W('  x%-5d %-30s  公司别名 %d 个: %s' % (
        j.get('dup_count', 1), j['title'][:30], len(aliases),
        ' / '.join(a[:16] for a in sorted(aliases)[:5])))
W()

# 去重后公司分布
comp = collections.Counter(j['company'] for j in dedup)

byte_like = sum(n for c, n in comp.items() if '字节' in c or 'TikTok' in c or 'CapCut' in c)
W('去重后公司数: %d' % len(comp))
W('  字节系岗位: %d (%.1f%%)' % (byte_like, 100.0 * byte_like / len(dedup)))
W('  非字节系  : %d (%.1f%%)' % (len(dedup) - byte_like, 100.0 * (len(dedup) - byte_like) / len(dedup)))
W()

W('去重后公司 Top 15:')
for c, n in comp.most_common(15):
    W('  %-5d %s' % (n, c[:46]))
W()

# 按岗位方向分桶（看用户关心的「数据分析 / AI 应用」有多少）
KW = {
    '数据分析/商业分析': ['数据分析', '商业分析', '数据科学', 'BI ', '数据运营', '业务分析', '数据产品'],
    '算法/AI': ['算法', '机器学习', '深度学习', '大模型', 'AI', '人工智能', 'NLP', '计算机视觉'],
    '开发/工程': ['开发工程师', '后端', '前端', '软件工程', '测试', '运维', '客户端'],
    '产品/运营': ['产品经理', '运营', '市场', '销售', '策划'],
    '硬件/制造': ['硬件', '电气', '机械', '结构', '工艺', '电控', '光学', '嵌入式'],
}
W('去重后岗位方向分布（按标题关键词，一条可命中多桶）:')
for label, kws in KW.items():
    n = sum(1 for j in dedup if any(k in j['title'] for k in kws))
    W('  %-18s %5d (%.1f%%)' % (label, n, 100.0 * n / len(dedup)))
W()

# 国企/军工线索
SOE = ['集团', '研究所', '重工', '航天', '航空', '兵器', '中船', '中电', '核工业',
       '轨道', '电力', '石化', '石油', '建筑', '建工', '水利', '汽车']
soe = [j for j in dedup if any(k in j['company'] for k in SOE)]
W('公司名含国企/制造特征的: %d 条, %d 家公司' % (len(soe), len({j['company'] for j in soe})))
soe_comp = collections.Counter(j['company'] for j in soe)
for c, n in soe_comp.most_common(18):
    W('  %-5d %s' % (n, c[:46]))
W()

# 去重后 JD 长度
lens = sorted(len(j['raw_text']) for j in dedup)
W('去重后 JD 长度: min=%d 中位=%d p95=%d max=%d' % (
    lens[0], statistics.median(lens), lens[int(len(lens) * .95)], lens[-1]))
W()
W('去重后预计切块数（按平均 3.2 块/岗位估算）: ~%d' % int(len(dedup) * 3.2))

txt = out.getvalue()
_REPORT = Path(__file__).resolve().parent.parent / 'docs' / 'corpus_analysis.txt'
_REPORT.parent.mkdir(parents=True, exist_ok=True)
_REPORT.write_text(txt, encoding='utf-8')
try:
    print(txt)
except UnicodeEncodeError:
    pass
print('[已写出] %s' % _REPORT)
