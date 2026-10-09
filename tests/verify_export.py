# -*- coding: utf-8 -*-
"""核验 prod_jobs.json 的真实性与可用性。"""
import json, re, collections, statistics, io, sys
from pathlib import Path

# 路径相对脚本，别写死绝对路径（别人克隆后路径完全不同）
ROOT = Path(__file__).resolve().parent.parent
P = ROOT / 'data' / 'prod_jobs.json'
if not P.is_file():
    # 也兼容合并后的语料
    P = ROOT / 'data' / 'corpus.json'
if not P.is_file():
    print('找不到 data/prod_jobs.json 或 data/corpus.json')
    print('先运行：python setup.py data')
    sys.exit(1)
jobs = json.loads(P.read_text(encoding='utf-8'))

out = io.StringIO()
W = lambda s='': out.write(s + '\n')

W('=' * 72)
W('prod_jobs.json 核验报告')
W('=' * 72)
W('文件大小 : %.2f MB' % (P.stat().st_size / 1024 / 1024))
W('岗位条数 : %d' % len(jobs))
W('字段     : %s' % sorted(jobs[0].keys()))
W()

# ---- 唯一性 ----
keys = [j['doc_key'] for j in jobs]
raws = [j['raw_text'] for j in jobs]
W('doc_key 唯一 : %d / %d' % (len(set(keys)), len(keys)))
W('raw_text 唯一: %d / %d' % (len(set(raws)), len(raws)))
W()

# ---- 正文质量 ----
lens = sorted(len(r) for r in raws)
W('JD 长度: min=%d p25=%d 中位=%d p75=%d p95=%d max=%d' % (
    lens[0], lens[len(lens)//4], statistics.median(lens),
    lens[len(lens)*3//4], lens[int(len(lens)*.95)], lens[-1]))
W()

# 是否有 HTML / 乱码 / JSON 垃圾
html = sum(1 for r in raws if re.search(r'<\s*(div|p|br|span|li)\b', r, re.I))
brack = sum(1 for r in raws if r.count('{') > 3 and r.count('"') > 10)
ctrl = sum(1 for r in raws if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', r))
repl = sum(1 for r in raws if '\ufffd' in r)
W('含 HTML 标签的 : %d' % html)
W('像 JSON/字典的 : %d' % brack)
W('含控制字符的   : %d' % ctrl)
W('含 U+FFFD 的   : %d' % repl)
W()

# 常见 JD 段名出现率（说明是结构化 JD 而不是碎片）
for name in ['岗位职责', '工作职责', '职位描述', '任职要求', '任职资格', '岗位要求',
             '加分项', '我们希望你', '你将', '负责']:
    n = sum(1 for r in raws if name in r)
    W('  含「%-6s」: %5d (%.1f%%)' % (name, n, 100.0*n/len(raws)))
W()

# ---- 公司与城市 ----
comp = collections.Counter(j['company'] for j in jobs)
W('公司数 %d，Top 20：' % len(comp))
for c, n in comp.most_common(20):
    W('  %-6d %s' % (n, c[:44]))
W()

city = collections.Counter((j['city'] or '(空)').strip() for j in jobs)
W('城市取值 %d 个，Top 20：' % len(city))
for c, n in city.most_common(20):
    W('  %-6d %s' % (n, c[:44]))
W()

src = collections.Counter(j['source'] for j in jobs)
W('来源: %s' % dict(src))
W()

# ---- 样例正文 ----
W('=' * 72)
W('样例 JD（各取前 700 字符）')
W('=' * 72)
for i in (0, 1, 2, 4000):
    j = jobs[i]
    W()
    W('--- [%d] %s | %s | %s | %s' % (i, j['company'], j['title'], j['city'], j['source']))
    W(j['raw_text'][:700].replace('\r', ''))
W()

txt = out.getvalue()
_REPORT = Path(__file__).resolve().parent.parent / 'docs' / 'export_verify.txt'
_REPORT.parent.mkdir(parents=True, exist_ok=True)
_REPORT.write_text(txt, encoding='utf-8')
try:
    print(txt)
except UnicodeEncodeError:
    pass
print('[已写出] %s' % _REPORT)
