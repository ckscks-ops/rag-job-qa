# -*- coding: utf-8 -*-
"""
用**真实语料**检查切块质量（tests/smoke_test.py 用的是带冒号的占位数据，
测不出"段名独立成行"这种真实写法）。

用法: python tests/check_chunking_real.py [语料.json] [样本数]
      默认语料 data/corpus.json（合并后的语料）
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from chunking import chunk_jd, _split_sections      # noqa: E402

_a = sys.argv[1:]
SRC = Path(_a[0]) if (_a and _a[0].endswith('.json')) else (ROOT / 'data' / 'corpus.json')
if not SRC.is_absolute():
    SRC = ROOT / SRC
if not SRC.is_file():
    SRC = ROOT / 'data' / 'prod_jobs.dedup.json'
LIMIT = int(_a[1]) if len(_a) > 1 else (int(_a[0]) if (_a and not _a[0].endswith('.json')) else 0)

jobs = json.loads(SRC.read_text(encoding='utf-8'))
if LIMIT:
    jobs = jobs[:LIMIT]

print('=' * 78)
print('真实语料切块检查')
print('=' * 78)
print('  语料: %s  (%d 条岗位)' % (SRC.name, len(jobs)))
print()

per_doc = []
sec_counter = Counter()
no_section = 0
oversize = 0
empty = 0

for j in jobs:
    pieces = chunk_jd(j['raw_text'], company=j.get('company'), title=j.get('title'))
    if not pieces:
        empty += 1
        continue
    per_doc.append(len(pieces))
    if all(p['section'] is None for p in pieces):
        no_section += 1
    for p in pieces:
        sec_counter[p['section'] or '(无段名)'] += 1
        if len(p['body']) > 480:
            oversize += 1

print('  岗位数            : %d' % len(per_doc))
print('  切块总数          : %d' % sum(per_doc))
print('  平均块/岗位       : %.2f' % (sum(per_doc) / len(per_doc) if per_doc else 0))
print('  中位块/岗位       : %d' % (statistics.median(per_doc) if per_doc else 0))
print('  只切出 1 块的岗位 : %d (%.1f%%)'
      % (sum(1 for n in per_doc if n == 1),
         100.0 * sum(1 for n in per_doc if n == 1) / len(per_doc) if per_doc else 0))
print('  完全没识别出段名的岗位: %d (%.1f%%)'
      % (no_section, 100.0 * no_section / len(per_doc) if per_doc else 0))
print('  超长块(>480字)    : %d' % oversize)
print('  切不出块的岗位    : %d' % empty)
print()
print('  段名分布:')
for name, n in sec_counter.most_common(12):
    print('    %-12s %6d' % (name, n))

print()
print('=' * 78)
print('样例（前 3 条，展示段名与正文开头）')
print('=' * 78)
for j in jobs[:3]:
    print()
    print('--- %s | %s' % (j['company'], j['title']))
    print('    原文长度 %d 字' % len(j['raw_text']))
    secs = _split_sections(j['raw_text'])
    print('    识别出的段: %s' % [(s or '(无)') + ':%d字' % len(t) for s, t in secs])
    for p in chunk_jd(j['raw_text'], company=j['company'], title=j['title']):
        print('      [%d] %-8s %d字 | %s'
              % (p['index'], p['section'] or '-', len(p['body']),
                 p['body'][:44].replace('\n', ' ')))

ok = (per_doc and sum(per_doc) / len(per_doc) >= 2.0)
print()
print('=' * 78)
if ok:
    print('  ✅ 平均块/岗位 %.2f >= 2.0，分段切块生效' % (sum(per_doc) / len(per_doc)))
else:
    print('  ❌ 平均块/岗位过低，段名识别仍有问题')
print('=' * 78)
sys.exit(0 if ok else 1)
