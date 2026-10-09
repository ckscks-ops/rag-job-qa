# -*- coding: utf-8 -*-
"""直接测试 _split_sections 的分段逻辑，定位为什么只切出 1 块。"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import chunking
from chunking import SECTION_PATTERNS, _split_sections, chunk_jd

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

JD = ('岗位职责：\n1. 负责数据清洗与建模；\n2. 输出分析报告。\n'
      '任职要求：\n1. 熟练使用 SQL；\n2. 掌握 Python 数据分析。\n'
      '加分项：\n有大数据平台经验者优先。')

print('=== 输入 ===')
print(repr(JD[:60]), '...')
print()

# 检查每个段名模式能否单独匹配
print('=== 各段名模式单独测试 ===')
for name, pat in SECTION_PATTERNS:
    m = re.search('(' + pat + r')\s*[:：]', JD)
    print('  %-8s 单独匹配: %s' % (name, bool(m)))

print()
print('=== 组合正则测试 ===')
alternation = '|'.join('(?:%s)' % pat for _, pat in SECTION_PATTERNS)
for lookbehind in ['(?<![A-Za-z0-9])', '(?<![A-Za-z0-9\\u4e00-\\u9fff])', '']:
    pat = lookbehind + '(' + alternation + r')\s*[:：]'
    try:
        hits = list(re.finditer(pat, JD))
        found = [h.group(1) for h in hits]
    except Exception as e:
        found = ['ERROR: %s' % e]
    print('  lookbehind=%-34s 命中 %d: %s'
          % (lookbehind or '(无)', len(found), found))

print()
print('=== 当前代码实际使用的正则 ===')
src = Path(chunking.__file__).read_text(encoding='utf-8')
m = re.search(r"pattern = re\.compile\((.*?)\)\n", src, re.S)
print('  %s' % (m.group(0).strip() if m else '未找到'))

print()
print('=== _split_sections 实际返回 ===')
secs = _split_sections(JD)
print('  段数: %d' % len(secs))
for name, body in secs:
    print('    [%s] len=%d  %r' % (name, len(body), body[:40]))

print()
print('=== chunk_jd 实际返回 ===')
cs = chunk_jd(JD, company='测试公司', title='数据分析师')
print('  块数: %d' % len(cs))
for c in cs:
    print('    #%d [%s] len=%d' % (c['index'], c['section'], len(c['body'])))
