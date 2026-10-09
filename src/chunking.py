"""
chunking.py —— 岗位 JD 的智能切块

为什么不能按固定字数切
====================
岗位 JD 天然是**分段结构化**的：岗位职责 / 任职要求 / 加分项 / 福利待遇。
如果按 500 字硬切，会把"任职要求"从中间切断，
检索出来的块语义残缺，生成答案时也会缺少关键条件。

本模块的策略（由粗到细）
====================
1. **先按小标题分段**：识别「岗位职责」「任职要求」「任职资格」「加分项」
   「岗位要求」「工作内容」「你将负责」等常见段名。
2. **段内再按语义单元切**：优先在句号 / 分号 / 换行处切，
   避免在句子中间断开。
3. **短段合并、长段细分**：小于 MIN_CHARS 的段与相邻合并；
   超过 MAX_CHARS 的段按句子贪心切分。
4. **保留上下文前缀**：每块前面带上「公司 · 岗位」，
   这样单独看这一块也知道它属于哪个岗位（检索命中后展示更可读）。

用法
----
    from chunking import chunk_jd
    chunks = chunk_jd(raw_text, company='某公司', title='数据分析师')
    for c in chunks:
        print(c['index'], c['section'], c['text'][:40])
"""

from __future__ import annotations

import re

# 段名模式：命中即认为是新段的开始
# ⚠️ 2026-10-08 扩充：真实语料实测「职位描述」占 60.1%，是最高频的段名，
#    初版没收录；「你将参与 / 对你的期待 / 我们希望你」这类外企写法也没有。
SECTION_PATTERNS = [
    ('岗位职责', r'岗\s*位\s*职\s*责|工\s*作\s*职\s*责|职\s*位\s*描\s*述|职\s*责\s*描\s*述'
                 r'|职\s*位\s*职\s*责|你\s*将\s*负\s*责|你\s*将\s*参\s*与|你\s*的\s*工\s*作\s*内\s*容'
                 r'|工\s*作\s*内\s*容|主\s*要\s*职\s*责|岗\s*位\s*描\s*述'),
    ('任职要求', r'任\s*职\s*要\s*求|岗\s*位\s*要\s*求|任\s*职\s*资\s*格|任\s*职\s*条\s*件'
                 r'|我\s*们\s*期\s*待|我\s*们\s*希\s*望\s*你|我\s*们\s*需\s*要\s*你|你\s*的\s*期\s*待'
                 r'|职\s*位\s*要\s*求|应\s*聘\s*要\s*求|应\s*聘\s*条\s*件|岗\s*位\s*任\s*职\s*要\s*求'
                 r'|任\s*职\s*要\s*求\s*及\s*条\s*件'),
    ('加分项',   r'加\s*分\s*项|加\s*分\s*条\s*件|优\s*势\s*条\s*件|以\s*下\s*条\s*件\s*优\s*先'
                 r'|preferred'),
    ('学历经验', r'学\s*历\s*要\s*求|工\s*作\s*经\s*验|经\s*验\s*要\s*求'),
    ('福利待遇', r'福\s*利\s*待\s*遇|薪\s*资\s*待\s*遇|我\s*们\s*提\s*供|你\s*将\s*获\s*得|待\s*遇'),
    ('公司介绍', r'公\s*司\s*简\s*介|关\s*于\s*我\s*们|团\s*队\s*介\s*绍|团\s*队\s*信\s*息'),
]

MIN_CHARS = 60      # 小于此长度的段尝试与相邻合并
MAX_CHARS = 480     # 大于此长度的段按句子再切
OVERLAP_CHARS = 0   # 段内切分时不重叠（JD 结构清晰，重叠反而引入噪声）


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    """
    按小标题把 JD 切成 (段名, 段文本) 列表。

    支持两种段名写法（真实语料两种都有）
    ------------------------------------
    A. **独立成行**（真实语料里的主流写法，实测占多数）
           岗位职责
           1、负责电机控制算法设计…
           任职要求
           1、硕士及以上…
       可带装饰：【你将参与什么】、（任职要求）、加分项： 都算。

    B. **行内加冒号**
           岗位职责：1、… 任职要求：1、… 加分项：…

    ⚠️ 两个踩过的坑（都是真实数据才暴露的）
    ------------------------------------
    1. 初版只逐行判断"整行是否等于段名"，对**行内段名**完全失效 → 已支持。
    2. 第二版改成"段名后必须跟冒号"，结果对**独立成行的段名**全部漏匹配：
       实测 2,116 条真实岗位 JD 只切出 1.13 块/岗位（应约 3 块），
       等于分段检索这个核心价值完全没生效。
       —— 占位示例数据是带冒号的，所以自测 17/17 全过却没发现。

    判定规则：
      A. 整行（允许首尾空白与【】（）等装饰）就是段名，后面可带可不带冒号
      B. 段名后紧跟「：」或「:」，且其前面不是字母数字
    """
    text = (text or '').replace('\r\n', '\n').replace('\r', '\n').strip()
    if not text:
        return []

    alternation = '|'.join('(?:%s)' % pat for _, pat in SECTION_PATTERNS)
    # 行内版：\s 可能跨行，所以这里保持 \s（段名本身不会含换行，风险可接受）
    inline_re = re.compile(r'(?<![A-Za-z0-9])(' + alternation + r')\s*[:：]')
    # 整行版：把段名模式里的 \s 换成 [ \t]，避免跨行匹配
    alt_line = '|'.join('(?:%s)' % pat.replace(r'\s', '[ \t]')
                        for _, pat in SECTION_PATTERNS)
    line_re = re.compile(
        r'^[ \t]*(?:[【\[（(][ \t]*)?(' + alt_line + r')'
        r'(?:[ \t]*[】\]）)])?[ \t]*[:：]?[ \t]*$',
        re.M)

    hits: list[tuple[int, int, str]] = []      # (段名起点, 正文起点, 原始段名)

    line_spans: list[tuple[int, int]] = []
    for m in line_re.finditer(text):
        nl = text.find('\n', m.end())
        body_start = len(text) if nl < 0 else nl + 1
        hits.append((m.start(), body_start, m.group(1)))
        line_spans.append((m.start(), m.end()))

    for m in inline_re.finditer(text):
        # 已被"整行版"覆盖的跳过
        if any(a <= m.start() and m.end() <= b for a, b in line_spans):
            continue
        hits.append((m.start(), m.end(), m.group(1)))

    if not hits:
        # 没识别出段名 -> 整篇作为一段（无结构 JD）
        return [(None, text)]

    hits.sort(key=lambda x: x[0])

    # 段名模式 -> 规范段名
    def canon(matched: str) -> str:
        flat = re.sub(r'[ \t]', '', matched)
        for name, pat in SECTION_PATTERNS:
            if re.fullmatch(pat, matched) or re.fullmatch(pat, flat):
                return name
        return matched

    sections: list[tuple[str | None, str]] = []

    # 段名之前的内容（若有）作为无段名开头
    if hits[0][0] > 0:
        head = text[:hits[0][0]].strip()
        if head:
            sections.append((None, head))

    for i, (_nstart, bstart, raw) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        body = text[bstart:end].strip()
        sections.append((canon(raw), body))

    # 合并同名相邻段（同一段名出现多次时）
    merged: list[tuple[str | None, str]] = []
    for name, body in sections:
        if not body:
            continue
        if merged and merged[-1][0] == name:
            merged[-1] = (name, merged[-1][1] + '\n' + body)
        else:
            merged.append((name, body))
    return merged


def _split_long(text: str) -> list[str]:
    """超长段按句子贪心切分。"""
    if len(text) <= MAX_CHARS:
        return [text]
    # 在中英文句末标点处切
    pieces = re.split(r'(?<=[。；;！!？?\n])', text)
    out, buf = [], ''
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) <= MAX_CHARS:
            buf += p
        else:
            if buf:
                out.append(buf)
            # 单句就超长 -> 硬切
            while len(p) > MAX_CHARS:
                out.append(p[:MAX_CHARS])
                p = p[MAX_CHARS:]
            buf = p
    if buf:
        out.append(buf)
    return out


def chunk_jd(raw_text: str, *, company: str | None = None,
             title: str | None = None,
             min_chars: int = MIN_CHARS,
             max_chars: int = MAX_CHARS) -> list[dict]:
    """
    把一份 JD 切成若干块。

    返回 list[dict]，每项含：
        index    序号（从 0 开始）
        section  段名（可能为 None）
        text     块文本（已带「公司 · 岗位」前缀）
        body     块正文（不含前缀，便于评测时对比）
    """
    global MIN_CHARS, MAX_CHARS
    MIN_CHARS, MAX_CHARS = min_chars, max_chars

    text = (raw_text or '').strip()
    if not text:
        return []

    sections = _split_sections(text)
    if not sections:
        sections = [(None, text)]

    # 1) 段内切分
    flat: list[tuple[str | None, str]] = []
    for name, body in sections:
        for piece in _split_long(body):
            piece = piece.strip()
            if piece:
                flat.append((name, piece))

    # 2) 合并策略（2026-10-08 二次修正）
    #    ⚠️ 初版规则是"短段就与相邻合并"，实测把 岗位职责(24字) + 任职要求(31字)
    #    + 加分项(12字) 全并成 1 块，语义段边界被抹掉——而分段检索正是 RAG 的价值所在。
    #    第二版改为"只有同名段或无名段才合并"，但实测又暴露两个新问题：
    #      (a) **段名丢失**：JD 开头的无段名总述会把紧跟的「岗位职责」整段吸进来，
    #          结果这一块 section=None，岗位职责标签没了（真实语料里 5% 受影响）
    #      (b) **超长**：合并在切分之后做，没有重新看长度，
    #          实测 490/4513 个块超过 MAX_CHARS（占 10.9%）
    #    现在：合并后若超长则放弃合并；无名段被有段名的段吸收时，**采用后者的段名**。
    merged: list[tuple[str | None, str]] = []
    for name, piece in flat:
        if merged:
            prev_name, prev_text = merged[-1]
            same_section = (prev_name == name)
            neutral_prev = (prev_name is None)
            if (same_section or neutral_prev) and \
                    (len(prev_text) + 1 + len(piece) <= max_chars):
                # 无名段被有段名的段吸收时，改用后者的段名（别丢标签）
                new_name = name if (neutral_prev and name is not None) else prev_name
                merged[-1] = (new_name, prev_text + '\n' + piece)
                continue
        merged.append((name, piece))

    # 3) 加前缀（提升可读性与独立可解释性）
    prefix = ''
    parts = [p for p in (company, title) if p]
    if parts:
        prefix = ' · '.join(parts) + '\n'

    out = []
    for i, (name, body) in enumerate(merged):
        out.append({
            'index': i,
            'section': name,
            'body': body,
            'text': prefix + body,
        })
    return out
