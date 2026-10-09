"""
schema.py —— RAG 岗位问答系统的数据层（SQLite + FTS5）

设计要点
========
1. **表结构预留向量列**：`chunks.embedding` 现在是 BLOB（numpy float32 原始字节），
   迁移到 PostgreSQL + pgvector 时改成 `vector(1024)` 即可，其余表结构不变。
2. **中文分词用 jieba**：SQLite 内置 unicode61 分词器把整句中文当一个词条
   （实测 '数据分析工程师，负责用户分层' 只切出 2 个词条，MATCH '数据分析' 命中 0 条），
   因此额外维护一张 `chunks_fts` 存**空格分隔的分词结果**，用 bm25() 排序。
3. **原文与分词分离**：`chunks.text` 存原文（用于展示与生成），
   `chunks_fts` 存分词（用于检索）。两者通过 rowid 对应。

用法
----
    from schema import connect, init_db, insert_document, insert_chunks
    con = connect('data/rag.db')
    init_db(con)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import jieba

# ─────────────── 分词 ───────────────
# ⚠️ 为什么不能只靠 jieba.add_word
#   jieba 的词典权重不保证"更长优先"。实测把「数据分析师」加入词典后，
#   文本「数据分析师，负责用户分层」仍被切成 ['数据', '分析师']——
#   领域词被切碎，检索时无法作为一个整体命中。
#   因此改用 **Aho-Corasick 式的多模式匹配**：
#   先把文本中出现的领域词按长度降序切开并加保护符，剩余部分再交给 jieba。
#   这样「数据分析师」会被整体识别，且不影响其他分词。
DOMAIN_WORDS = [
    # 数据方向
    '数据分析师', '数据分析', '数据挖掘', '数据开发', '数据仓库', '数据治理',
    '数据建模', '数据科学', '数据工程', '商业分析', '用户分层', '用户画像',
    '指标体系', 'A/B 实验', 'AB 实验', '假设检验', '回归分析', '数据可视化',
    # 算法 / AI 方向
    '机器学习', '深度学习', '自然语言处理', '计算机视觉', '推荐算法',
    '大模型', '提示词', '向量检索', '向量数据库', '知识库', '智能体',
    '检索增强', '语义检索', '微调',
    # 开发方向
    '后端开发', '前端开发', '全栈开发', '微服务', '分布式', '嵌入式',
    # 岗位 / 招聘词
    '产品经理', '项目管理', '需求分析', '用户增长',
    '校招', '秋招', '提前批', '实习', '转正', '应届生', '本科及以上', '硕士及以上',
    # 动作词
    '熟练使用', '熟练', '熟悉', '掌握', '了解', '负责', '参与', '独立完成',
    # 技术栈
    'Hadoop', 'Spark', 'Flink', 'Kafka', 'Hive', 'ClickHouse', 'Elasticsearch',
    'PostgreSQL', 'pgvector', 'MySQL', 'Redis', 'MongoDB', 'Milvus', 'FAISS',
    'PyTorch', 'TensorFlow', 'scikit-learn', 'LightGBM', 'LangChain', 'LlamaIndex',
    'FastAPI', 'Django', 'Spring', 'Go 语言', 'C++', 'STM32',
    'Power BI', 'Tableau', 'Excel',
]

# 按长度降序，保证长词先匹配
_SORTED_WORDS = sorted(set(DOMAIN_WORDS), key=len, reverse=True)
_loaded = False


def load_domain_words():
    """把领域词加入 jieba 词典（辅助，主要依靠 _protect_domain_words）。"""
    global _loaded
    if _loaded:
        return
    for w in DOMAIN_WORDS:
        jieba.add_word(w)
    _loaded = True


# 保护符：不会出现在正常文本里，且 jieba 会把它当分隔
_GUARD = '\ue000'


def _protect_domain_words(text: str) -> str:
    """
    用长度降序的多模式匹配，把领域词两边加保护符，
    使 jieba 无法把领域词从中间切开。
    """
    for w in _SORTED_WORDS:
        if w in text:
            text = text.replace(w, '%s%s%s' % (_GUARD, w, _GUARD))
    return text


def tokenize(text: str) -> str:
    """中文分词，返回空格分隔的词串，供 FTS5 检索。"""
    load_domain_words()
    if not text:
        return ''
    protected = _protect_domain_words(str(text))
    words = [w.strip() for w in jieba.cut(protected) if w.strip()]
    # 去掉纯标点/空白，保留字母数字与中文；去掉保护符
    out = []
    for w in words:
        w = w.replace(_GUARD, '')
        if w and any(c.isalnum() or '\u4e00' <= c <= '\u9fff' for c in w):
            out.append(w)
    return ' '.join(out)


# ─────────────── 表结构 ───────────────
SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- 公司（可选，来自岗位数据）
CREATE TABLE IF NOT EXISTS companies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    company_type TEXT,                      -- 外企/国企/民企/大厂 等
    city        TEXT
);

-- 岗位文档：一条 = 一个岗位
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_key      TEXT    NOT NULL UNIQUE,   -- 业务唯一键（如 公司+岗位+城市）
    company      TEXT,
    title        TEXT,                      -- 岗位名称
    city         TEXT,
    job_type     TEXT,                      -- 校招/实习/提前批
    source       TEXT,                      -- 数据来源
    raw_text     TEXT    NOT NULL,          -- JD 原文
    metadata     TEXT,                      -- JSON，扩展字段
    created_at   TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_documents_company ON documents(company);
CREATE INDEX IF NOT EXISTS idx_documents_title   ON documents(title);

-- 切块：一条岗位可切成多块（按职责/要求分段）
CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,          -- 在文档内的序号
    section      TEXT,                      -- 段名，如 "岗位职责" / "任职要求"
    text         TEXT    NOT NULL,          -- 原文
    token_count  INTEGER,
    embedding    BLOB,                      -- ★ 预留：numpy float32 原始字节
    embedding_model TEXT,                   -- ★ 记录用了哪个模型，便于换模型后区分
    UNIQUE(doc_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- FTS5 全文索引：存**分词结果**，用 bm25() 排序
--
-- ⚠️ contentless_delete=1 是必需的，不是可选优化（2026-10-09 踩坑）
--    这个表是 **contentless**（content=''），默认**不允许 DELETE**——
--    但 insert_chunk 需要"先删同 rowid 再插"来支持 (doc_id, chunk_index) 重复插入。
--    实测行为很隐蔽：
--        DELETE 一个**不存在**的 rowid  -> 成功（SQLite 直接跳过 xUpdate）
--        DELETE 一个**存在**的 rowid    -> sqlite3.OperationalError:
--                                          cannot DELETE from contentless fts5 table
--    所以它在"全新入库"时一直不报错（DELETE 总在 INSERT 之前、rowid 还不存在），
--    直到某次出现重复插入才炸。加 contentless_delete=1 后两者都正常。
--    该选项需要 SQLite >= 3.43（本机 3.53.1）。
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    tokens,
    content='',                             -- 不存原文，只存索引
    contentless_delete=1,                   -- 允许按 rowid 删除
    tokenize='unicode61'
);

-- 检索日志（用于评测与调试）
CREATE TABLE IF NOT EXISTS query_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    query      TEXT NOT NULL,
    mode       TEXT,                        -- keyword / vector / hybrid
    top_ids    TEXT,                        -- JSON 数组
    latency_ms REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- 评测集：查询 -> 相关文档
CREATE TABLE IF NOT EXISTS eval_set (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query       TEXT NOT NULL,
    relevant_keys TEXT NOT NULL,            -- JSON 数组，存 doc_key
    note        TEXT
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """打开数据库连接（自动建父目录）。"""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    return con


def init_db(con: sqlite3.Connection) -> None:
    """建表 + 必要时迁移 FTS 表结构。"""
    con.executescript(SCHEMA_SQL)
    _migrate_fts_if_needed(con)
    con.commit()


def _migrate_fts_if_needed(con: sqlite3.Connection) -> None:
    """
    老库里的 chunks_fts 是 contentless 且**没有** contentless_delete=1，
    那种表不支持按 rowid 删除，会让重复插入的入库直接崩掉。
    这里检测出来就重建并从 chunks 重新灌一遍索引。
    """
    row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
    ).fetchone()
    if not row or not row[0] or 'contentless_delete' in row[0]:
        return

    print('  [迁移] chunks_fts 缺少 contentless_delete，正在重建索引 …')
    n_before = con.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
    con.execute('DROP TABLE IF EXISTS chunks_fts')
    con.executescript(SCHEMA_SQL)               # 用新定义重建
    rows = con.execute('SELECT id, text FROM chunks').fetchall()
    for cid, text in rows:
        con.execute('INSERT INTO chunks_fts (rowid, tokens) VALUES (?,?)',
                    (cid, tokenize(text)))
    con.commit()
    print('  [迁移] 完成，已重建 %d 条索引（chunks 原有 %d 条）'
          % (len(rows), n_before))


def reset_db(con: sqlite3.Connection) -> None:
    """
    清空所有数据（保留表结构）。评测/重跑入库前常用。

    ⚠️ chunks_fts 是 **contentless FTS5 表**（`content=''`），
    对它执行 `DELETE FROM chunks_fts` **不会真正清空**，旧版 SQLite 会直接报错——
    而初版把异常 try/except 吞掉了，于是：
        - FTS 行数 > chunks 行数（实测 2453 vs 2384）
        - bm25 会命中已经不存在的 chunk，关键词检索结果被污染
    正确做法是 DROP 掉再重建。
    """
    for t in ('chunks', 'documents', 'companies', 'query_log'):
        try:
            con.execute('DELETE FROM %s' % t)
        except sqlite3.OperationalError as e:
            print('  [警告] 清空 %s 失败: %s' % (t, e))
    con.execute('DROP TABLE IF EXISTS chunks_fts')
    con.executescript(SCHEMA_SQL)     # 重建 FTS 表
    con.commit()


# ─────────────── 写入 ───────────────
def insert_document(con: sqlite3.Connection, *, doc_key: str, raw_text: str,
                    company: str | None = None, title: str | None = None,
                    city: str | None = None, job_type: str | None = None,
                    source: str | None = None,
                    metadata: dict | None = None,
                    commit: bool = True) -> int:
    """插入一条岗位文档，返回 doc_id（已存在则返回既有 id）。"""
    cur = con.execute(
        """INSERT INTO documents (doc_key, company, title, city, job_type, source, raw_text, metadata)
           VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(doc_key) DO UPDATE SET
             company=excluded.company, title=excluded.title, city=excluded.city,
             job_type=excluded.job_type, source=excluded.source,
             raw_text=excluded.raw_text, metadata=excluded.metadata
           RETURNING id""",
        (doc_key, company, title, city, job_type, source, raw_text,
         json.dumps(metadata or {}, ensure_ascii=False)),
    )
    row = cur.fetchone()
    if row is None:                       # RETURNING 在旧版本不可用
        row = con.execute('SELECT id FROM documents WHERE doc_key=?', (doc_key,)).fetchone()
    if commit:
        con.commit()
    return row[0]


def insert_chunk(con: sqlite3.Connection, *, doc_id: int, chunk_index: int,
                 text: str, section: str | None = None,
                 embedding: bytes | None = None,
                 embedding_model: str | None = None,
                 commit: bool = True) -> int:
    """插入一个切块，并同步写入 FTS 索引。返回 chunk_id。"""
    toks = tokenize(text)
    cur = con.execute(
        """INSERT INTO chunks (doc_id, chunk_index, section, text, token_count,
                               embedding, embedding_model)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(doc_id, chunk_index) DO UPDATE SET
             section=excluded.section, text=excluded.text,
             token_count=excluded.token_count,
             embedding=excluded.embedding, embedding_model=excluded.embedding_model
           RETURNING id""",
        (doc_id, chunk_index, section, text, len(toks.split()),
         embedding, embedding_model),
    )
    row = cur.fetchone()
    chunk_id = row[0] if row else con.execute(
        'SELECT id FROM chunks WHERE doc_id=? AND chunk_index=?',
        (doc_id, chunk_index)).fetchone()[0]

    # 维护 FTS：先删同 rowid 再插
    con.execute('DELETE FROM chunks_fts WHERE rowid=?', (chunk_id,))
    con.execute('INSERT INTO chunks_fts (rowid, tokens) VALUES (?,?)', (chunk_id, toks))
    if commit:
        con.commit()
    return chunk_id


def update_chunk_embedding(con: sqlite3.Connection, chunk_id: int,
                           embedding: bytes | None,
                           embedding_model: str | None = None,
                           commit: bool = True) -> None:
    """回写某个切块的向量（配合批量向量化：先入库、后统一算向量）。"""
    con.execute(
        'UPDATE chunks SET embedding=?, embedding_model=? WHERE id=?',
        (embedding, embedding_model, chunk_id),
    )
    if commit:
        con.commit()


def update_chunk_embeddings_bulk(con: sqlite3.Connection,
                                 rows: list[tuple[bytes, str | None, int]]) -> int:
    """批量回写向量。rows = [(blob, model, chunk_id), ...]。返回写入条数。"""
    if not rows:
        return 0
    con.executemany(
        'UPDATE chunks SET embedding=?, embedding_model=? WHERE id=?', rows)
    con.commit()
    return len(rows)


def stats(con: sqlite3.Connection) -> dict:
    """返回当前库的统计信息。"""
    g = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
    return {
        'documents': g('SELECT COUNT(*) FROM documents'),
        'chunks': g('SELECT COUNT(*) FROM chunks'),
        'fts_rows': g('SELECT COUNT(*) FROM chunks_fts'),
        'with_embedding': g('SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL'),
        'companies': g('SELECT COUNT(DISTINCT company) FROM documents WHERE company IS NOT NULL'),
    }
