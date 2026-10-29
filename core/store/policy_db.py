# -*- coding: utf-8 -*-
"""制度条款结构化库（MySQL）访问层。

设计说明：
    - policy_document  一行 = 一份制度（含文号、效力状态、施行/废止日期）
    - policy_clause    一行 = 一个"条"（含所属章节、条款号、正文、所属条线）
    - faq_standard      一行 = 一条经过审定的标准问法及其指向的条款

为什么条款要落到 MySQL 而不是只放向量库：
    制度问答有三个需求是向量检索做不到的——
    1) 效力硬过滤：问"贷款额度上限"时，已废止的暂行办法条款必须彻底排除，
       这是 where 条件，不是相似度能解决的问题；
    2) 精确溯源：回答要能追到"《XX办法》第X条"，需要结构化字段而非文本切片；
    3) 版本比对：同一主题的现行版与废止版要能并排取出，供人工复核变更点。
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date
from typing import Any, Iterable, Iterator

import pymysql
from pymysql.cursors import DictCursor

from core.config import config
from core.logger import logger

# ---------- 建表语句 ----------
DDL_POLICY_DOCUMENT = """
CREATE TABLE IF NOT EXISTS policy_document (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    doc_no        VARCHAR(64)  NOT NULL COMMENT '发文字号，如 恒信银发〔2024〕27号',
    title         VARCHAR(255) NOT NULL COMMENT '制度全称',
    issuer        VARCHAR(128)          DEFAULT NULL COMMENT '发文机构',
    business_line VARCHAR(64)           DEFAULT NULL COMMENT '所属业务条线',
    version_tag   VARCHAR(32)           DEFAULT NULL COMMENT '版本标识，如 2024修订',
    status        VARCHAR(16)  NOT NULL DEFAULT 'active' COMMENT 'active/repealed/pending',
    effective_date DATE                 DEFAULT NULL COMMENT '施行日期',
    repeal_date   DATE                  DEFAULT NULL COMMENT '废止日期',
    supersedes    VARCHAR(64)           DEFAULT NULL COMMENT '被本制度替代的旧文号',
    source_file   VARCHAR(255)          DEFAULT NULL COMMENT '原始文件名',
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_doc_no (doc_no),
    KEY idx_status (status),
    KEY idx_line (business_line)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='制度主表';
"""

DDL_POLICY_CLAUSE = """
CREATE TABLE IF NOT EXISTS policy_clause (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    clause_key    VARCHAR(128) NOT NULL COMMENT '全局唯一键：文号#条款号',
    doc_no        VARCHAR(64)  NOT NULL COMMENT '所属制度文号',
    title         VARCHAR(255) NOT NULL COMMENT '所属制度名称，便于引用时直接拼装',
    chapter       VARCHAR(128)          DEFAULT NULL COMMENT '所属章节，如 第三章 授信审批与额度',
    clause_no     VARCHAR(64)  NOT NULL COMMENT '条款号，如 第七条',
    clause_title  VARCHAR(255)          DEFAULT NULL COMMENT '条款小标题',
    content       TEXT         NOT NULL COMMENT '条款正文',
    business_line VARCHAR(64)           DEFAULT NULL COMMENT '所属业务条线',
    status        VARCHAR(16)  NOT NULL DEFAULT 'active' COMMENT '随主表冗余，供检索时直接过滤',
    effective_date DATE                 DEFAULT NULL COMMENT '随主表冗余，供时效过滤',
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_clause_key (clause_key),
    KEY idx_doc_no (doc_no),
    KEY idx_status (status),
    KEY idx_chapter (chapter)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='制度条款表';
"""

DDL_FAQ_STANDARD = """
CREATE TABLE IF NOT EXISTS faq_standard (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    question     VARCHAR(500) NOT NULL COMMENT '审定后的标准问法',
    answer       TEXT         NOT NULL COMMENT '审定答案（人工确认，可绕过生成直接返回）',
    clause_keys  VARCHAR(512)          DEFAULT NULL COMMENT '对应条款键，JSON 数组，如 ["恒信银发〔2024〕27号#第七条"]',
    business_line VARCHAR(64)          DEFAULT NULL,
    source       VARCHAR(32)  NOT NULL DEFAULT 'manual' COMMENT '来源：seed 批量生成 / manual 人工录入',
    hit_count    INT          NOT NULL DEFAULT 0 COMMENT '命中次数，用于调度复审',
    last_verified DATE                 DEFAULT NULL COMMENT '最后一次人工复核日期，超期不再直接返回',
    invalid_at   DATETIME              DEFAULT NULL COMMENT '失效时间。指向条款废止或答案确认错误时置位，置位后不再参与匹配',
    updated_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_question (question),
    KEY idx_invalid (invalid_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='标准问答表';
"""

# 表结构演进：老库里的 faq_standard 缺这三列，启动时补齐。
# 用 ADD COLUMN IF NOT EXISTS 在 MySQL 8.0 不支持，所以先查 information_schema。
FAQ_MIGRATIONS = {
    "clause_keys_type": "ALTER TABLE faq_standard MODIFY COLUMN clause_keys VARCHAR(512) DEFAULT NULL COMMENT '对应条款键，JSON 数组'",
    "source": "ALTER TABLE faq_standard ADD COLUMN source VARCHAR(32) NOT NULL DEFAULT 'manual' COMMENT '来源：seed / manual' AFTER business_line",
    "last_verified": "ALTER TABLE faq_standard ADD COLUMN last_verified DATE DEFAULT NULL COMMENT '最后一次人工复核日期' AFTER hit_count",
    "invalid_at": "ALTER TABLE faq_standard ADD COLUMN invalid_at DATETIME DEFAULT NULL COMMENT '失效时间' AFTER last_verified",
}


class PolicyDB:
    """制度库读写。连接按需创建，用完即关。"""

    def __init__(self) -> None:
        self._conn_params = {
            "host": config.MYSQL_HOST,
            "port": config.MYSQL_PORT,
            "user": config.MYSQL_USER,
            "password": config.MYSQL_PASSWORD,
            "database": config.MYSQL_DATABASE,
            "charset": "utf8mb4",
            "cursorclass": DictCursor,
            "autocommit": False,
        }

    @contextmanager
    def connection(self) -> Iterator[pymysql.connections.Connection]:
        conn = pymysql.connect(**self._conn_params)
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_database(self) -> None:
        """库不存在时先建库。首次部署不该卡在一个 CREATE DATABASE 上。"""
        params = {k: v for k, v in self._conn_params.items()
                  if k not in ("database", "cursorclass")}
        conn = pymysql.connect(**params)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{config.MYSQL_DATABASE}` "
                    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci"
                )
            conn.commit()
        finally:
            conn.close()

    # ---------- 初始化 ----------
    def init_schema(self) -> None:
        """建立三张业务表。可重复执行。"""
        try:
            with self.connection() as conn:
                pass
        except pymysql.err.OperationalError as exc:
            if exc.args and exc.args[0] == 1049:  # Unknown database
                logger.info("数据库 %s 不存在，自动创建", config.MYSQL_DATABASE)
                self._ensure_database()
            else:
                raise

        with self.connection() as conn:
            with conn.cursor() as cur:
                for ddl in (DDL_POLICY_DOCUMENT, DDL_POLICY_CLAUSE, DDL_FAQ_STANDARD):
                    cur.execute(ddl)
            conn.commit()
        self._migrate_faq_columns()
        logger.info("制度库表结构检查完成")

    def _migrate_faq_columns(self) -> None:
        """给老库的 faq_standard 补齐新增列。

        直接用 ALTER TABLE ADD COLUMN 在列已存在时会报 1060，
        所以先查 information_schema 再决定执行哪些。
        """
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COLUMN_NAME FROM information_schema.COLUMNS
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'faq_standard'
                    """,
                    (config.MYSQL_DATABASE,),
                )
                existing = {row["COLUMN_NAME"] for row in cur.fetchall()}
                if not existing:
                    return
                applied = 0
                for column, ddl in FAQ_MIGRATIONS.items():
                    if column == "clause_keys_type":
                        cur.execute(ddl)
                        applied += 1
                    elif column not in existing:
                        cur.execute(ddl)
                        applied += 1
            conn.commit()
        if applied:
            logger.info("faq_standard 补齐 %d 处表结构改动", applied)

    # ---------- 写入 ----------
    def upsert_document(self, doc: dict[str, Any]) -> None:
        sql = """
        INSERT INTO policy_document
            (doc_no, title, issuer, business_line, version_tag, status,
             effective_date, repeal_date, supersedes, source_file)
        VALUES
            (%(doc_no)s, %(title)s, %(issuer)s, %(business_line)s, %(version_tag)s,
             %(status)s, %(effective_date)s, %(repeal_date)s, %(supersedes)s, %(source_file)s)
        ON DUPLICATE KEY UPDATE
            title = VALUES(title),
            issuer = VALUES(issuer),
            business_line = VALUES(business_line),
            version_tag = VALUES(version_tag),
            status = VALUES(status),
            effective_date = VALUES(effective_date),
            repeal_date = VALUES(repeal_date),
            supersedes = VALUES(supersedes),
            source_file = VALUES(source_file)
        """
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, doc)
            conn.commit()

    def replace_clauses(self, doc_no: str, clauses: Iterable[dict[str, Any]]) -> int:
        """整篇替换某制度的条款，避免重复导入产生脏数据。"""
        rows = list(clauses)
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM policy_clause WHERE doc_no = %s", (doc_no,))
                if rows:
                    cur.executemany(
                        """
                        INSERT INTO policy_clause
                            (clause_key, doc_no, title, chapter, clause_no,
                             clause_title, content, business_line, status, effective_date)
                        VALUES
                            (%(clause_key)s, %(doc_no)s, %(title)s, %(chapter)s,
                             %(clause_no)s, %(clause_title)s, %(content)s,
                             %(business_line)s, %(status)s, %(effective_date)s)
                        """,
                        rows,
                    )
            conn.commit()
        logger.info("制度 %s 写入条款 %d 条", doc_no, len(rows))
        return len(rows)

    def mark_repealed(self, doc_no: str, repeal_date) -> None:
        """把某制度标记为已废止，条款表同步失效。"""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE policy_document SET status='repealed', repeal_date=%s WHERE doc_no=%s",
                    (repeal_date, doc_no),
                )
                cur.execute(
                    "UPDATE policy_clause SET status='repealed' WHERE doc_no=%s",
                    (doc_no,),
                )
            conn.commit()
        logger.info("制度 %s 标记为已废止", doc_no)

    # ---------- 读取 ----------
    def get_clause(self, clause_key: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM policy_clause WHERE clause_key = %s", (clause_key,))
                return cur.fetchone()

    def get_clauses_by_keys(self, clause_keys: list[str]) -> list[dict[str, Any]]:
        if not clause_keys:
            return []
        placeholders = ",".join(["%s"] * len(clause_keys))
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM policy_clause WHERE clause_key IN ({placeholders})",
                    clause_keys,
                )
                return list(cur.fetchall())

    def list_documents(self, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM policy_document"
        params: tuple = ()
        if status:
            sql += " WHERE status = %s"
            params = (status,)
        sql += " ORDER BY effective_date DESC"
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())

    def find_history_chain(self, year: int | None = None, doc_no_hint: str = "") -> list[dict[str, Any]]:
        """找出与"某个历史版本"相关的制度，按新旧排序。

        制度换版时，旧办法的文档记录里通常不会写"我被谁废止了"，
        新办法却一定在 supersedes 字段里写"我废止了谁"。
        所以沿 supersedes 反向展开，才能把一条完整的换版链拉出来。

        两条线索合并使用：
            year       问题里出现的年份，匹配旧办法的施行日期；
            doc_no_hint 问题里直接点名的文号，如"〔2021〕9号"。
        命中任一即纳入，最后按施行日期升序——最早的版本在前。
        """
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM policy_document")
                docs = {row["doc_no"]: row for row in cur.fetchall()}

        if not docs:
            return []

        hits: set[str] = set()

        # 线索一：文号直接点名
        if doc_no_hint:
            for doc_no in docs:
                if doc_no_hint in doc_no:
                    hits.add(doc_no)

        # 线索二：年份匹配施行日期
        if year is not None:
            for doc_no, doc in docs.items():
                eff = doc.get("effective_date")
                if eff and getattr(eff, "year", None) == year:
                    hits.add(doc_no)

        # 线索三：向后展开已被现行制度取代的前身版本
        frontier = list(hits)
        while frontier:
            cursor_doc = frontier.pop()
            for doc_no, doc in docs.items():
                if doc_no in hits:
                    continue
                if doc.get("supersedes") == cursor_doc:
                    hits.add(doc_no)
                    frontier.append(doc_no)

        chain = [docs[doc_no] for doc_no in hits]
        chain.sort(key=lambda d: (d.get("effective_date") or date.min))
        return chain

    def get_active_version_of(self, doc_no: str) -> dict[str, Any] | None:
        """查某份历史制度当前被哪份制度取代，用于在答案里指向现行版本。"""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM policy_document WHERE supersedes = %s AND status = 'active' LIMIT 1",
                    (doc_no,),
                )
                return cur.fetchone()

    def stats(self) -> dict[str, int]:
        """首页展示用的统计口径。"""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS c FROM policy_document")
                doc_total = cur.fetchone()["c"]
                cur.execute("SELECT COUNT(*) AS c FROM policy_document WHERE status='active'")
                doc_active = cur.fetchone()["c"]
                cur.execute("SELECT COUNT(*) AS c FROM policy_clause")
                clause_total = cur.fetchone()["c"]
                cur.execute("SELECT COUNT(*) AS c FROM policy_clause WHERE status='active'")
                clause_active = cur.fetchone()["c"]
                cur.execute("SELECT COUNT(*) AS c FROM faq_standard WHERE invalid_at IS NULL")
                faq_total = cur.fetchone()["c"]
        return {
            "document_total": doc_total,
            "document_active": doc_active,
            "clause_total": clause_total,
            "clause_active": clause_active,
            "faq_total": faq_total,
        }

    # ---------- FAQ ----------
    def upsert_faq(self, faq: dict[str, Any]) -> None:
        """写入一条标准问答。question 唯一，重复写入覆盖答案。

        clause_keys 统一按 JSON 数组存：条款键里含「〔〕」和「#」，
        用逗号切分迟早出问题（条款键本身也可能带逗号）。
        """
        keys = faq.get("clause_keys") or []
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split(",") if k.strip()]
        sql = """
        INSERT INTO faq_standard
            (question, answer, clause_keys, business_line, source, last_verified)
        VALUES
            (%(question)s, %(answer)s, %(clause_keys)s, %(business_line)s,
             %(source)s, %(last_verified)s)
        ON DUPLICATE KEY UPDATE
            answer = VALUES(answer),
            clause_keys = VALUES(clause_keys),
            business_line = VALUES(business_line),
            source = VALUES(source),
            last_verified = VALUES(last_verified),
            invalid_at = NULL
        """
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, {
                "question": faq["question"],
                "answer": faq["answer"],
                "clause_keys": json.dumps(keys, ensure_ascii=False),
                "business_line": faq.get("business_line"),
                "source": faq.get("source", "manual"),
                "last_verified": faq.get("last_verified") or date.today(),
            })
            conn.commit()

    def list_faq(self, include_invalid: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM faq_standard"
        if not include_invalid:
            sql += " WHERE invalid_at IS NULL"
        sql += " ORDER BY id"
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(sql)
            return list(cur.fetchall())

    def invalidate_faq(self, question: str, reason: str = "") -> None:
        """标记一条 FAQ 失效。不删除——保留是为了事后能查"它当初写了什么"。"""
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE faq_standard SET invalid_at = NOW() WHERE question = %s",
                (question,),
            )
            conn.commit()
        logger.warning("FAQ 已标记失效：%s（%s）", question[:30], reason or "未注明原因")

    def check_faq_validity(self) -> dict[str, int]:
        """扫全部 FAQ，检查其指向的条款是否仍现行有效、是否超期未复核。

        这是 FAQ 层最容易被忽略、也最容易出事的一环：
        审定答案写死之后，制度改版了它不会跟着变，而因为它绕过检索，
        错的答案会稳定地错很久。

        返回 {"unchecked": 未被检查的, "invalidated": 本次置失效的, "stale": 超期未复核的}
        """
        stale_days = config.FAQ_VERIFY_TTL_DAYS
        today = date.today()
        invalidated, stale = [], []

        for row in self.list_faq():
            keys = row.get("clause_keys") or []
            if isinstance(keys, str):
                try:
                    keys = json.loads(keys)
                except json.JSONDecodeError:
                    keys = [k.strip() for k in keys.split(",") if k.strip()]

            # ① 指向的条款若已不现行有效，这条 FAQ 的答案就不可信
            dead = []
            for key in keys:
                doc_no, _, clause_no = str(key).partition("#")
                with self.connection() as conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT c.status AS clause_status, d.status AS doc_status
                        FROM policy_clause c
                        JOIN policy_document d ON d.doc_no = c.doc_no
                        WHERE c.doc_no = %s AND c.clause_no = %s
                        LIMIT 1
                        """,
                        (doc_no, clause_no),
                    )
                    hit = cur.fetchone()
                if not hit or hit["clause_status"] != "active" or hit["doc_status"] != "active":
                    dead.append(str(key))

            if dead:
                self.invalidate_faq(row["question"], f"指向条款已失效：{'、'.join(dead[:2])}")
                invalidated.append(row["question"])
                continue

            # ② 超期未复核：不置失效，只记录，命中时由调用方降级处理
            verified = row.get("last_verified")
            if not verified or (today - verified).days > stale_days:
                stale.append(row["question"])

        return {"invalidated": len(invalidated), "stale": len(stale)}


db = PolicyDB()
