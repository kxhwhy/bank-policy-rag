# -*- coding: utf-8 -*-
"""系统编排层：一次问答的完整链路。

    query
      → 缓存命中？（同一问题同一基准日直接返回，省一次模型调用）
      → 策略判定 → 查询改写（子查询 / 回溯 / 上下文补全）
      → 多路召回 → 时效过滤 → 阈值截断
      → 无依据 → 拒答
      → 有依据 → 生成 → 引用校验 → 未通过则降级为条款直读
      → 写缓存 → 返回（含条款溯源信息）

会话历史只保留最近若干轮，且只用于补全指代，不作为回答依据——
制度问答里把历史当依据，很容易把上一轮回错的内容带进本轮。
"""
from __future__ import annotations

import uuid
from collections import defaultdict, deque
from datetime import date
from typing import Iterator

from core.config import config
from core.rag.answer import generator
from core.logger import logger
from core.rag.faq_search import faq_search
from core.rag.query_analyzer import STRATEGY_HISTORY, STRATEGY_SUBQUERY, analyzer
from core.rag.retriever import RetrievedClause as Clause
from core.rag.retriever import retriever
from core.store.policy_db import db
from core.store.qa_cache import cache

MAX_HISTORY_TURNS = 3


class PolicyQASystem:
    def __init__(self) -> None:
        self.logger = logger
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY_TURNS * 2))

    # ---------- 历史 ----------
    def _history_text(self, session_id: str) -> str:
        return "\n".join(self._history[session_id])

    def _remember(self, session_id: str, question: str, answer: str) -> None:
        self._history[session_id].append(f"用户：{question}")
        # 只留回答开头，避免历史把上下文预算撑爆
        self._history[session_id].append(f"助手：{answer[:200]}")

    def reset_session(self, session_id: str) -> None:
        self._history.pop(session_id, None)

    # ---------- 检索 ----------
    def _collect_clauses(
        self,
        queries: list[str],
        business_line: str | None,
        as_of: date | None,
        include_repealed: bool = False,
    ) -> list[Clause]:
        """多路查询分别召回后合并去重，取相似度最高的若干条。"""
        merged: dict[str, Clause] = {}
        for q in queries:
            result = retriever.retrieve(
                q,
                business_line=business_line,
                as_of=as_of,
                include_repealed=include_repealed,
            )
            for clause in result.clauses:
                existing = merged.get(clause.clause_key)
                if existing is None or clause.similarity > existing.similarity:
                    merged[clause.clause_key] = clause

        clauses = sorted(merged.values(), key=lambda c: c.similarity, reverse=True)
        return clauses[: config.FINAL_TOP_M]

    def _resolve_scope(
        self,
        question: str,
        strategy: str,
        business_line: str | None,
    ) -> tuple[str | None, bool]:
        """按策略决定检索范围：条线过滤与是否放行废止条款。

        放在一处统一判定，是因为这条链路上有两个独立的"过滤"开关，
        分散判断很容易出现"问了旧版本但仍然被条线过滤掉"的错配。
        """
        line = business_line or analyzer.infer_business_line(question)
        include_repealed = strategy == STRATEGY_HISTORY
        if include_repealed:
            logger.info("历史版本检索：放行已废止条款 ← %s", question[:30])
        return line, include_repealed

    # ---------- 主入口（非流式） ----------
    def ask(
        self,
        question: str,
        business_line: str | None = None,
        session_id: str | None = None,
        as_of: date | None = None,
        use_cache: bool = True,
    ) -> dict:
        question = (question or "").strip()
        if not question:
            return {"error": "问题不能为空"}

        session_id = session_id or str(uuid.uuid4())
        as_of_str = (as_of or date.today()).isoformat()
        cache_key = cache.make_answer_key(question, business_line, as_of_str)
        use_cache = use_cache and config.ENABLE_ANSWER_CACHE

        if use_cache:
            hit = cache.get(cache_key)
            if hit:
                hit["session_id"] = session_id
                hit["from_cache"] = True
                return hit

        # FAQ 层：审定过的标准问法直接返回，省掉检索 + 重排 + 生成的整条链路。
        # 放在策略判定之前，因为 FAQ 命中时根本不需要判断策略。
        faq_payload = self._try_faq(question, business_line, as_of_str, session_id)
        if faq_payload is not None:
            if use_cache:
                cache.set(cache_key, faq_payload)
            self._remember(session_id, question, faq_payload["answer"])
            return faq_payload

        has_history = len(self._history[session_id]) > 0
        strategy = analyzer.select(question, has_history=has_history)
        queries = analyzer.rewrite(question, strategy, self._history_text(session_id))
        self.logger.info("问题=%s | 策略=%s | 检索式=%s", question[:40], strategy, queries)

        line, include_repealed = self._resolve_scope(question, strategy, business_line)
        clauses = self._collect_clauses(queries, line, as_of, include_repealed)
        answer = generator.generate(question, clauses)

        payload = {
            "session_id": session_id,
            "question": question,
            "strategy": strategy,
            "queries": queries,
            "business_line": line,
            "as_of": as_of_str,
            "from_cache": False,
            **answer.to_dict(),
        }

        if use_cache and not answer.refused:
            cache.set(cache_key, payload)

        self._remember(session_id, question, answer.text)
        return payload

    # ---------- 主入口（流式） ----------
    def ask_stream(
        self,
        question: str,
        business_line: str | None = None,
        session_id: str | None = None,
        as_of: date | None = None,
    ) -> Iterator[dict]:
        """逐事件产出，前端按 event 类型渲染。

        event 取值：
            meta      本次问答的策略与检索式，先发，前端可展示"系统按X策略检索"
            clauses   命中的条款列表，在正文之前发，便于前端先渲染依据卡片
            token     正文片段
            done      结束，附带完整溯源信息
        """
        question = (question or "").strip()
        if not question:
            yield {"event": "error", "message": "问题不能为空"}
            return

        session_id = session_id or str(uuid.uuid4())
        as_of_str = (as_of or date.today()).isoformat()
        cache_key = cache.make_answer_key(question, business_line, as_of_str)
        use_cache = config.ENABLE_ANSWER_CACHE

        # FAQ 层与 ask() 保持同一顺序：先缓存、再 FAQ、最后完整链路。
        hit = cache.get(cache_key) if use_cache else None
        if hit:
            yield {"event": "meta", "strategy": hit.get("strategy"), "queries": hit.get("queries", []),
                   "from_cache": True, "session_id": session_id}
            yield {"event": "clauses", "clauses": hit.get("clauses", [])}
            yield {"event": "token", "token": hit.get("answer", "")}
            yield {"event": "done", "from_cache": True, "session_id": session_id,
                   "refused": hit.get("refused", False), "citations": hit.get("citations", [])}
            return

        faq_payload = self._try_faq(question, business_line, as_of_str, session_id)
        if faq_payload is not None:
            if use_cache:
                cache.set(cache_key, faq_payload)
            self._remember(session_id, question, faq_payload["answer"])
            yield {"event": "meta", "strategy": "FAQ 精确命中", "queries": [],
                   "business_line": faq_payload["business_line"], "from_faq": True,
                   "session_id": session_id}
            yield {"event": "clauses", "clauses": faq_payload["clauses"]}
            yield {"event": "token", "token": faq_payload["answer"]}
            yield {"event": "done", "session_id": session_id, "refused": False,
                   "grounded": True, "issues": [], "from_cache": False, "from_faq": True}
            return

        has_history = len(self._history[session_id]) > 0
        strategy = analyzer.select(question, has_history=has_history)
        queries = analyzer.rewrite(question, strategy, self._history_text(session_id))
        line, include_repealed = self._resolve_scope(question, strategy, business_line)
        clauses = self._collect_clauses(queries, line, as_of, include_repealed)

        yield {"event": "meta", "strategy": strategy, "queries": queries,
               "business_line": line, "include_repealed": include_repealed,
               "from_cache": False, "session_id": session_id}
        yield {
            "event": "clauses",
            "clauses": [
                {
                    "citation": c.citation,
                    "clause_key": c.clause_key,
                    "title": c.title,
                    "doc_no": c.doc_no,
                    "chapter": c.chapter,
                    "clause_no": c.clause_no,
                    "clause_title": c.clause_title,
                    "content": c.content,
                    "similarity": c.similarity,
                }
                for c in clauses
            ],
        }

        collected: list[str] = []
        for piece in generator.stream(question, clauses):
            collected.append(piece)
            yield {"event": "token", "token": piece}

        full_text = "".join(collected)
        ok, issues = generator.check_grounding(full_text, clauses) if clauses else (True, [])
        payload = {
            "session_id": session_id,
            "question": question,
            "strategy": strategy,
            "queries": queries,
            "business_line": line,
            "as_of": as_of_str,
            "answer": full_text,
            "refused": not clauses,
            "grounded": ok,
            "citations": [],
            "clauses": [
                {
                    "citation": c.citation,
                    "clause_key": c.clause_key,
                    "title": c.title,
                    "doc_no": c.doc_no,
                    "chapter": c.chapter,
                    "clause_no": c.clause_no,
                    "clause_title": c.clause_title,
                    "content": c.content,
                    "similarity": c.similarity,
                }
                for c in clauses
            ],
        }
        if use_cache and clauses and not payload["refused"]:
            cache.set(cache_key, payload)

        self._remember(session_id, question, full_text)

        yield {
            "event": "done",
            "session_id": session_id,
            "refused": payload["refused"],
            "grounded": ok,
            "issues": issues,
            "from_cache": False,
        }

    # ---------- FAQ 层 ----------
    def _try_faq(
        self,
        question: str,
        business_line: str | None,
        as_of_str: str,
        session_id: str,
    ) -> dict | None:
        """尝试用审定好的标准问法直接作答，命中则返回完整 payload。

        为什么不直接返回答案文本：FAQ 答案虽然是审定过的，但制度会换版，
        前端仍需要条款依据做什么用、溯源窗口要能点开原文。所以这里按
        clause_keys 回表取出条款，跟正常链路一样带上 clauses 列表。

        两个必须挡住的情况：
        - 条款已失效（废止/被取代）：审定答案可能已是旧口径，不能直接返回
        - 审定超过复核周期：措辞确定反而比模型现生成更危险，降级走完整链路
        """
        if not config.ENABLE_FAQ:
            return None
        match = faq_search.search(question)
        if match is None:
            return None

        clauses = self._faq_clauses(match.clause_keys)
        if not clauses:
            self.logger.info("FAQ 命中但条款已失效或取不到原文，降级走检索链路：%s", match.question[:30])
            return None

        if not faq_search.is_answer_fresh(match.question):
            self.logger.info("FAQ 命中但答案超过复核周期，降级走检索链路：%s", match.question[:30])
            return None

        faq_search.bump_hit(match.question)
        line = business_line or (clauses[0].business_line if clauses[0].business_line else None)
        self.logger.info("FAQ 命中（score=%.3f）：%s", match.score, match.question[:30])
        return {
            "session_id": session_id,
            "question": question,
            "strategy": "FAQ 精确命中",
            "queries": [],
            "business_line": line,
            "as_of": as_of_str,
            "from_cache": False,
            "from_faq": True,
            "faq_question": match.question,
            "answer": match.answer,
            "refused": False,
            "grounded": True,
            "citations": [c.citation for c in clauses],
            "clauses": [
                {
                    "citation": c.citation,
                    "clause_key": c.clause_key,
                    "title": c.title,
                    "doc_no": c.doc_no,
                    "chapter": c.chapter,
                    "clause_no": c.clause_no,
                    "clause_title": c.clause_title,
                    "content": c.content,
                    "similarity": None,
                }
                for c in clauses
            ],
        }

    @staticmethod
    def _faq_clauses(clause_keys: list[str]) -> list:
        """按 FAQ 记录的 clause_key 回表取现行条款。

        只要有一条取不到或已失效就整体返回空——FAQ 答案通常由一条条款支撑，
        缺了依据还返回答案等于无凭据作答，不如降级走完整链路。
        返回的是 retriever 侧的 Clause 对象，这样上游渲染逻辑只有一套。
        """
        if not clause_keys:
            return []
        rows = db.get_clauses_by_keys(clause_keys)

        by_key = {r["clause_key"]: r for r in rows}
        clauses = []
        for key in clause_keys:
            row = by_key.get(key)
            if row is None or row["status"] != config.POLICY_STATUS_ACTIVE:
                return []
            clauses.append(
                Clause(
                    clause_key=row["clause_key"],
                    doc_no=row["doc_no"],
                    title=row["title"],
                    chapter=row.get("chapter") or "",
                    clause_no=row["clause_no"],
                    clause_title=row.get("clause_title") or "",
                    content=row["content"],
                    business_line=row.get("business_line") or "",
                    status=row["status"],
                    # FAQ 命中不经过召回，没有相似度可分；引用格式由 Clause.citation 拼
                    similarity=0.0,
                    effective_date=row.get("effective_date"),
                )
            )
        return clauses

    # ---------- 元信息 ----------
    @staticmethod
    def overview() -> dict:        return {
            "stats": db.stats(),
            "business_lines": config.BUSINESS_LINES,
            "llm_available": bool(config.LLM_API_KEY),
            "retrieval_k": config.RETRIEVAL_K,
            "min_similarity": config.MIN_SIMILARITY,
        }


qa_system = PolicyQASystem()
