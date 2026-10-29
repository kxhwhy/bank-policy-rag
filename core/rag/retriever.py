# -*- coding: utf-8 -*-
"""检索层：召回 → 时效过滤 → 重排 → 阈值截断。

制度问答的检索与通用 RAG 的差别在于"能不能答"比"答得好不好"更重要：
宁可只返回 1 条高置信条款，也不要凑 5 条不相关的塞给模型，
因为模型会把凑数的条款当成依据编出答案。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from core.config import config
from core.logger import logger
from core.rag.reranker import get_reranker
from core.store.policy_db import db
from core.store.vector_store import vector_store


@dataclass
class RetrievedClause:
    """对外统一暴露的检索结果。"""

    clause_key: str
    doc_no: str
    title: str
    chapter: str
    clause_no: str
    clause_title: str
    content: str
    business_line: str
    status: str
    similarity: float
    effective_date: date | None = None
    # 废止信息：只有命中历史版本时才可能非空，生成层据此强制加失效声明
    repeal_date: date | None = None
    superseded_by: str = ""
    # 交叉编码器给出的相关性分，仅在启用重排时存在
    rerank_score: float = 0.0

    @property
    def is_repealed(self) -> bool:
        return self.status != config.POLICY_STATUS_ACTIVE

    @property
    def citation(self) -> str:
        """标准引用格式：《制度名》（文号）第X条。"""
        return f"《{self.title}》（{self.doc_no}）{self.clause_no}"

    @property
    def display_head(self) -> str:
        parts = [f"{self.clause_no}"]
        if self.clause_title:
            parts.append(self.clause_title)
        return "　".join(parts)

    @property
    def validity_note(self) -> str:
        """给生成层用的效力说明，现行条款返回空串。

        这段话会拼进 prompt 里紧贴条款正文，而不是放在回答末尾的免责声明里。
        放在末尾的声明模型容易忘，贴在条款上它就跑不掉。
        """
        if not self.is_repealed:
            return ""
        if self.repeal_date:
            tail = f"自 {self.repeal_date.isoformat()} 起废止"
        else:
            tail = "已废止"
        if self.superseded_by:
            tail += f"，由《{self.superseded_by}》取代"
        return f"【已失效】{tail}"


@dataclass
class RetrievalResult:
    """一次检索的完整产物，包含被过滤掉的内容，便于前端说明"为什么不给答案"。"""

    clauses: list[RetrievedClause] = field(default_factory=list)
    dropped_repealed: int = 0
    max_similarity: float = 0.0
    query_used: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.clauses

    @property
    def confidence(self) -> str:
        """给前端展示的置信度标签，不参与业务判定。"""
        best = self.clauses[0].similarity if self.clauses else 0.0
        if best >= 0.75:
            return "高"
        if best >= 0.5:
            return "中"
        return "低"


class PolicyRetriever:
    def __init__(self, store=None, clause_db=None) -> None:
        # 依赖以参数形式注入，默认走全局实例；离线自测时可替换成内存实现，
        # 不必为了跑一次断言去启动 Milvus 和 MySQL。
        self.store = store if store is not None else vector_store
        self.db = clause_db if clause_db is not None else db
        self.top_k = config.RETRIEVAL_K
        self.min_similarity = config.MIN_SIMILARITY
        # 制度文档的效力信息在一次检索里会被反复询问，缓存一份，
        # 避免同一条款在候选池里出现多次就查多次 MySQL
        self._doc_meta: dict[str, dict[str, Any]] = {}

    def _load_doc_meta(self) -> None:
        if self._doc_meta:
            return
        try:
            for doc in self.db.list_documents():
                self._doc_meta[doc["doc_no"]] = doc
        except Exception as exc:  # noqa: BLE001
            logger.warning("加载制度元信息失败：%s", exc)

    def _repeal_date_of(self, doc_no: str) -> date | None:
        self._load_doc_meta()
        return (self._doc_meta.get(doc_no) or {}).get("repeal_date")

    def _superseded_by(self, doc_no: str) -> str:
        """现行制度全称——用于生成层写"由《XX》取代"。"""
        self._load_doc_meta()
        for doc in self._doc_meta.values():
            if doc.get("supersedes") == doc_no and doc.get("status") == config.POLICY_STATUS_ACTIVE:
                return doc.get("title", "")
        return ""

    def retrieve(
        self,
        query: str,
        business_line: str | None = None,
        as_of: date | None = None,
        include_repealed: bool = False,
    ) -> RetrievalResult:
        """主检索入口。

        as_of：查询基准日。制度会换版，用户的"现在"必须是确定的，
        默认取系统当天。若传入历史日期，则允许命中当时有效的旧制度。

        include_repealed：是否放行已废止条款。
        默认 False——问"额度上限是多少"时，废止的暂行办法条款必须彻底排除，
        这是效力硬过滤，不是相似度能解决的问题。
        但用户在明确问某个旧版本时（"2021年那个办法规定多少"），
        滤掉就等于把一条可查的事实变成查不到，这时候要放行，
        由下游 prompt 强制标注废止状态。
        """
        query = (query or "").strip()
        if not query:
            return RetrievalResult(query_used=query)

        raw_hits = self.store.hybrid_search(
            query=query,
            top_k=self.top_k,
            business_line=business_line,
            skip_repealed=False,
        )
        logger.info("检索「%s」原始召回 %d 条", query[:40], len(raw_hits))

        # ---------- 时效二次校验 ----------
        # 向量层不做效力过滤，一律交回这里判断：既要看 status，
        # 也要看施行日期是否早于基准日，避免"已发文未施行"的制度被当成现行制度引用。
        candidates: list[RetrievedClause] = []
        dropped = 0
        for hit in raw_hits:
            clause = self.db.get_clause(hit["id"].split("#p")[0])
            if clause is None:
                # MySQL 未同步时以向量库内容为准，但打日志便于排查
                logger.warning("条款 %s 在结构化库中缺失", hit["id"])
                continue

            if not self._is_effective(clause, as_of, include_repealed):
                dropped += 1
                continue

            candidates.append(
                RetrievedClause(
                    clause_key=clause["clause_key"],
                    doc_no=clause["doc_no"],
                    title=clause["title"],
                    chapter=clause["chapter"] or "",
                    clause_no=clause["clause_no"],
                    clause_title=clause["clause_title"] or "",
                    content=clause["content"],
                    business_line=clause["business_line"] or "",
                    status=clause["status"],
                    similarity=hit["similarity"],
                    effective_date=clause["effective_date"],
                    repeal_date=self._repeal_date_of(clause["doc_no"]),
                    superseded_by=self._superseded_by(clause["doc_no"]),
                )
            )

        if dropped:
            logger.info("时效校验剔除 %d 条非现行条款", dropped)

        # ---------- 同条款去重 + 阈值截断 ----------
        seen: dict[str, RetrievedClause] = {}
        for item in candidates:
            existing = seen.get(item.clause_key)
            if existing is None or item.similarity > existing.similarity:
                seen[item.clause_key] = item

        filtered = [c for c in seen.values() if c.similarity >= self.min_similarity]
        filtered.sort(key=lambda x: x.similarity, reverse=True)
        # 先按向量分取一个较宽的候选池，交给重排决定最终顺序与取舍
        pool = filtered[: max(config.RETRIEVAL_K, config.RERANK_TOP_M)]
        filtered = pool

        max_sim = max((c.similarity for c in candidates), default=0.0)
        if not filtered and candidates:
            logger.info(
                "召回 %d 条但全部低于相似度阈值 %.2f，按无依据处理",
                len(candidates),
                self.min_similarity,
            )

        # ---------- 交叉编码器重排 ----------
        # 向量分衡量"整体像不像"，重排衡量"能不能回答这个问题"，后者更贴近实际需求。
        # 重排不可用时自动退回向量排序，不阻断问答。
        if filtered and config.ENABLE_RERANK:
            filtered = get_reranker().rerank(query, filtered, top_m=config.RERANK_TOP_M)

        return RetrievalResult(
            clauses=filtered,
            dropped_repealed=dropped,
            max_similarity=max_sim,
            query_used=query,
        )

    @staticmethod
    def _is_effective(
        clause: dict[str, Any], as_of: date | None, include_repealed: bool = False
    ) -> bool:
        """判断条款在基准日能否作为依据。

        include_repealed=True 时放行已废止条款，但"已发文未施行"仍然拦住：
        还没施行的制度无论怎么问都不是历史版本，它属于未来，不是过去。
        """
        status_ok = clause["status"] == config.POLICY_STATUS_ACTIVE
        if not status_ok and not include_repealed:
            return False

        ref = as_of or date.today()
        eff = clause["effective_date"]
        if eff and eff > ref:
            # 制度已发文但尚未施行
            return False
        return True


retriever = PolicyRetriever()
