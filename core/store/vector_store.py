# -*- coding: utf-8 -*-
"""条款向量库（Milvus）。

一个集合 policy_clause，字段设计对应"条款"这一级语义单元：

    id            主键 = 文号#条款号，与 MySQL 的 clause_key 对齐
    dense_vector  BGE-M3 稠密向量（1024 维），负责语义泛化
    sparse_vector BGE-M3 稀疏向量，负责专有名词、数字、文号的精确命中
    parent_content 父块正文（整条），召回后送给大模型的是完整条款而非切片
    child_text    子块正文（条内切片），用于更细粒度的相似度匹配

稠密 + 稀疏双路各自建索引，检索时按权重融合。制度问答里"5万元以上"
"第X条""恒信银发〔2024〕27号"这类字面信息，纯稠密向量容易糊掉，
必须靠稀疏一路兜住。
"""
from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore as LangchainVectorStore

from core.config import config
from core.logger import logger

try:  # 装了 pymilvus 才建真实连接，解析、切块、引用校验不需要它
    from pymilvus import DataType, MilvusClient
    MILVUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    DataType = None  # type: ignore[assignment]
    MilvusClient = None  # type: ignore[assignment]
    MILVUS_AVAILABLE = False

# BGE-M3 稠密向量维度
DENSE_DIM = 1024


class PolicyVectorStore(LangchainVectorStore):
    """条款向量库。

    继承 langchain_core 的 VectorStore，是让这套自研检索能进 LangChain 生态的
    最小代价：父类只有一个抽象方法 similarity_search，实现它之后，
    similarity_search_with_score、as_retriever、LCEL 链式调用全部免费获得。

    之所以不直接用 langchain_milvus.Milvus 现成的实现，是因为它管不住本项目的
    两个硬需求：
        1. 时效过滤要拼 Milvus boolean expr（status == "active"），
           且要与业务条线过滤同时生效；
        2. 召回结果必须带 parent_content 回表校验，而不是只返回子块。
    自己实现一个 VectorStore 子类，比在现成实现外面再包一层适配更短、更可控。
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._embedder = None
        self.collection = config.MILVUS_COLLECTION

    # ---------- 懒加载 ----------
    @property
    def client(self):
        if self._client is None:
            if not MILVUS_AVAILABLE:
                raise RuntimeError(
                    "未安装 pymilvus，无法建立向量库连接；请先 pip install -r requirements.txt"
                )
            self._client = MilvusClient(uri=config.MILVUS_URI, db_name=config.MILVUS_DB)
            logger.info("已连接 Milvus：%s/%s", config.MILVUS_URI, config.MILVUS_DB)
        return self._client

    @property
    def embedder(self):
        if self._embedder is None:
            from core.ingest.embedder import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    # ---------- 建集合 ----------
    def ensure_collection(self, drop: bool = False) -> None:
        if drop and self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
            logger.warning("已删除旧集合 %s", self.collection)

        if self.client.has_collection(self.collection):
            logger.info("复用已存在的集合 %s", self.collection)
        else:
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=160)
            schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DENSE_DIM)
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field("parent_content", DataType.VARCHAR, max_length=8192)
            schema.add_field("child_text", DataType.VARCHAR, max_length=2048)
            # 标量字段：检索时直接做 where 过滤，避免捞回已废止条款再丢弃
            schema.add_field("doc_no", DataType.VARCHAR, max_length=64)
            schema.add_field("clause_no", DataType.VARCHAR, max_length=64)
            schema.add_field("business_line", DataType.VARCHAR, max_length=64)
            schema.add_field("status", DataType.VARCHAR, max_length=16)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_name="idx_dense",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="idx_sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2},
            )
            self.client.create_collection(
                collection_name=self.collection,
                schema=schema,
                index_params=index_params,
            )
            logger.info("已创建集合 %s", self.collection)

        self.client.load_collection(self.collection)

    # ---------- 写入 ----------
    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """VectorStore 接口：接 LangChain Document 列表。

        split_document() 的输出可以直接灌进来，metadata 里的溯源字段
        会原样落进 Milvus 的标量列，召回时再原样带回。
        """
        if not documents:
            return []

        rows = []
        texts = [d.page_content for d in documents]
        embeddings = self.embedder.encode(texts)
        for doc, dense, sparse in zip(documents, embeddings["dense"], embeddings["sparse"]):
            meta = doc.metadata
            chunk_id = meta.get("id") or hashlib.md5(
                f"{meta.get('doc_no','')}#{meta.get('clause_no','')}#{doc.page_content}".encode("utf-8")
            ).hexdigest()[:16]
            rows.append(
                {
                    "id": chunk_id,
                    "dense_vector": dense.tolist(),
                    "sparse_vector": sparse,
                    "parent_content": meta.get("parent_content", ""),
                    "child_text": doc.page_content,
                    "doc_no": meta.get("doc_no", ""),
                    "clause_no": meta.get("clause_no", ""),
                    "business_line": meta.get("business_line", "其他"),
                    "status": meta.get("status", "active"),
                }
            )
        self.client.upsert(collection_name=self.collection, data=rows)
        logger.info("Document 批量写入 %d 条", len(rows))
        return [r["id"] for r in rows]

    def upsert_clauses(self, clauses: list[dict[str, Any]], batch_size: int = 64) -> int:
        """条款入库。每个条款同时写父块（整条）与子块（切片）。"""
        if not clauses:
            return 0

        total = 0
        for start in range(0, len(clauses), batch_size):
            batch = clauses[start : start + batch_size]
            texts = [item["child_text"] for item in batch]
            embeddings = self.embedder.encode(texts)

            rows = []
            for item, dense, sparse in zip(batch, embeddings["dense"], embeddings["sparse"]):
                rows.append(
                    {
                        "id": item["id"],
                        "dense_vector": dense.tolist(),
                        "sparse_vector": sparse,
                        "parent_content": item["parent_content"],
                        "child_text": item["child_text"],
                        "doc_no": item["doc_no"],
                        "clause_no": item["clause_no"],
                        "business_line": item["business_line"],
                        "status": item["status"],
                    }
                )
            self.client.upsert(collection_name=self.collection, data=rows)
            total += len(rows)
            logger.info("条款向量写入进度 %d/%d", total, len(clauses))

        return total

    # ---------- 检索 ----------
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """VectorStore 抽象方法：按问题检索条款。

        kwargs 支持 business_line / as_of / skip_repealed，
        与 hybrid_search 的参数一一对应。
        """
        hits = self.hybrid_search(
            query=query,
            top_k=k,
            business_line=kwargs.get("business_line"),
            skip_repealed=kwargs.get("skip_repealed", True),
        )
        return [
            Document(
                page_content=hit.get("parent_content") or hit.get("child_text", ""),
                metadata={
                    "id": hit.get("id", ""),
                    "doc_no": hit.get("doc_no", ""),
                    "clause_no": hit.get("clause_no", ""),
                    "business_line": hit.get("business_line", ""),
                    "status": hit.get("status", ""),
                    "similarity": hit.get("similarity", 0.0),
                },
            )
            for hit in hits
        ]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 8,
        business_line: str | None = None,
        skip_repealed: bool = True,
    ) -> list[dict[str, Any]]:
        """稠密 + 稀疏双路召回并按权重融合。

        返回项统一带 similarity 字段，供上层做阈值截断。
        """
        emb = self.embedder.encode([query])

        # 时效过滤是硬条件：已废止条款不进候选集
        expr_parts = []
        if skip_repealed:
            expr_parts.append('status == "active"')
        if business_line:
            expr_parts.append(f'business_line == "{business_line}"')
        expr = " and ".join(expr_parts) if expr_parts else None

        dense_hits = self.client.search(
            collection_name=self.collection,
            data=[emb["dense"][0].tolist()],
            anns_field="dense_vector",
            limit=top_k,
            filter=expr,
            output_fields=["parent_content", "child_text", "doc_no", "clause_no", "business_line", "status"],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        )[0]

        sparse_hits = self.client.search(
            collection_name=self.collection,
            data=[emb["sparse"][0]],
            anns_field="sparse_vector",
            limit=top_k,
            filter=expr,
            output_fields=["parent_content", "child_text", "doc_no", "clause_no", "business_line", "status"],
            search_params={"metric_type": "IP"},
        )[0]

        return self._fuse(dense_hits, sparse_hits)

    def _fuse(self, dense_hits, sparse_hits) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        for hit in dense_hits:
            entity = hit.get("entity", {})
            merged[hit["id"]] = {
                "id": hit["id"],
                "dense_score": float(hit["distance"]),
                "sparse_score": 0.0,
                **entity,
            }

        for hit in sparse_hits:
            entity = hit.get("entity", {})
            item = merged.setdefault(
                hit["id"],
                {
                    "id": hit["id"],
                    "dense_score": 0.0,
                    **entity,
                },
            )
            item["sparse_score"] = float(hit["distance"])

        # 两路分数量纲不同，各自归一化后加权，避免稀疏分数压过稠密
        max_dense = max((r["dense_score"] for r in merged.values()), default=1.0) or 1.0
        max_sparse = max((r["sparse_score"] for r in merged.values()), default=1.0) or 1.0

        results = []
        for row in merged.values():
            score = (
                config.DENSE_WEIGHT * (row["dense_score"] / max_dense)
                + config.SPARSE_WEIGHT * (row["sparse_score"] / max_sparse)
            )
            row["similarity"] = round(score, 4)
            results.append(row)

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def count(self) -> int:
        try:
            return self.client.query(
                collection_name=self.collection,
                filter='status != ""',
                output_fields=["count(*)"],
            )[0]["count(*)"]
        except Exception as exc:  # 集合未建立时直接返回 0
            logger.warning("统计向量条数失败：%s", exc)
            return 0

    # ---------- LangChain 接口 ----------
    @classmethod
    def from_texts(cls, texts: list[str], embedding: Any = None, metadatas: Any = None, **kwargs: Any):
        """VectorStore 抽象类方法：从纯文本批量建库。

        本项目的正式入库走 scripts/ingest（切块后 upsert_clauses），
        这个类方法是为 LangChain 生态里的通用调用路径留的入口：
        外部代码只拿到一段文本时，也能通过标准接口把数据灌进来。
        """
        store = cls()
        store.ensure_collection(drop=kwargs.get("drop", False))
        documents = [
            Document(page_content=text, metadata=(metadatas or [{}] * len(texts))[i])
            for i, text in enumerate(texts)
        ]
        store.add_documents(documents)
        return store


vector_store = PolicyVectorStore()
