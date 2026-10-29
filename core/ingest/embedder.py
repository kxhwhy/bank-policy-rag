# -*- coding: utf-8 -*-
"""BGE-M3 向量编码器。

项目里所有向量化都从这里取单例，保证入库与检索用的是同一个模型、同一套预处理。
模型文件缺失时，如果配置允许，会退化为确定性的哈希向量——
这只用于没有 GPU、没有权重文件的离线自测，向量质量必然差，
所以 config 里 ALLOW_HASH_FALLBACK 默认关闭，正式部署前请确认模型已下载。
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np

from core.config import config
from core.logger import logger

DENSE_DIM = 1024


class BGEM3Embedder:
    """BGE-M3 稠密 + 稀疏双向量输出。"""

    def __init__(self, model_path: str) -> None:
        from milvus_model.hybrid import BGEM3EmbeddingFunction

        device = "cuda" if self._has_cuda() else "cpu"
        logger.info("加载 BGE-M3 模型：%s（device=%s）", model_path, device)
        self._fn = BGEM3EmbeddingFunction(model_name_or_path=model_path, device=device, use_fp16=device == "cuda")
        self.dim = self._fn.dim["dense"]
        logger.info("模型加载完成，稠密维度 %d", self.dim)

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def encode(self, texts: list[str]) -> dict[str, Any]:
        if not texts:
            return {"dense": np.zeros((0, DENSE_DIM), dtype=np.float32), "sparse": []}

        result = self._fn(texts)
        dense = np.asarray(result["dense"], dtype=np.float32)

        # 稀疏向量转成 {token_id: weight} 字典列表，Milvus 接受这种格式
        sparse_matrix = result["sparse"]
        sparse_list = []
        for i in range(len(texts)):
            # 新版 scipy 返回 csr_array（无 getrow），旧版返回 csr_matrix，两种都要兼容
            row = sparse_matrix[i]
            if hasattr(row, "tocoo"):
                coo = row.tocoo()
                indices, data = coo.col.tolist(), coo.data.tolist()
            else:
                indices, data = row.indices.tolist(), row.data.tolist()
            sparse_list.append({int(k): float(v) for k, v in zip(indices, data) if v > 0})

        return {"dense": dense, "sparse": sparse_list}


class HashEmbedder:
    """离线占位编码器：仅用于模型缺失时跑通流程，不具备语义能力。"""

    dim = DENSE_DIM

    def encode(self, texts: list[str]) -> dict[str, Any]:
        dense_rows = []
        sparse_list = []
        for text in texts:
            vector = np.zeros(DENSE_DIM, dtype=np.float32)
            sparse: dict[int, float] = {}
            # 按字二元组散列到固定维度，相同字串得到相同向量
            for i in range(len(text) - 1):
                token = text[i : i + 2]
                idx = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % DENSE_DIM
                vector[idx] += 1.0
                sparse[idx] = sparse.get(idx, 0.0) + 1.0
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            dense_rows.append(vector)
            sparse_list.append(sparse)
        return {"dense": np.vstack(dense_rows), "sparse": sparse_list}


_embedder = None


def get_embedder():
    """取编码器单例。"""
    global _embedder
    if _embedder is not None:
        return _embedder

    model_path = config.EMBEDDING_MODEL
    local_dir = os.path.join(config.MODEL_DIR, os.path.basename(model_path))

    for candidate in (local_dir, model_path):
        if os.path.isdir(candidate):
            _embedder = BGEM3Embedder(candidate)
            return _embedder

    if config.ALLOW_HASH_FALLBACK:
        logger.warning(
            "未找到 BGE-M3 模型（尝试过 %s）。已按配置退化为哈希向量，"
            "仅能验证流程连通性，语义检索效果不可用。",
            local_dir,
        )
        _embedder = HashEmbedder()
        return _embedder

    raise FileNotFoundError(
        f"未找到 BGE-M3 模型：{local_dir}。请将模型权重放入 models/ 目录，"
        f"或设置环境变量 POLICY_EMBEDDING_MODEL 指向本地路径。"
    )
