# -*- coding: utf-8 -*-
"""重排层：用 bge-reranker-large 对召回结果重新打分。

为什么需要这一层：向量检索算的是"问题和条款整体像不像"，
而制度问答真正要判断的是"这条条款能不能回答这个问题"。
两者经常不一致——问"借钱最多能批多少"时，"审批权限"整条在讲审批流程，
向量相似度可能高于"额度上限"，但前者答不了这个问题。
交叉编码器把问题和条款拼在一起逐对判定，正好补上这个缺口。

重排只处理已经召回的少数候选（默认 8 条），成本可控；
调用失败时退回向量分数，不让重排成为新的故障点。
"""
from __future__ import annotations

import os
from typing import Any

from core.config import config
from core.logger import logger


class PolicyReranker:
    def __init__(self) -> None:
        self._model = None
        self._failed = False

    @property
    def available(self) -> bool:
        return config.ENABLE_RERANK and not self._failed

    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker

            path = self._resolve_path()
            logger.info("加载重排模型：%s（device=%s）", path, "cuda" if self._has_cuda() else "cpu")
            self._model = FlagReranker(path, use_fp16=self._has_cuda())
            logger.info("重排模型加载完成")
        return self._model

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def _resolve_path() -> str:
        """优先用配置里的路径，其次找 models/ 下的同名目录。"""
        candidate = config.RERANK_MODEL_PATH
        if os.path.isdir(candidate):
            return candidate
        fallback = os.path.join(config.MODEL_DIR, "bge-reranker-large")
        if os.path.isdir(fallback):
            return fallback
        raise FileNotFoundError(
            f"未找到重排模型：{candidate} 或 {fallback}。"
            f"可将 config.ini 的 enable_rerank 置为 false 跳过重排。"
        )

    def rerank(self, query: str, clauses: list[Any], top_m: int | None = None) -> list[Any]:
        """按重排分数排序并截断。

        clauses 里每项需带 content 字段（条款正文）。
        返回的对象会被就地写入 rerank_score，方便前端和日志展示打分依据。
        """
        if not clauses:
            return clauses
        if not self.available:
            return clauses[: top_m or config.FINAL_TOP_M]

        pairs = [[query, c.content] for c in clauses]
        try:
            scores = self.model.compute_score(pairs, normalize=True)
        except Exception as exc:
            # 重排是增强项，任何异常都退回原顺序，不能让问答挂掉
            self._failed = True
            logger.error("重排失败，退回向量排序：%s", exc)
            return clauses[: top_m or config.FINAL_TOP_M]

        if isinstance(scores, float):
            scores = [scores]

        for clause, score in zip(clauses, scores):
            clause.rerank_score = float(score)

        ordered = sorted(clauses, key=lambda c: getattr(c, "rerank_score", 0.0), reverse=True)
        logger.info(
            "重排完成，最高分 %.3f（向量分 %.3f）",
            getattr(ordered[0], "rerank_score", 0.0),
            ordered[0].similarity,
        )
        return ordered[: top_m or config.FINAL_TOP_M]


_reranker = None


def get_reranker() -> PolicyReranker:
    global _reranker
    if _reranker is None:
        _reranker = PolicyReranker()
    return _reranker
