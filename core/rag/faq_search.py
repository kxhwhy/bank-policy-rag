"""FAQ 精确命中：BM25 打分 + 阈值判定。

设计取向与另一条路（用 BGE-M3 稠密向量算余弦）不同，这里要的是"同一句话被
不同人问出来也能认出来"，而不是"语义相近就返回"。BM25 对词面重合敏感、
对同义改写不敏感，这正是 FAQ 需要的：宁可漏命中回落 RAG，也不要误命中给出
一条不相关的审定答案。

三层缓存里的第二层和第三层就落在本文件：

    第二层  标准问法列表   qa_original_questions   避免每次启动都从 MySQL 拉全表
    第三层  问法分词结果   qa_tokenized_questions  避免每次启动都重新跑一遍 jieba

这两层和第一层（问答结果 answer:{query}）的差别在于**失效时机**：第一层随
制度换版失效，第二三层随 FAQ 库内容变更失效。所以键分开、清理分开——
改一条标准问法不该顺手把整库答案缓存清掉。

加载顺序与失效策略：
    1. 先读 Redis 的第二三层，两个都在则直接建 BM25，跳过 MySQL 和分词
    2. 任一缺失则从 MySQL 取全部标准问法，分词后回写第二三层
    3. 标准问法有变更时必须 active 调 refresh() 清索引，否则会一直用旧的

索引缓存的键带一个版本号，库内容变化时改版本号即可整体失效，
不需要逐键扫描删除。
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from core.config import config
from core.rag.faq.preprocess import tokenize
from core.logger import logger
from core.store.policy_db import db
from core.store.redis_client import redis_client

# 第二三层缓存：问法列表 / 分词列表共享同一版本号，改版本号即整体失效
_INDEX_VERSION = "v1"
_REDIS_KEY_QUESTIONS = f"qa_original_questions:{_INDEX_VERSION}"
_REDIS_KEY_TOKENIZED = f"qa_tokenized_questions:{_INDEX_VERSION}"
_CACHE_PREFIX = "qa_original_questions:*"


class FaqMatch:
    """一次 FAQ 匹配的结果。"""

    __slots__ = ("question", "answer", "score", "clause_keys")

    def __init__(self, question: str, answer: str, score: float, clause_keys: list[str]) -> None:
        self.question = question
        self.answer = answer
        self.score = score
        self.clause_keys = clause_keys

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "score": round(self.score, 4),
            "clause_keys": self.clause_keys,
        }


class FaqSearch:
    """BM25 标准问答匹配器。

    单例使用：全局只维护一份索引，避免每次问答都重算 BM25 的 IDF。
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._questions: list[str] = []
        self._answers: list[str] = []
        self._clause_keys: list[list[str]] = []
        self._tokenized: list[list[str]] = []
        self._loaded = False

    # ---------- 索引 ----------

    def refresh(self) -> int:
        """强制重建索引并清空 Redis 缓存。FAQ 数据变更后必须调用。"""
        self._loaded = False
        self._clear_redis_index()
        return self.load_index(force=True)

    def load_index(self, force: bool = False) -> int:
        """加载标准问法并构建 BM25。返回索引里的问法条数。"""
        if self._loaded and not force:
            return len(self._questions)

        rows = self._load_rows()
        if not rows:
            logger.warning("FAQ 库为空，FAQ 层不生效，所有问题都会回落 RAG")
            self._questions, self._answers, self._clause_keys, self._tokenized = [], [], [], []
            self._bm25 = None
            self._loaded = True
            return 0

        self._questions = [r["question"] for r in rows]
        self._answers = [r["answer"] for r in rows]
        self._clause_keys = [self._parse_clause_keys(r.get("clause_keys")) for r in rows]

        tokenized = self._load_tokenized(self._questions)
        self._tokenized = tokenized
        self._bm25 = BM25Okapi(tokenized)
        self._loaded = True
        logger.info("FAQ 索引就绪：%d 条标准问法", len(self._questions))
        return len(self._questions)

    def _load_rows(self) -> list[dict[str, Any]]:
        """从 MySQL 取全部标准问法。只取生效的行。"""
        try:
            with db.connection() as conn, conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT question, answer, clause_keys
                    FROM faq_standard
                    WHERE invalid_at IS NULL
                    ORDER BY id
                    """
                )
                return list(cursor.fetchall())
        except Exception as exc:  # noqa: BLE001 - 库不可用时 FAQ 层应降级而非中断
            logger.warning("FAQ 库读取失败，FAQ 层本次不可用：%s", exc)
            return []

    def _load_tokenized(self, questions: list[str]) -> list[list[str]]:
        """加载第二三层缓存：标准问法列表 + 分词结果。

        两个键都要在且问法条数对得上才算命中——只命中一个没有意义，
        反正 MySQL 那一趟已经跑了，不如两个一起重建。
        """
        if not config.ENABLE_FAQ_INDEX_CACHE:
            logger.info("FAQ 索引缓存已关闭，本次直接分词")
            return [tokenize(q) for q in questions]

        cached_questions = self._read_redis(_REDIS_KEY_QUESTIONS)
        cached_tokenized = self._read_redis(_REDIS_KEY_TOKENIZED)

        if (
            isinstance(cached_questions, list)
            and isinstance(cached_tokenized, list)
            and len(cached_questions) == len(questions)
            and len(cached_tokenized) == len(questions)
            and cached_questions == questions
        ):
            logger.info("FAQ 问法列表与分词结果均命中缓存（%d 条），跳过分词", len(questions))
            return [list(x) for x in cached_tokenized]

        tokenized = [tokenize(q) for q in questions]
        self._write_redis(_REDIS_KEY_QUESTIONS, questions)
        self._write_redis(_REDIS_KEY_TOKENIZED, tokenized)
        logger.info("FAQ 问法列表与分词结果已写入缓存（%d 条）", len(questions))
        return tokenized

    # ---------- Redis（第二三层） ----------

    @staticmethod
    def _read_redis(key: str) -> Any:
        """读索引缓存。Redis 不可用时返回 None，由上层回落到本地重算。"""
        return redis_client.get(key)

    @staticmethod
    def _write_redis(key: str, value: Any) -> None:
        # 不过期：问法库变了由 refresh() 主动清，不靠时间兜底
        redis_client.set(key, value, ttl=None)

    @staticmethod
    def _clear_redis_index() -> None:
        redis_client.delete(_REDIS_KEY_QUESTIONS, _REDIS_KEY_TOKENIZED)

    def cache_stats(self) -> dict[str, Any]:
        """第二三层的自检信息，供排查"缓存到底有没有生效"用。"""
        return {
            "questions_cached": redis_client.exists(_REDIS_KEY_QUESTIONS),
            "tokenized_cached": redis_client.exists(_REDIS_KEY_TOKENIZED),
            "version": _INDEX_VERSION,
            "indexed": len(self._questions),
        }

    # ---------- 匹配 ----------

    def search(self, query: str, threshold: float | None = None) -> FaqMatch | None:
        """按双判据判定是否命中标准问法：独占度 或 绝对相似度，满足其一即可。

        为什么需要两个判据：只用 softmax 时有反直觉的漏判。
        softmax 衡量的是"这一条占全部候选的概率份额"，最像的那条分到 0.999
        自然没问题；但同义改写（"个人消费贷最高能批多少" vs "个人消费贷款最多
        能贷多少钱"）会同时激活好几条候选，分数被摊平到 0.54 左右，明明最接近
        的是对的也被判成未命中。反过来只看原始 BM25 分数也不行：它没有上界，
        随库大小和词长漂移，0.85 这样的固定阈值换个库就失效。

        所以对两个量分别定阈值：
        - share：softmax 份额 → 判"是不是独占最像的那条"（逐字命中走这条）
        - similarity：原始 BM25 除以理论满分的归一化值 → 判"绝对有多像"（改写走这条）
        两个阈值都偏高，误命中主要靠 similarity 那道管住；而 share 那道在候选
        极多时依然可靠，两者互补。
        """
        if not query or not query.strip():
            return None

        if self.load_index() == 0 or self._bm25 is None:
            return None

        min_share = config.FAQ_HIT_THRESHOLD if threshold is None else threshold
        min_similarity = config.FAQ_MIN_SIMILARITY
        tokens = tokenize(query)
        if not tokens:
            return None

        raw = np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)
        shares = self._softmax(raw)
        best = int(shares.argmax())
        share = float(shares[best])
        similarity = float(raw[best]) / self._max_possible_score(tokens)

        # 信息量闸门：查询去噪后剩不下两个实义词时，任何候选都"100% 覆盖"，
        # 分数判据在这类输入上完全没有分辨力（「银行年假怎么休」只剩"银行"，
        # 会撞上任何带"银行"的标准问法）。这种时候不猜，交给完整检索链路。
        if len(tokens) < 2:
            logger.info("FAQ 未命中：查询实义词不足（%s），交回检索链路", "/".join(tokens))
            return None

        if share <= min_share and similarity <= min_similarity:
            logger.info(
                "FAQ 未命中：share=%.4f<=%.4f 且 similarity=%.4f<=%.4f（最接近：%s）",
                share, min_share, similarity, min_similarity, self._questions[best][:24],
            )
            return None

        # 覆盖闸门：光靠分数比例不够。查询被去噪后可能只剩两个实义词，
        # 只要这两个词在候选里出现过，BM25 就能拿到接近满分的比例，
        # 于是「银行年假怎么休」会撞上「手机银行转账限额」（只共享"银行"）。
        # 所以再要求候选把查询的实义词覆盖到一定比例——漏命中可以退回检索链路，
        # 误命中却是直接把不相干的审定答案返回给用户，两者代价不对等。
        coverage = self._coverage(tokens, self._tokenized[best])
        if coverage < config.FAQ_MIN_COVERAGE:
            logger.info(
                "FAQ 未命中：覆盖度 %.2f<%.2f（最接近：%s）",
                coverage, config.FAQ_MIN_COVERAGE, self._questions[best][:24],
            )
            return None

        # 冲突闸门：查询里有业务词是候选完全没有的，就不算命中。
        # 「手机银行怎么开通」和「手机银行转账限额」共享"手机银行"，覆盖度算 100%，
        # 但前者问的是开通流程、后者答的是限额，属于业务意图不同。
        # 判定方式：查询里的业务词（_DOMAIN_WORDS 命中）如果有一个候选里完全没有，
        # 说明两者在谈论不同的业务对象，宁可退回检索链路。
        missing = self._missing_domain_terms(tokens, self._tokenized[best])
        if missing:
            logger.info(
                "FAQ 未命中：候选缺少业务词 %s（最接近：%s）",
                "/".join(missing), self._questions[best][:24],
            )
            return None

        logger.info(
            "FAQ 命中：share=%.4f similarity=%.4f coverage=%.2f → %s",
            share, similarity, coverage, self._questions[best][:30],
        )
        return FaqMatch(
            question=self._questions[best],
            answer=self._answers[best],
            score=round(similarity, 4),
            clause_keys=self._clause_keys[best],
        )

    @staticmethod
    def _missing_domain_terms(query_tokens: list[str], candidate_tokens: list[str]) -> list[str]:
        """查询里出现、但候选问法里完全没出现的业务词。"""
        from core.rag.faq.preprocess import domain_terms

        candidate = set(candidate_tokens)
        return [tok for tok in query_tokens if tok in domain_terms() and tok not in candidate]

    @staticmethod
    def _coverage(query_tokens: list[str], candidate_tokens: list[str]) -> float:
        """查询实义词有多大比例真的出现在候选问法里。

        分母用查询词数而不是候选词数：我们关心的是"用户问的东西候选答没答"，
        候选多出来的词不该拉低覆盖度。
        """
        if not query_tokens:
            return 0.0
        candidate = set(candidate_tokens)
        return sum(1 for tok in query_tokens if tok in candidate) / len(query_tokens)

    def _max_possible_score(self, tokens: list[str]) -> float:
        """BM25 的理论满分，用来把原始分数归一化到 0～1。

        单条文本在某个词上能拿到的最高分是 idf × (k1+1)，其中
        tf 项 tf·(k1+1)/(tf + k1·(1-b+b·dl/avgdl)) 在 tf 极大且 dl≈avgdl 时趋近 k1+1。
        把所有查询词的这个上限加起来就是满分，超不过它，所以做分母是安全的。
        k1、b 直接从 rank_bm25 实例上读，避免和库的默认值写死两处。
        """
        k1 = float(getattr(self._bm25, "k1", 1.5))
        idf = getattr(self._bm25, "idf", {})
        ceiling = 0.0
        for token in tokens:
            weight = float(idf.get(token, 0.0))
            if weight <= 0:
                # 词表外（或出现在过半文档里被压成 0）的词对本条无贡献，跳过
                continue
            ceiling += weight * (k1 + 1.0)
        # 全部查询词都是词表外时给一个极小分母，避免除零后 similarity 爆成 inf 而误命中
        return ceiling if ceiling > 1e-6 else 1e-6

    def is_answer_fresh(self, question: str) -> bool:
        """审定答案是否还在复核周期内。

        为什么超期就不返回：制度会换版，一条两年前审定的答案措辞越确定、
        用户越容易信，但它可能已经跟不上现行制度。这时候宁可降级让模型
        基于现行条款现答，也不要直接吐出过期结论。
        """
        try:
            with db.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT last_verified FROM faq_standard WHERE question = %s AND invalid_at IS NULL",
                    (question,),
                )
                row = cur.fetchone()
        except Exception as exc:  # noqa: BLE001 - 查不到就当过期，保守处理
            logger.warning("FAQ 复核日期查询失败，按过期处理：%s", exc)
            return False

        if not row or not row.get("last_verified"):
            return False
        age = (date.today() - row["last_verified"]).days
        return age <= config.FAQ_VERIFY_TTL_DAYS

    def bump_hit(self, question: str) -> None:
        """命中计数。用于识别高频问题，也用于发现"命中多但答案可能已旧"的条目。"""
        try:
            with db.connection() as conn, conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE faq_standard SET hit_count = hit_count + 1 WHERE question = %s",
                    (question,),
                )
        except Exception as exc:  # noqa: BLE001 - 计数失败不影响返回答案
            logger.warning("FAQ 命中计数更新失败，忽略：%s", exc)

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        """数值稳定版 softmax。减最大值避免 exp 溢出。"""
        if scores.size == 0:
            return scores
        shifted = np.exp(scores - np.max(scores))
        total = shifted.sum()
        return shifted / total if total else shifted

    @staticmethod
    def _parse_clause_keys(raw: Any) -> list[str]:
        """clause_keys 存 JSON 数组；兼容早期逗号分隔的写法。"""
        if not raw:
            return []
        if isinstance(raw, (list, tuple)):
            return [str(x) for x in raw if str(x).strip()]
        text = str(raw).strip()
        if text.startswith("["):
            try:
                return [str(x) for x in json.loads(text)]
            except json.JSONDecodeError:
                logger.warning("clause_keys 不是合法 JSON，按逗号切分：%s", text[:40])
        return [x.strip() for x in text.split(",") if x.strip()]


faq_search = FaqSearch()
