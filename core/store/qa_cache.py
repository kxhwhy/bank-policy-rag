# -*- coding: utf-8 -*-
"""第一层缓存：问答结果缓存（answer:{query}）。

三层缓存里的第一层，键形如 `answer:{问题}|{条线}|{基准日}`。第三层（FAQ
分词索引）在 core/rag/faq_search.py，第二层（标准问法列表）同样在
faq_search.py 里一起加载。

命中这一层时不再做检索、不做时效校验、不做引用校验——因为写入时这些校验
都已经过了。所以键必须把所有会改变答案的维度都算进去，漏一个就等于埋了
一条走错的捷径。这里把两个维度都算进了键：

1. 业务条线。不同条线能看到的条款范围不同，同一条款在不同条线下答案可能
   不一样。只按问题原文做键，会把个人信贷条线的答案错给运营柜面条线。
2. 查询基准日。制度会换版。今天缓存"额度 50 万"，半年后制度改成 60 万，
   检索层明明正确地把旧条款标为废止并过滤了，用户却从缓存拿到半年前的
   答案——缓存把整条链路的时效控制绕过去了。

另外明确一条规则：拒答结果不写缓存。拒答有两种含义，一种是"知识层面确实
没有"，一种是"这一次运行时降级了"（模型抖动、重生成没过）。后者如果被缓存，
那个问题就被永久锁死成拒答，用户会以为是系统能力不足。缓存的是关于制度的
知识，不是某一次调用的结果。

同时实现 langchain_core 的 BaseCache 接口：LangChain 的模型与链在配置了
cache 之后会自动查这里，省掉重复的模型调用。两者共用同一个 Redis 连接，
但键空间分开——业务缓存按业务维度哈希，LangChain 缓存按 prompt 内容哈希。
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

import redis
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.outputs import Generation

from core.config import config
from core.logger import logger
from core.store.redis_client import redis_client

# 问句里对语义没有贡献的修饰词，归一化时先剥掉，提高缓存命中率
_NOISE_PREFIX = (
    "请问", "问一下", "想问", "想问下", "咨询", "你好", "您好",
    "麻烦", "帮我", "告诉我", "查一下", "查下", "我想知道",
)

# 第一层缓存的键前缀，用 answer: 这个命名空间，便于和 FAQ 索引缓存、
# LangChain 缓存分开清理。
_ANSWER_PREFIX = "answer:"

# LangChain 缓存用独立前缀，与业务缓存键隔开，避免 invalidate 时互相误删
_LC_PREFIX = "lc_cache:"


class QACache(BaseCache):
    """业务缓存 + LangChain 缓存后端。

    继承 BaseCache 后，把实例传给 LangChain 的 cache 参数即可让模型调用走缓存。
    LangChain 只要求实现 lookup / update / clear，其余默认行为由基类提供。
    """

    def __init__(self) -> None:
        self._ttl = config.CACHE_TTL_SECONDS

    @property
    def client(self) -> redis.Redis:
        return redis_client.client

    @staticmethod
    def normalize(question: str) -> str:
        q = question.strip()
        for prefix in _NOISE_PREFIX:
            if q.startswith(prefix):
                q = q[len(prefix):].lstrip("，,。 ？?")
        # 去掉空白和标点，只留语义字符
        return "".join(ch for ch in q if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")

    def make_answer_key(
        self,
        question: str,
        business_line: str | None = None,
        as_of: str | None = None,
    ) -> str:
        """问题的完整缓存键。

        问题原文直接进键（可读、便于排查），条线与基准日拼在后面：
            answer:{归一化问题}|{条线}|{基准日}

        为什么问题原文不哈希：调试时要能直接看出某个键对应哪句话，哈希之后
        只能对着日志反推。这里没有键过长的风险——问题本来就不长。
        """
        return f"{_ANSWER_PREFIX}{self.normalize(question)}|{business_line or 'all'}|{as_of or 'now'}"

    def get(self, key: str) -> dict[str, Any] | None:
        payload = redis_client.get(key)
        if payload:
            logger.info("缓存命中 %s", key)
        return payload

    def set(self, key: str, payload: dict[str, Any], ttl: int | None = None) -> None:
        redis_client.set(key, payload, ttl=self._ttl if ttl is None else ttl)

    def invalidate(self, pattern: str = f"{_ANSWER_PREFIX}*") -> int:
        """制度换版后需要清空第一层，避免返回旧条款。

        只清 answer:* 前缀，不动 FAQ 索引缓存（那是第二三层，与条款版本无关）。
        """
        return redis_client.delete_pattern(pattern)

    def stats(self) -> dict[str, Any]:
        """缓存层自检信息，供 /api/overview 和排查用。"""
        try:
            info = self.client.info("keyspace")
            return {
                "available": redis_client.ping(),
                "db": config.REDIS_DB,
                "answer_keys": len(list(self.client.scan_iter(match=f"{_ANSWER_PREFIX}*", count=500))),
                "lc_keys": len(list(self.client.scan_iter(match=f"{_LC_PREFIX}*", count=500))),
                "ttl_seconds": self._ttl,
                "keyspace": info,
            }
        except redis.RedisError as exc:
            return {"available": False, "error": str(exc)}

    # ---------- langchain_core.caches.BaseCache 接口 ----------
    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        """LangChain 在调用模型前查这里。

        llm_string 是 LangChain 拼出的模型标识（含模型名、参数），
        与 prompt 一起构成键——同一 prompt 换模型不该命中。

        返回结构是 list[list[Generation]]：外层是"第几次生成"，
        内层是这一次生成的多个候选。两层都不能省，只还原一层会丢候选。
        """
        key = self._lc_key(prompt, llm_string)
        value = redis_client.get(key)
        if not value:
            return None
        try:
            return [
                [Generation(**gen) for gen in candidates]
                for candidates in value
            ]
        except (TypeError, ValueError) as exc:
            logger.warning("LangChain 缓存反序列化失败，降级为直连模型：%s", exc)
            return None

    def update(self, prompt: str, llm_string: str, return_val: Sequence[RETURN_VAL_TYPE]) -> None:
        """模型返回后写回。return_val 是多次生成的列表，每次生成是 Generation 列表。"""
        key = self._lc_key(prompt, llm_string)
        payload = [[gen.model_dump() for gen in candidates] for candidates in return_val]
        redis_client.set(key, payload, ttl=self._ttl)

    def clear(self, **kwargs: Any) -> None:
        """只清 LangChain 自己的键空间，不动业务缓存。"""
        redis_client.delete_pattern(f"{_LC_PREFIX}*")

    @staticmethod
    def _lc_key(prompt: str, llm_string: str) -> str:
        digest = hashlib.sha256(f"{llm_string}||{prompt}".encode("utf-8")).hexdigest()
        return f"{_LC_PREFIX}{digest}"


cache = QACache()
