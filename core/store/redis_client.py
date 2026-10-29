# -*- coding: utf-8 -*-
"""Redis 通用客户端：三层缓存共用的连接与读写原语。

这一层只做"键 → 值"的搬运，不掺业务判断。三层缓存各自的键怎么拼、
值存什么结构、什么时候失效，分别在调用方决定：

    第一层  问答结果缓存   core/store/qa_cache.py   answer:{query}
    第二层  标准问法缓存   core/rag/faq_search.py  qa_original_questions
    第三层  问法分词缓存   core/rag/faq_search.py  qa_tokenized_questions

为什么不把三层合并进一个类：它们的失效时机完全不同。问法库变了要清二三
层，制度换版要清第一层，混在一起写会出现"改了一条 FAQ 顺手把整库答案缓存
清了"这种连带失效。

连接上的两个约定：
- 全部方法在 Redis 不可用时返回 None / False 而不抛异常。缓存是加速手段，
  不能成为问答链路的单点故障。
- 显式指定 protocol=2（RESP2）。redis-py 8.x 默认先用 RESP3 的 HELLO 握手，
  对设了 requirepass 的老版本服务端会握手失败，报"必须已认证才能调用 HELLO"。
"""

from __future__ import annotations

import json
from typing import Any

import redis

from core.config import config
from core.logger import logger


class RedisClient:
    """轻量 Redis 封装。惰性连接，首次访问时才建连。"""

    def __init__(self) -> None:
        self._client: redis.Redis | None = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD or None,
                db=config.REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=2,
                protocol=2,
            )
        return self._client

    # ---------- 字符串 ----------

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """写值。ttl 为 None 表示不过期，由调用方的主动失效负责。"""
        try:
            payload = json.dumps(value, ensure_ascii=False)
            if ttl:
                self.client.setex(key, ttl, payload)
            else:
                self.client.set(key, payload)
            return True
        except (redis.RedisError, TypeError) as exc:
            logger.warning("Redis 写入失败（%s）：%s", key, exc)
            return False

    def get(self, key: str) -> Any:
        try:
            raw = self.client.get(key)
            return json.loads(raw) if raw else None
        except (redis.RedisError, json.JSONDecodeError) as exc:
            logger.warning("Redis 读取失败（%s）：%s", key, exc)
            return None

    def delete(self, *keys: str) -> int:
        try:
            return int(self.client.delete(*keys)) if keys else 0
        except redis.RedisError as exc:
            logger.warning("Redis 删除失败：%s", exc)
            return 0

    def delete_pattern(self, pattern: str) -> int:
        """按前缀批量删。用 scan_iter 而不是 keys，避免大库阻塞。"""
        try:
            keys = list(self.client.scan_iter(match=pattern, count=500))
            return int(self.client.delete(*keys)) if keys else 0
        except redis.RedisError as exc:
            logger.warning("Redis 批量删除失败（%s）：%s", pattern, exc)
            return 0

    def exists(self, key: str) -> bool:
        try:
            return bool(self.client.exists(key))
        except redis.RedisError:
            return False

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except redis.RedisError:
            return False


redis_client = RedisClient()
