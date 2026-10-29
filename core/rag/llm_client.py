# -*- coding: utf-8 -*-
"""大模型调用封装。

底层用 langchain_openai.ChatOpenAI，统一走 OpenAI 兼容协议，
在通义千问、DeepSeek、行内私有化部署之间只需换 base_url。

对外仍暴露 complete() / stream() 两个方法、与原来完全一致，
上层 answer.py、query_analyzer.py、grounding.py 不需要跟着改。

用 ChatOpenAI 而非 openai SDK 直连的理由：
1. 消息对象、参数、返回结构都由 LangChain 的标准化类型收口，
   换供应商时不用逐处改解包代码；
2. 上层如果要接 LCEL（prompt | llm | parser），ChatOpenAI 直接能上管道，
   SDK 直连还得再包一层；
3. 流式与非流式共用同一套消息构造，减少两处写法漂移的风险。

未配置 API Key 时 available 为 False，上层自动降级为"只返回条款原文"，
不会因为模型不可用而让整个问答挂掉。
"""
from __future__ import annotations

from typing import Iterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.config import config
from core.logger import logger


class LLMClient:
    def __init__(self) -> None:
        self._client = None

    @property
    def available(self) -> bool:
        return bool(config.LLM_API_KEY)

    @property
    def client(self) -> ChatOpenAI:
        """惰性构造，避免没配 key 时启动就报错。"""
        if self._client is None:
            self._client = ChatOpenAI(
                model=config.LLM_MODEL,
                api_key=config.LLM_API_KEY or "not-configured",
                base_url=config.LLM_BASE_URL,
                temperature=config.LLM_TEMPERATURE,
                timeout=60,
                max_retries=1,
            )
        return self._client

    def complete(self, prompt: str, temperature: float | None = None, max_tokens: int = 1024) -> str:
        """非流式调用，失败返回空串。

        temperature 显式传入时走 bind 覆盖默认值，不在实例上改状态，
        避免同一 client 被不同调用线程改坏。
        """
        if not self.available:
            return ""
        try:
            target = self.client
            if temperature is not None:
                target = target.bind(temperature=temperature, max_tokens=max_tokens)
            resp = target.invoke([HumanMessage(content=prompt)])
            content = resp.content
            # 部分兼容端点会返回 content 列表（多模态格式），拼成纯文本
            if isinstance(content, list):
                content = "".join(
                    part if isinstance(part, str) else part.get("text", "")
                    for part in content
                )
            return content or ""
        except Exception as exc:
            logger.error("模型调用失败：%s", exc)
            return ""

    def chain(self, temperature: float | None = None, max_tokens: int = 1024):
        """返回可接进 LCEL 管道的可调用对象。

        用途是 `PromptTemplate | llm.chain() | StrOutputParser()`：
        PromptTemplate 负责变量校验，这里负责"把 prompt 变成模型回复"，
        StrOutputParser 负责把 AIMessage 拆成纯文本。三段各自可替换。

        与 complete() 的分工：complete() 是给"已经有整段 prompt 字符串"的
        调用方用的（如 answer.py 里 system + user 两段式），
        chain() 是给"prompt 由模板产出"的调用方用的。两者共用同一个底层 client。
        """
        target = self.client
        if temperature is not None:
            target = target.bind(temperature=temperature, max_tokens=max_tokens)
        return target

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """流式调用，逐段产出文本。"""
        if not self.available:
            logger.warning("未配置模型 API Key，跳过生成")
            return
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            for chunk in self.client.stream(messages):
                text = chunk.content
                if isinstance(text, list):
                    text = "".join(
                        part if isinstance(part, str) else part.get("text", "")
                        for part in text
                    )
                if text:
                    yield text
        except Exception as exc:
            logger.error("模型流式调用失败：%s", exc)
            yield "\n\n[生成中断：模型服务异常，请稍后重试]"


llm = LLMClient()
