# -*- coding: utf-8 -*-
"""问题理解与检索策略选择。

制度问答的提问方式和通用问答不同，柜员问的往往是"我这个客户能不能办"，
里面混着业务动作、金额、客户类型三类信息。直接拿原句去检索，
稠密向量会被"能不能办"这种口语拉偏，所以先做一次结构化拆解：

    策略          适用场景                          示例
    ------------  --------------------------------  ------------------------------------
    直接检索      要素齐全、能直接定位条款           "个人消费贷款单户额度上限是多少"
    子查询检索    一句话里问了多个额度/期限/权限     "消费贷和经营贷的额度、期限分别是多少"
    回溯检索      问题太具体、含大量个案细节         "客户56岁有房贷能不能贷10年"
    情境补全      含指代、省略，需结合上下文         "那这个额度需要谁审批"
    历史版本      明确在问某个旧版本的规定           "2021年那个办法规定的消费贷额度是多少"

"历史版本"与其余四个策略不是同一维度：那四个决定"怎么改写检索式"，
这一个决定"能不能看到已废止条款"。制度会换版，合规检查与争议追溯经常要
回答"当时的规定是什么"，而默认的效力硬过滤会把废止条款整份滤掉，
导致系统只能回答"未检索到"——这不是更安全，是把一种可查的信息变成了查不到。

策略选择走规则优先、模型兜底：规则能判的（含"和/分别"、含年龄金额等个案要素
或明确的时间线索）直接判，判不出来才调模型，既省调用又避免模型把简单问题复杂度化。
"""
from __future__ import annotations

import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from core.config import config
from core.logger import logger

STRATEGY_DIRECT = "直接检索"
STRATEGY_SUBQUERY = "子查询检索"
STRATEGY_BACKTRACK = "回溯检索"
STRATEGY_CONTEXT = "情境补全"
STRATEGY_HISTORY = "历史版本"

# 多个问询对象 → 拆子查询
_MULTI_TARGET = re.compile(r"(和|与|、|以及|分别|各自|还有)")
# 个案要素 → 需要抽象成规则问题
_CASE_FACTOR = re.compile(r"\d+\s*(岁|万|元|天|个月|年|%)|能不能|可不可以|行不行|我客户|这个客户")
# 指代 → 需要结合上下文
_CONTEXT_HINT = re.compile(r"^(那|那么|这个|该|它|上面|刚才|还有)|这些|那些")

# 历史版本线索：年份、文号、或明确指向旧规定的措辞。
# 只认"20XX年"不认"XX年"，因为"3年""5年"是期限不是版本；
# 单独一个"年"字会被"消费贷款期限几年"这类问题误伤。
_YEAR_HINT = re.compile(r"(20\d{2})\s*年")
_DOC_NO_HINT = re.compile(r"〔\s*(20\d{2})\s*〕\s*\d+\s*号")
_LEGACY_HINT = re.compile(
    r"(旧版|老版|旧的|老的|旧办法|旧规定|老办法|老规定|原办法|原规定|原来|之前的|以前的"
    r"|废止|失效|作废|修订前|修改前|没改之前)"
)
# 问"现在/最新/现行"时说明用户在问现行口径，不是历史版本
_CURRENT_HINT = re.compile(r"(现在|目前|当前|最新|现行|今年)")

# 条线名 → 检索过滤值。用户在问题里点名条线时必须过滤，
# 否则"受益所有人"这种跨条线都可能出现的词会被别的制度抢走。
_LINE_ALIASES: dict[str, tuple[str, ...]] = {
    "个人信贷": ("个人贷款", "消费贷", "经营贷", "信贷", "贷款", "授信", "催收", "抵押"),
    "反洗钱": ("反洗钱", "洗钱", "受益所有人", "尽职调查", "可疑交易", "身份识别", "风险等级"),
    "运营柜面": ("柜面", "开户", "挂失", "转账", "现金", "柜员", "网点", "银行卡", "账户"),
}

# 四个模板在模块装载时构造一次。PromptTemplate 在这里的价值不是省几行字符串拼接，
# 而是把变量声明成契约：模板里写了 {query}，调用时漏传会直接抛错，
# 而不是像 f-string 那样拼出一个缺了关键信息的 prompt 却静默跑下去。
STRATEGY_TEMPLATE = PromptTemplate(
    template="""你是银行制度问答系统的查询分析器。请判断下面这个问题应该采用哪种检索策略，
只输出策略名称，不要解释。

可选策略：
1. {direct}：问题要素齐全，直接检索原文即可定位条款。
2. {subquery}：一个问题里比较了多个对象或多个维度，需要拆成多个子问题分别检索。
3. {backtrack}：问题夹杂大量个案细节（客户年龄、具体金额、个案情形），
   需要先抽象成对应的制度规则问题才能检索到条款。
4. {context}：问题含指代或省略，必须结合上一轮对话才能确定问的是什么。
5. {history}：问题明确在问某个旧版本制度的规定（提到具体年份、旧文号，
   或用了"旧办法""原规定""废止前"这类说法），需要取到已废止的条款。

问题：{query}

策略名称：""",
    input_variables=["query", "direct", "subquery", "backtrack", "context", "history"],
)

SUBQUERY_TEMPLATE = PromptTemplate(
    template="""把下面这个银行制度问题拆成若干个独立的子问题，每行一个，最多3个。
只输出子问题本身，不要编号，不要解释。

问题：{query}
子问题：""",
    input_variables=["query"],
)

BACKTRACK_TEMPLATE = PromptTemplate(
    template="""把下面这个问题抽象成对应的银行制度规则问题，去掉个案细节（具体年龄、金额、姓名），
保留业务类型和要问的规则点。只输出一句问题，不要解释。

原问题：{query}
抽象后的问题：""",
    input_variables=["query"],
)

CONTEXT_TEMPLATE = PromptTemplate(
    template="""根据对话历史，把下面这个省略了指代的问题补全成一个完整独立的问题。
只输出补全后的问题，不要解释。

对话历史：
{history}

当前问题：{query}
补全后的问题：""",
    input_variables=["query", "history"],
)


class QueryAnalyzer:
    def __init__(self) -> None:
        self._llm = None
        self._parser = StrOutputParser()

    # ---------- 规则判别 ----------
    @staticmethod
    def is_history_query(query: str) -> bool:
        """问题是否在问某个历史版本。

        判据是"有没有指向过去的线索"，而不是"有没有出现年份"：
        "消费贷款期限最长几年"里没有年份，"2021年那个办法规定的额度"里有。
        同时出现"现在/目前"时判为问现行口径，直接短路——
        "2021年的规定现在还算数吗"问的是效力，不是当年数字，不能当历史题。
        """
        q = query or ""
        if _CURRENT_HINT.search(q) and not _LEGACY_HINT.search(q):
            return False
        return bool(_YEAR_HINT.search(q) or _DOC_NO_HINT.search(q) or _LEGACY_HINT.search(q))

    @staticmethod
    def extract_year(query: str) -> int | None:
        """取问题里出现的年份，用于定位制度版本。

        优先取文号里的年份（"〔2021〕9号"比"2021年"更可信），
        其次取显式年份。年份本身在制度语境里是明确的，不需要做歧义消解。
        """
        q = query or ""
        hit = _DOC_NO_HINT.search(q)
        if hit:
            return int(hit.group(1))
        hit = _YEAR_HINT.search(q)
        if hit:
            return int(hit.group(1))
        return None

    @staticmethod
    def extract_doc_no_hint(query: str) -> str:
        """取问题里直接点名的文号片段，如"〔2021〕9号"，用于精确匹配制度。"""
        hit = _DOC_NO_HINT.search(query or "")
        return hit.group(0).replace(" ", "") if hit else ""

    @staticmethod
    def infer_business_line(query: str) -> str | None:
        """从问题措辞推断业务条线。

        只做过滤而不做"猜"，因为条线是制度的固有属性：
        反洗钱细则里的"客户"和信贷办法里的"客户"是两个概念，
        不过滤的话，问"客户风险等级"会把信贷办法里的客户分类条款一起召回。
        多个条线同时命中时不猜，交回给用户或按全域检索。
        """
        q = query or ""
        hits = [
            line
            for line, aliases in _LINE_ALIASES.items()
            if any(alias in q for alias in aliases)
        ]
        return hits[0] if len(hits) == 1 else None

    @classmethod
    def rule_based(cls, query: str, has_history: bool = False) -> str | None:
        """能靠规则判定的直接返回策略，判不了返回 None。"""
        q = query.strip()

        if has_history and (_CONTEXT_HINT.search(q) or len(q) <= 8):
            return STRATEGY_CONTEXT

        # 历史版本优先于其余策略判定：它决定的是"看不看得到废止条款"，
        # 属于检索范围问题，一旦判错后面怎么改写都补不回来。
        if cls.is_history_query(q):
            return STRATEGY_HISTORY

        if _MULTI_TARGET.search(q) and len(q) > 12:
            return STRATEGY_SUBQUERY

        if _CASE_FACTOR.search(q):
            return STRATEGY_BACKTRACK

        if len(q) <= 30:
            return STRATEGY_DIRECT

        return None

    # ---------- 模型兜底 ----------
    @property
    def llm(self):
        if self._llm is None:
            from core.rag.llm_client import llm

            self._llm = llm
        return self._llm

    def _run_chain(self, template: PromptTemplate, variables: dict, temperature: float = 0.0) -> str:
        """模板 → 模型 → 纯文本，走 LCEL 管道。

        用管道而不是手写 format + complete，是因为这三段的分工固定且可替换：
        换模型只改 client，换解析只改 parser，模板本身不动。
        """
        chain = template | self.llm.chain(temperature=temperature) | self._parser
        try:
            return (chain.invoke(variables) or "").strip()
        except Exception as exc:  # noqa: BLE001
            # 改写失败不该阻断问答，退回原问题继续检索
            logger.warning("查询改写失败，退回原问题：%s", exc)
            return ""

    def select(self, query: str, has_history: bool = False) -> str:
        strategy = self.rule_based(query, has_history)
        if strategy:
            logger.info("策略判定（规则）：%s ← %s", strategy, query[:30])
            return strategy

        if not config.ENABLE_QUERY_REWRITE or not self.llm.available:
            return STRATEGY_DIRECT

        answer = self._run_chain(
            STRATEGY_TEMPLATE,
            {
                "query": query,
                "direct": STRATEGY_DIRECT,
                "subquery": STRATEGY_SUBQUERY,
                "backtrack": STRATEGY_BACKTRACK,
                "context": STRATEGY_CONTEXT,
                "history": STRATEGY_HISTORY,
            },
        )
        for candidate in (
            STRATEGY_DIRECT,
            STRATEGY_SUBQUERY,
            STRATEGY_BACKTRACK,
            STRATEGY_CONTEXT,
            STRATEGY_HISTORY,
        ):
            if candidate in answer:
                logger.info("策略判定（模型）：%s ← %s", candidate, query[:30])
                return candidate
        return STRATEGY_DIRECT

    # ---------- 改写 ----------
    def rewrite(self, query: str, strategy: str, history: str = "") -> list[str]:
        """把原始问题改写成适合检索的若干条查询。

        返回列表：单条查询返回 1 个元素，子查询返回多个。
        """
        if not config.ENABLE_QUERY_REWRITE or not self.llm.available:
            return [query]

        if strategy == STRATEGY_SUBQUERY:
            return self._split_subqueries(query)
        if strategy == STRATEGY_BACKTRACK:
            return self._backtrack(query)
        if strategy == STRATEGY_CONTEXT:
            return self._resolve_context(query, history)
        return [query]

    def _split_subqueries(self, query: str) -> list[str]:
        text = self._run_chain(SUBQUERY_TEMPLATE, {"query": query})
        subs = [line.strip(" -、0123456789.") for line in text.splitlines() if line.strip()]
        subs = [s for s in subs if len(s) >= 4][:3]
        return subs or [query]

    def _backtrack(self, query: str) -> list[str]:
        text = self._run_chain(BACKTRACK_TEMPLATE, {"query": query})
        return [text] if len(text) >= 4 else [query]

    def _resolve_context(self, query: str, history: str) -> list[str]:
        if not history:
            return [query]
        text = self._run_chain(CONTEXT_TEMPLATE, {"query": query, "history": history})
        return [text] if len(text) >= 4 else [query]


analyzer = QueryAnalyzer()
