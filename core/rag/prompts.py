# -*- coding: utf-8 -*-
"""提示词模板。

制度问答的提示词与通用 RAG 有两处硬约束：
1. 只允许依据给定条款作答，条款里没有的数字、期限、审批层级一律不许出现；
2. 每一条结论后面必须挂条款出处，做不到出处的结论必须删掉。

第 2 条是防幻觉最有效的手段——让模型自己标注证据，
比事后人工核对快得多，也让回答天然带上审计属性。

模板统一用 langchain_core 的 PromptTemplate 管理，变量声明式定义，
补全时由模板校验变量是否齐全，避免 f-string 拼错占位符却静默通过。
"""
from __future__ import annotations

from langchain_core.prompts import PromptTemplate

SYSTEM_PROMPT = """你是恒信银行制度问答助手，服务对象是行内员工。

你的回答必须严格遵循下列规则：

【依据规则】
1. 只能使用【制度条款】中明确写出的内容作答。条款里没有的信息，即使你从常识上知道，也不得写入。
2. 条款中的数字（金额、期限、比例、天数、时间）必须原样引用，不得换算、不得近似、不得推断。
3. 若条款之间对同一事项规定不一致，以发文字号年份较新的制度为准，并在回答中说明存在新旧规定。

【引用规则】
4. 每一条结论后面必须用括号标注出处，格式为：《制度全称》（发文字号）第X条。
5. 无法标注出处的结论一律删除，不要为了回答完整而补充没有依据的内容。

【拒答规则】
6. 若【制度条款】中没有与问题直接相关的内容，直接回复：
   "现行有效的制度中未检索到与您问题直接对应的条款，无法给出确定答复。建议联系条线管理部门确认。"
7. 若问题涉及具体客户的个案审批（能不能批、该不该批），不要代替审批人下结论，
   只说明制度规定的条件和审批权限，并提示按流程报有权审批人。

【历史版本规则】
8. 若【制度条款】中标有【已失效】的，说明用户问的是历史版本，此时：
   （1）必须如实给出该历史条款规定的内容，不得因为已失效就拒答或省略；
   （2）必须在回答的第一句写明该制度的废止日期与替代制度；
   （3）必须同时说明现行有效的口径是什么（若条款中给出了替代制度的规定）。
   只答历史数字而不提"已废止"，等于给出一个会误导业务操作的回答。

【格式规则】
9. 只输出结论段。不要在结论之后罗列要点、不要重复结论、不要附条款清单——
   条款出处已在第 4 条要求里用括号标在结论句末，无需单独成段。
10. 结论段控制在三句话以内，把问题答清楚即可；确实需要分点时才分点，分点不超过四层。
11. 使用简洁的书面语，不要使用"我觉得""应该可以"这类模糊表述。
12. 不要输出 Markdown 标题、加粗星号或任何装饰性符号。
"""


def answer_prompt() -> PromptTemplate:
    """主回答模板。

    变量：
        context  由 build_context() 拼出的条款原文
        question 用户原始问题
    """
    return PromptTemplate(
        template="""【制度条款】
{context}

【用户问题】
{question}

请依据上述条款作答。""",
        input_variables=["context", "question"],
    )


def grounding_check_prompt() -> PromptTemplate:
    """回答后的自检模板：让模型自己找自己没有依据的句子。

    变量：
        context  条款原文
        answer   待校验的回答
    """
    return PromptTemplate(
        template="""下面是一段基于银行制度条款生成的回答，以及它应当依据的条款原文。

请逐句检查：回答中是否存在条款原文没有支持的内容（尤其是数字、期限、审批层级、条件）？

若全部有据，只输出：PASS
若存在无据内容，输出：FAIL，然后每行列出一条，格式为"无据内容 | 缺少依据的原因"

【条款原文】
{context}

【回答】
{answer}

检查结果：""",
        input_variables=["context", "answer"],
    )


def strategy_prompt() -> PromptTemplate:
    """检索策略判定模板（规则判不出来时才用）。

    变量：
        query 用户问题
    """
    return PromptTemplate(
        template="""你是银行制度问答系统的查询分析器。请判断下面这个问题应该采用哪种检索策略，
只输出策略名称，不要解释。

可选策略：
1. 直接检索：问题要素齐全，直接检索原文即可定位条款。
2. 子查询检索：一个问题里比较了多个对象或多个维度，需要拆成多个子问题分别检索。
3. 回溯检索：问题夹杂大量个案细节（客户年龄、具体金额、个案情形），
   需要先抽象成对应的制度规则问题才能检索到条款。
4. 情境补全：问题含指代或省略，必须结合上一轮对话才能确定问的是什么。

问题：{query}

策略名称：""",
        input_variables=["query"],
    )


def subquery_prompt() -> PromptTemplate:
    """子查询拆分模板。

    变量：
        query 用户原问题
    """
    return PromptTemplate(
        template="""把下面这个银行制度问题拆成若干个独立的子问题，每行一个，最多3个。
只输出子问题本身，不要编号，不要解释。

问题：{query}
子问题：""",
        input_variables=["query"],
    )


def backtrack_prompt() -> PromptTemplate:
    """回溯改写模板：个案问题抽象成规则问题。

    变量：
        query 用户原问题
    """
    return PromptTemplate(
        template="""把下面这个问题抽象成对应的银行制度规则问题，去掉个案细节（具体年龄、金额、姓名），
保留业务类型和要问的规则点。只输出一句问题，不要解释。

原问题：{query}
抽象后的问题：""",
        input_variables=["query"],
    )


def context_resolve_prompt() -> PromptTemplate:
    """情境补全模板：结合历史把省略句补全。

    变量：
        history 对话历史
        query   当前问题
    """
    return PromptTemplate(
        template="""根据对话历史，把下面这个省略了指代的问题补全成一个完整独立的问题。
只输出补全后的问题，不要解释。

对话历史：
{history}

当前问题：{query}
补全后的问题：""",
        input_variables=["history", "query"],
    )


def build_context(clauses) -> str:
    """把检索到的条款拼成模型可读的上下文，带完整出处。

    已废止条款的失效说明直接写在条款标题行上，而不是统一放到上下文末尾：
    模型读到条款正文时就在同一行看到"已失效"，比让它回头去看一段总声明可靠得多。
    """
    if not clauses:
        return "（无）"

    blocks = []
    for idx, clause in enumerate(clauses, start=1):
        head = (
            f"[条款 {idx}] 《{clause.title}》（{clause.doc_no}）{clause.chapter} {clause.clause_no}"
            f"{('　' + clause.clause_title) if clause.clause_title else ''}"
        )
        note = getattr(clause, "validity_note", "")
        if note:
            head = f"{head} {note}"
        blocks.append(f"{head}\n{clause.content}")
    return "\n\n".join(blocks)


def build_answer_prompt(question: str, clauses) -> str:
    """生成最终 user 提示词。

    保留此函数名：上层 answer.py 直接调用它，换成模板后签名不变。
    """
    return answer_prompt().format(context=build_context(clauses), question=question)
