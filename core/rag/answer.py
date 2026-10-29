# -*- coding: utf-8 -*-
"""答案生成与引用校验。

流程：
    检索结果为空 → 直接拒答（不调模型，省一次调用也避免幻觉）
    检索有结果   → 生成 → 自检是否每句有据 → 无据则重生成一次 → 仍无据则降级为条款原文

降级策略很重要：制度问答宁可给原文让柜员自己读，也不给一段编好的漂亮答案。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from core.config import config
from core.rag import prompts
from core.rag.llm_client import llm
from core.logger import logger
from core.rag.retriever import RetrievedClause

# 回答中应出现的引用格式：《XX办法》（恒信银发〔2024〕27号）第七条
CITATION_PATTERN = re.compile(r"《[^》]{2,60}》\s*[（(][^）)]{4,40}[）)]\s*第[一二三四五六七八九十百零\d]+条")
# 常见幻觉信号：条款里没有却出现的具体数字
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*(?:万元|元|%|天|个月|年|个工作日|倍)")
# 失效声明：引用废止条款时，回答里必须出现其中之一
REPEAL_DECLARATION_PATTERN = re.compile(r"(废止|失效|作废|已由|取代|替代|不再执行|不再适用)")


@dataclass
class Answer:
    text: str
    clauses: list[RetrievedClause] = field(default_factory=list)
    refused: bool = False
    grounded: bool = True
    citations_found: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.text,
            "refused": self.refused,
            "grounded": self.grounded,
            "citations": self.citations_found,
            "clauses": [
                {
                    "citation": c.citation,
                    "clause_key": c.clause_key,
                    "doc_no": c.doc_no,
                    "title": c.title,
                    "chapter": c.chapter,
                    "clause_no": c.clause_no,
                    "clause_title": c.clause_title,
                    "content": c.content,
                    "similarity": c.similarity,
                }
                for c in self.clauses
            ],
        }


class AnswerGenerator:
    # ---------- 拒答判定 ----------
    @staticmethod
    def _is_refusal_text(text: str) -> bool:
        """模型是否按拒答规则给出了标准拒答话术。

        命中关键词就读作"模型判定条款与问题无关"，
        这是模型的正常产出，不是失败，不应被引用校验当成幻觉再生成一次。
        """
        if not text:
            return False
        head = text.strip()[:120]
        markers = ("未检索到", "无法给出确定答复", "没有与问题直接对应", "无法回答")
        return any(marker in head for marker in markers)

    def _relevance_is_low(self, clauses: list[RetrievedClause]) -> bool:
        """最高相似度偏低时，认为召回结果不足以支撑作答。"""
        if not clauses:
            return True
        return clauses[0].similarity < config.LOW_RELEVANCE_SIMILARITY

    # ---------- 非流式 ----------
    def generate(self, question: str, clauses: list[RetrievedClause]) -> Answer:
        if not clauses:
            return Answer(text=config.REFUSAL_TEMPLATE, refused=True, grounded=True)

        if not llm.available:
            # 模型不可用时退化为条款直读，至少保证信息真实
            logger.warning("模型不可用，降级为条款直读")
            return Answer(text=self._fallback_text(clauses), clauses=clauses, grounded=True)

        user_prompt = prompts.build_answer_prompt(question, clauses)
        answer_text = llm.complete(f"{prompts.SYSTEM_PROMPT}\n\n{user_prompt}", max_tokens=1200)

        if not answer_text:
            return Answer(text=self._fallback_text(clauses), clauses=clauses, grounded=True)

        # 模型主动拒答：直接透传，不参与引用校验
        if self._is_refusal_text(answer_text):
            logger.info("模型判定条款与问题无关，按拒答返回")
            return Answer(text=answer_text.strip(), clauses=clauses, refused=True, grounded=True)

        grounded, issues = self.check_grounding(answer_text, clauses)
        if not grounded:
            logger.warning("自检发现无据内容：%s", issues)
            # 带着问题清单再生成一次，明确要求删掉无据部分
            retry_prompt = (
                f"{prompts.SYSTEM_PROMPT}\n\n{user_prompt}\n\n"
                f"注意：上一版回答中出现下列无依据内容，请删除或修正后重新作答：\n"
                + "\n".join(f"- {item}" for item in issues)
            )
            retry_text = llm.complete(retry_prompt, max_tokens=1200)
            if not retry_text:
                return Answer(text=self._fallback_text(clauses), clauses=clauses, grounded=False)

            if self._is_refusal_text(retry_text):
                logger.info("二次生成转为拒答，条款不足以支撑作答")
                return Answer(text=retry_text.strip(), clauses=clauses, refused=True, grounded=True)

            grounded2, _ = self.check_grounding(retry_text, clauses)
            if grounded2:
                answer_text = retry_text
                grounded = True
            else:
                # 两次都没过关：若召回本身就不相关，如实拒答；
                # 只有当召回确实相关、只是模型没写好处时，才退回条款原文。
                if self._relevance_is_low(clauses):
                    logger.warning("二次生成仍未通过校验且召回相关性偏低，转为拒答")
                    return Answer(text=config.REFUSAL_TEMPLATE, clauses=clauses, refused=True, grounded=True)
                logger.warning("二次生成仍未通过引用校验，降级为条款直读")
                return Answer(text=self._fallback_text(clauses), clauses=clauses, grounded=False)

        citations = CITATION_PATTERN.findall(answer_text)
        return Answer(
            text=answer_text.strip(),
            clauses=clauses,
            refused=False,
            grounded=grounded,
            citations_found=citations,
        )

    # ---------- 流式 ----------
    def stream(self, question: str, clauses: list[RetrievedClause]) -> Iterator[str]:
        """流式生成。流式模式下不做逐句自检，改为在流结束后由调用方校验。"""
        if not clauses:
            yield config.REFUSAL_TEMPLATE
            return

        if not llm.available:
            yield self._fallback_text(clauses)
            return

        user_prompt = prompts.build_answer_prompt(question, clauses)
        produced = []
        for piece in llm.stream(prompts.SYSTEM_PROMPT, user_prompt):
            produced.append(piece)
            yield piece

        # 流结束后做一次整体校验，异常时在末尾追加提示
        full_text = "".join(produced)
        ok, issues = self.check_grounding(full_text, clauses)
        if not ok:
            logger.warning("流式回答未通过引用校验：%s", issues)
            yield "\n\n> 提示：本回答未通过自动引用校验，请以文末条款原文为准。"

    # ---------- 校验 ----------
    def check_grounding(self, answer_text: str, clauses: list[RetrievedClause]) -> tuple[bool, list[str]]:
        """检查数字是否有据，以及引用了废止条款时是否声明了失效。

        只校验数字类硬信息，因为这是制度问答里代价最高的幻觉：
        把"50万元"说成"80万元"是会直接产生业务损失的。
        """
        context = prompts.build_context(clauses)
        issues: list[str] = []

        for match in NUMBER_PATTERN.finditer(answer_text):
            literal = match.group(0).replace(" ", "")
            if literal not in context.replace(" ", ""):
                issues.append(f"{literal}（条款原文中未出现该数值）")

        # 引用格式检查：有条款但回答里一条出处都没有，说明模型没按规则输出
        if CITATION_PATTERN.search(answer_text) is None and len(answer_text) > 40:
            issues.append("整段回答未标注任何条款出处")

        # 失效声明检查：拿到废止条款却只答数字不提废止，是这里最危险的一种"对而有害"。
        # 数字全对、引用也在，业务上却会把旧口径当现行口径用，所以必须当作错误拦下，
        # 交给上层重生成，而不是靠 prompt 里一句"记得声明"。
        repealed = [c for c in clauses if getattr(c, "is_repealed", False)]
        if repealed and answer_text and not REPEAL_DECLARATION_PATTERN.search(answer_text):
            names = "、".join(f"《{c.title}》（{c.doc_no}）" for c in repealed)
            issues.append(f"引用了已废止条款 {names}，但回答未声明该制度已失效")

        return (not issues), issues

    # ---------- 降级 ----------
    @staticmethod
    def _fallback_text(clauses: list[RetrievedClause]) -> str:
        has_repealed = any(getattr(c, "is_repealed", False) for c in clauses)
        head = (
            "未取得模型生成结果，以下为检索到的条款原文（含已废止条款，请注意效力）："
            if has_repealed
            else "未取得模型生成结果，以下为检索到的现行有效条款原文："
        )
        lines = [head, ""]
        for idx, clause in enumerate(clauses, start=1):
            note = getattr(clause, "validity_note", "")
            tag = f" {note}" if note else ""
            lines.append(f"{idx}. {clause.citation}　{clause.clause_title}{tag}")
            lines.append(clause.content.strip())
            lines.append("")
        lines.append("如需完整口径，请对照制度原文或联系条线管理部门确认。")
        return "\n".join(lines)


generator = AnswerGenerator()
