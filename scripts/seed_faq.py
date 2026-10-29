# -*- coding: utf-8 -*-
"""生成 FAQ 种子数据的候选清单。

做法：对每个条款内置若干"柜员/客户真会这么问"的问法，答案直接从条款原文
摘取（按句切分后取支撑该问法的那一两句），再输出到 CSV 供人工审定。

为什么不让模型生成答案：FAQ 的答案一经审定就会绕过整条检索链路直接返回，
写错一个数字没人拦得住。所以答案一律来自条款原文，脚本只负责摘取和组装。

用法：
    python -m scripts.seed_faq            # 生成 data/faq/faq_seed.csv
    python -m scripts.seed_faq --apply    # 审定后写库
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.logger import logger  # noqa: E402
from core.store.policy_db import db  # noqa: E402

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "faq" / "faq_seed.csv"

# 一条 FAQ 答案的上限。FAQ 命中后直接返回，超过这个长度说明锚点取得太宽，
# 应该收窄锚点而不是让用户去读半条制度。
MAX_ANSWER_CHARS = 160

DOC_27 = "恒信银发〔2024〕27号"
DOC_3 = "恒信银发〔2025〕3号"
DOC_58 = "恒信银运〔2024〕58号"

# (标准问法, 条款键, 锚点元组, 业务条线)
# 锚点是条款原文里的特征词；命中后向后连取 take 个分句（默认 1）。
# 需要"引导句 + 分项"一起答的时候写 ("引导词",) 命中后 take=2；
# 需要拼两段不相邻的内容时写多个锚点。
SEED_CANDIDATES: list[tuple[str, str, tuple, str]] = [
    # ---- 个人信贷 ----
    ("个人消费贷款最多能贷多少钱？", f"{DOC_27}#第七条", ("信用类消费贷款单户最高", 2), "个人信贷"),
    ("抵押类消费贷款能贷多少？抵押率有要求吗？", f"{DOC_27}#第七条", ("抵押类消费贷款单户最高",), "个人信贷"),
    ("个人经营贷款的最高额度是多少？", f"{DOC_27}#第七条", ("个人经营贷款单户最高",), "个人信贷"),
    ("个人消费贷款最长可以贷几年？", f"{DOC_27}#第八条", ("个人消费贷款期限最长",), "个人信贷"),
    ("个人经营贷款的期限最长多久？", f"{DOC_27}#第八条", ("个人经营贷款期限最长",), "个人信贷"),
    ("抵押类个人贷款的期限加年龄有什么限制？", f"{DOC_27}#第八条", ("借款人年龄之和",), "个人信贷"),
    ("多少钱以内的贷款分行自己就能批？", f"{DOC_27}#第九条", ("50万元（含）以下",), "个人信贷"),
    ("超过200万的个人贷款要报到哪一级审批？", f"{DOC_27}#第九条", ("200万元以上",), "个人信贷"),
    ("个人贷款必须面签吗？", f"{DOC_27}#第十一条", ("面谈面签制度",), "个人信贷"),
    ("电子渠道签的贷款合同有效吗？", f"{DOC_27}#第十一条", ("电子签名法",), "个人信贷"),
    ("月债务收入比的上限是多少？", f"{DOC_27}#第六条", ("债务收入比",), "个人信贷"),
    ("收入证明怎么认定？要单位盖章吗？", f"{DOC_27}#第六条", ("收入证明的认定方式", 2), "个人信贷"),
    ("自己开公司的，收入怎么认定？", f"{DOC_27}#第六条", ("适用于自雇人士",), "个人信贷"),
    ("贷款能不能用来买房付首付？", f"{DOC_27}#第五条", ("不得用于下列用途", 2), "个人信贷"),
    ("贷款资金可以拿去炒股吗？", f"{DOC_27}#第五条", ("股票、债券、期货",), "个人信贷"),
    ("贷款资金可以由别人代收吗？", f"{DOC_27}#第五条", ("不得由第三方代收",), "个人信贷"),
    ("贷款逾期了多久会开始催收？", f"{DOC_27}#第十五条", ("3个工作日内启动催收",), "个人信贷"),
    ("催收电话可以在晚上打吗？", f"{DOC_27}#第十五条", ("催收时间限定",), "个人信贷"),
    ("催收的录音要保存多久？", f"{DOC_27}#第十五条", ("录音录像资料保存期限",), "个人信贷"),
    ("个人贷款的利率怎么定？", f"{DOC_27}#第十条", ("利率定价管理办法",), "个人信贷"),
    # ---- 反洗钱 ----
    ("新客户建立业务关系后多久要完成风险等级评定？", f"{DOC_3}#第七条", ("10个工作日内",), "反洗钱"),
    ("客户风险等级分几级？", f"{DOC_3}#第七条", ("五级",), "反洗钱"),
    ("客户身份资料要保存多少年？", f"{DOC_3}#第十条", ("至少保存5年",), "反洗钱"),
    ("哪些客户需要识别受益所有人？", f"{DOC_3}#第五条", ("受益所有人", 2), "反洗钱"),
    ("高风险客户多久要重新审核一次？", f"{DOC_3}#第八条", ("每半年至少重新审核一次",), "反洗钱"),
    ("哪些情况要启动可疑交易分析？", f"{DOC_3}#第九条", ("启动可疑交易分析", 2), "反洗钱"),
    ("单日现金存取超过多少要关注？", f"{DOC_3}#第九条", ("单日现金存取累计超过",), "反洗钱"),
    # ---- 运营柜面 ----
    ("开个人账户需要什么材料？", f"{DOC_58}#第三条", ("由本人持有效身份证件办理",), "运营柜面"),
    ("一个人能在一家银行开几个Ⅰ类户？", f"{DOC_58}#第三条", ("只能开立一个Ⅰ类户",), "运营柜面"),
    ("未成年人开户怎么办理？", f"{DOC_58}#第四条", ("无民事行为能力人",), "运营柜面"),
    ("大额现金存取要登记吗？多少算大额？", f"{DOC_58}#第六条", ("人民币5万元以上现金存取",), "运营柜面"),
    ("20万以上现金存取需要复核吗？", f"{DOC_58}#第六条", ("人民币20万元以上",), "运营柜面"),
    ("手机银行转账限额是多少？", f"{DOC_58}#第九条", ("短信验证方式",), "运营柜面"),
    ("U盾转账的单笔和日累计限额是多少？", f"{DOC_58}#第九条", ("数字证书或U盾",), "运营柜面"),
    ("柜面转账有金额限制吗？", f"{DOC_58}#第九条", ("柜面转账",), "运营柜面"),
    ("银行卡丢了怎么挂失？", f"{DOC_58}#第十二条", ("临时挂失或正式挂失",), "运营柜面"),
    ("挂失后多久能补办新卡？", f"{DOC_58}#第十二条", ("正式挂失后7天",), "运营柜面"),
    ("老人要给陌生账户大额转账该怎么办？", f"{DOC_58}#第十一条", ("老年客户大额转账",), "运营柜面"),
]

# 句号、分号、换行都算切分点。换行必须算：制度正文里的分项列表
# （"不得用于下列用途：\n\n（一）……；"）是靠换行分段的，分项句末虽有分号，
# 但引出分项的那句以冒号结尾，只按句号/分号切会把主干和分项切成两块。
_SPLIT_RE = re.compile(r"(?<=[。；])|\n+")

# 以冒号结尾、或明显在引出下文的句子，摘答案时要连同它的后续分项一起带
_LEAD_IN_RE = re.compile(r"[：:]$")
# 分项序号：（一）（二）;(1)1. 等。分项脱离主干就没有主语，不能单独当答案
_ITEM_NO_RE = re.compile(r"^[（(]\s*[一二三四五六七八九十百\d]+\s*[）)]")


def _fetch_clause(doc_no: str, clause_no: str) -> str:
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT content, status FROM policy_clause WHERE doc_no = %s AND clause_no = %s",
            (doc_no, clause_no),
        )
        row = cur.fetchone()
    if not row:
        raise LookupError(f"条款不存在：{doc_no} {clause_no}")
    if row["status"] != "active":
        raise ValueError(f"条款已失效，不能作为 FAQ 依据：{doc_no} {clause_no}")
    return row["content"].strip()


def _pick_answer(content: str, anchor: tuple) -> str:
    """从条款正文里摘出含锚点的句子，锚点可带 take 表示向后连取几个分句。

    为什么不用整条条款当答案：FAQ 命中后会绕过检索直接返回，答案太长用户
    找不到重点，而且条款里总有与问题无关的分项，贴回去等于让用户自己筛。
    也绝不退回整条条款——摘不到就报错，说明锚点写错了，该修锚点不该放宽策略。
    """
    word, take = anchor[0], (anchor[1] if len(anchor) > 1 else 1)
    pieces = [s.strip() for s in _SPLIT_RE.split(content)]
    pieces = [s for s in pieces if s]
    for index, piece in enumerate(pieces):
        if word in piece:
            # 锚点落在分项里时，把引出这些分项的主干一并带上。
            # 只看相邻上一段是不够的——锚点常是「（二）」，而「（一）」横在中间，
            # 所以要从锚点往前扫到第一个非分项段落为止。
            start = index
            while start > 0 and _ITEM_NO_RE.match(pieces[start]):
                start -= 1
            picked = pieces[start:index + take]
            answer = "".join(picked).strip()
            # 锚点本身就是引出句（"不得用于下列用途："），答案在后续分项里，整组带上
            if _LEAD_IN_RE.search(picked[-1]):
                tail = []
                for extra in pieces[index + 1:]:
                    if not _ITEM_NO_RE.match(extra):
                        break
                    tail.append(extra)
                answer = "".join(picked + tail).strip()
            return answer
    raise LookupError(f"锚点未在条款中找到：{word}")


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for question, key, anchor, line in SEED_CANDIDATES:
        doc_no, _, clause_no = key.partition("#")
        try:
            content = _fetch_clause(doc_no, clause_no)
            answer = _pick_answer(content, anchor)
        except (LookupError, ValueError) as exc:
            logger.error("跳过 [%s]：%s", question, exc)
            continue
        if len(answer) > MAX_ANSWER_CHARS:
            logger.error("跳过 [%s]：答案 %d 字过长", question, len(answer))
            continue
        rows.append({
            "question": question,
            "answer": answer,
            "clause_keys": key,
            "business_line": line,
            "source": "seed",
            "clause_title": f"{doc_no} {clause_no}",
        })
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["question", "answer", "clause_keys", "business_line", "source", "clause_title"],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("已写出 %d 条候选到 %s", len(rows), OUTPUT)


def apply_csv() -> None:
    if not OUTPUT.exists():
        raise SystemExit(f"找不到 {OUTPUT}，请先执行 python -m scripts.seed_faq 生成并审定")
    with OUTPUT.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if not row.get("answer", "").strip():
            logger.warning("跳过空答案：%s", row["question"][:20])
            continue
        db.upsert_faq(row)
    from core.rag.faq_search import faq_search

    count = faq_search.refresh()
    logger.info("已写入 %d 条 FAQ，索引当前 %d 条", len(rows), count)


def main() -> None:
    parser = argparse.ArgumentParser(description="FAQ 种子数据")
    parser.add_argument("--apply", action="store_true", help="把审定后的 CSV 写入数据库")
    args = parser.parse_args()

    db.init_schema()
    if args.apply:
        apply_csv()
    else:
        write_csv(build_rows())


if __name__ == "__main__":
    main()
