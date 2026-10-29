# -*- coding: utf-8 -*-
"""第一段：跑真实系统，把问答结果落盘。

为什么单独拆一个脚本：
    RAGAS 只能在独立的 venv 里跑（它要求 langchain 0.3 系列，
    而主链路用的是 1.x）。如果评估脚本直接 import ragas，
    主环境就跑不了任何东西。拆成两段后：

        段一 runner  → 主环境跑，只依赖本项目自己的代码
        段二 ragas   → venv 跑，只依赖落盘的 JSON

    这样评估不会反向约束生产代码的依赖版本。

产出 data/eval/raw_results.json，字段与 RAGAS 的输入对齐：
    question / answer / contexts / ground_truth

用法：
    python -m core.assessment.runner
    python -m core.assessment.runner --no-cache --out data/eval/raw_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.assessment.eval_set import EVAL_SET, retrieval_only, stats, stats_by_line
from core.logger import logger
from core.qa_system import PolicyQASystem

DEFAULT_OUT = os.path.join("data", "eval", "raw_results.json")


def run_one(system: PolicyQASystem, item: dict, use_cache: bool) -> dict:
    """跑一条问答，保留评估需要的全部信息。"""
    payload = system.ask(item["question"], use_cache=use_cache)

    if "error" in payload:
        logger.error("问题执行失败：%s → %s", item["question"], payload["error"])
        return {
            "question": item["question"],
            "answer": "",
            "contexts": [],
            "ground_truth": item["ground_truth"],
            "error": payload["error"],
        }

    # contexts 用父块正文：评估要衡量"交给模型的材料对不对"，
    # 而不是子块切片——真正进入 prompt 的就是父块。
    contexts = [c["content"] for c in payload.get("clauses", [])]

    return {
        "question": item["question"],
        "answer": payload.get("answer", ""),
        "contexts": contexts,
        "ground_truth": item["ground_truth"],
        # 以下字段不进 RAGAS，只用来自查
        "clause_key": item["clause_key"],
        "doc_no": item["doc_no"],
        "line": item.get("line", ""),
        "type": item["type"],
        "expect_refuse": item["expect_refuse"],
        "refused": payload.get("refused", False),
        "citations": payload.get("citations", []),
        "retrieved_keys": [
            f"{c['doc_no']}#{c['clause_no']}" for c in payload.get("clauses", [])
        ],
        "strategy": payload.get("strategy", ""),
        "from_cache": payload.get("from_cache", False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="跑评估集，产出 RAGAS 输入")
    parser.add_argument("--out", default=DEFAULT_OUT, help="结果落盘路径")
    parser.add_argument("--no-cache", action="store_true", help="绕过问答缓存，强制真实生成")
    parser.add_argument("--only", choices=["all", "retrieval"], default="all",
                        help="all=含拒答题；retrieval=只跑有依据条款的题")
    args = parser.parse_args()

    items = EVAL_SET if args.only == "all" else retrieval_only()
    system = PolicyQASystem()

    logger.info("评估开始：%d 题，覆盖类型 %s，覆盖条线 %s", len(items), stats(), stats_by_line())
    results = []
    for idx, item in enumerate(items, 1):
        record = run_one(system, item, use_cache=not args.no_cache)
        results.append(record)
        logger.info(
            "[%d/%d] %s | 拒答=%s | 召回=%d条 | 策略=%s",
            idx, len(items), item["question"][:24],
            record.get("refused"), len(record.get("contexts", [])),
            record.get("strategy", ""),
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    # ---------- 自查：不依赖 RAGAS 也能看出系统是否正常 ----------
    print("\n" + "=" * 72)
    print("检索层自查")
    print("=" * 72)
    hit = total = 0
    print(f"{'题目':30s} {'期望条款':26s} {'命中':6s}")
    print("-" * 72)
    for r in results:
        if not r.get("clause_key"):
            continue
        total += 1
        got = r["clause_key"] in r.get("retrieved_keys", [])
        hit += got
        print(f"{r['question'][:28]:30s} {r['clause_key']:26s} {'✓' if got else '✗'}")

    if total:
        print(f"\n条款召回命中率：{hit}/{total} = {hit / total:.1%}")

    ref = [r for r in results if r.get("expect_refuse")]
    if ref:
        ok = sum(1 for r in ref if r.get("refused"))
        print(f"拒答准确率：{ok}/{len(ref)} = {ok / len(ref):.1%}")

    grounded = [r for r in results if not r.get("expect_refuse")]
    if grounded:
        ok = sum(1 for r in grounded if not r.get("refused"))
        print(f"可答题作答率：{ok}/{len(grounded)} = {ok / len(grounded):.1%}")
        with_cite = sum(1 for r in grounded if r.get("citations"))
        print(f"带条款引用：{with_cite}/{len(grounded)} = {with_cite / len(grounded):.1%}")

    # ---------- 按业务条线拆分：总分掩盖不了条线之间的差异 ----------
    print("\n" + "=" * 72)
    print("按业务条线分层自查")
    print("=" * 72)
    print(f"{'条线':12s} {'题量':6s} {'条款召回':12s} {'作答/拒答正确':14s} {'带引用':10s}")
    print("-" * 72)
    lines: dict[str, list[dict]] = {}
    for r in results:
        lines.setdefault(r.get("line") or "未标注", []).append(r)

    for line_name, rows in lines.items():
        cited_rows = [r for r in rows if r.get("clause_key")]
        hit_n = sum(1 for r in cited_rows if r["clause_key"] in r.get("retrieved_keys", []))
        recall = f"{hit_n}/{len(cited_rows)}" if cited_rows else "—"

        # 拒答题看"拒对了没"，可答题看"答了没"
        refuse_rows = [r for r in rows if r.get("expect_refuse")]
        answer_rows = [r for r in rows if not r.get("expect_refuse")]
        ok_n = sum(1 for r in refuse_rows if r.get("refused")) + sum(
            1 for r in answer_rows if not r.get("refused")
        )
        graded = len(refuse_rows) + len(answer_rows)
        graded_txt = f"{ok_n}/{graded}" if graded else "—"

        cite_n = sum(1 for r in answer_rows if r.get("citations"))
        cite_txt = f"{cite_n}/{len(answer_rows)}" if answer_rows else "—"

        print(f"{line_name:12s} {len(rows):<6d} {recall:12s} {graded_txt:14s} {cite_txt:10s}")

    print(f"\n结果已写入 {args.out}（共 {len(results)} 条）")
    print("下一步：.venv-ragas/Scripts/python.exe -m core.assessment.rag_as")


if __name__ == "__main__":
    main()
