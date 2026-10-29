# -*- coding: utf-8 -*-
"""对照实验：验证"引用括号是否被 RAGAS 误判为无据论断"。

背景：
    回答按制度问答的要求，每条结论句末都挂《全称》（文号）第X条。
    这个括号里的字符串不会出现在条款原文里，RAGAS 拆论断时很可能
    把它当成一条独立断言，找不到依据就判为幻觉，从而系统性压低 faithfulness。

做法：
    同一批回答跑两遍，一遍带引用括号，一遍去掉。
    两次的差额，就是"引用格式"对指标的污染量。

这个实验只改评估输入，不改系统输出——生产链路的引用规则一个字不动。
"""
from __future__ import annotations

import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from datasets import Dataset  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.metrics import faithfulness  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

from core.assessment.rag_as import make_judge, require_api_key  # noqa: E402

DEFAULT_IN = os.path.join(PROJECT_ROOT, "data", "eval", "raw_results.json")

# 引用格式：《制度全称》（发文字号）第X条，外面再套一层中文括号。
# 全称里可能含书名号，文号里含六角括号，所以按"最外层括号配对"来削，
# 正则处理不了嵌套，改用字符扫描。
def strip_citations(text: str) -> str:
    out = []
    depth = 0
    for ch in text:
        if ch == "（":
            depth += 1
            continue
        if ch == "）":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out).strip()


def run(items: list[dict], label: str) -> float:
    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in items],
            "answer": [r["answer"] for r in items],
            "contexts": [r["contexts"] for r in items],
            "ground_truth": [r["ground_truth"] for r in items],
        }
    )
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness],
        llm=make_judge(),
        run_config=RunConfig(max_workers=2, timeout=240),
        raise_exceptions=False,
    )
    score = float(result.to_pandas()["faithfulness"].mean())
    print(f"  {label:16s} faithfulness = {score:.4f}")
    return score


def main() -> None:
    require_api_key()
    with open(DEFAULT_IN, encoding="utf-8") as fp:
        items = [r for r in json.load(fp) if not r.get("expect_refuse") and r.get("answer")]

    stripped = [
        {**r, "answer": strip_citations(r["answer"])}
        for r in items
    ]

    print(f"\n对照实验：{len(items)} 条可答题\n")
    raw = run(items, "带引用括号")
    clean = run(stripped, "去引用括号")

    print(f"\n差值：{clean - raw:+.4f}")
    print("\n去引用后的答案示例：")
    for r in stripped[:2]:
        print(f"  {r['answer'][:150]}")


if __name__ == "__main__":
    main()
