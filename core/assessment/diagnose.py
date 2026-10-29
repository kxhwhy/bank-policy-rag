# -*- coding: utf-8 -*-
"""逐条诊断：把 RAGAS 每条指标的单项得分和理由导出。

主报告只给均值，看不出是哪一条、哪句话在掉分。
这个脚本按题打印分数，并把 faithfulness 的判定理由原样输出——
RAGAS 会说清楚它认为哪段论断在条款里找不到依据。
"""
from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd  # noqa: E402
from datasets import Dataset  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig  # noqa: E402

from core.assessment.rag_as import make_embeddings, make_judge, require_api_key  # noqa: E402

DEFAULT_IN = os.path.join(PROJECT_ROOT, "data", "eval", "raw_results.json")


def main() -> None:
    inp = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
    require_api_key()

    with open(inp, encoding="utf-8") as fp:
        results = [r for r in json.load(fp) if not r.get("expect_refuse") and r.get("answer")]

    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "ground_truth": [r["ground_truth"] for r in results],
        }
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=make_judge(),
        embeddings=make_embeddings(),
        run_config=RunConfig(max_workers=2, timeout=240),
        raise_exceptions=False,
    )

    df: pd.DataFrame = result.to_pandas()
    cols = ["user_input", "question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    cols = [c for c in cols if c in df.columns]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)

    print("\n" + "=" * 100)
    print("逐题得分")
    print("=" * 100)
    print(df[cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("faithfulness 判定理由（RAGAS 认为哪些论断在条款里找不到依据）")
    print("=" * 100)
    qcol = "user_input" if "user_input" in df.columns else "question"
    for i, (_, row) in enumerate(df.iterrows()):
        q = str(results[i]["question"]) if i < len(results) else "?"
        print(f"\n[{row.get('faithfulness')}] {q[:60]}")
        for col in df.columns:
            if col.startswith("_") and "faith" in col.lower():
                print(f"   {col}: {str(row[col])[:800]}")

    out = os.path.join(PROJECT_ROOT, "data", "eval", "per_item.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n逐题明细已写入 {out}")


if __name__ == "__main__":
    main()
