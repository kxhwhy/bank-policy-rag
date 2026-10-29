# -*- coding: utf-8 -*-
"""第二段：RAGAS 四指标评估。

运行环境必须是 .venv-ragas（ragas 0.2 要求 langchain 0.3 系列，
与主链路的 1.x 不兼容），所以这个文件不 import 项目里任何 langchain 相关模块，
只读 runner.py 落盘的 JSON：

    主环境：python -m core.assessment.runner
    venv  ：.venv-ragas/Scripts/python.exe -m core.assessment.rag_as

四个指标各自回答什么问题：
    faithfulness      回答里的每个论断，能不能在给它的条款原文里找到依据
                      —— 衡量"胡编"，RAG 系统的第一道底线
    answer_relevancy  回答有没有答到问题上
                      —— 衡量"跑题"，答得很规范但答非所问会在这里掉分
    context_precision 召回的条款里，有多少是真用得上 的
                      —— 衡量"白召回"，排在前面的是噪声就扣分
    context_recall    标准答案涉及的信息，有多少被召回覆盖
                      —— 衡量"漏召回"，这是制度问答最致命的失分项

指标不是越高越好：拒答题的 faithfulness 天然偏低，因为答案本身
就该说"制度里没有"。所以报告按可答题 / 拒答题分组给分，不混算。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# 只把项目根塞进路径，不 import core 包里的任何模块——
# core 下的代码会连锁 import langchain 1.x，与 venv 里的 0.3 冲突。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_IN = os.path.join(PROJECT_ROOT, "data", "eval", "raw_results.json")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "data", "eval", "ragas_report.json")


def load_results(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def build_dataset(items: list[dict]):
    """装成 RAGAS 要求的四列结构。"""
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "question": [r["question"] for r in items],
            "answer": [r["answer"] for r in items],
            "contexts": [r["contexts"] for r in items],
            "ground_truth": [r["ground_truth"] for r in items],
        }
    )


def make_judge():
    """评估模型。不直接用 ChatOpenAI 是为了显式指定 base_url。

    评估用的模型可以和生成用的不同——用一个更强的模型当裁判，
    才不会被生成模型自己的口径带着走。
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    judge = ChatOpenAI(
        model=os.getenv("POLICY_EVAL_MODEL", "deepseek-chat"),
        api_key=os.environ["POLICY_LLM_API_KEY"],
        base_url=os.getenv("POLICY_LLM_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.0,
    )
    return LangchainLLMWrapper(judge)


def require_api_key() -> None:
    """在调用 ragas 之前就拦住缺 key 的情况。

    不能把这项检查放在 make_judge 里：ragas 的 evaluate 会捕获指标内部抛出的
    异常并降级成警告，SystemExit 会被吞掉，最后打出一堆没有意义的分数。
    """
    if os.getenv("POLICY_LLM_API_KEY", "").strip():
        return
    print(
        "缺少 POLICY_LLM_API_KEY。RAGAS 要调模型当裁判，没有 key 无法打分。\n"
        "Windows：  set POLICY_LLM_API_KEY=sk-xxxx\n"
        "PowerShell：$env:POLICY_LLM_API_KEY=\"sk-xxxx\"",
        file=sys.stderr,
    )
    sys.exit(2)


def make_embeddings():
    """评估用的 embedding：直接复用项目里已下载的 BGE-M3。

    为什么不用 OpenAIEmbeddings：
        answer_relevancy 需要算语义相似度，缺 embedding 时 ragas 会自己
        去建 OpenAIEmbeddings，结果拿不到 key 直接崩。而且评估用的 embedding
        应当和检索用的保持一致，否则"相关性"衡量的就不是同一套空间。
        BGE-M3 已经在本机 models/ 下，不用联网下载。
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    model_path = os.getenv("POLICY_EVAL_EMBEDDING", os.path.join(PROJECT_ROOT, "models", "bge-m3"))
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )


def strip_citations(text: str) -> str:
    """削掉句末的引用括号。

    制度问答要求每条结论挂《制度全称》（文号）第X条，这串编号不在条款
    正文里，RAGAS 逐句判断依据时会把它当成独立断言判为无据，
    系统性压低 faithfulness。评估时两套口径都算，差值写进报告。
    """
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


def evaluate_group(items: list[dict], label: str) -> dict:
    """跑一组题的四个指标。"""
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    if not items:
        return {}

    print(f"\n正在评估：{label}（{len(items)} 题）")
    dataset = build_dataset(items)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=make_judge(),
        embeddings=make_embeddings(),
        # 逐条跑会慢到不可用，适度并发；失败的条目会被跳过而不是中断整批
        run_config=RunConfig(max_workers=4, timeout=180),
        raise_exceptions=False,
    )

    df = result.to_pandas()
    scores = {k: round(float(v), 4) for k, v in df.mean(numeric_only=True).items()}
    print(f"  {label}: {scores}")

    # 再跑一遍去引用的 faithfulness，把"引用格式污染"显式量化
    if label == "可答题" and items:
        cleaned = [{**r, "answer": strip_citations(r["answer"])} for r in items]
        clean_result = evaluate(
            dataset=build_dataset(cleaned),
            metrics=[faithfulness],
            llm=make_judge(),
            run_config=RunConfig(max_workers=4, timeout=180),
            raise_exceptions=False,
        )
        clean_score = round(float(clean_result.to_pandas()["faithfulness"].mean()), 4)
        scores["faithfulness_no_citation"] = clean_score
        scores["citation_bias"] = round(scores.get("faithfulness", 0.0) - clean_score, 4)
        print(f"  {label}（去引用括号）faithfulness = {clean_score}"
              f"（引用格式带来的偏差 {scores['citation_bias']:+.4f}）")

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 评估")
    parser.add_argument("--in", dest="inp", default=DEFAULT_IN, help="runner 产出的结果文件")
    parser.add_argument("--out", default=DEFAULT_OUT, help="报告落盘路径")
    args = parser.parse_args()

    results = load_results(args.inp)
    answerable = [r for r in results if not r.get("expect_refuse") and r.get("answer")]
    refusable = [r for r in results if r.get("expect_refuse")]

    degraded = sum(1 for r in results if "条款原文" in (r.get("answer") or ""))
    if degraded:
        print(
            f"警告：{degraded}/{len(results)} 条回答是降级产物（模型不可用时的条款直读）。\n"
            "      这些回答不是模型生成的，faithfulness 会虚高、answer_relevancy 会虚低。\n"
            "      要拿可用数字，先配好 POLICY_LLM_API_KEY 重跑 runner。\n"
        )

    require_api_key()

    report = {
        "answerable": evaluate_group(answerable, "可答题"),
        # 拒答题不打 RAGAS 分：它们的答案是"制度里没有"，和 ground_truth
        # 的语义相似度天然接近 0，跑出来全是 0 只会让报告看起来像系统坏了。
        # 拒答题要衡量的是"该拒的时候有没有拒"，这是精确率问题，不是生成质量问题。
        "refusable": {
            "total": len(refusable),
            "refused_correctly": sum(1 for r in refusable if r.get("refused")),
            "refusal_accuracy": round(
                sum(1 for r in refusable if r.get("refused")) / len(refusable), 4
            ) if refusable else 0.0,
        },
        "meta": {
            "total": len(results),
            "answerable": len(answerable),
            "refusable": len(refusable),
            "degraded": degraded,
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    print(f"\n报告已写入 {args.out}")
    if degraded:
        print("注意：以上分数含降级回答，只能证明链路通，不能当成绩写进简历。")


if __name__ == "__main__":
    main()
