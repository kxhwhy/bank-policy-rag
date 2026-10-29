# -*- coding: utf-8 -*-
"""命令行问答入口。

    python main.py                          # 交互式
    python main.py -q "个人消费贷款额度上限"   # 单次问答
    python main.py -q "..." -l 个人信贷 --show-clauses -v

加 -v 会打印命中条款的相似度、章节、原文，用来核对答案到底引了哪一条。
这是自测和调阈值时最常用的开关。
"""
from __future__ import annotations

import argparse
import json
import sys

from core.config import config
from core.generate.llm_client import llm
from core.logger import logger
from core.qa_system import qa_system


def print_result(result: dict, show_clauses: bool, verbose: bool) -> None:
    if result.get("error"):
        print(f"[错误] {result['error']}")
        return

    print()
    print("=" * 72)
    print(f"问题：{result['question']}")
    print(f"策略：{result['strategy']}    基准日：{result['as_of']}"
          f"{'    [缓存命中]' if result.get('from_cache') else ''}")
    if verbose and result.get("queries") and result["queries"] != [result["question"]]:
        print("检索式：")
        for q in result["queries"]:
            print(f"  - {q}")
    print("-" * 72)

    if result.get("refused"):
        print(f"[拒答] {result['answer']}")
    else:
        print(result["answer"])

    if show_clauses and result.get("clauses"):
        print("-" * 72)
        print("依据条款：")
        for idx, clause in enumerate(result["clauses"], start=1):
            sim = f"{clause['similarity']:.3f}"
            print(f"  [{idx}] {clause['citation']}　{clause['clause_title']}　(相似度 {sim})")
            if verbose:
                print(f"      {clause['chapter']}")
                body = clause["content"].replace("\n", "\n      ")
                print(f"      {body}")
                print()

    if verbose:
        print("-" * 72)
        print(f"引用校验：{'通过' if result.get('grounded') else '未通过'}")

    print("=" * 72)


def interactive(business_line: str | None, show_clauses: bool, verbose: bool) -> None:
    overview = qa_system.overview()
    stats = overview["stats"]
    print()
    print("恒信银行制度智能问答系统")
    print(f"知识库：{stats['document_total']} 份制度 / {stats['clause_total']} 条条款"
          f"（现行 {stats['document_active']} 份 / {stats['clause_active']} 条）")
    print(f"可选条线：{'、'.join(overview['business_lines']) or '全部'}")
    if business_line:
        print(f"当前条线：{business_line}")
    print(f"模型：{config.LLM_MODEL if overview['llm_available'] else '未配置（仅返回条款原文）'}")
    print("输入 exit 退出，输入 /reset 清空会话上下文")
    print()

    session_id = None
    while True:
        try:
            question = input("问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break
        if question == "/reset":
            if session_id:
                qa_system.reset_session(session_id)
            session_id = None
            print("会话上下文已清空")
            continue

        result = qa_system.ask(question, business_line=business_line, session_id=session_id)
        session_id = result.get("session_id", session_id)
        print_result(result, show_clauses=show_clauses, verbose=verbose)


def main() -> None:
    parser = argparse.ArgumentParser(description="恒信银行制度智能问答")
    parser.add_argument("-q", "--question", help="单次提问，不进入交互模式")
    parser.add_argument("-l", "--line", dest="business_line", help="限定业务条线")
    parser.add_argument("--show-clauses", action="store_true", help="显示命中的依据条款")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示检索细节与条款原文")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    args = parser.parse_args()

    if args.business_line and args.business_line not in config.BUSINESS_LINES:
        print(f"[错误] 未知业务条线：{args.business_line}，可选：{'、'.join(config.BUSINESS_LINES)}")
        sys.exit(1)

    if args.question:
        result = qa_system.ask(
            args.question,
            business_line=args.business_line,
            use_cache=not args.verbose,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_result(result, show_clauses=args.show_clauses or True, verbose=args.verbose)
    else:
        interactive(args.business_line, show_clauses=args.show_clauses, verbose=args.verbose)


if __name__ == "__main__":
    main()
