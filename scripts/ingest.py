# -*- coding: utf-8 -*-
"""入库脚本：把 data/raw 下的制度原文导入结构化库与向量库。

用法：
    python -m scripts.ingest                  # 全量重建
    python -m scripts.ingest --keep-vectors   # 只重建 MySQL，不动向量库
    python -m scripts.ingest --drop           # 先删集合再建
    python -m scripts.ingest --check          # 只做解析预检，不写任何库

支持的文件格式：.md / .txt / .pdf / .docx。四种格式统一由
core.ingest.file_reader 抽出纯文本，再走同一条解析链路，结果一致。

执行顺序刻意安排为先写 MySQL 再写 Milvus：
向量库只负责召回，最终引用以结构化库为准，先把"真相"落盘。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date

# 允许以脚本方式直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import config
from core.ingest.chunker import build_chunks, build_clause_rows
from core.ingest.file_reader import SUPPORTED_SUFFIXES, read_text
from core.ingest.parser import ParsedDocument, parse_document
from core.logger import logger
from core.store.policy_db import db
from core.store.vector_store import vector_store


def load_files() -> list[tuple[str, str]]:
    """读取 data/raw 下所有支持格式的制度文件，统一返回纯文本。"""
    if not os.path.isdir(config.RAW_DIR):
        raise FileNotFoundError(f"制度目录不存在：{config.RAW_DIR}")

    files: list[tuple[str, str]] = []
    skipped: list[str] = []
    for name in sorted(os.listdir(config.RAW_DIR)):
        path = os.path.join(config.RAW_DIR, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith(tuple(SUPPORTED_SUFFIXES)):
            skipped.append(name)
            continue
        files.append((name, read_text(path)))
    if skipped:
        # 明确告知被忽略的文件，避免"放进去了却没入库"这种静默失败
        logger.warning("跳过不支持的文件：%s", "、".join(skipped))
    if not files:
        raise FileNotFoundError(
            f"{config.RAW_DIR} 下没有找到制度文件（支持 {'/'.join(sorted(SUPPORTED_SUFFIXES))}）"
        )
    return files


def check_only() -> None:
    """解析预检：只报告每份文件的格式、条款数与元数据抽取情况，不写库。"""
    logger.info("=" * 60)
    logger.info("解析预检（不写任何库）")

    failed = 0
    for name, text in load_files():
        suffix = os.path.splitext(name)[1].lower()
        try:
            doc = parse_document(text, source_file=name)
        except Exception as exc:  # 预检要把问题全部列出来，不能第一个就中断
            failed += 1
            logger.error("✗ %s（%s）解析失败：%s", name, suffix, exc)
            continue

        warn = []
        if not doc.clauses:
            warn.append("无条款")
        if doc.doc_no.startswith("恒信银发〔0000〕"):
            warn.append("文号靠占位")
        if not doc.effective_date:
            warn.append("无施行日期")
        chunks = build_chunks(doc)
        logger.info(
            "✓ %-30s %-6s %2d条 / %d切片  %s  施行 %s  %s",
            name,
            suffix,
            len(doc.clauses),
            len(chunks),
            doc.doc_no,
            doc.effective_date or "未识别",
            ("⚠ " + "、".join(warn)) if warn else "",
        )

    if failed:
        logger.error("预检完成：%d 份文件解析失败", failed)
        sys.exit(1)
    logger.info("预检完成：全部文件解析正常")


def ingest(drop_vectors: bool = False, keep_vectors: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("开始导入制度文件")

    # ---------- 1. 结构化库 ----------
    db.init_schema()

    docs: list[ParsedDocument] = []
    for name, text in load_files():
        doc = parse_document(text, source_file=name)
        db.upsert_document(doc.to_document_row())
        rows = build_clause_rows(doc)
        db.replace_clauses(doc.doc_no, rows)
        docs.append(doc)

    # ---------- 2. 处理废止关系 ----------
    # 新制度的正文里声明了"原《XX》同时废止"，据此把旧制度标记失效。
    for doc in docs:
        if doc.supersedes_no:
            db.mark_repealed(doc.supersedes_no, doc.effective_date or date.today())
            logger.info("依据《%s》废止旧制度 %s", doc.title, doc.supersedes_no)

    # ---------- 3. 向量库 ----------
    if not keep_vectors:
        vector_store.ensure_collection(drop=drop_vectors)
        all_chunks = []
        for doc in docs:
            all_chunks.extend(build_chunks(doc))
        written = vector_store.upsert_clauses(all_chunks)
        logger.info("向量库写入完成，共 %d 个切片", written)

    # ---------- 4. 缓存失效 ----------
    # 只清第一层问答结果缓存。第二三层是 FAQ 索引，和制度条款版本无关，
    # 重灌制度不该顺手把 FAQ 分词缓存也清掉——那会让下次启动重新分词。
    from core.store.qa_cache import cache

    cleared = cache.invalidate()
    if cleared:
        logger.info("已清理问答结果缓存 %d 条", cleared)

    stats = db.stats()
    logger.info(
        "导入完成：制度 %d 份（现行 %d），条款 %d 条（现行 %d）",
        stats["document_total"],
        stats["document_active"],
        stats["clause_total"],
        stats["clause_active"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="制度知识库导入")
    parser.add_argument("--drop", action="store_true", help="删除已有向量集合并重建")
    parser.add_argument("--keep-vectors", action="store_true", help="只重建结构化库，跳过向量库")
    parser.add_argument("--check", action="store_true", help="只做解析预检，不写任何库")
    args = parser.parse_args()

    if args.check:
        check_only()
        return

    ingest(drop_vectors=args.drop, keep_vectors=args.keep_vectors)


if __name__ == "__main__":
    main()
