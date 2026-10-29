# -*- coding: utf-8 -*-
"""切块器：中文制度文本的父子两级切分。

本文件实现中文专用的递归切分器（ChineseRecursiveTextSplitter）。
LangChain 原版 RecursiveCharacterTextSplitter 在底层用 re.split(separator, text)，
分隔符本身会被丢弃，对中文标点是硬伤：

    keep_separator=True  → 句号跑到下一块开头，子块变成"。审贷会由分行…"
    keep_separator=False → 句号整块消失，两块拼起来粘成"审批审贷会由分行…"

子块要单独送进 bge-reranker 打分，以标点开头会被判成语义残缺、打分偏低。
因此覆盖 _split_text，把切分改成"分隔符归前段"的配对式切法：

    re.split("(。)", "AAA。BBB。CCC") → ['AAA', '。', 'BBB', '。', 'CCC']
    zip(_splits[0::2], _splits[1::2])  → 'AAA。'  'BBB。'  + 尾部 'CCC'

这个 override 在 langchain-text-splitters 1.x 上依然成立：
_split_text 的签名 (self, text: str, separators: list[str]) -> list[str] 自 0.2 至今未变。

为什么制度问答不用固定长度切块：
    固定 300 字窗口会把"（一）……（二）……"切到两半，模型只看到半个枚举项，
    就会把"50万元"错配到另一条的限制条件上。制度文本的天然边界就是"条"，
    所以父块严格等于一条，子块只在条内部滑动，保证召回粒度细、
    但送给模型的永远是完整条款。
"""
from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import config
from core.ingest.parser import ParsedDocument


def _split_text_with_regex_from_end(
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    """切开文本，并让分隔符归到前一段末尾。"""
    if separator:
        if keep_separator:
            # 括号让 re.split 保留分隔符，再奇偶配对挂到前一段
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """中文友好的递归切分器：分隔符归前段，句末标点不丢。"""

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            r"\.\s|\!\s|\?\s",
            r"；|;\s",
            r"，|,\s",
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归切分主流程。"""
        final_chunks: List[str] = []

        # 1. 从上往下挑第一个能在文本中找到的分隔符
        separator = separators[-1]
        new_separators: List[str] = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        # 2. 配对式切分，标点跟随前一段
        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # 3. 短段累积合并，超长段用下一级分隔符递归
        _good_splits: List[str] = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)

        # 仅去首尾空白，保留内部换行，父类处理 overlap 时才不会错位
        return [chunk.strip() for chunk in final_chunks if chunk.strip() != ""]


def _make_splitter(size: int, overlap: int) -> ChineseRecursiveTextSplitter:
    return ChineseRecursiveTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=size,
        chunk_overlap=overlap,
    )


def split_parent(content: str, max_chars: int) -> list[str]:
    """长条款按枚举项拆父块，短条款整条作为一个父块。"""
    if len(content) <= max_chars:
        return [content]
    segments = _make_splitter(max_chars, 0).split_text(content)
    return [seg for seg in segments if seg.strip()]


def split_child(parent: str, size: int, overlap: int) -> list[str]:
    """条内递归切分。切点落在句末，重叠区保证跨句限定条件不被截断。"""
    if len(parent) <= size:
        return [parent]
    children = _make_splitter(size, overlap).split_text(parent)
    return [c for c in children if c.strip()]


def split_document(doc: ParsedDocument) -> list[Document]:
    """把整份制度切成条款级 Document，供 VectorStore 直接消费。

    与 build_chunks 的区别：这个返回 LangChain Document，
    metadata 已带上溯源字段，接 Milvus / Retriever 时无需再转换。
    """
    documents: list[Document] = []
    for clause in doc.clauses:
        clause_key = f"{doc.doc_no}#{clause.clause_no}"
        parents = split_parent(clause.content, config.PARENT_MAX_CHARS)

        for p_idx, parent_text in enumerate(parents):
            children = split_child(parent_text, config.CHILD_CHUNK_SIZE, config.CHILD_OVERLAP)
            head = (
                f"{clause.clause_no}　{clause.clause_title}"
                if clause.clause_title
                else clause.clause_no
            )
            parent_content = f"{head}\n{parent_text}"

            for c_idx, child_text in enumerate(children):
                documents.append(
                    Document(
                        page_content=child_text,
                        metadata={
                            "id": f"{clause_key}#p{p_idx}c{c_idx}",
                            "parent_content": parent_content,
                            "doc_no": doc.doc_no,
                            "title": doc.title,
                            "chapter": clause.chapter,
                            "clause_no": clause.clause_no,
                            "clause_title": clause.clause_title,
                            "clause_key": clause_key,
                            "business_line": doc.business_line,
                            "status": "active",
                        },
                    )
                )
    return documents


def build_chunks(doc: ParsedDocument) -> list[dict[str, Any]]:
    """生成入 Milvus 的条款切片。父块与子块共用同一 clause_key。

    输出为 dict 列表，写入层（ingest 脚本与 vector_store）按这些字段落库。
    """
    chunks: list[dict[str, Any]] = []

    for clause in doc.clauses:
        clause_key = f"{doc.doc_no}#{clause.clause_no}"
        parents = split_parent(clause.content, config.PARENT_MAX_CHARS)

        for p_idx, parent_text in enumerate(parents):
            children = split_child(parent_text, config.CHILD_CHUNK_SIZE, config.CHILD_OVERLAP)
            # 父块正文带上条款抬头，模型看到的内容与制度原文一致
            head = f"{clause.clause_no}　{clause.clause_title}" if clause.clause_title else clause.clause_no
            parent_content = f"{head}\n{parent_text}"

            for c_idx, child_text in enumerate(children):
                chunk_id = f"{clause_key}#p{p_idx}c{c_idx}"
                chunks.append(
                    {
                        "id": chunk_id,
                        "parent_content": parent_content,
                        "child_text": child_text,
                        "doc_no": doc.doc_no,
                        "clause_no": clause.clause_no,
                        "business_line": doc.business_line,
                        "status": "active",
                        # 供 MySQL 落库使用
                        "clause_key": clause_key,
                        "title": doc.title,
                        "chapter": clause.chapter,
                        "clause_title": clause.clause_title,
                        "content": parent_text,
                    }
                )

    return chunks


def build_clause_rows(doc: ParsedDocument) -> list[dict[str, Any]]:
    """生成入 MySQL policy_clause 的行。一条一行，不切片。"""
    rows: list[dict[str, Any]] = []

    for clause in doc.clauses:
        rows.append(
            {
                "clause_key": f"{doc.doc_no}#{clause.clause_no}",
                "doc_no": doc.doc_no,
                "title": doc.title,
                "chapter": clause.chapter,
                "clause_no": clause.clause_no,
                "clause_title": clause.clause_title,
                "content": clause.content,
                "business_line": doc.business_line,
                "status": "active",
                "effective_date": doc.effective_date,
            }
        )

    return rows
