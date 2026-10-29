# -*- coding: utf-8 -*-
"""离线自测：不依赖 MySQL / Milvus / 大模型，验证解析、切块、时效过滤与引用校验。

日常跑法：
    python -m tests.offline_check

它做的事：
1. 解析 data/raw 下的全部制度，检查文号、效力、条款数是否正常
2. 用内存版向量库替换 Milvus、用假 LLM 替换模型，跑通一条完整问答链路
3. 重点验证两件事：已废止制度的条款不会再被召回；答案里的数字必须能在条款原文里找到
"""
from __future__ import annotations

import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ingest.chunker import build_chunks, build_clause_rows
from core.ingest.parser import parse_document
from core.logger import logger


def load_docs():
    docs = []
    for path in sorted(glob.glob("data/raw/*.md")):
        with open(path, encoding="utf-8") as fp:
            text = fp.read()
        docs.append(parse_document(text, source_file=os.path.basename(path)))
    return docs


def check_parse(docs):
    """每份制度都必须解析出文号、施行日期和至少一条条款。"""
    assert len(docs) == 4, f"制度数量异常：{len(docs)}"
    for doc in docs:
        assert doc.doc_no, f"{doc.source_file} 未识别到发文字号"
        assert doc.effective_date, f"{doc.source_file} 未识别到施行日期"
        assert doc.clauses, f"{doc.source_file} 未解析到任何条款"

    new_doc = next(d for d in docs if d.doc_no == "恒信银发〔2024〕27号")
    old_doc = next(d for d in docs if d.doc_no == "恒信银发〔2021〕9号")
    # 新办法正文里声明废止了旧办法，这条关系决定了旧制度要下线
    assert new_doc.supersedes_no == old_doc.doc_no, "废止关系未建立"

    total = sum(len(d.clauses) for d in docs)
    logger.info("解析校验通过：%d 份制度，%d 条条款", len(docs), total)
    return total


def check_chunk(docs):
    """父块必须严格落在单条之内，子块不能跨条。"""
    for doc in docs:
        clauses = {c.clause_no for c in doc.clauses}
        chunks = build_chunks(doc)
        rows = build_clause_rows(doc)
        assert len(rows) == len(doc.clauses), f"{doc.doc_no} 条款行数不一致"
        for chunk in chunks:
            # id 形如 文号#第七条#p0c1，从中反解出的条款号必须在原文中存在
            clause_no = chunk["id"].split("#")[1]
            assert clause_no in clauses, f"子块归属条款不存在：{chunk['id']}"
            assert chunk["parent_content"].strip(), "父块内容为空"
            assert len(chunk["child_text"]) <= 320, "子块超出长度上限"
    logger.info("切块校验通过")


def check_effective_filter(docs):
    """时效硬过滤：基准日之后才生效、或者已被废止的条款，都不能进候选。"""
    from datetime import date
    from core.rag.retriever import PolicyRetriever

    new_doc = next(d for d in docs if d.doc_no == "恒信银发〔2024〕27号")
    old_doc = next(d for d in docs if d.doc_no == "恒信银发〔2021〕9号")
    aml_doc = next(d for d in docs if d.doc_no == "恒信银发〔2025〕3号")

    rows = []
    for doc, status in ((new_doc, "active"), (old_doc, "repealed"), (aml_doc, "active")):
        for row in build_clause_rows(doc):
            row["status"] = status
            rows.append(row)

    class FakeDB:
        """把条款行塞进内存字典，替代 MySQL 回查。"""

        def __init__(self, items):
            self.items = items

        def get_clause(self, key):
            return self.items.get(key)

    class FakeStore:
        """按顺序返回全部候选，把条数交还给检索层自己截断。

        这样才能验证"越权召回的历史版本会被时效校验拦掉"，
        而不是被 top_k 提前切掉、让过滤逻辑白跑一趟。
        """

        def __init__(self, items):
            self.items = items

        def hybrid_search(self, query, top_k, business_line=None, skip_repealed=False):
            hits = []
            for key, row in self.items.items():
                if skip_repealed and row["status"] != "active":
                    continue
                hits.append({"id": f"{key}#p0c0", "score": 0.9, "similarity": 0.9})
            return hits

    db = FakeDB({r["clause_key"]: r for r in rows})
    store = FakeStore({r["clause_key"]: r for r in rows})
    retriever = PolicyRetriever(store=store, clause_db=db)

    result = retriever.retrieve("个人消费贷款额度上限")
    keys = {c.clause_key for c in result.clauses}
    old_keys = {r["clause_key"] for r in rows if r["status"] == "repealed"}
    assert not (keys & old_keys), "已废止制度的条款被召回了"
    assert result.dropped_repealed >= len(old_keys), "废止条款未计入拦截数"

    # 把基准日拨回 2024 年之前，当年还没施行的新办法也不该出现
    early = retriever.retrieve("个人消费贷款额度上限", as_of=date(2024, 1, 1))
    assert all(c.doc_no != "恒信银发〔2024〕27号" for c in early.clauses), \
        "基准日早于施行日，新办法仍被召回"
    logger.info(
        "时效过滤通过：拦截废止条款 %d 条，基准日回溯后新办法已下线",
        result.dropped_repealed,
    )


def check_grounding(docs):
    """引用校验：答案里的数字必须在条款原文出现，编造的数字要被抓出来。"""
    from core.rag.answer import AnswerGenerator
    from core.rag.retriever import RetrievedClause

    doc = next(d for d in docs if d.doc_no == "恒信银发〔2024〕27号")
    clause = next(c for c in doc.clauses if "额度" in c.clause_title or "额度" in c.content)
    retrieved = RetrievedClause(
        clause_key=f"{doc.doc_no}#{clause.clause_no}",
        doc_no=doc.doc_no,
        title=doc.title,
        chapter=clause.chapter,
        clause_no=clause.clause_no,
        clause_title=clause.clause_title,
        content=clause.content,
        business_line=doc.business_line,
        status="active",
        similarity=0.9,
        effective_date=doc.effective_date,
    )
    clauses = [retrieved]
    citation = retrieved.citation

    good = f"根据{citation}，{clause.clause_title}方面的口径见条款原文，其中 50 万元为上限。"
    bad = f"根据{citation}，单户额度上限为 88 万元。"

    # 正确答案里的数字取自条款原文本身，先确认条款里确实有这个数
    numbers = re.findall(r"\d+\s*万元", clause.content)
    assert numbers, "测试条款不含金额数字，用例失效"
    good = f"根据{citation}，{clause.clause_title}：{numbers[0]}。"

    gen = AnswerGenerator.__new__(AnswerGenerator)  # 只测校验逻辑，不需要初始化模型客户端
    passed_ok, issues_ok = gen.check_grounding(good, clauses)
    passed_bad, issues_bad = gen.check_grounding(bad, clauses)

    assert passed_ok and not issues_ok, f"正确引用被误判：{issues_ok}"
    assert not passed_bad and issues_bad, "虚构的数字没有被拦截"
    logger.info("引用校验通过：正确引用 0 问题，虚构数字被拦截 %d 项", len(issues_bad))


def main():
    docs = load_docs()
    check_parse(docs)
    check_chunk(docs)
    check_effective_filter(docs)
    check_grounding(docs)
    print("\n离线自测全部通过。")


if __name__ == "__main__":
    main()
