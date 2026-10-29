# -*- coding: utf-8 -*-
"""RAG 问答层：一次问答从"问题"到"答案"要经过的全部环节都在这里。

    faq/               标准问法的文本预处理（分词、业务词表）
    faq_search.py      FAQ 层：BM25 打分 + 三重闸门判定是否命中标准问法
    query_analyzer.py  策略判定 + 查询改写（子查询 / 回溯 / 上下文补全 / 历史版本）
    retriever.py       混合检索（Milvus 稠密 + 稀疏）→ 父块回取 → 时效校验 → 阈值截断
    reranker.py        交叉编码器精排（bge-reranker-large）
    prompts.py         Prompt 模板
    llm_client.py      模型调用（ChatOpenAI，兼容 DeepSeek / 通义 / OpenAI）
    answer.py          生成 + 拒答判定 + 引用校验 + 降级为条款直读

练习要点：一次问答的分层顺序是 缓存 → FAQ → 检索重排 → 生成校验。
前两层不需要模型，第三层才是"RAG"这个词真正指的那部分。
"""
