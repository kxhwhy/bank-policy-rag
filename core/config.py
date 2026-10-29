# -*- coding: utf-8 -*-
"""恒信银行制度知识库 - 全局配置

配置读取顺序：环境变量 > config.ini > 代码默认值。
生产部署时密钥一律从环境变量注入，不写入版本库。
"""
import configparser
import os

# 工程根目录（core/ 的上级）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.ini")

_parser = configparser.ConfigParser()
_parser.read(CONFIG_PATH, encoding="utf-8")


def _get(section: str, key: str, env: str, default: str) -> str:
    """优先取环境变量，其次 config.ini，最后默认值。"""
    value = os.environ.get(env)
    if value:
        return value
    return _parser.get(section, key, fallback=default)


def _get_int(section: str, key: str, env: str, default: int) -> int:
    raw = _get(section, key, env, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _get_list(section: str, key: str, env: str, default: list[str]) -> list[str]:
    raw = _get(section, key, env, ",".join(default))
    return [item.strip() for item in raw.split(",") if item.strip()]


class Config:
    """配置聚合类，实例化即读取当前环境。"""

    # ---------- 基础路径 ----------
    BASE_DIR = BASE_DIR
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    LOG_FILE = os.path.join(BASE_DIR, "logs", "policy_rag.log")

    # ---------- MySQL：制度条款结构化库 ----------
    MYSQL_HOST = _get("mysql", "host", "POLICY_MYSQL_HOST", "127.0.0.1")
    MYSQL_PORT = _get_int("mysql", "port", "POLICY_MYSQL_PORT", 3306)
    MYSQL_USER = _get("mysql", "user", "POLICY_MYSQL_USER", "root")
    MYSQL_PASSWORD = _get("mysql", "password", "POLICY_MYSQL_PASSWORD", "")
    MYSQL_DATABASE = _get("mysql", "database", "POLICY_MYSQL_DATABASE", "bank_policy")

    # ---------- Redis：三层缓存 ----------
    REDIS_HOST = _get("redis", "host", "POLICY_REDIS_HOST", "127.0.0.1")
    REDIS_PORT = _get_int("redis", "port", "POLICY_REDIS_PORT", 6379)
    REDIS_PASSWORD = _get("redis", "password", "POLICY_REDIS_PASSWORD", "")
    REDIS_DB = _get_int("redis", "db", "POLICY_REDIS_DB", 0)
    CACHE_TTL_SECONDS = _get_int("redis", "cache_ttl", "POLICY_CACHE_TTL", 86400)
    # 问答结果缓存（第一层 answer:{query}）是否启用。关闭后每次问答都走完整链路
    ENABLE_ANSWER_CACHE = _get("redis", "enable_answer_cache", "POLICY_ENABLE_ANSWER_CACHE", "true").lower() == "true"
    # FAQ 索引缓存（第二三层 问法列表 / 分词结果）是否启用。
    # 关掉后每次进程启动都会重读 MySQL 并重新分词，FAQ 库很大时启动会变慢
    ENABLE_FAQ_INDEX_CACHE = _get(
        "redis", "enable_faq_index_cache", "POLICY_ENABLE_FAQ_INDEX_CACHE", "true"
    ).lower() == "true"

    # ---------- Milvus：条款向量库 ----------
    MILVUS_URI = _get("milvus", "uri", "POLICY_MILVUS_URI", "http://127.0.0.1:19530")
    MILVUS_DB = _get("milvus", "database", "POLICY_MILVUS_DB", "bank_policy")
    MILVUS_COLLECTION = _get("milvus", "collection", "POLICY_MILVUS_COLLECTION", "policy_clause")

    # ---------- 向量模型 ----------
    EMBEDDING_MODEL = _get("embedding", "model_name", "POLICY_EMBEDDING_MODEL", "BAAI/bge-m3")
    # 模型不存在时是否允许回退到哈希向量（仅用于离线自测，生产必须为 False）
    ALLOW_HASH_FALLBACK = _get("embedding", "allow_hash_fallback", "POLICY_ALLOW_HASH_FALLBACK", "true").lower() == "true"

    # ---------- LLM ----------
    LLM_MODEL = _get("llm", "model", "POLICY_LLM_MODEL", "deepseek-chat")
    LLM_API_KEY = _get("llm", "api_key", "POLICY_LLM_API_KEY", "")
    LLM_BASE_URL = _get("llm", "base_url", "POLICY_LLM_BASE_URL", "https://api.deepseek.com/v1")
    LLM_TEMPERATURE = float(_get("llm", "temperature", "POLICY_LLM_TEMPERATURE", "0.1"))

    # ---------- 切块参数 ----------
    # 父块 = 一个完整"条"（第X条），保证送给模型的是完整条款而不是断句
    PARENT_MAX_CHARS = _get_int("chunk", "parent_max_chars", "POLICY_PARENT_MAX_CHARS", 1200)
    CHILD_CHUNK_SIZE = _get_int("chunk", "child_chunk_size", "POLICY_CHILD_CHUNK_SIZE", 300)
    CHILD_OVERLAP = _get_int("chunk", "child_overlap", "POLICY_CHILD_OVERLAP", 60)

    # ---------- 检索参数 ----------
    RETRIEVAL_K = _get_int("retrieval", "top_k", "POLICY_RETRIEVAL_K", 8)
    FINAL_TOP_M = _get_int("retrieval", "final_top_m", "POLICY_FINAL_TOP_M", 3)
    DENSE_WEIGHT = float(_get("retrieval", "dense_weight", "POLICY_DENSE_WEIGHT", "0.7"))
    SPARSE_WEIGHT = float(_get("retrieval", "sparse_weight", "POLICY_SPARSE_WEIGHT", "0.3"))
    # 低于该相似度的召回一律丢弃，宁缺毋滥
    MIN_SIMILARITY = float(_get("retrieval", "min_similarity", "POLICY_MIN_SIMILARITY", "0.35"))
    # 最高相似度低于该值视为"条款不足以支撑作答"，模型硬凑时直接拒答而非退回原文
    LOW_RELEVANCE_SIMILARITY = float(
        _get("retrieval", "low_relevance_similarity", "POLICY_LOW_RELEVANCE_SIMILARITY", "0.55")
    )
    # 是否启用交叉编码器重排：召回后按"问题-条款是否真的能回答"重新打分
    ENABLE_RERANK = _get("retrieval", "enable_rerank", "POLICY_ENABLE_RERANK", "true").lower() == "true"
    RERANK_MODEL_PATH = _get(
        "retrieval", "rerank_model_path", "POLICY_RERANK_MODEL_PATH", "models/bge-reranker-large"
    )
    # 重排后保留的条款数
    RERANK_TOP_M = _get_int("retrieval", "rerank_top_m", "POLICY_RERANK_TOP_M", 3)
    # 启用查询改写（HYDE / 子查询 / 回溯）
    ENABLE_QUERY_REWRITE = _get("retrieval", "enable_query_rewrite", "POLICY_ENABLE_QUERY_REWRITE", "true").lower() == "true"

    # ---------- FAQ 层 ----------
    # 是否启用标准问答精确命中。关闭后所有问题都走完整检索链路
    ENABLE_FAQ = _get("faq", "enabled", "POLICY_ENABLE_FAQ", "true").lower() == "true"
    # 判据一：BM25 原始分 softmax 归一化后的"独占份额"阈值，用于逐字/近逐字命中。
    # 这不是余弦相似度，取值含义见 core/rag/faq_search.py 的 search 注释。
    FAQ_HIT_THRESHOLD = float(_get("faq", "hit_threshold", "POLICY_FAQ_HIT_THRESHOLD", "0.85"))
    # 判据二：BM25 原始分除以理论满分的"绝对相似度"阈值，用于同义改写命中。
    # 为什么要两条判据：多条候选一起激活时 softmax 份额会被摊薄，
    # 只靠判据一漏掉常见的口语化改写；只靠判据二又因 BM25 没有上界而换库失效。
    FAQ_MIN_SIMILARITY = float(_get("faq", "min_similarity", "POLICY_FAQ_MIN_SIMILARITY", "0.40"))
    # 关于这一层的已知边界：BM25 是字面匹配，不理解"能批"≈"额度"这类同义替换，
    # 所以口语化改写会走完整检索链路而不是 FAQ。这是有意的取舍——FAQ 命中后
    # 直接返回审定答案、绕过生成与校验，宁可漏命中退回检索，也不能猜。
    # 想提高命中率靠扩充标准问法的别称，而不是放松阈值。
    # 判据三：查询实义词被候选问法覆盖的最低比例。
    # 单靠分数比例会在查询很短时失效（去噪后只剩"银行"两个字也能撞上
    # 「手机银行转账限额」），要求覆盖度才能真正挡住不相关的问题。
    FAQ_MIN_COVERAGE = float(_get("faq", "min_coverage", "POLICY_FAQ_MIN_COVERAGE", "0.6"))
    # 审定答案的复核周期。超过该天数的 FAQ 命中时降级为提示，不再直接返回。
    # 制度会换版，审定答案过期后措辞确定反而比模型现生成更危险。
    FAQ_VERIFY_TTL_DAYS = _get_int("faq", "verify_ttl_days", "POLICY_FAQ_VERIFY_TTL_DAYS", 90)

    # ---------- 业务维度 ----------
    # 制度效力状态：现行有效 / 已废止 / 尚未生效
    POLICY_STATUS_ACTIVE = "active"
    POLICY_STATUS_REPEALED = "repealed"
    POLICY_STATUS_PENDING = "pending"

    # 可选的业务条线，用于前端筛选（对应制度的所属条线字段）
    BUSINESS_LINES = _get_list(
        "app", "business_lines", "POLICY_BUSINESS_LINES",
        ["个人信贷", "反洗钱", "运营柜面"],
    )

    # 拒答口径：检索不到依据时统一话术，避免模型自由发挥
    REFUSAL_TEMPLATE = _get(
        "app", "refusal_template", "POLICY_REFUSAL_TEMPLATE",
        "现行有效的制度中未检索到与您问题直接对应的条款，无法给出确定答复。建议联系条线管理部门确认，或查阅相关制度原文。",
    )


config = Config()
