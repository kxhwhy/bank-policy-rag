"""FAQ 索引与查询共用的分词。

BM25 是词袋模型，查询和索引必须用完全一致的切分方式，
否则同一个词在两边切法不同就永远匹配不上。所以这里只暴露一个函数，
建库和查询都调它。
"""

import re

import jieba

from core.logger import logger

# 中文标点与常见疑问助词。BM25 的 IDF 会给这些词很低的权重，
# 但它们会稀释问题的有效词项，建库时先剔掉能让打分更稳。
_NOISE_CHARS = "，。？！、；：“”‘’（）《》【】,.?!;:\"'()<>[] \t\n\r"
_NOISE_WORDS = {
    "吗", "呢", "吧", "啊", "呀", "嘛", "的", "了", "是", "有",
    "请问", "我想问", "想问", "问一下", "一下", "多少", "什么", "怎么", "如何",
    # 下面这些是问句里的功能词，不含业务信息。留着会让两个毫不相关的问题
    # 因为共用"在哪里/可以/能"而算出一个不低的 BM25 分，造成误命中。
    "在哪", "在哪里", "哪里", "哪个", "哪些", "哪种",
    "可以", "能不能", "能否", "需要", "要不要", "是否",
    "我们", "我", "你", "他", "她", "它", "这", "那", "该",
}

_tokenized_cache: dict[str, tuple[str, ...]] = {}
_CACHE_LIMIT = 4096

# 业务自定义词典。jieba 的通用词典不认这些银行条线的固定搭配，
# 不登记的话"个人消费贷能贷多少"会被切成 个人消费 / 贷能 / 贷，
# "贷能"根本不存在于任何标准问法里，白白占掉一个词项并稀释有效信号。
# 单个字"贷"也登记成词，避免它跟后面的字粘成 "贷能""贷多" 这类假词。
_DOMAIN_WORDS = [
    "个人消费贷款", "个人经营贷款", "消费贷款", "经营贷款", "信用贷款",
    "抵押贷款", "信用类", "抵押类", "贷款", "借款", "贷",
    "额度", "授信", "审批权限", "审批", "还款", "逾期", "催收",
    "利率", "期限", "抵押率", "评估价值", "债务收入比", "流水",
    "身份证明", "收入证明", "面谈面签", "电子签名",
    "反洗钱", "受益所有人", "尽职调查", "风险等级", "可疑交易",
    "现金存取", "大额交易", "转账限额", "日累计", "单笔",
    "挂失", "补办", "开户", "销户", "账户", "身份证件", "监护人",
    # 组合词单独登记，避免"银行"这种泛词把几乎所有问句都连起来。
    # 登记后「手机银行怎么开通」和「手机银行转账限额」会因为共享"手机银行"
    # 而覆盖度满分，但前者带"开通"这个业务词、后者没有，冲突闸门能拦住。
    "手机银行", "网上银行", "柜面转账", "开通", "销户", "解冻",
]
_DICT_LOADED = False


def _ensure_dict() -> None:
    """把业务词一次性灌进 jieba 用户词典。"""
    global _DICT_LOADED
    if _DICT_LOADED:
        return
    for word in _DOMAIN_WORDS:
        # freq 给高一点，压过通用词典里的切分偏好；tag 留空。
        jieba.add_word(word, freq=100000)
    _DICT_LOADED = True


def _cut(text: str) -> list[str]:
    """精确模式切词，不用 HMM。

    为什么关掉 HMM：HMM 负责"发现词典里没有的新词"，在业务问句上它净帮倒忙——
    "个人消费贷能贷多少" 会被它拼出 "贷能" 这种不存在的词，吃掉一个词项。
    我们已经有 _DOMAIN_WORDS 兜住业务词，不需要它再猜。
    """
    _ensure_dict()
    return [tok.strip() for tok in jieba.cut(text, cut_all=False, HMM=False) if tok.strip()]


def normalize(text: str) -> str:
    """只做字符级清洗，不改变词序。

    注意不要在这里去掉“多少/什么”这类疑问词——建库时剔掉、查询时不剔，
    会造成两边词项不对称。疑问词统一在 tokenize 里按噪声词表处理。
    """
    if not text:
        return ""
    cleaned = text.strip().lower()
    for ch in _NOISE_CHARS:
        cleaned = cleaned.replace(ch, " ")
    return " ".join(cleaned.split())


def domain_terms() -> set[str]:
    """业务词集合。FAQ 命中判定用它区分"问的是同一件事"和"只是共用了一个词"。"""
    _ensure_dict()
    return set(_DOMAIN_WORDS)


def tokenize(text: str) -> list[str]:
    """切词并去噪。

    结果做了缓存：FAQ 库每次启动都要对所有标准问法分词，
    标准问法数量有限但会被反复用到，缓存能省掉重复的 jieba 计算。

    三层过滤，缺一层都会造成误命中：
    1) 标点与空白（normalize 处理）
    2) 噪声词：疑问词、功能词（_NOISE_WORDS）
    3) 单字词 —— "在""打""下"这种单字在任何问句里都能撞上。实测
       「制度文件在哪里下载」与「催收电话可以在晚上打吗」只凭一个"在"
       就能越过阈值命中，这是最危险的一类错误：答非所问且用户无从察觉。
    业务上真正有意义的单字（贷、卡、户）已在 _DOMAIN_WORDS 登记，
    会作为双字及以上词被切出，不受这层影响。
    """
    if not text:
        return []

    cached = _tokenized_cache.get(text)
    if cached is not None:
        return list(cached)

    normalized = normalize(text)
    if not normalized:
        return []

    tokens = [tok for tok in _cut(normalized) if tok not in _NOISE_WORDS and len(tok) > 1]

    # 全被过滤掉时退回未过滤的切分，避免空词袋让 BM25 直接失效
    if not tokens:
        tokens = _cut(normalized)

    if len(_tokenized_cache) >= _CACHE_LIMIT:
        _tokenized_cache.clear()
    _tokenized_cache[text] = tuple(tokens)
    return tokens
