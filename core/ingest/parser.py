# -*- coding: utf-8 -*-
"""制度文件解析：把一份制度原文拆成带完整元数据的条款结构。

分层设计：
    PolicyFileLoader (langchain BaseLoader)
        负责"文件 → Document"，只做读取与元数据标注；
    parse_document(text)
        负责"文本 → ParsedDocument"，用规则解析出章节层级与效力关系。

两层分开的理由：LangChain 的 Loader 体系只解决"怎么把文件搬进内存并包成
统一的 Document"，它不解决"怎么识别发文字号和废止声明"。后者是本项目的
核心资产，必须自己控制，不能被框架的 loader 抽象挡住。

银行制度文本有三个必须靠规则解析、不能靠 LLM 猜的地方：

1. **发文字号**：恒信银发〔2024〕27号 —— 这是制度唯一标识，也是废止关系的锚点。
   注意〔〕是全角六角括号（U+3014/U+3015），很多制度的 PDF 转换后会变成
   〔〕"[""]" 混用，必须统一。

2. **层级**：章 → 条。一章含若干条，"条"是最小完整语义单元。
   检索时按"条"召回，绝不跨条拼接，否则引用会串号。

3. **效力**：一条制度可能在正文里声明"原《XX办法》（文号）同时废止"，
   这条声明决定了旧制度是否还能被检索到。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from core.logger import logger
from core.ingest import file_reader

# ---------- 正则 ----------
# 发文字号：机关代字 + 〔年份〕 + 序号 + 号
# 机关代字只取紧邻〔 之前的那一段。PDF 转文本时机构全称常和代字连成一行
# （"恒信银行股份有限公司恒信银发〔2024〕27号"），因此先匹配一段，
# 再用 _clean_issuer_code() 把机构通用词从前面剥掉。
DOC_NO_PATTERN = re.compile(
    r"([\u4e00-\u9fa5]{1,20})\s*[〔\[\(]\s*(\d{4})\s*[〕\]\)]\s*(\d{1,4})\s*号"
)

# 公文代字前面的机构通用词，按长度从长到短剥
_ISSUER_GENERIC = (
    "股份有限公司", "有限责任公司", "集团有限公司",
    "股份公司", "有限公司", "集团",
    "银行", "公司",
)


def _clean_issuer_code(raw: str) -> str:
    """把"银行股份有限公司恒信银发"这类粘连串裁成"恒信银发"。

    公文代字一般是"机构简称 + 文种简称"，长度在 2~8 字。这里倒着找：
    从右往左看，只要左边还留着机构通用词（银行/公司/股份…），就剥掉它
    以及它左边的全部内容——因为真正的代字紧贴〔 左侧。
    """
    code = raw.strip()
    cut = 0
    for word in _ISSUER_GENERIC:
        idx = code.rfind(word)
        if idx >= 0:
            cut = max(cut, idx + len(word))
    if cut:
        code = code[cut:]
    # 兜底：代字不会超过 8 字，超长就从右截断
    return code[-8:] if len(code) > 8 else code
# 章标题
CHAPTER_PATTERN = re.compile(r"^#{0,4}\s*第([一二三四五六七八九十百零]+)章\s*(.+)$")
# 条标题：第X条 可选小标题
CLAUSE_PATTERN = re.compile(r"^#{0,4}\s*第([一二三四五六七八九十百零]+)条\s*(.*)$")
# 施行日期
EFFECTIVE_PATTERN = re.compile(
    r"自\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日起施行"
)
# 废止声明：原《制度名》（文号）同时废止 / 予以废止
REPEAL_PATTERN = re.compile(
    r"《([^》]{2,60})》\s*[（\(]([^）\)]{2,40})[）\)][^。；]{0,20}?废止"
)
# 标题行（一级标题）
TITLE_PATTERN = re.compile(r"^#\s+(.+)$")

# 支持的制度文件后缀。格式读取（PDF/Word/纯文本）交给 file_reader，
# 本模块只负责"文本 → 条款结构"。
SUPPORTED_SUFFIXES = set(file_reader.SUPPORTED_SUFFIXES)


@dataclass
class Clause:
    """一个"条"的完整结构。"""

    clause_no: str          # 第七条
    clause_title: str       # 贷款额度上限
    chapter: str            # 第三章 授信审批与额度
    content: str            # 正文（不含"第七条"抬头）
    order: int              # 全文顺序，用于判断条款先后

    @property
    def full_text(self) -> str:
        """还原成原文形态，供大模型读取。"""
        head = f"{self.clause_no}"
        if self.clause_title:
            head = f"{head}　{self.clause_title}"
        return f"{head}\n{self.content}"

    def to_document(self, doc: "ParsedDocument") -> Document:
        """转成 LangChain Document，供向量化与重排使用。

        page_content 用条款全文（含条号与章名），让嵌入模型能理解它的上下文；
        metadata 保留全部溯源字段——召回后要据此回表校验时效。
        """
        return Document(
            page_content=f"《{doc.title}》{self.chapter} {self.full_text}",
            metadata={
                "clause_no": self.clause_no,
                "clause_title": self.clause_title,
                "chapter": self.chapter,
                "order": self.order,
                "doc_no": doc.doc_no,
                "doc_title": doc.title,
                "business_line": doc.business_line,
                "effective_date": doc.effective_date.isoformat() if doc.effective_date else "",
                "source_file": doc.source_file,
            },
        )


@dataclass
class ParsedDocument:
    """解析后的整份制度。"""

    title: str
    doc_no: str
    business_line: str
    issuer: str = ""
    version_tag: str = ""
    effective_date: date | None = None
    supersedes_no: str = ""
    supersedes_title: str = ""
    clauses: list[Clause] = field(default_factory=list)
    source_file: str = ""

    def to_document_row(self, status: str = "active") -> dict[str, Any]:
        return {
            "doc_no": self.doc_no,
            "title": self.title,
            "issuer": self.issuer,
            "business_line": self.business_line,
            "version_tag": self.version_tag,
            "status": status,
            "effective_date": self.effective_date,
            "repeal_date": None,
            "supersedes": self.supersedes_no or None,
            "source_file": self.source_file,
        }

    def to_documents(self) -> list[Document]:
        """整份制度转成条款级 Document 列表。"""
        return [c.to_document(self) for c in self.clauses]


# ---------- LangChain 加载器 ----------
class PolicyFileLoader(BaseLoader):
    """制度文件加载器。

    继承 BaseLoader 而不是自己 open()，好处是能被 LangChain 的
    Document 体系直接消费：后续接 Milvus VectorStore、Retriever、
    或者 DirectoryLoader 批量加载，都不用再包一层适配。

    lazy_load() 是 BaseLoader 的抽象方法，load() 会把它转成列表，
    这里只实现 lazy_load 就够了。
    """

    def __init__(self, file_path: str | Path, encoding: str = "utf-8") -> None:
        self.file_path = Path(file_path)
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        if self.file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"不支持的制度文件格式：{self.file_path.suffix}"
                f"（支持 {sorted(SUPPORTED_SUFFIXES)}）"
            )
        text = file_reader.read_text(self.file_path, encoding=self.encoding)
        yield Document(
            page_content=text,
            metadata={
                "source": str(self.file_path),
                "file_name": self.file_path.name,
                "suffix": self.file_path.suffix.lower(),
            },
        )


# ---------- 工具函数 ----------
_CN_NUM = {c: i for i, c in enumerate("零一二三四五六七八九", start=0)}
_CN_UNIT = {"十": 10, "百": 100}


def cn_to_int(text: str) -> int:
    """中文数字转阿拉伯数字，覆盖 1~999（制度条号不会更大）。"""
    if not text:
        return 0
    if text.isdigit():
        return int(text)

    total = 0
    section = 0
    for ch in text:
        if ch in _CN_NUM:
            section = _CN_NUM[ch]
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            # "十七" 这种省略了前导"一"
            total += (section or 1) * unit
            section = 0
        else:
            return 0
    return total + section


def normalize_brackets(text: str) -> str:
    """统一全角/半角括号，避免文号解析因括号变体漏匹配。"""
    table = str.maketrans({"【": "〔", "】": "〕", "[": "〔", "]": "〕", "(": "（", ")": "）"})
    return text.translate(table)


def normalize_clause_no(raw: str) -> str:
    """把"第 7 条"这类变体统一成"第七条"。"""
    raw = raw.strip().replace(" ", "")
    return raw


# 条款抬头之后的行内小标题：通常很短，且不与正文粘连。
# 制度里的写法有三种：
#     "第七条 贷款额度上限"                    ← 有标题、正文另起一行
#     "第七条 贷款额度上限。信用类为……"        ← 有标题但正文粘在同一行
#     "第七条 为规范本行……，……"               ← 无标题，整行都是正文
# 判据是行内标题的长度与断句符号。**不能用逗号切**：
# "制定依据为规范本行个人贷款业务经营行为，加强……" 里第一个逗号在 20 字之后，
# 但它前面那一段是正文而不是标题，用逗号切会把正文前半句误判成标题丢掉。
# 只有句号、冒号、分号、问号这类"句子结束"的符号才可能是标题的边界。
_CLAUSE_TITLE_MAX = 20
_CLAUSE_TITLE_BREAK = re.compile(r"[。；：;:！？!?]")

# 条款内部的分项序号：（一）、（1）、1.、①—⑳ 等
_SUB_ITEM_PATTERN = re.compile(
    r"^[（(][一二三四五六七八九十\d]+[）)]"
    r"|^\d+[.、．]"
    r"|^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]"
)


def _normalize_body(parts: list[str]) -> str:
    """把条款正文的多行拼接成规范形态。

    规则：
    * 空行 = 段落分隔，保留一个空行
    * 相邻的两个分项（（一）（二）…）之间用换行分隔，不插空行
    * 其余情况用换行
    """
    out: list[str] = []
    for part in parts:
        if part == "":
            if out and out[-1] != "":
                out.append("")
            continue
        # 上一个元素是分项、本行也是分项 → 已有分隔，不再加空行
        if out and out[-1] == "":
            # 若空白前后的两块都是分项，把这个空白吃掉
            prev = next((x for x in reversed(out[:-1]) if x != ""), "")
            if _SUB_ITEM_PATTERN.match(prev) and _SUB_ITEM_PATTERN.match(part):
                out.pop()
        out.append(part)

    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out).strip()


def _split_clause_head(rest: str) -> tuple[str, str]:
    """把"第X条"后面的内容拆成（小标题，行内正文）。

    返回的正文部分会作为该条正文的第一段，避免被当成标题丢弃。
    """
    rest = rest.strip()
    if not rest:
        return "", ""

    m = _CLAUSE_TITLE_BREAK.search(rest)
    first_seg = rest[: m.start()] if m else rest

    if m is not None and len(first_seg) <= _CLAUSE_TITLE_MAX:
        # 有标题，正文从断句符号之后开始（保留该符号，正文需要它）
        body = rest[m.start():].strip()
        return first_seg.strip(), body

    if m is None and len(rest) <= _CLAUSE_TITLE_MAX:
        # 整行都很短且无断句符号，当标题处理，正文为空
        return rest, ""

    # 没标题，整行都是正文
    return "", rest


def infer_business_line(title: str) -> str:
    """按制度名称推断业务条线。

    实际生产中条线由制度管理员在录入时选择，这里按关键词兜底，
    保证脚本批量导入时不会因缺字段而失败。
    """
    rules = [
        (("反洗钱", "可疑", "受益所有人", "客户身份"), "反洗钱"),
        (("柜面", "现金", "账户", "结算", "挂失", "尾箱"), "运营柜面"),
        (("贷款", "授信", "额度", "信贷", "消费金融"), "个人信贷"),
    ]
    for keywords, line in rules:
        if any(kw in title for kw in keywords):
            return line
    return "其他"


# ---------- 主解析函数 ----------
def parse_document(text: str, source_file: str = "") -> ParsedDocument:
    """把制度原文解析成 ParsedDocument。"""
    text = normalize_brackets(text)
    lines = [line.rstrip() for line in text.splitlines()]

    title = ""
    issuer = ""
    doc_no = ""
    effective_date = None
    supersedes_no = ""
    supersedes_title = ""

    clauses: list[Clause] = []
    current_chapter = ""
    buffer_head: str | None = None      # 当前条款号
    buffer_title = ""
    buffer_body: list[str] = []
    order = 0

    def emit_blank() -> None:
        """在正文里留一个段落分隔（连续空行只留一个）。"""
        if buffer_head is None:
            return
        if not buffer_body or buffer_body[-1] == "":
            return
        buffer_body.append("")

    def flush() -> None:
        nonlocal buffer_head, buffer_title, buffer_body, order
        if buffer_head is None:
            return
        body = _normalize_body(buffer_body)
        if body:
            order += 1
            clauses.append(
                Clause(
                    clause_no=buffer_head,
                    clause_title=buffer_title.strip(),
                    chapter=current_chapter,
                    content=body,
                    order=order,
                )
            )
        buffer_head = None
        buffer_title = ""
        buffer_body = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            emit_blank()
            continue

        # 一级标题
        title_match = TITLE_PATTERN.match(line)
        if title_match and not title:
            title = title_match.group(1).strip()
            continue

        # 发文字号（取第一个出现的）
        if not doc_no:
            doc_match = DOC_NO_PATTERN.search(line)
            if doc_match and "号" in line and len(line) < 80:
                code = _clean_issuer_code(doc_match.group(1))
                doc_no = f"{code}〔{doc_match.group(2)}〕{doc_match.group(3)}号"
                continue

        # 发文机构：出现在文号之前的单行机构名
        if not issuer and not doc_no and 4 <= len(line) <= 30 and line.endswith(("银行", "公司", "部", "中心")):
            issuer = line
            continue

        # 废止声明（可能在正文任意位置）
        if "废止" in line:
            for m in REPEAL_PATTERN.finditer(line):
                supersedes_title, supersedes_no = m.group(1), m.group(2)
                supersedes_no = normalize_brackets(supersedes_no)
                logger.info("识别到废止声明：%s %s", supersedes_title, supersedes_no)

        # 施行日期
        if effective_date is None:
            eff = EFFECTIVE_PATTERN.search(line)
            if eff:
                try:
                    effective_date = date(int(eff.group(1)), int(eff.group(2)), int(eff.group(3)))
                except ValueError:
                    logger.warning("施行日期非法，已忽略：%s", eff.group(0))

        # 章
        chapter_match = CHAPTER_PATTERN.match(line)
        if chapter_match:
            flush()
            current_chapter = f"第{chapter_match.group(1)}章 {chapter_match.group(2).strip()}"
            continue

        # 条
        clause_match = CLAUSE_PATTERN.match(line)
        if clause_match:
            flush()
            num = cn_to_int(clause_match.group(1))
            buffer_head = normalize_clause_no(f"第{clause_match.group(1)}条")
            buffer_title, inline_body = _split_clause_head(clause_match.group(2).strip())
            # "第七条 贷款额度上限为规范本行……" 这种小标题与正文粘连的情况，
            # 把正文部分还给 buffer_body，否则整段正文会被当成标题丢掉。
            buffer_body = [inline_body] if inline_body else []
            order_hint = num
            continue

        # 分项（（一）、1.、① 等）：必须与所属条款一起切。
        # 分项之前若已有正文段落，留一个空行分开——否则"……下列条件："
        # 会和"（一）……"粘成一段，阅读与切片都受影响。
        sub_match = _SUB_ITEM_PATTERN.match(line)
        if sub_match and buffer_head is not None:
            if buffer_body and buffer_body[-1] != "":
                buffer_body.append("")
            buffer_body.append(line)
            continue

        # 正文
        if buffer_head is not None:
            buffer_body.append(line)

    flush()

    if not title:
        # 文件名兜底
        title = source_file.rsplit(".", 1)[0] if source_file else "未命名制度"
        logger.warning("未解析到制度标题，使用文件名替代：%s", title)

    if not doc_no:
        logger.warning("制度《%s》未解析到发文字号，将使用文件指纹生成占位文号", title)
        import hashlib

        digest = hashlib.md5(source_file.encode("utf-8")).hexdigest()[:6]
        doc_no = f"恒信银发〔0000〕{digest}号"

    business_line = infer_business_line(title)
    version_tag = ""
    year_match = re.search(r"〔(\d{4})〕", doc_no)
    if year_match:
        version_tag = f"{year_match.group(1)}版"

    doc = ParsedDocument(
        title=title,
        doc_no=doc_no,
        business_line=business_line,
        issuer=issuer,
        version_tag=version_tag,
        effective_date=effective_date,
        supersedes_no=supersedes_no,
        supersedes_title=supersedes_title,
        clauses=clauses,
        source_file=source_file,
    )
    logger.info("解析完成：《%s》 %s，共 %d 条", title, doc_no, len(clauses))
    return doc


def load_document(file_path: str | Path) -> ParsedDocument:
    """一步到位：文件 → ParsedDocument。

    这是入库脚本唯一需要调用的入口，内部走 loader 再走解析，
    调用方不需要关心 Document 这一层。
    """
    path = Path(file_path)
    docs = PolicyFileLoader(path).load()
    if not docs:
        raise ValueError(f"制度文件为空：{path}")
    return parse_document(docs[0].page_content, source_file=path.name)


def load_directory(directory: str | Path) -> list[ParsedDocument]:
    """批量加载一个目录下的制度文件。"""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"制度目录不存在：{root}")
    results = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                results.append(load_document(path))
            except Exception as exc:
                logger.error("加载失败 %s：%s", path.name, exc)
    if not results:
        raise FileNotFoundError(f"{root} 下没有找到制度文件")
    return results
