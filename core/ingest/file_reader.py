# -*- coding: utf-8 -*-
"""制度文件读取：把 PDF / Word / 纯文本统一转成纯文本，交给 parser 做结构解析。

为什么不把这些逻辑写进 parser.py：
    parser 的职责是"文本 → 条款结构"，它需要的是干净的文本流，
    不应该关心文本是从 .pdf 还是 .docx 来的。格式差异属于 IO 层，
    放在这里，parser 的测试可以继续用纯字符串跑，不用造 PDF fixture。

三种格式各自的坑：

1. **PDF（PyMuPDF）**
   制度 PDF 几乎都是 Word 导出的。直接按页提取会带来两个问题：
   - 每页末尾的页眉页脚（文件名、页码、"内部资料"）会混进正文，
     切块后变成噪声，甚至被当成条款的一部分；
   - 跨页的段落被硬切成两段，切块时会出现半句话。
   处理办法：逐页提取后按文本块（block）判定，丢弃位于页面上下
   边缘且重复出现的行，再把页与页之间用换行拼起来交给后续处理。

2. **扫描件 PDF**
   图片型 PDF 提取出来的文本是空的。这种情况必须报错而不是返回空字符串，
   否则下游会静默入库 0 条条款，看起来"导入成功"其实什么都没进去。
   本模块不内置 OCR，检测到无文本层时明确报错并提示需要 OCR。

3. **Word（python-docx）**
   docx 里段落和表格是两个独立集合，`doc.paragraphs` 拿不到表格内容。
   而制度里的额度表、利率表恰恰在表格里，漏掉等于漏掉核心数据。
   处理办法：按文档顺序遍历 body 元素，段落和表格交错输出；
   表格按行转成"第一列：第二列；第三列：……"的自解释句子，
   而不是拼成一行用空格分开——后者切块后无法还原列与值的对应关系。
"""
from __future__ import annotations

import re
from pathlib import Path

from core.logger import logger

# 可读取的文件后缀。markdown/txt 本身已是纯文本，直接读。
PLAIN_SUFFIXES = {".md", ".txt"}
PDF_SUFFIXES = {".pdf"}
DOCX_SUFFIXES = {".docx"}

SUPPORTED_SUFFIXES = PLAIN_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES

# 判定页眉页脚的行：出现在页面上下 8% 区域内，且长度不超过该阈值
_PAGE_EDGE_RATIO = 0.08
_HEADER_FOOTER_MAX_CHARS = 40

# 抬头剩余部分的最长字数，超过就认为抬头后面还粘着正文
_BARE_HEAD_MAX = 20
# "第X条"抬头本身
_CLAUSE_HEAD_RE = re.compile(r"^第[一二三四五六七八九十百零\d]+[条章节]")
# 判断小标题是否已收尾的句读符号（逗号不算：标题里可能有顿号、逗号）
_CLAUSE_HEAD_BREAK = re.compile(r"[。；：;:！？!?]")


# ---------- PDF ----------
def _is_edge_block(block_bbox: tuple[float, float, float, float], page_height: float) -> bool:
    """判断一个文本块是否落在页面上下边缘（页眉/页脚区）。"""
    _, y0, _, y1 = block_bbox
    top_limit = page_height * _PAGE_EDGE_RATIO
    bottom_limit = page_height * (1 - _PAGE_EDGE_RATIO)
    return y1 <= top_limit or y0 >= bottom_limit


def _read_pdf(path: Path) -> str:
    """PDF → 纯文本。

    用 pymupdf（PyMuPDF 2.x 起 fitz 别名已弃用），按文本块提取而不是
    page.get_text() 的纯文本模式，因为块级 bbox 才能判断页面边缘。
    """
    import pymupdf

    pages: list[str] = []
    all_lines: list[str] = []
    dropped = 0
    page_count = 0

    with pymupdf.open(str(path)) as doc:
        page_count = doc.page_count
        if page_count == 0:
            raise ValueError(f"PDF 没有页面：{path.name}")

        for page in doc:
            height = page.rect.height or 1.0
            blocks = page.get_text("blocks") or []
            kept: list[str] = []

            for block in blocks:
                # blocks 每项为 (x0, y0, x1, y1, text, block_no, block_type)
                # block_type == 0 是文本块，1 是图片块
                if len(block) >= 7 and block[6] != 0:
                    continue
                text = (block[4] or "").strip()
                if not text:
                    continue
                if _is_edge_block(block[:4], height) and len(text) <= _HEADER_FOOTER_MAX_CHARS:
                    dropped += 1
                    continue
                kept.extend(text.split("\n"))

            if kept:
                pages.append("\n".join(kept))
                all_lines.extend(kept)
                # 跨页的段落要在页与页之间留一个分隔标记，交给全局处理
                all_lines.append("")
    if dropped:
        logger.info("PDF「%s」丢弃页眉页脚 %d 处", path.name, dropped)

    # 折行合并与段落补空行必须**跨页执行**：
    # 一条制度可能跨页断开（"（三）……1000万元。"在第 1 页末、
    # "各分行不得以拆分……"在第 2 页首），按页各自处理就补不出段落边界。
    text = "\n".join(_join_wrapped_lines(all_lines)).strip()

    if not text:
        raise ValueError(
            f"PDF「{path.name}」提取不到任何文本层，可能是扫描件。"
            "本项目不内置 OCR，请先对这份文件做 OCR 转成文本后入库，"
            "或另存为带文本层的 PDF。"
        )

    # 低文本量提示：可能文本层缺失或排版异常
    per_page = len(text) / max(page_count, 1)
    if per_page < 100:
        logger.warning(
            "PDF「%s」平均每页仅 %.0f 字，可能存在文本层缺失或排版异常，建议人工抽检",
            path.name,
            per_page,
        )

    return text


# ---------- Word ----------
def _table_to_sentences(table) -> list[str]:
    """把 docx 表格转成自解释句子，一行一句。

    为什么不拼成一行用空格分开：
        切块后空格分隔的多列会被语义平均掉，"额度上限 50 万 期限 3 年"
        这种拼接在检索阶段无法判断 50 万对应哪个指标。
    为什么不取首行做表头：
        制度表格往往有合并单元格或两级表头，首行未必是列名。
        这里采用"列序 + 值"的保守写法，配合表格前的引导段使用，
        引导段里通常已经说明了这张表在列什么。
    """
    lines: list[str] = []
    for row in table.rows:
        cells = [c.text.strip().replace("\n", " ") for c in row.cells]
        # 合并单元格会重复文本，去重但保留顺序
        uniq: list[str] = []
        for cell in cells:
            if cell and (not uniq or uniq[-1] != cell):
                uniq.append(cell)
        if not uniq:
            continue
        if len(uniq) == 1:
            lines.append(uniq[0])
        else:
            head = uniq[0]
            rest = "；".join(uniq[1:])
            lines.append(f"{head}：{rest}")
    return lines


def _read_docx(path: Path) -> str:
    """Word → 纯文本，段落与表格按文档原顺序交错输出。

    doc.paragraphs 和 doc.tables 是两个独立集合，按各自索引直接遍历会丢失
    二者的相对位置（表格通常插在特定段落之间，位置就是语义）。
    所以遍历 body 的直接子元素，按出现顺序处理 w:p / w:tbl。
    """
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    parts: list[str] = []

    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(child, document).text.strip()
            if text:
                parts.append(text)
        elif tag == "tbl":
            rows = _table_to_sentences(Table(child, document))
            if rows:
                parts.append("\n".join(rows))
        # 其余（节属性、批注等）跳过

    text = "\n\n".join(parts).strip()
    if not text:
        raise ValueError(f"Word 文档「{path.name}」没有可提取的文本")
    return text


def _is_bare_head(line: str) -> bool:
    """判断"第X条 小标题"这种抬头是否只剩抬头本身（后面没跟正文）。

    PDF 排版常把「第一条 制定依据」抬在上一行、正文另起；也常把
    「第七条 贷款额度上限。信用类……」整段排在一行。两种情况都要支持，
    所以判据是：去掉"第X条"之后剩下的部分，若不含句读符号、且不超过
    20 字，就认为它只是小标题——真正的正文会作为下一行跟进来。
    """
    rest = _CLAUSE_HEAD_RE.sub("", line, count=1).strip()
    if not rest:
        return True
    if len(rest) > _BARE_HEAD_MAX:
        return False
    return _CLAUSE_HEAD_BREAK.search(rest) is None


def _join_wrapped_lines(lines: list[str]) -> list[str]:
    """把 PDF 排版折行造成的断句接回去。

    为什么需要这一步：
        同一个自然段在 PDF 里是一行一行排的，排满一行就折到下一行。
        不同 PDF 生成工具在提取时的分块粒度不一样：
            · Word 导出的 PDF 往往按段落分块，块内部还是多行；
            · 用 insert_text 逐行绘制、或部分扫描件识别结果，则按**行**分块，
              一个段落变成好几个 block。
        所以不能只做"块内合并"，必须让合并逻辑同时能跨 block 工作。

    不能无脑合并的原因：
        「第一条 制定依据」这类条款抬头本身就是独立一行，紧跟其后的
        是正文。如果把抬头和正文粘成一句，条款就切不出来了——实测踩过，
        条款数会从 16 掉到 10，文号也会被拼成"XX银行股份有限公司XX银发…"。

    判定规则（按序检查）：
        1. 本行是条款/章节/分项抬头 → 必须独立成行；
        2. 上一行以句末标点结尾 → 上一行已收尾，本行另起，保留断行；
        3. 其余情况视为同一段被折行 → 相连。中英文之间补一个空格，
           中文之间直接拼（否则「促进」会被拆成「促 进」，中文分词
           会把它当成两个单字词，污染 BM25 稀疏向量）。
    """
    if not lines:
        return lines

    endings = tuple("。！？；：、）》」』】…”“")
    # 条款/章节/分项抬头：这些行必须独占一行。
    # 注意"第X条 小标题"这种形态：抬头后面可能还跟着小标题，
    # 但只要本行以"第X条"开头，就一定是新的一条，不能再和上一行粘。
    head_pattern = re.compile(
        r"^(第[一二三四五六七八九十百零\d]+[条章节]"
        r"|[（(][一二三四五六七八九十\d]+[）)]"
        r"|[一二三四五六七八九十]+、"
        r"|\d+[.、])"
    )

    merged: list[str] = []
    pending_book = False   # 上一行以"》"结尾，且其后出现空行
    for raw in lines:
        line = raw.strip()
        if not line:
            # 空行通常是段落分隔，PDF 里也可能是文本块之间的排版空隙。
            # 若上一行以"》"收尾，紧接着的下一行以"《"开头，说明并列书目被切开，
            # 该合并（见规则 0）。这里只记录状态，不输出空行。
            pending_book = bool(merged and merged[-1].endswith("》"))
            continue

        if not merged:
            merged.append(line)
            continue

        prev = merged[-1]

        # 规则 0：上一行以"》"收尾、本行以"《"开头 → 并列书目被排版切开，合并。
        # PDF 把并列的制度名当成独立文本块输出：《商业银行法》《监督管理法》
        # 换行后接上《个人贷款管理暂行办法》，中间可能还有空行。这不是新段落。
        if prev.endswith("》") and line.startswith(("《", "〈")):
            merged[-1] = prev + line
            continue

        # 规则 1：本行是抬头，另起
        if head_pattern.match(line):
            merged.append(line)
            continue

        # 规则 2：上一行**是**抬头（"第X条"开头且后面只有短小标题）→ 另起。
        # PDF 里抬头与正文常被排在同一行，靠这一步把它们分开：
        # "第一条 制定依据" 后面若跟的是正文，正文必须是新的一行。
        if head_pattern.match(prev) and _is_bare_head(prev):
            merged.append(line)
            continue

        # 规则 3：上一行已收尾，另起
        if prev and prev[-1] in endings:
            merged.append(line)
            continue

        # 规则 3.5：跨页折行的特例——上一行以"》"收尾、本行以"《"开头时
        # 合并。PDF 分页会把并列的制度名截断在两页或两行上：
        #   ……《中华人民共和国银行业监督管理法》
        #   《个人贷款管理暂行办法》等法律法规……
        # 这实际是一句话，断开后在切片时会被当成两段，必须接回去。
        if prev.endswith("》") and line.startswith(("《", "〈")):
            merged[-1] = prev + line
            continue

        # 规则 4：折行，接回去
        joiner = " " if (prev and prev[-1].isascii() and line[0].isascii()) else ""
        merged[-1] = prev + joiner + line

    # 收尾：把段落边界补回成空行。
    # PDF 按版面输出，段间空行常常丢失。真实制度里"一条"内部的段落形态是：
    #   主干说明：          ← 以冒号或句号结尾
    #   （一）……；          ← 分项列表，各行之间只换行
    #   （四）……。
    #   后续规定……          ← 分项结束后主干重新开始，这一段前面有空行
    # 判据：上一行以句末标点收尾、且不是分项，而本行也不是分项 → 补空行。
    result: list[str] = []
    for line in merged:
        if result:
            prev = result[-1]
            prev_is_sub = bool(_SUB_ITEM_RE.match(prev))
            line_is_sub = bool(_SUB_ITEM_RE.match(line))
            ends_sentence = bool(prev) and prev[-1] in "。；：！？"

            # 分项列表结束 → 主干段落重新开始
            if prev_is_sub and not line_is_sub:
                result.append("")
            # 主干段落结束 → 另一个主干段落开始（如"自……施行。"后接"原《……》废止。"）
            elif not prev_is_sub and not line_is_sub and ends_sentence:
                result.append("")
        result.append(line)

    return result


_SUB_ITEM_RE = re.compile(
    r"^[（(][一二三四五六七八九十\d]+[）)]"
    r"|^\d+[.、．]"
    r"|^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]"
)


# ---------- 统一入口 ----------
def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    """按后缀选择读取方式，统一返回纯文本。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"制度文件不存在：{p}")

    suffix = p.suffix.lower()

    if suffix in PLAIN_SUFFIXES:
        return p.read_text(encoding=encoding)
    if suffix in PDF_SUFFIXES:
        return _read_pdf(p)
    if suffix in DOCX_SUFFIXES:
        return _read_docx(p)

    raise ValueError(
        f"不支持的制度文件格式：{suffix}（支持 {sorted(SUPPORTED_SUFFIXES)}）"
    )
