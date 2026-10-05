"""页码溯源：把 MinerU 产出的 Markdown 重新对齐回源 PDF 的真实页码。

背景
----
MinerU 的 ``full.md`` 是整篇文档合并后的 Markdown，**不含任何分页标记**。
但下游三件事全都依赖真实页码：

* ``parent_document_retrieval`` —— 需要 ``content["pages"]`` 才能回溯整页
* ``_validate_page_references``  —— 需要真实页码才能判断引用是否为幻觉
* ``references[].page_index``   —— 最终提交物里的引用出处

在引入本模块之前，分块产物里没有 ``page`` 字段，检索侧 ``chunk.get("page", 0)``
恒返回 0，于是所有引用最终都退化成 ``page_index: -1``。

对齐原理
--------
MinerU 走的是 OCR，但源 PDF 通常仍带文本层。因此这里走"顺序对齐"：

1. 用 pdfium 逐页抽取文本，做字符归一化（只保留 CJK / 字母 / 数字）；
2. 对 Markdown 的每一行取字符 6-gram 集合（中文无需分词）；
3. 以"文档顺序单调"为约束做贪心前向匹配，仅允许小幅回退以抑制累积漂移；
4. 未锚定的行（空行、表格 HTML 碎片等）用左右锚点单调插值补齐。

对齐结果会用 PDF 自带页脚（形如 ``12 / 222``）做一次独立交叉校验，
一致率低于 95% 时告警。
"""

from __future__ import annotations

import glob
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_log = logging.getLogger(__name__)

# 只保留 CJK、统一表意文字（含中文标点）、字母和数字，其余全部丢弃
_KEEP_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
# 页脚形如 "12 / 222"。出现在页面首尾（页眉/页脚区），故在全页范围内搜索
_FOOTER_RE = re.compile(r"(\d+)\s*/\s*(\d+)")

SHINGLE_SIZE = 6
MIN_LINE_CHARS = SHINGLE_SIZE * 2
# 与 MinerU content_list 交叉校验时，锚串至少要有这么多个归一化字符才可比对
MIN_ANCHOR_CHARS = 12
# 贪心前向窗口：MinerU 输出顺序与 PDF 页序一致，窗口无需很大
FORWARD_WINDOW = 12
BACKTRACK_WINDOW = 3
BACKTRACK_MARGIN = 0.15
# 重合度阈值：>= ANCHOR_SCORE 认为锚定可靠并推进指针
ANCHOR_SCORE = 0.30
WEAK_SCORE = 0.12


def normalize(text: str) -> str:
    """归一化文本：去掉空白、Markdown 记号与 HTML 标签，只留可比对的字符。"""
    return _KEEP_RE.sub("", text)


def shingles(text: str, n: int = SHINGLE_SIZE) -> set:
    """字符 n-gram 集合。中文没有词边界，字符 n-gram 是免分词的稳妥做法。"""
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def overlap(a: set, b: set) -> float:
    """两个 shingle 集合的重合度，按较小集合归一化，避免长文本拉高分数。"""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def extract_pdf_pages(pdf_path: Path) -> List[str]:
    """用 pdfium 抽取 PDF 每一页的文本。pypdfium2 随 docling 一起安装。"""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出可执行的提示
        raise RuntimeError(
            "页码对齐需要 pypdfium2（docling 的依赖）：pip install pypdfium2"
        ) from exc

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        return [doc[i].get_textpage().get_text_bounded() for i in range(len(doc))]
    finally:
        doc.close()


def _align_lines(
    lines: Sequence[str], page_pool: Sequence[set]
) -> Tuple[List[Optional[int]], Dict[str, int]]:
    """顺序对齐：返回每行命中的 1-based 页码（未命中为 None）与统计信息。"""
    line_pages: List[Optional[int]] = []
    stats = {"strong": 0, "weak": 0, "unmatched": 0}
    ptr = 0
    last_page = len(page_pool)

    for line in lines:
        normalized = normalize(line)
        if len(normalized) < MIN_LINE_CHARS:
            line_pages.append(None)
            stats["unmatched"] += 1
            continue

        query = shingles(normalized)
        best_score, best_index = 0.0, None

        for i in range(ptr, min(last_page, ptr + FORWARD_WINDOW)):
            score = overlap(query, page_pool[i])
            if score > best_score:
                best_score, best_index = score, i

        # 允许小幅回退：单页文本与相邻页高度相似时，防止指针被噪声推得过远
        for i in range(max(0, ptr - BACKTRACK_WINDOW), ptr):
            score = overlap(query, page_pool[i])
            if score > best_score + BACKTRACK_MARGIN:
                best_score, best_index = score, i

        if best_index is None or best_score < WEAK_SCORE:
            line_pages.append(None)
            stats["unmatched"] += 1
            continue

        line_pages.append(best_index + 1)
        if best_score >= ANCHOR_SCORE:
            ptr = best_index
            stats["strong"] += 1
        else:
            stats["weak"] += 1

    return line_pages, stats


def _interpolate(line_pages: Sequence[Optional[int]]) -> List[Optional[int]]:
    """把未锚定的行用左右锚点单调插值补齐，文档首尾则沿用最近的锚点。"""
    filled: List[Optional[int]] = list(line_pages)
    anchors = [i for i, page in enumerate(filled) if page is not None]
    if not anchors:
        return filled

    first, last = anchors[0], anchors[-1]
    for i in range(0, first):
        filled[i] = filled[first]
    for i in range(last + 1, len(filled)):
        filled[i] = filled[last]

    for left, right in zip(anchors, anchors[1:]):
        start, end = filled[left], filled[right]
        span = right - left
        for step in range(1, span):
            filled[left + step] = start + round((end - start) * step / span)

    return filled


def verify_against_pdf_footers(
    line_pages: Sequence[Optional[int]], pages_text: Sequence[str]
) -> Dict[str, object]:
    """用 PDF 自带页脚 "N / 总页数" 独立校验对齐结果。

    这是不依赖对齐算法自身的旁证：若某行被判定在第 k 页，而 PDF 第 k 页的
    页脚恰好印着 k，则说明该行归属正确。多数排版把页眉页脚放在页面首尾，
    因此在全页范围内搜索，并用"分母等于总页数"排除正文里的偶然数字。
    """
    total = len(pages_text)
    printed: Dict[int, int] = {}
    for i, text in enumerate(pages_text):
        for match in _FOOTER_RE.finditer(text):
            if int(match.group(2)) == total:
                printed[i + 1] = int(match.group(1))
                break

    checked = agreed = 0
    mismatches: List[Tuple[int, int, int]] = []
    for line_index, page in enumerate(line_pages):
        if page is None or page not in printed:
            continue
        checked += 1
        if printed[page] == page:
            agreed += 1
        else:
            mismatches.append((line_index, page, printed[page]))

    return {
        "verifiable": bool(printed),
        "pages_with_footer": len(printed),
        "lines_checked": checked,
        "lines_agreeing": agreed,
        # 无法校验时返回 None，避免把"没有页脚"误报成"一致率 0%"
        "agreement": round(agreed / checked, 4) if checked else None,
        "mismatches": mismatches[:10],
    }


def load_content_list(md_path: Path) -> Optional[List[Dict]]:
    """读取与 markdown 同名的 ``*.content_list.json``。

    MinerU 的 batch 上传会一并产出这个文件，其中每个 block 带**权威的**
    ``page_idx``（0-based）。它的价值在于：纯扫描件页没有文本层，无法靠上面的
    对齐得到页码，只有它能给出答案。
    """
    candidate = md_path.with_suffix(".content_list.json")
    if not candidate.exists():
        return None
    try:
        with open(candidate, encoding="utf-8") as handle:
            blocks = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        _log.warning("无法读取 %s：%s", candidate, exc)
        return None
    return blocks if isinstance(blocks, list) else None


def cross_check_with_content_list(
    md_path: Path,
    line_pages: Sequence[Optional[int]],
    pages_text: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, object]]:
    """用 MinerU 的权威 ``page_idx`` 校验顺序对齐的结果。

    返回 None 表示没有 content_list 或无法比对；否则返回一致率与不一致样本。
    """
    blocks = load_content_list(md_path)
    if not blocks:
        return None

    lines = read_markdown_lines(md_path)
    line_norm = [normalize(l) for l in lines]

    # 按页收集锚串（取该页最长的一条正文）
    anchors: List[Tuple[int, str]] = []
    best_by_page: Dict[int, str] = {}
    for block in blocks:
        if not isinstance(block, dict) or "page_idx" not in block:
            continue
        body = block.get("text") or ""
        if block.get("type") not in ("text", "title", "discrete_formula"):
            continue
        normalized = normalize(body)
        if len(normalized) < MIN_ANCHOR_CHARS:
            continue
        page = int(block["page_idx"]) + 1
        if page not in best_by_page or len(normalized) > len(best_by_page[page]):
            best_by_page[page] = normalized
    for page in sorted(best_by_page):
        anchors.append((page, best_by_page[page]))

    if not anchors:
        return {"comparable": 0, "reason": "content_list 中没有可用的文本 block"}

    compared = agreed = 0
    mismatches: List[Dict[str, object]] = []
    cursor = 0
    for page, anchor in anchors:
        # 在 markdown 里顺序定位该页锚串（单调前进，避免错配到后文的重复句）
        pos = -1
        upper = len(line_norm)
        for probe_len in (len(anchor), 40, 24):
            needle = anchor[:probe_len]
            if len(needle) < MIN_ANCHOR_CHARS:
                break
            for i in range(cursor, min(upper, cursor + 400)):
                if needle in line_norm[i]:
                    pos = i
                    break
            if pos >= 0:
                break
        if pos < 0:
            continue
        cursor = pos
        inferred = line_pages[pos] if pos < len(line_pages) else None
        compared += 1
        if inferred == page:
            agreed += 1
        else:
            mismatches.append({"line": pos, "inferred": inferred, "content_list": page})

    return {
        "comparable": compared,
        "agreed": agreed,
        "agreement": round(agreed / compared, 4) if compared else None,
        "mismatches": mismatches[:10],
    }


def read_markdown_lines(md_path: Path) -> List[str]:
    """按行读取 markdown —— **"第 N 行"口径的唯一权威来源**。

    为什么必须是单一入口
    ------------------
    本项目的下游有三处按"行"索引同一个文件，且必须对"第 N 行"给出完全一致的
    定义：页码对齐产出的 ``line_pages``、分块的 ``split_markdown_file``、父文档
    聚合的 ``build_pages``（直接用 ``line_pages`` 的下标去取 ``lines[index]``）。
    口径一旦差 1，引用页码会整体错位 —— 而且是**静默**错位。

    历史上这里出过一次真实故障：``align_markdown_to_pages`` 用
    ``read().split("\\n")``，而 ``split_markdown_file`` 用 ``readlines()``。
    前者在文件以换行结尾时会多出一个**尾部空元素**，后者不会，于是长度校验
    直接报 ``line_pages 长度(1262)与 markdown 行数(1261)不一致``，
    ``chunk_reports`` 整个阶段失败。触发条件不是"文件末尾有换行"这么简单：
    Python 文本模式的 universal newlines 会把 ``\\r\\n`` 与**孤立 ``\\r``**
    都归一为 ``\\n``，所以只要最后一个字符是换行类字节就会差 1。而文本文件
    以换行结尾是通行惯例 —— 按 MinerU 的正常产出，9 份文档**全部**会触发。

    为什么不用 ``splitlines()``
    ------------------------
    ``splitlines()`` 确实也不会产生尾部空元素，看似等价，但它额外会在
    ``\\x0c``（form feed）、``\\x0b``、``\\x1c``、``\\u2028`` 等 Unicode 行界符处
    切分，而文本模式的 ``readlines()`` **不会**。PDF 抽取文本里出现 form feed
    很常见，用 ``splitlines()`` 等于把这个 bug 换成一个更隐蔽的版本。

    因此统一使用 ``readlines()``：它的语义与 ``split_markdown_file`` /
    ``build_pages`` 原本的行为**逐字节一致**，修复只消除分歧、不改变分块结果。
    """
    with open(md_path, "r", encoding="utf-8") as handle:
        return handle.readlines()


def align_markdown_to_pages(
    md_path: Path, pdf_path: Optional[Path]
) -> Tuple[List[Optional[int]], Dict[str, object]]:
    """把 markdown 的每一行映射到 1-based 真实页码。

    参数
    ----
    md_path
        MinerU 产出的 ``full.md``。
    pdf_path
        源 PDF。缺失或没有文本层时返回全 ``None``，调用方应降级处理。

    返回
    ----
    ``(line_pages, report)``；``line_pages`` 与 markdown 行一一对应。
    """
    # 行口径必须与 split_markdown_file / build_pages 完全一致，否则长度校验
    # 会报"不一致"，或下标越界。见 read_markdown_lines 的说明。
    lines = read_markdown_lines(md_path)

    if pdf_path is None or not Path(pdf_path).exists():
        _log.warning("未找到源 PDF %s，跳过页码对齐（引用将不可用）", pdf_path)
        return [None] * len(lines), {"aligned": False, "reason": "missing_pdf"}

    try:
        pages_text = extract_pdf_pages(Path(pdf_path))
    except Exception as exc:  # noqa: BLE001 - 损坏的 PDF 不应中断整条流水线
        _log.warning("从 %s 抽取文本失败：%s", pdf_path, exc)
        return [None] * len(lines), {"aligned": False, "reason": f"extract_failed: {exc}"}

    pool = [shingles(normalize(t)) for t in pages_text]
    if not any(pool):
        _log.warning("%s 没有可用文本层（疑似纯扫描件），跳过页码对齐", pdf_path)
        return [None] * len(lines), {"aligned": False, "reason": "no_text_layer"}

    raw_pages, stats = _align_lines(lines, pool)
    line_pages = _interpolate(raw_pages)

    report: Dict[str, object] = {
        "aligned": True,
        "pdf": str(pdf_path),
        "pdf_pages": len(pages_text),
        "total_lines": len(lines),
        "directly_matched": stats["strong"] + stats["weak"],
        "interpolated": sum(1 for i, p in enumerate(raw_pages) if p is None and line_pages[i]),
        **stats,
    }
    report["footer_check"] = verify_against_pdf_footers(raw_pages, pages_text)

    footer_check = report["footer_check"]
    if isinstance(footer_check, dict) and footer_check.get("lines_checked"):
        agreement = footer_check["agreement"]
        _log.info(
            "页码对齐：直接命中 %d/%d 行，PDF 页脚交叉校验一致率 %.1f%%",
            report["directly_matched"],
            report["total_lines"],
            100 * agreement,
        )
        if agreement < 0.95:
            _log.warning(
                "页码对齐一致率偏低（%.1f%%），不一致样本：%s",
                100 * agreement,
                footer_check["mismatches"],
            )
    else:
        _log.info(
            "该 PDF 未检出可用页脚，跳过交叉校验（对齐仍已完成，"
            "但缺少独立旁证，建议人工抽查若干页）"
        )

    # 与 MinerU content_list 的权威 page_idx 对照（若有该文件）
    content_check = cross_check_with_content_list(md_path, raw_pages, pages_text)
    if content_check is not None:
        report["content_list_check"] = content_check
        if content_check.get("agreement") is not None:
            _log.info(
                "与 MinerU content_list 交叉校验：可比对 %d 页，一致率 %.1f%%",
                content_check["comparable"],
                100 * content_check["agreement"],
            )
            if content_check["agreement"] < 0.95:
                _log.warning(
                    "与 content_list 一致率偏低，不一致样本：%s",
                    content_check["mismatches"],
                )

    return line_pages, report


def resolve_pdf_for_markdown(
    md_path: Path, pdf_reports_dir: Optional[Path]
) -> Optional[Path]:
    """按同名文件在 pdf_reports 目录里找到 markdown 对应的源 PDF。"""
    if pdf_reports_dir is None or not Path(pdf_reports_dir).exists():
        return None

    stem = md_path.stem
    exact = Path(pdf_reports_dir) / f"{stem}.pdf"
    if exact.exists():
        return exact

    # 少数文件名在转换过程中可能被规范化，退化为按前缀匹配
    matches = sorted(glob.glob(str(Path(pdf_reports_dir) / f"{stem}*.pdf")))
    if matches:
        return Path(matches[0])
    return None
