"""引文式引用：把模型返回的原文片段解析为页码。

为什么改用引文
--------------
原先模型直接返回页码（``relevant_pages: List[int]``），系统照单校验。这要求
模型在"上下文里的第几段"和"第几页"之间做一次映射，于是引入了一整类**页码
差一**错误：模型把第 N 段读成第 N-1 页或第 N+1 页。

差一无法被现有校验拦住 —— 实测（eval/score_citations.py 专项）：
构造 58 个差一引用，落在 top-10 检索面内的 17 个**全部通过校验，拦截率 0**。
上下文检查只能靠巧合挡下其余 71%，而父文档检索返回整页、相邻页主题相似，
差一页落进 top-10 是常见情况。

改成让模型返回**原文片段**，页码由本模块用字符串匹配得出，整类差一错误
从源头消失：模型不再参与任何页码运算。

解析策略刻意保守 —— 只做精确子串匹配，不做模糊匹配
----------------------------------------------------
模糊匹配（编辑距离 / n-gram 相似度）会把引文解析到**错误的页**，那比解析失败
更糟：失败会被丢弃并记日志，错误会带着合法 pdf_sha1 和合法 page_index 进入
提交物。因此这里的"放宽"仅限两种确定安全的操作：

1. **省略号切分**：模型引文中间用「……」截断时，取最长片段单独匹配；
2. **前缀缩短**：模型引文尾部被截断时，逐步缩短前缀再精确匹配。

两者都仍是精确子串匹配，只是换了更短的针。

为什么引用必须带文档身份
------------------------
多文档检索下，**页码本身不再是唯一标识**：中芯国际的年报、深度报告、调研纪要
里都存在"第 5 页"。只记页码的引用无法区分来源，会把不同 PDF 的同一页混为一谈
—— 而提交格式要求 ``(pdf_sha1, page_index)`` 成对出现。

因此本模块产出一律是 :class:`Citation`（文档身份 + 页码），不再产出裸页码。
``sha1`` 为 ``None`` 表示单文档场景（未提供文档身份），此时语义与旧版一致。
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from src.pdf_page_map import normalize

# 引文过短时无法唯一定位，低于此长度直接判为不可解析
MIN_QUOTE_CHARS = 8
# 单条引文最多采纳多少个页。引文命中多页时全部采纳虽属准确，但像"本公司"
# 这类高频短语会命中几十页，引用列表随之失去指向性；因此按检索排名取前若干个。
# build_page_index 已按检索得分排序，因此直接取前 N 个即保留排名。
MAX_PAGES_PER_QUOTE = 3
# 前缀缩短的比例序列（保留最长的先试）
PREFIX_RATIOS = (1.0, 0.85, 0.7, 0.55, 0.4)
# 中英文省略号 / 三个点
_ELLIPSIS_RE = re.compile(r"(?:…{1,3}|⋯{1,3}|\.{3,})")


class Citation(NamedTuple):
    """一条引用：文档身份 + 页码。

    多文档检索下页码不唯一 —— 年报和调研纪要都有"第 5 页"。提交格式要求
    ``(pdf_sha1, page_index)`` 成对出现，因此引用必须携带文档身份。

    ``sha1`` 为 ``None`` 表示单文档场景（检索结果未带 ``pdf_sha1``）。
    """

    sha1: Optional[str]
    page: int


def _fragments(quote: str) -> List[str]:
    """把引文拆成候选片段：去省略号分段 + 前缀逐步缩短，均保持原顺序。"""
    seen = set()
    candidates: List[str] = []

    def push(text: str) -> None:
        text = text.strip()
        if len(text) >= MIN_QUOTE_CHARS and text not in seen:
            seen.add(text)
            candidates.append(text)

    # 先按省略号分段，取最长片段优先
    parts = [p for p in _ELLIPSIS_RE.split(quote) if p and p.strip()]
    parts.sort(key=len, reverse=True)
    for part in parts:
        push(part)

    # 再对整条引文做前缀缩短
    normalized_full = quote.strip()
    for ratio in PREFIX_RATIOS:
        push(normalized_full[: max(MIN_QUOTE_CHARS, int(len(normalized_full) * ratio))])

    return candidates


def build_page_index(retrieval_results: Sequence[Dict]) -> List[Tuple[Optional[str], int, str]]:
    """从检索结果构造 (sha1, 页码, 归一化文本) 列表。

    只在**模型实际被展示的页面**里查找。引文若解析到检索面之外的页，说明模型
    引用了没被给予的内容，按未解析处理 —— 与上下文检查保持同一口径。

    ``sha1`` 取自检索结果的 ``pdf_sha1``；缺失时为 ``None``，表示单文档场景。
    """
    index: List[Tuple[Optional[str], int, str]] = []
    for result in retrieval_results:
        page = result.get("page")
        text = result.get("text") or ""
        if page is None:
            continue
        index.append((result.get("pdf_sha1"), page, normalize(text)))
    return index


def resolve_quotes_to_pages(
    claimed_quotes: Sequence[str],
    retrieval_results: Sequence[Dict],
) -> Tuple[List[Citation], List[str]]:
    """把引文解析为 ``(sha1, 页码)`` 引用。

    返回 ``(citations, unresolved)``：``citations`` 去重后保序；``unresolved``
    是解析失败的引文，仅用于日志，不进入引用。

    多文档下同一条引文可能同时出现在多份 PDF 里（例如各份都写"营业收入"），
    此时全部采纳并按检索排名截断到 :data:`MAX_PAGES_PER_QUOTE` 条 —— 因为
    无法凭引文判断哪一份才是模型所指的文档，猜测比多列几条更危险。
    """
    index = build_page_index(retrieval_results)
    if not index or not claimed_quotes:
        return [], [q for q in claimed_quotes if q]

    resolved: List[Citation] = []
    seen: set = set()
    unresolved: List[str] = []
    ambiguous: List[str] = []

    for quote in claimed_quotes:
        if not isinstance(quote, str) or not quote.strip():
            continue
        hit: List[Citation] = []
        for candidate in _fragments(quote):
            needle = normalize(candidate)
            if len(needle) < MIN_QUOTE_CHARS:
                continue
            hit = [Citation(sha1, page) for sha1, page, text in index if needle in text]
            if hit:
                break
        if hit:
            if len(hit) > MAX_PAGES_PER_QUOTE:
                # 引文区分度不足（如高频短语），按检索排名取前若干并记录
                ambiguous.append(quote)
                hit = hit[:MAX_PAGES_PER_QUOTE]
            for citation in hit:
                if citation not in seen:
                    seen.add(citation)
                    resolved.append(citation)
        else:
            unresolved.append(quote)

    if ambiguous:
        print(f"Warning: {len(ambiguous)} quote(s) matched more than "
              f"{MAX_PAGES_PER_QUOTE} pages (likely low-discriminative text such as "
              f"company names); kept the highest-ranked pages only.")

    return resolved, unresolved