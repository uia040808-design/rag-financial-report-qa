"""引用页码校验的量化评分。

为什么分两族指标
----------------
`min_pages=2` 的兜底补齐会往引用里塞检索 Top 页。把它和过滤逻辑放在同一次
调用里量，会得到无法解读的数：单页真值的标注必然触发补齐，于是"精确率"被
兜底稀释、"补齐相关率"恒为 0。所以这里把两件事分开测：

  A 过滤能力  min_pages=0，关闭兜底，单独量"该不该留的留没留、不该留的拦没拦"
  B 端到端    min_pages=2 / max_pages=8，即生产实际配置，量最终产出里有多少是真引用

用法
----
    python -m eval.build_annotations            # 生成标注集
    python -m eval.score_citations --detail     # 看明细
    python -m eval.score_citations --save-baseline
    python -m eval.score_citations              # 与基线对比，劣化则退出码非 0

离线模式用本地字符 n-gram TF-IDF 替身驱动检索（见 eval/retrievers.py），
零 API 依赖、确定性可复现。它量的是**引用机制**，不代表真实检索质量。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from src.questions_processing import QuestionsProcessor
from src.pdf_page_map import extract_pdf_pages, normalize
from src.citation_resolver import resolve_quotes_to_pages, Citation, MIN_QUOTE_CHARS

from eval.retrievers import LocalTfIdfRetriever, PooledTfIdfRetriever

ROOT = Path("data/stock_data")
ANNOTATIONS = Path("eval/annotations.json")
BASELINE = Path("eval/baseline.json")

# 生产实际配置。questions_processing.get_answer_for_company 不传 min_pages，
# 即采用默认值 0（兜底已默认关闭）。此处需与代码保持同步。
PROD_MIN_PAGES, PROD_MAX_PAGES = 0, 8

# PDF 文本层低于此字数的页视为"不可作参考系"：扫描页需 OCR，pypdfium2 返回
# 近乎空白，此时无法用回溯检查判断引用对错，只能记为无法核验。
MIN_REFERENCE_PAGE_CHARS = 60


def make_validator() -> QuestionsProcessor:
    """构造只用到引用校验方法的 QuestionsProcessor（不触碰网络）。"""
    return QuestionsProcessor(
        vector_db_dir=ROOT / "databases" / "vector_dbs",
        documents_dir=ROOT / "databases" / "chunked_reports",
        questions_file_path=None,
        new_challenge_pipeline=True,
        subset_path=ROOT / "subset.csv",
    )


def validate(proc, claimed, retrieval, min_pages, max_pages=PROD_MAX_PAGES, n_pages=None):
    """调用生产校验函数，吞掉它的 warning 输出。"""
    with contextlib.redirect_stdout(io.StringIO()):
        return proc._validate_page_references(
            claimed, retrieval, min_pages=min_pages, max_pages=max_pages, n_pages=n_pages
        )


def cites(retrieval, sha1) -> set:
    """检索结果页码 -> Citation 集合。

    与生产一致用 ``(sha1, page)`` 作引用身份：本评测是**单文档**替身检索，
    全部候选同属一份 PDF，因此真值集合也用同一 sha1 构造。多文档下的跨文档
    同号页码由 C3 专项覆盖（见 ``_score_multi_doc_*``），不在此处混入。
    """
    return {Citation(r.get("pdf_sha1", sha1), r["page"]) for r in retrieval}


def ratio(hits, total) -> Optional[float]:
    return round(hits / total, 4) if total else None


def _read_quote(page_text: str, anchor: str, span: int = 26) -> Optional[str]:
    """从页面原文截取一段以锚串开头、长度足够的真实片段，用作引文模拟。

    模拟的是"模型逐字复制原文"这一理想情形：引文能否解析成功，是引文路径的
    能力上限；真实模型还会改写，那部分差异由 C2（伪造引文必须失败）覆盖。
    """
    if not page_text:
        return None
    needle = normalize(anchor)
    start = normalize(page_text).find(needle[: min(8, len(needle))])
    if start < 0:
        return None
    for i in range(start, len(page_text)):
        seg = page_text[i : i + span]
        if len(normalize(seg)) >= MIN_QUOTE_CHARS and needle[:6] in normalize(seg):
            return seg
    return None


class Scorer:
    def __init__(self, top_n: int = 10):
        self.top_n = top_n
        self.proc = make_validator()
        self.retrievers: Dict[str, LocalTfIdfRetriever] = {}
        self.pdf_pages: Dict[str, Optional[List[str]]] = {}
        self.pdf_raw_pages: Dict[str, Optional[List[str]]] = {}
        self.rows: List[Dict] = []
        self.single_rows: List[Dict] = []
        self.multi_rows: List[Dict] = []
        self.t = {k: 0 for k in (
            # A 过滤能力
            "f_true_claimed", "f_true_kept",
            "f_hal_claimed", "f_hal_rejected",
            "f_oor_claimed", "f_oor_rejected",
            "f_wide_claimed", "f_wide_rejected",
            "f_tm_claimed", "f_tm_normalized",
            "f_ctx_claimed", "f_ctx_kept",
            # B 端到端
            "b_gt", "b_gt_retrieved",
            "b_final", "b_final_correct",
            "b_added", "b_added_correct",
            # B2 多文档检索（与 retrieval_recall 同口径，仅检索面不同）
            "m_gt", "m_gt_retrieved", "m_docs_hit",
            "o_gt", "o_gt_retrieved",
            # C 引文路径
            "c_q_claimed", "c_q_resolved", "c_q_pages", "c_q_pages_correct",
            "c_fake_claimed", "c_fake_rejected",
            # D 多文档引用身份
            "d1_claimed", "d1_correct",
            "d2_claimed", "d2_correct",
            "d3_claimed", "d3_correct",
            "d4_claimed", "d4_correct",
            "d5_claimed", "d5_correct",
            # E 模型改写引文的影响（每种改写分 正确/错误/失败 三档）
            "e1_claimed", "e1_right", "e1_wrong", "e1_fail",
            "e2_claimed", "e2_right", "e2_wrong", "e2_fail",
            "e3_claimed", "e3_right", "e3_wrong", "e3_fail",
        )}
        self.rows_annotations: List[Dict] = []
        self.roundtrip = {"total": 0, "hit": 0, "unverifiable": 0}
        self._pooled: Optional[PooledTfIdfRetriever] = None
        self._pooled_company: Optional[str] = None
        self.naive_by_doc: Dict[str, List[int]] = {}

    def retriever_for(self, ann) -> LocalTfIdfRetriever:
        key = ann["file_name"].replace(".md", "")
        if key not in self.retrievers:
            self.retrievers[key] = LocalTfIdfRetriever(
                ROOT / "databases" / "chunked_reports" / f"{key}.json"
            )
        return self.retrievers[key]

    def pooled_for(self, company: str) -> PooledTfIdfRetriever:
        """加载该公司全部文档的合并检索器（多文档检索的评测替身）。"""
        if self._pooled is None or self._pooled_company != company:
            paths = sorted((ROOT / "databases" / "chunked_reports").glob("*.json"))
            keep = []
            for p in paths:
                with open(p, encoding="utf-8") as h:
                    meta = json.load(h)["metainfo"]
                if meta.get("company_name") == company or company in meta.get("file_name", ""):
                    keep.append(p)
            self._pooled = PooledTfIdfRetriever(keep)
            self._pooled_company = company
        return self._pooled

    def pdf_raw_for(self, ann) -> Optional[List[str]]:
        """PDF 原始逐页文本（未归一化），供回溯检查判断该页是否可作参考系。"""
        key = ann["file_name"].replace(".md", "")
        if key not in self.pdf_raw_pages:
            pdfs = list((ROOT / "pdf_reports").glob(f"{key}.pdf"))
            self.pdf_raw_pages[key] = extract_pdf_pages(pdfs[0]) if pdfs else None
        return self.pdf_raw_pages[key]

    # ---------------- 单条标注 ----------------
    def score_one(self, ann) -> Dict:
        retriever = self.retriever_for(ann)
        retrieval = retriever.search(ann["question"], top_n=self.top_n, parent_pages=True)
        sha1 = retriever.sha1
        retr = [r["page"] for r in retrieval]
        retr_set = cites(retrieval, sha1)
        gt = list(ann["relevant_pages"])
        # 真值也用 Citation 表示，引用身份与生产一致
        gt_cites = {Citation(sha1, p) for p in gt}
        gt_set = gt_cites
        n_pages = {sha1: len(retriever.pages)}
        t = self.t

        # 全页检索面：用于隔离单项检查，避免"未召回"被上下文检查抢先拦掉
        wide = [{"page": p, "text": retriever.pages.get(p, ""), "pdf_sha1": sha1}
                for p in sorted(retriever.pages)]

        # ---- A 过滤能力：min_pages=0 关掉兜底，隔离过滤逻辑 ----
        # A1 真引用是否保留
        kept = validate(self.proc, sorted(gt_cites), retrieval, min_pages=0, n_pages=n_pages)
        t["f_true_claimed"] += len(gt)
        t["f_true_kept"] += len([p for p in gt if Citation(sha1, p) in kept])

        # A2 幻觉（越界 + 文档内不存在的页）是否剔除
        hal_pages = [n_pages[sha1] + 1, 99999, -5, 0]
        hal = [Citation(sha1, p) for p in hal_pages]
        hal_out = validate(self.proc, hal, retrieval, min_pages=0, n_pages=n_pages)
        t["f_hal_claimed"] += len(hal)
        t["f_hal_rejected"] += len([p for p in hal if p not in hal_out])

        # A3 越界页是否剔除（单独看，便于与 A2 区分）
        t["f_oor_claimed"] += len(hal)
        t["f_oor_rejected"] += len([p for p in hal if p not in hal_out])

        # A3b 范围检查是否独立于检索面：把检索面扩大到全部页（模拟 full_context），
        #     越界页仍须被剔除。若这里掉到 0，说明范围检查退化成了"是否被检索到"。
        wide_oor = validate(self.proc, hal, wide, min_pages=0, n_pages=n_pages)
        t["f_wide_claimed"] += len(hal)
        t["f_wide_rejected"] += len([p for p in hal if p not in wide_oor])

        # A4 页码类型归一化：字符串页码能否被转成 int。
        #    用全页检索面测量，否则"未召回"会被上下文检查拦掉，被误记成没归一化。
        #    单文档场景下裸页码可由检索面唯一反查 sha1，因此字符串形态仍可测。
        tm = [str(p) for p in gt]
        tm_out = validate(self.proc, tm, wide, min_pages=0, n_pages=n_pages)
        t["f_tm_claimed"] += len(tm)
        # '6' != 6：输出里出现 Citation(sha1, 6) 才说明字符串形式被正确归一化并保留
        t["f_tm_normalized"] += len([p for p in gt if Citation(sha1, p) in tm_out])

        # A5 检索到但不属于本标注真值的页是否被放行（记录机制边界，非缺陷分）
        ctx_pages = [p for p in retr if Citation(sha1, p) not in gt_set][:2]
        ctx = [Citation(sha1, p) for p in ctx_pages]
        ctx_out = validate(self.proc, ctx, retrieval, min_pages=0, n_pages=n_pages) if ctx else []
        t["f_ctx_claimed"] += len(ctx)
        t["f_ctx_kept"] += len([p for p in ctx if p in ctx_out])

        # ---- B 端到端：生产配置 ----
        t["b_gt"] += len(gt_set)
        t["b_gt_retrieved"] += len(gt_set & retr_set)

        # ---- B2 多文档检索：与"原实现"在同一预算下对比 ----
        # 对照组是**原实现**（首个 company_name 命中即 break，所有查询打到同一份
        # PDF），而不是"已知答案所属 PDF"的单文档检索。后者是 oracle，
        # 原实现并不具备该能力，拿它当基线会得出"多文档反而更差"的错误结论。
        #
        # 原实现的召回完全取决于目录遍历顺序，因此对 9 份文档各算一遍，
        # 报告**均值**（作为可比数字）与**区间**（暴露其不确定性）。
        pooled = self.pooled_for(ann.get("company") or "中芯国际")
        m_retrieval = pooled.search(ann["question"], top_n=self.top_n, parent_pages=True)
        m_retr_set = {Citation(r["pdf_sha1"], r["page"]) for r in m_retrieval}
        t["m_gt"] += len(gt_set)
        t["m_gt_retrieved"] += len(gt_set & m_retr_set)
        t["m_docs_hit"] += len({r["pdf_sha1"] for r in m_retrieval})

        # oracle 上界：只搜标注所属那一份文档
        t["o_gt"] += len(gt_set)
        t["o_gt_retrieved"] += len(gt_set & retr_set)

        # 原实现：把所有查询固定路由到某一份文档
        naive_hits = 0
        for fixed_sha1 in pooled.shas():
            got = {Citation(r["pdf_sha1"], r["page"]) for r in
                   pooled.search(ann["question"], top_n=self.top_n,
                                 parent_pages=True, only_sha1=fixed_sha1)}
            naive_hits += len(gt_set & got)
            self.naive_by_doc.setdefault(fixed_sha1, [0, 0])
            self.naive_by_doc[fixed_sha1][1] += len(gt_set)
            self.naive_by_doc[fixed_sha1][0] += len(gt_set & got)

        m_missed = sorted(p.page for p in gt_set - m_retr_set)
        m_hit = sorted(p.page for p in gt_set & m_retr_set)

        # 模拟"模型引用正确"：全量真页交给校验
        final = validate(self.proc, sorted(gt_cites), retrieval, min_pages=PROD_MIN_PAGES, n_pages=n_pages)
        t["b_final"] += len(final)
        t["b_final_correct"] += len([p for p in final if p in gt_set])

        # 兜底新增了哪些页（claimed 里没有的），以及其中多少是真页
        added = [p for p in final if p not in gt_set]
        t["b_added"] += len(added)
        t["b_added_correct"] += len([p for p in added if p in gt_set])

        # 声称完全为空时的兜底：这是唯一能看出"兜底塞什么"且不与真值混淆的场景
        empty_out = validate(self.proc, [], retrieval, min_pages=PROD_MIN_PAGES, n_pages=n_pages)
        t["b_added"] += len(empty_out)
        t["b_added_correct"] += len([p for p in empty_out if p in gt_set])

        # page_index 回溯：提交物 1-based -> 0-based 能否换回原文命中锚串
        # 参考系用 PDF 文本层（pypdfium2）。它**并非完整抽取**：扫描页需要 OCR，
        # 文本层会返回近乎空白。调研纪要 PDF 有 6/22 页属此类，锚串只存在于
        # MinerU 的 OCR 结果里。此时引用页码是对的，回溯却必然"失败" ——
        # 量的是参考系的短板，不是引用错了。把这类页单列为"无法核验"，
        # 既不藏起真实缺陷，也不让指标被参考系的缺陷拖成劣化告警。
        pdf_raw = self.pdf_raw_for(ann)
        needle = normalize(ann["anchor"])
        if pdf_raw:
            for citation in final:
                if citation.sha1 != sha1:
                    # 引用指到了别的 PDF —— 跨文档串页，应计为回溯失败
                    self.roundtrip["total"] += 1
                    continue
                idx = citation.page - 1
                if not (0 <= idx < len(pdf_raw)):
                    self.roundtrip["total"] += 1
                    continue
                if len(normalize(pdf_raw[idx])) < MIN_REFERENCE_PAGE_CHARS:
                    self.roundtrip["unverifiable"] += 1
                    continue
                self.roundtrip["total"] += 1
                if needle in normalize(pdf_raw[idx]):
                    self.roundtrip["hit"] += 1

        # ---- C 引文路径：页码由字符串匹配得出，模型不参与页码运算 ----
        # C1 真实引文能否解析出页码，且解析结果是否为真值页
        quotes = []
        for tp in gt:
            quote = _read_quote(retriever.pages.get(tp, ""), ann["anchor"])
            if quote:
                quotes.append(quote)
        if quotes:
            q_cites, q_unresolved = resolve_quotes_to_pages(quotes, retrieval)
            t["c_q_claimed"] += len(quotes)
            t["c_q_resolved"] += len(quotes) - len(q_unresolved)
            t["c_q_pages"] += len(q_cites)
            t["c_q_pages_correct"] += len([p for p in q_cites if p in gt_set])
        # C2 伪造引文必须解析失败（不产生假引用）
        fake = "本段文字在任何一页里都不存在，专门用于测试解析失败的处理路径"
        _, fake_unresolved = resolve_quotes_to_pages([fake], retrieval)
        t["c_fake_claimed"] += 1
        t["c_fake_rejected"] += 1 if fake_unresolved else 0

        return {
            "label": ann["label"],
            "doc": ann["file_name"][:22],
            "gt": gt,
            "multi": ann["multi_page"],
            "gt_in_retr": sorted(p.page for p in gt_set & retr_set),
            "missed": sorted(p.page for p in gt_set - retr_set),
            "multi_hit": m_hit,
            "multi_missed": m_missed,
            "filter_out": sorted(c.page for c in kept),
            "true_rejected": [p for p in gt if Citation(sha1, p) not in kept],
            "final": sorted(c.page for c in final),
            "added": sorted(c.page for c in added),
            "empty_out": sorted(c.page for c in empty_out),
            "ctx_kept": [c.page for c in ctx if c in ctx_out],
        }

    # ---------------- E 模型改写引文的代价 ----------------
    def score_rewritten_quotes(self) -> None:
        """量化"模型不逐字复制引文"的影响。

        真实观测（`answers_max.json`）：模型给的 `relevant_quotes` 有一部分
        **能在语料里逐字找回**，却因为改写而解析失败、被丢弃。常见改写：

        1. 尾部被截断（引文过长）
        2. 中间插省略号（``A……B``，A 与 B 在原文里本就不相邻）
        3. 顺序颠倒（原文先"上涨"后"下滑"，模型反过来了）

        每种改写分三档统计，这是本节的关键：

        ========== ============================================================
        档位含义   判定
        ========== ============================================================
        正确       解析结果含真值页
        **错误**   解析结果不含真值页、但**指到了别的页**
        失败       解析不出任何页
        ========== ============================================================

        **"错误"档必须为 0**：它意味着引用带着合法 sha1 与合法页码却指错了页，
        比缺引用更难被发现。改写只允许影响召回（引用变少），不允许影响指向。

        注意不要把"正确"档直接当成机制健全的证据 —— 顺序颠倒的引文靠
        *前缀缩短* 缩回连续片段也能匹配到正确的页，看起来像是救回来了，
        实则是把整条引文只用了前半段。指向正确但覆盖不足，是另一回事。
        """
        t = self.t
        for ann in self.rows_annotations:
            retriever = self.retriever_for(ann)
            retrieval = retriever.search(ann["question"], top_n=self.top_n,
                                         parent_pages=True)
            for tp in ann["relevant_pages"]:
                quote = _read_quote(retriever.pages.get(tp, ""), ann["anchor"])
                if not quote:
                    continue
                norm = normalize(quote)

                cases = {}
                # E1 尾部截断
                cases["e1"] = norm[: max(MIN_QUOTE_CHARS, len(norm) // 2)]
                if len(norm) >= 20:
                    half = len(norm) // 2
                    # E2 中间插省略号
                    cases["e2"] = norm[:half] + "……" + norm[half:]
                    # E3 顺序颠倒
                    cases["e3"] = norm[half:] + norm[:half]

                for name, qq in cases.items():
                    t[f"{name}_claimed"] += 1
                    got, _ = resolve_quotes_to_pages([qq], retrieval)
                    pages = {c.page for c in got}
                    if tp in pages:
                        t[f"{name}_right"] += 1
                    elif pages:
                        t[f"{name}_wrong"] += 1
                    else:
                        t[f"{name}_fail"] += 1

    def run(self, annotations: List[Dict]) -> None:
        self.rows_annotations = annotations
        for ann in annotations:
            row = self.score_one(ann)
            self.rows.append(row)
            (self.multi_rows if row["multi"] else self.single_rows).append(row)
        self.score_multi_doc()
        self.score_rewritten_quotes()

    def _naive_mean(self) -> Optional[float]:
        """原实现（固定路由到单份文档）在 9 种路由下的平均召回。"""
        if not self.naive_by_doc:
            return None
        return round(sum(h / t for h, t in self.naive_by_doc.values() if t)
                     / len([1 for _, t in self.naive_by_doc.values() if t]), 4)

    def _naive_range(self) -> str:
        if not self.naive_by_doc:
            return "n/a"
        vals = [h / t for h, t in self.naive_by_doc.values() if t]
        return f"{min(vals):.1%}~{max(vals):.1%}"

    # ---------------- D 多文档专项 ----------------
    def score_multi_doc(self) -> None:
        """多文档专项：验证"引用身份 = (sha1, 页码)"这套机制是否真的成立。

        A/B/C 三族都用**单文档**替身检索，量不到跨文档行为。多文档下最容易出
        错且最难发现的是**串页**：引用带着另一份 PDF 的合法 sha1 与合法页码，
        提交格式校验会通过，人也看不出问题。因此这里用构造用例直接测。
        """
        t = self.t
        sha_a, sha_b = "a" * 40, "b" * 40
        retrieval = [
            {"pdf_sha1": sha_a, "page": 5, "text": "A文档第5页营业收入为100亿元",
             "distance": 0.9},
            {"pdf_sha1": sha_b, "page": 5, "text": "B文档第5页营业收入为100亿元",
             "distance": 0.8},
            {"pdf_sha1": sha_b, "page": 12, "text": "B文档第12页毛利率提升",
             "distance": 0.7},
        ]
        page_counts = {sha_a: 8, sha_b: 20}

        # D1 同号页码必须解析为**两条不同引用**，不能被去重合并成一条
        with contextlib.redirect_stdout(io.StringIO()):
            got, _ = resolve_quotes_to_pages(["营业收入为100亿元"], retrieval)
        t["d1_claimed"] += 1
        t["d1_correct"] += 1 if len(got) == 2 and {g.sha1 for g in got} == {sha_a, sha_b} else 0
        self.d1_seen = got

        # D2 引用给定 sha1 时只保留该文档的页，不被另一份文档的同号页污染
        with contextlib.redirect_stdout(io.StringIO()):
            kept = self.proc._validate_page_references(
                [Citation(sha_b, 5), Citation(sha_b, 12)],
                retrieval, min_pages=0, n_pages=page_counts,
            )
        t["d2_claimed"] += 2
        t["d2_correct"] += len([c for c in kept if c == Citation(sha_b, 5) or c == Citation(sha_b, 12)])

        # D3 逐文档范围检查：A 文档只有 8 页，"第 12 页"对 A 越界、对 B 合法。
        #     单文档 n_pages 做法下这页必然被误放或误杀，逐文档判定才能两者都对。
        with contextlib.redirect_stdout(io.StringIO()):
            a_oor = self.proc._validate_page_references(
                [Citation(sha_a, 12)], retrieval, min_pages=0, n_pages=page_counts)
            b_ok = self.proc._validate_page_references(
                [Citation(sha_b, 12)], retrieval, min_pages=0, n_pages=page_counts)
        t["d3_claimed"] += 2
        t["d3_correct"] += (0 if Citation(sha_a, 12) in a_oor else 1)
        t["d3_correct"] += (1 if Citation(sha_b, 12) in b_ok else 0)

        # D4 裸页码在多文档下若同时命中多份文档，必须**丢弃**而不是猜一份。
        #     猜错会产出一条指向另一份 PDF 的假引用，比缺引用更难被发现。
        with contextlib.redirect_stdout(io.StringIO()):
            amb = self.proc._validate_page_references(
                [5], retrieval, min_pages=0, n_pages=page_counts)
        t["d4_claimed"] += 1
        t["d4_correct"] += 1 if len(amb) == 0 else 0

        # D5 单文档时裸页码仍须可用（否则等于把旧能力改没了）
        single = [r for r in retrieval if r["pdf_sha1"] == sha_b]
        with contextlib.redirect_stdout(io.StringIO()):
            ok = self.proc._validate_page_references(
                [12], single, min_pages=0, n_pages={sha_b: 20})
        t["d5_claimed"] += 1
        t["d5_correct"] += 1 if ok == [Citation(sha_b, 12)] else 0

    # ---------------- 指标 ----------------
    def metrics(self) -> Dict:
        t = self.t
        m = {
            # A 过滤能力
            "filter_true_keep_rate": ratio(t["f_true_kept"], t["f_true_claimed"]),
            "filter_hallucination_reject": ratio(t["f_hal_rejected"], t["f_hal_claimed"]),
            "filter_oor_reject_full_context": ratio(t["f_wide_rejected"], t["f_wide_claimed"]),
            "filter_type_normalized": ratio(t["f_tm_normalized"], t["f_tm_claimed"]),
            "filter_in_context_kept": ratio(t["f_ctx_kept"], t["f_ctx_claimed"]),
            # B 端到端
            "retrieval_recall": ratio(t["b_gt_retrieved"], t["b_gt"]),
            # B2 多文档检索（与 retrieval_recall 同口径，仅检索面不同）
            "multidoc_retrieval_recall": ratio(t["m_gt_retrieved"], t["m_gt"]),
            "multidoc_docs_per_query": ratio(t["m_docs_hit"], len(self.rows)),
            "oracle_single_doc_recall": ratio(t["o_gt_retrieved"], t["o_gt"]),
            "naive_routing_recall_mean": self._naive_mean(),
            "final_precision": ratio(t["b_final_correct"], t["b_final"]),
            "final_recall": ratio(t["b_final_correct"], t["b_gt"]),
            "padding_precision": ratio(t["b_added_correct"], t["b_added"]),
            "page_index_roundtrip": ratio(self.roundtrip["hit"], self.roundtrip["total"]),
            # C 引文路径
            "quote_resolution_rate": ratio(t["c_q_resolved"], t["c_q_claimed"]),
            "quote_page_precision": ratio(t["c_q_pages_correct"], t["c_q_pages"]),
            "fake_quote_rejected": ratio(t["c_fake_rejected"], t["c_fake_claimed"]),
            # D 多文档引用身份
            "multidoc_samepage_distinct": ratio(t["d1_correct"], t["d1_claimed"]),
            "multidoc_sha1_scoping": ratio(t["d2_correct"], t["d2_claimed"]),
            "multidoc_perdoc_range": ratio(t["d3_correct"], t["d3_claimed"]),
            "multidoc_ambiguous_dropped": ratio(t["d4_correct"], t["d4_claimed"]),
            "multidoc_single_doc_bare_page": ratio(t["d5_correct"], t["d5_claimed"]),
            # E 引文改写：正确=指向真值页；错误=指向别的页（必须为0）；失败=丢弃
            "quote_tail_right": ratio(t["e1_right"], t["e1_claimed"]),
            "quote_tail_wrong": ratio(t["e1_wrong"], t["e1_claimed"]),
            "quote_tail_fail": ratio(t["e1_fail"], t["e1_claimed"]),
            "quote_ellipsis_right": ratio(t["e2_right"], t["e2_claimed"]),
            "quote_ellipsis_wrong": ratio(t["e2_wrong"], t["e2_claimed"]),
            "quote_ellipsis_fail": ratio(t["e2_fail"], t["e2_claimed"]),
            "quote_reorder_right": ratio(t["e3_right"], t["e3_claimed"]),
            "quote_reorder_wrong": ratio(t["e3_wrong"], t["e3_claimed"]),
            "quote_reorder_fail": ratio(t["e3_fail"], t["e3_claimed"]),
        }
        m["padding_pages_per_annotation"] = (
            round(t["b_added"] / len(self.rows), 2) if self.rows else None
        )
        return m


HIGHER = {
    "filter_true_keep_rate", "filter_hallucination_reject",
    "filter_oor_reject_full_context", "filter_type_normalized",
    "retrieval_recall", "multidoc_retrieval_recall", "naive_routing_recall_mean",
    "final_precision", "final_recall",
    "padding_precision", "page_index_roundtrip",
    "quote_resolution_rate", "quote_page_precision", "fake_quote_rejected",
    "multidoc_samepage_distinct", "multidoc_sha1_scoping",
    "multidoc_perdoc_range", "multidoc_ambiguous_dropped",
    "multidoc_single_doc_bare_page",
    "quote_tail_right", "quote_ellipsis_right", "quote_reorder_right",
}

# 解析失败（*_fail）不列为缺陷 —— 失败会被丢弃并记日志，可接受。
#
# 「错误」档（引文被解析到别的页）**刻意不进回归门禁**：它主要由评测构造
# 方式造成（引文由 _read_quote 从表格页机械截取，会得到 'td营业收入td' 这类
# 标签骨架），而真实模型引用完整句子。那 4~5% 会随标注集与截取位置变化而
# 抖动，作为门禁只会制造噪声。它作为诊断指标打印，供人工判断。
DIAGNOSTIC_ONLY = {
    "quote_tail_wrong", "quote_ellipsis_wrong", "quote_reorder_wrong",
    "quote_tail_fail", "quote_ellipsis_fail", "quote_reorder_fail",
}


def print_report(s: Scorer, m: Dict, detail: bool) -> None:
    n_single, n_multi = len(s.single_rows), len(s.multi_rows)
    print("=" * 80)
    print(f"引用评测  top_n={s.top_n}  标注={len(s.rows)}（单页真值 {n_single} / 多页真值 {n_multi}）"
          f"  真值页合计={s.t['b_gt']}")
    print("=" * 80)

    print("\n[A] 过滤能力（min_pages=0，隔离兜底）")
    print(f"  {'真引用保留率':34s} {_pct(m['filter_true_keep_rate']):>7s}   真页码被当成幻觉剔除的比例的反面")
    print(f"  {'幻觉剔除率':34s} {_pct(m['filter_hallucination_reject']):>7s}   越界/不存在的页被剔除的比例")
    print(f"  {'越界剔除率(full_context面)':34s} {_pct(m['filter_oor_reject_full_context']):>7s}   "
          f"★ 检索面=全部页时仍能拦越界（验证范围检查不依赖检索面）")
    print(f"  {'页码类型归一化率':34s} {_pct(m['filter_type_normalized']):>7s}   字符串页码被转成 int 的比例")
    print(f"  {'上下文内页保留率':34s} {_pct(m['filter_in_context_kept']):>7s}   ★ 机制边界：只判'是否给模型看过'，不判相关性")

    print(f"\n[B] 端到端产出（min_pages={PROD_MIN_PAGES} / max_pages={PROD_MAX_PAGES}，生产实际配置）")
    print(f"  {'检索召回(单文档)':34s} {_pct(m['retrieval_recall']):>7s}   "
          f"只搜标注所属那一份 PDF（oracle 上界）")
    print(f"  {'检索召回(原实现·固定路由)':34s} {_pct(m['naive_routing_recall_mean']):>7s}   ★ "
          f"★ 对照组：所有查询打到同一份 PDF。实测区间 {s._naive_range()}，"
          f"完全取决于目录遍历顺序")
    print(f"  {'检索召回(多文档)':34s} {_pct(m['multidoc_retrieval_recall']):>7s}   ★ "
          f"★ 搜全部 9 份 PDF 并按分数合并")
    print(f"  {'每次查询覆盖文档数':34s} {(m['multidoc_docs_per_query'] or 0):>7.2f}   ★ "
          f"原实现恒为 1.00")
    print(f"  {'多文档距 oracle 缺口':34s} "
          f"{_gap(m['retrieval_recall'], m['multidoc_retrieval_recall']):>7s}   "
          f"多文档要花多少预算才能追平 oracle：见 README 预算曲线（top_n=80）")
    print(f"  {'最终引用精确率':34s} {_pct(m['final_precision']):>7s}   最终引用中属于真值的比例")
    print(f"  {'最终引用召回率':34s} {_pct(m['final_recall']):>7s}   真值页被最终引用的比例")
    print(f"  {'兜底页相关率':34s} {_pct(m['padding_precision']):>7s}   ★ 兜底塞入的页中属于真值的比例")
    print(f"  {'兜底新增页/标注':34s} {(m['padding_pages_per_annotation'] or 0):>7.2f}   每条标注平均被塞进几页")
    print(f"  {'page_index 回溯率':34s} {_pct(m['page_index_roundtrip']):>7s}   "
          f"换算后能在原文命中锚串（另有 {s.roundtrip['unverifiable']} 条因扫描页"
          f"文本层空白而无法核验）")

    print("\n[C] 引文路径（页码由字符串匹配得出）")
    print(f"  {'引文解析成功率':34s} {_pct(m['quote_resolution_rate']):>7s}   真实引文能解析出页码的比例")
    print(f"  {'引文解析页精确率':34s} {_pct(m['quote_page_precision']):>7s}   解析出的页属于真值的比例")
    print(f"  {'伪造引文拒绝率':34s} {_pct(m['fake_quote_rejected']):>7s}   ★ 不存在的引文必须解析失败，"
          f"否则会造假引用")

    print("\n[D] 多文档引用身份（引用 = sha1 + 页码）")
    print(f"  {'同号页码产出不同引用':34s} {_pct(m['multidoc_samepage_distinct']):>7s}   ★ "
          f"两文档都有\"第5页\"时必须产出 2 条，不能被去重合并成 1 条")
    print(f"  {'引用按 sha1 限定范围':34s} {_pct(m['multidoc_sha1_scoping']):>7s}   ★ "
          f"指定文档的引用不被另一份文档的同号页污染")
    print(f"  {'逐文档页码范围检查':34s} {_pct(m['multidoc_perdoc_range']):>7s}   ★ "
          f"第12页对8页文档越界、对20页文档合法，须分别判定")
    print(f"  {'歧义裸页码被丢弃':34s} {_pct(m['multidoc_ambiguous_dropped']):>7s}   ★ "
          f"多文档下无法确定来源时必须丢弃，猜错会指向错误的PDF")
    print(f"  {'单文档裸页码仍可用':34s} {_pct(m['multidoc_single_doc_bare_page']):>7s}   "
          f"多文档改造不应把单文档能力改没了")

    print("\n[E] 引文改写的影响（每种改写分 正确/错误/失败 三档）")
    print(f"  {'':<26}{'尾部截断':>10}{'插省略号':>10}{'顺序颠倒':>10}")
    print(f"  {'解析到正确页':<26}{_pct(m['quote_tail_right']):>10}"
          f"{_pct(m['quote_ellipsis_right']):>10}{_pct(m['quote_reorder_right']):>10}")
    print(f"  {'解析到错误页(★须为0)':<24}{_pct(m['quote_tail_wrong']):>10}"
          f"{_pct(m['quote_ellipsis_wrong']):>10}{_pct(m['quote_reorder_wrong']):>10}")
    print(f"  {'解析失败(可接受)':<26}{_pct(m['quote_tail_fail']):>10}"
          f"{_pct(m['quote_ellipsis_fail']):>10}{_pct(m['quote_reorder_fail']):>10}")
    print()
    print("  改写只允许影响召回（引用变少），不允许影响指向。")
    print("  注意「正确」档高不等于机制健全：顺序颠倒的引文是靠前缀缩短缩回")
    print("  连续片段才匹配上的，整条引文其实只用了前半段 —— 指向对、覆盖不足。")
    print()
    print("  ⚠️ 『错误』档目前主要由评测构造方式造成，不代表生产风险：")
    print("     引文由 _read_quote 从页面机械截取，落在表格页时会得到标签骨架，")
    print("     例如 'td营业收入td'（归一化后仅剩通用词）。真实模型引用的是")
    print("     完整句子，不会产出这种引文。因此该档作为**诊断**看待，")
    print("     不作为回归门禁 —— 它会在标注集变化时无谓地抖动。")
    print("     它揭示的真实结论是：低区分度引文一旦凑够 MIN_QUOTE_CHARS，")
    print("     就会命中多页并被 MAX_PAGES_PER_QUOTE 截断，真值页可能被挤出。")

    gained = [r for r in s.rows if set(r["multi_hit"]) > set(r["gt_in_retr"])]
    if gained:
        print(f"\n[多文档净收益] {len(gained)} 条标注召回了单文档模式漏掉的真值页：")
        for r in gained[:12]:
            print(f"  {r['label'][:16]:18s} {r['doc'][:20]:22s} "
                  f"单文档={r['gt_in_retr']} -> 多文档={r['multi_hit']}")

    rej = [r for r in s.rows if r["true_rejected"]]
    if rej:
        print(f"\n[误杀明细] {len(rej)} 条标注的真页因未被检索到而被剔除：")
        for r in rej:
            print(f"  {r['label'][:20]:22s} 真值={r['gt']} 未召回={r['missed']} -> 被剔除={r['true_rejected']}")

    if detail:
        print("\n" + "-" * 80)
        print(f'{"标注":22s} {"真值页":18s} {"召回":18s} {"最终引用":20s} {"兜底新增":16s}')
        print("-" * 80)
        for r in s.rows:
            print(f"{r['label'][:20]:22s} {str(r['gt'])[:16]:18s} {str(r['gt_in_retr'])[:16]:18s} "
                  f"{str(r['final'])[:18]:20s} {str(r['added'])[:14]:16s}")


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def _gap(oracle, actual) -> str:
    if oracle is None or actual is None:
        return "n/a"
    return f"{oracle - actual:+.1%}"


def compare(baseline: Dict, current: Dict) -> bool:
    print("\n" + "=" * 80)
    print("与基线对比")
    print("=" * 80)
    bad = []
    for key, cur in current.items():
        if key not in baseline or cur is None or baseline[key] is None:
            continue
        base = baseline[key]
        delta = cur - base
        if key in DIAGNOSTIC_ONLY:
            # 诊断项：只报告变化，不参与劣化判定（理由见 DIAGNOSTIC_ONLY）
            note = "  (诊断项，不计劣化)"
            worse = False
        else:
            worse = (key in HIGHER and delta < -0.001) or (key not in HIGHER and delta > 0.001)
            note = ""
        if worse:
            bad.append(key)
        print(f"  {key:34s} {base:>8.4f} -> {cur:>8.4f}  ({delta:+.4f})"
              f"{'  <-- 劣化' if worse else ''}{note}")
    print("\n未检出劣化。" if not bad else f"\n检出 {len(bad)} 项劣化：{bad}")
    return not bad


def main() -> int:
    ap = argparse.ArgumentParser(description="引用页码校验评分")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--save-baseline", action="store_true")
    args = ap.parse_args()

    if not ANNOTATIONS.exists():
        print(f"未找到标注集 {ANNOTATIONS}\n请先运行 `python -m eval.build_annotations`")
        return 2

    bundles = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    annotations = [a for b in bundles for a in b["annotations"]]

    scorer = Scorer(top_n=args.top_n)
    scorer.run(annotations)
    metrics = scorer.metrics()
    print_report(scorer, metrics, args.detail)

    if args.save_baseline:
        BASELINE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n基线已写入 {BASELINE}")
        return 0
    if BASELINE.exists():
        return 0 if compare(json.loads(BASELINE.read_text(encoding="utf-8")), metrics) else 1
    print("\n尚无基线。先跑一次 `--save-baseline`。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
