"""生成并校验标注集。

做三件事：
1. 在 `content.pages` 原文里检索每个锚串，解析出**真实页码**（不经模型、不手写）；
2. 命中页数不在 [1,4] 的种子自动丢弃，并说明原因 —— 避免把"到处都是的词"
   当成标注，那会让精确率指标失真；
3. 为每条标注保存原文上下文片段，供人工抽查。

用法：
    python -m eval.build_annotations
    python -m eval.build_annotations --report   # 打印上下文供人工复核
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List

from src.pdf_page_map import normalize

from eval.seeds import SEEDS

ROOT = Path("data/stock_data")
CHUNKED_DIR = ROOT / "databases" / "chunked_reports"
OUT_PATH = Path("eval/annotations.json")

MAX_PAGES = 4
MIN_PAGES = 1
CTX_BEFORE = 60
CTX_AFTER = 60

# 锚串在**全部文档**里允许命中的总页数上限。只按单文档计数是不够的：
# '22.5%' 在每份文档里各命中 <=4 页，逐文档检查全部通过，可它归一化成 '225'
# 后在 9 份文档里共命中 31 页（多为 1,225、225.0 这类无关数字），标注指错了地方。
MAX_CORPUS_PAGES = 6
# 归一化后的最短锚串。过短则任何巧合匹配都可能命中。
MIN_ANCHOR_CHARS = 3
# 纯数字锚串的额外下限。`normalize()` 会剔除小数点与百分号，纯数字锚串会退化
# 成无区分度的数字串：'22.5%' -> '225'。这类锚串必须更长才可信 ——
# '56.1/68.0/78.7' -> '561680787' 归一化后 9 位且只命中 2 页，仍然可用。
MIN_DIGIT_ANCHOR_CHARS = 8


def load_reports() -> List[Dict]:
    reports = []
    for path in sorted(glob.glob(str(CHUNKED_DIR / "*.json"))):
        with open(path, encoding="utf-8") as handle:
            reports.append(json.load(handle))
    if not reports:
        raise SystemExit(
            f"未找到分块报告：{CHUNKED_DIR}\n"
            f"请先运行 `python -m src.pipeline` 的 chunk_reports 阶段。"
        )
    return reports


def resolve(needle: str, page_norm: Dict[int, str]) -> List[int]:
    """返回锚串命中的页码列表。页码即标注答案。"""
    target = normalize(needle)
    if not target:
        return []
    return [page for page in sorted(page_norm) if target in page_norm[page]]


def screen_seeds(corpus_norm: List[str]) -> tuple:
    """对每个种子做**全局**区分度筛查。

    返回 ``(banned, warnings)``：``banned`` 是 ``label -> 原因``，``warnings``
    是需要人工过目的纯数字锚串。

    刻意做成全局一次性筛查，而不是每份文档各判一次：
      - 逐文档判定看不出"跨文档无区分度"——'22.5%' 在每份文档里各命中 <=4 页，
        逐文档检查全部通过，可它归一化成 '225' 后在全语料里命中 31 页；
      - 逐文档判定还会把同一个种子的判定重复输出 9 遍，噪音掩盖真问题。
    """
    banned: Dict[str, str] = {}
    warnings: List[Dict] = []
    for _q, needle, label, _kind in SEEDS:
        norm = normalize(needle)
        if len(norm) < MIN_ANCHOR_CHARS:
            banned[label] = (f"归一化后仅 {len(norm)} 字，锚串过短："
                             f"{needle!r} -> {norm!r}")
            continue
        digits_only = norm.isdigit()
        if digits_only and len(norm) < MIN_DIGIT_ANCHOR_CHARS:
            banned[label] = (f"归一化后为纯数字且仅 {len(norm)} 位，区分度不足："
                             f"{needle!r} -> {norm!r}（小数点与百分号已被剔除）")
            continue
        # 注意：比较的两侧都必须是归一化后的文本。拿归一化锚串去搜**原始**
        # markdown（含 `#`、`<table>` 等记号）会永远命中 0 次，
        # 于是上限检查形同虚设 —— 这是本检查最初实现里的实际 bug。
        hits = sum(1 for t in corpus_norm if norm in t)
        if hits > MAX_CORPUS_PAGES:
            banned[label] = (f"在全部文档中共命中 {hits} 页（>{MAX_CORPUS_PAGES}），"
                             f"锚串无区分度：{needle!r} -> {norm!r}")
            continue
        if digits_only:
            warnings.append({"label": label, "needle": needle, "norm": norm,
                             "hits": hits})
    return banned, warnings


def build(report: Dict, banned: Dict[str, str]) -> Dict:
    pages = report["content"].get("pages", [])
    page_norm = {p["page"]: normalize(p["text"]) for p in pages}

    accepted, rejected = [], []

    for question, needle, label, kind in SEEDS:
        if label in banned:
            rejected.append({"label": label, "needle": needle,
                             "reason": banned[label]})
            continue

        hits = resolve(needle, page_norm)

        if len(hits) < MIN_PAGES:
            rejected.append({"label": label, "needle": needle,
                             "reason": f"原文中未命中（0 页）"})
            continue
        if len(hits) > MAX_PAGES:
            rejected.append({"label": label, "needle": needle,
                             "reason": f"命中 {len(hits)} 页过于分散，无法作为页码标注"})
            continue

        # 取首个命中页的上下文，供人工抽查锚串是否真的指向该事实
        anchor_page = hits[0]
        norm_text = page_norm[anchor_page]
        idx = norm_text.find(normalize(needle))
        context = norm_text[max(0, idx - CTX_BEFORE) : idx + len(normalize(needle)) + CTX_AFTER]

        accepted.append({
            "id": f"{report['metainfo']['sha1']}::{label}",
            "label": label,
            "question": question,
            "kind": kind,
            "anchor": needle,
            "relevant_pages": hits,
            "n_pages": len(hits),
            "multi_page": len(hits) > 1,
            "company": report["metainfo"].get("company_name", ""),
            "file_name": report["metainfo"].get("file_name", ""),
            "pdf_sha1": report["metainfo"].get("sha1", ""),
            "context_preview": context,
        })

    return {
        "document": {
            "file_name": report["metainfo"].get("file_name", ""),
            "company": report["metainfo"].get("company_name", ""),
            "pdf_sha1": report["metainfo"].get("sha1", ""),
            "n_pages": len(pages),
        },
        "annotations": accepted,
        "rejected": rejected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成引用评测标注集")
    parser.add_argument("--report", action="store_true", help="打印原文上下文供人工复核")
    args = parser.parse_args()

    reports = load_reports()

    # 全语料页面（已归一化），用于种子的跨文档区分度筛查
    corpus_norm = [normalize(p["text"])
                   for r in reports
                   for p in r["content"].get("pages", [])]
    banned, seed_warnings = screen_seeds(corpus_norm)

    bundles = [build(r, banned) for r in reports]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(bundles, handle, ensure_ascii=False, indent=2)

    total = sum(len(b["annotations"]) for b in bundles)
    dropped = sum(len(b["rejected"]) for b in bundles)
    multi = sum(1 for b in bundles for a in b["annotations"] if a["multi_page"])

    print("=" * 76)
    print("标注集生成完成")
    print("=" * 76)
    for b in bundles:
        doc = b["document"]
        print(f"\n文档: {doc['file_name']}  页数={doc['n_pages']}")
        print(f"  采用 {len(b['annotations'])} 条（其中多页标注 {multi} 条），丢弃 {len(b['rejected'])} 条")
        for r in b["rejected"]:
            if r["label"] in banned:
                continue  # 全局剔除的种子统一在末尾汇总，不逐文档重复
            print(f"    [丢弃] {r['label']}: {r['reason']}")

    print(f"\n合计 {total} 条标注 -> {OUT_PATH}")

    # 纯数字锚串即使通过了区分度检查也值得人工过目：归一化剔除小数点后，
    # 它可能指向的是另一个数字（例如 '22.5%' -> '225' 命中 225.0）。
    if seed_warnings:
        print(f"\n[需人工确认] {len(seed_warnings)} 条纯数字锚串"
              f"（已通过区分度筛查，仅提示复核）：")
        for w in seed_warnings:
            print(f"  {w['label']:22s} {w['needle']!r:24s} -> {w['norm']!r:18s} "
                  f"全语料命中 {w['hits']} 页")

    if banned:
        print(f"\n[全局剔除] {len(banned)} 个锚串区分度不足，已在所有文档中禁用：")
        for label, reason in banned.items():
            print(f"  {label:22s} {reason}")

    if args.report:
        print("\n" + "=" * 76)
        print("原文上下文（人工复核用）")
        print("=" * 76)
        for b in bundles:
            for a in b["annotations"]:
                print(f"\n[{a['label']}] pages={a['relevant_pages']}  ({'多页' if a['multi_page'] else '单页'})")
                print(f"  Q: {a['question']}")
                print(f"  锚: {a['anchor']}")
                print(f"  上下文: ...{a['context_preview']}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
