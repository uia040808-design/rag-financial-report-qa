"""引用评测用的替身检索器。

用途
----
完整链路评测需要 DashScope 的 embedding 与 LLM。本模块提供一个**纯本地、
零依赖、确定性**的检索器来驱动「检索 → 引用校验 → 页码换算」这条链路，
使引用机制本身可被离线、可复现地度量。

它**不代表真实检索质量**：字符 n-gram TF-IDF 与 text-embedding-v2 的排序
结果会有差异。因此凡是由它产生的指标都只用于验证引用机制，真实检索质量
需用 `--mode online` 跑真 pipeline 得到。

结构上刻意与生产检索器保持一致，便于对比：
    chunk 级打分 → 取 top_k*oversample → 按 parent page 去重 → top_n
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

NGRAM = 3


def _counts(text: str) -> collections.Counter:
    from src.pdf_page_map import normalize

    norm = normalize(text)
    return collections.Counter(norm[i : i + NGRAM] for i in range(max(0, len(norm) - NGRAM + 1)))


class LocalTfIdfRetriever:
    """字符 n-gram TF-IDF 检索器，接口与 VectorRetriever 对齐。"""

    def __init__(self, chunked_report_path: Path):
        with open(chunked_report_path, encoding="utf-8") as handle:
            report = json.load(handle)

        self.chunks: List[Dict] = report["content"]["chunks"]
        self.pages: Dict[int, str] = {p["page"]: p["text"] for p in report["content"].get("pages", [])}
        self.metainfo = report["metainfo"]
        # 与生产检索器对齐：检索结果必须携带文档身份，否则引用无法区分
        # 不同 PDF 的同号页码（多文档检索下的核心难点）。
        self.sha1 = self.metainfo.get("sha1")

        raw = [_counts(chunk["text"]) for chunk in self.chunks]

        doc_freq: collections.Counter = collections.Counter()
        for counts in raw:
            doc_freq.update(counts.keys())
        n_docs = len(raw)
        self._idf = {t: math.log(n_docs / (1 + c)) for t, c in doc_freq.items()}

        self._vectors: List[Dict[str, float]] = []
        for counts in raw:
            vec = {t: (1.0 + math.log(c)) * self._idf.get(t, 0.0) for t, c in counts.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self._vectors.append({t: v / norm for t, v in vec.items()})

    def search(self, query: str, top_n: int = 10, parent_pages: bool = True,
               oversample: int = 12) -> List[Dict]:
        counts = _counts(query)
        qv = {t: 1.0 + math.log(c) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in qv.values())) or 1.0
        qv = {t: v / norm for t, v in qv.items()}

        scored = []
        for i, vec in enumerate(self._vectors):
            common = qv.keys() & vec.keys()
            scored.append((sum(qv[t] * vec[t] for t in common), i))
        scored.sort(reverse=True)

        if not parent_pages:
            return [
                {"page": self.chunks[i]["page"], "text": self.chunks[i]["text"],
                 "distance": round(s, 4), "pdf_sha1": self.sha1,
                 "file_name": self.metainfo.get("file_name", "")}
                for s, i in scored[:top_n]
            ]

        out: List[Dict] = []
        seen = set()
        for score, i in scored[: max(top_n * oversample, top_n)]:
            page = self.chunks[i]["page"]
            if page in seen:
                continue
            seen.add(page)
            out.append({"page": page, "text": self.pages.get(page, ""),
                        "distance": round(score, 4), "pdf_sha1": self.sha1,
                        "file_name": self.metainfo.get("file_name", "")})
            if len(out) >= top_n:
                break
        return out


class PooledTfIdfRetriever:
    """跨多份文档检索，接口与 :class:`LocalTfIdfRetriever` 对齐。

    对应生产侧 ``VectorRetriever`` 的多文档行为：把该公司的**全部**文档放进
    同一个排序里竞争，而不是每份文档各取 top_n 再拼起来。IDF 在合并后的语料
    上统一计算，与"单一向量空间全局排序"一致。

    与单文档检索器的关键差别在去重键：``(sha1, page)``。9 份文档里普遍存在
    同号页码，只按页码去重会把不同 PDF 的同一页误判为重复。
    """

    def __init__(self, chunked_paths: List[Path]):
        self.docs: List[Dict] = []
        for path in chunked_paths:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
            meta = report["metainfo"]
            self.docs.append({
                "sha1": meta.get("sha1"),
                "file_name": meta.get("file_name", ""),
                "n_pages": len(report["content"].get("pages", [])),
                "pages": {p["page"]: p["text"] for p in report["content"].get("pages", [])},
                "chunks": report["content"]["chunks"],
            })

        # 全局 IDF
        raw = [(d, i, _counts(c["text"]))
               for d in self.docs for i, c in enumerate(d["chunks"])]
        doc_freq: collections.Counter = collections.Counter()
        for _, _, counts in raw:
            doc_freq.update(counts.keys())
        n_docs = len(raw)
        idf = {t: math.log(n_docs / (1 + c)) for t, c in doc_freq.items()}

        self.vectors: List[Dict[str, float]] = []
        self.owners: List[tuple] = []
        for doc, i, counts in raw:
            vec = {t: (1.0 + math.log(c)) * idf.get(t, 0.0) for t, c in counts.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.vectors.append({t: v / norm for t, v in vec.items()})
            self.owners.append((doc, i))

    def page_counts(self) -> Dict[str, int]:
        return {d["sha1"]: d["n_pages"] for d in self.docs}

    def get(self, sha1: str) -> Optional[Dict]:
        return next((d for d in self.docs if d["sha1"] == sha1), None)

    def shas(self) -> List[str]:
        return [d["sha1"] for d in self.docs]

    def search(self, query: str, top_n: int = 10, parent_pages: bool = True,
               oversample: int = 12, quota: bool = False,
               only_sha1: Optional[str] = None) -> List[Dict]:
        """检索。

        ``only_sha1`` 只搜指定文档 —— 用来模拟**原实现**的行为：所有查询都被
        路由到同一份 PDF（首个 ``company_name`` 命中即 ``break``）。这不是
        oracle 基线，而是原代码真实的表现；把它与多文档检索对比才看得出改动
        的净收益。

        ``quota=True`` 启用逐文档配额。**实测有害，已默认关闭**：相同预算下
        召回从 73.0% 掉到 55.6%，因为强制每份文档出场会把名额花在无关文档上。
        保留开关是为了让这个反直觉的结论可被复现，而不是留一个没人验证的选项。
        """
        counts = _counts(query)
        qv = {t: 1.0 + math.log(c) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in qv.values())) or 1.0
        qv = {t: v / norm for t, v in qv.items()}

        scored = []
        for idx, vec in enumerate(self.vectors):
            if only_sha1 is not None and self.owners[idx][0]["sha1"] != only_sha1:
                continue
            common = qv.keys() & vec.keys()
            scored.append((sum(qv[t] * vec[t] for t in common), idx))
        scored.sort(reverse=True)

        pool = scored[: max(top_n * oversample, top_n)] if parent_pages else scored[:top_n]

        if quota and self.docs:
            # 逐文档保底：每份文档至少 ceil(top_n / 文档数) 个候选
            per_doc = max(1, math.ceil(top_n / len(self.docs)))
            taken: Dict[str, int] = {d["sha1"]: 0 for d in self.docs}
            kept = []
            for score, idx in pool:
                doc, _ci = self.owners[idx]
                if taken[doc["sha1"]] < per_doc:
                    taken[doc["sha1"]] += 1
                    kept.append((score, idx))
                if len(kept) >= top_n:
                    break
            if len(kept) < top_n:
                chosen = {id(k) for k in kept}
                kept.extend(k for k in pool if id(k) not in chosen)
            pool = kept[: max(top_n * oversample, top_n)]

        out: List[Dict] = []
        seen = set()
        for score, idx in pool:
            doc, ci = self.owners[idx]
            chunk = doc["chunks"][ci]
            page = chunk["page"]
            key = (doc["sha1"], page)
            if key in seen:
                continue
            seen.add(key)
            text = doc["pages"].get(page, "") if parent_pages else chunk["text"]
            out.append({"page": page, "text": text,
                        "distance": round(score, 4), "pdf_sha1": doc["sha1"],
                        "file_name": doc["file_name"]})
            if len(out) >= top_n:
                break
        return out
