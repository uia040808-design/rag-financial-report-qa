import json
import logging
from typing import List, Dict
from pathlib import Path
import faiss
import time
import numpy as np
from src.reranking import LLMReranker

_log = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        # 初始化向量检索器，加载所有向量库和文档
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.all_dbs = self._load_dbs()

    def _get_embedding(self, text: str):
        """取查询文本的向量表示。

        原本这里还有一个 ``embedding_provider="openai"`` 分支，走
        ``text-embedding-3-large``。它只能由外部直接传
        ``VectorRetriever(embedding_provider=...)`` 才会生效 —— 全项目的调用方
        （``questions_processing`` / ``HybridRetriever``）都没有传，因此是不可达
        分支，已删除。切换 embedding 后端请改
        :func:`src.ingestion.embedding_model`，并**重建向量库**。
        """
        import dashscope
        from src.ingestion import embedding_model
        rsp = dashscope.TextEmbedding.call(
            model=embedding_model(),   # 必须与建库时同一个模型
            input=[text]
        )
        # 兼容 dashscope 返回格式，不能用 resp.output，需用 resp['output']
        if 'output' in rsp and 'embeddings' in rsp['output']:
            # 多条输入（本处只有一条）
            emb = rsp['output']['embeddings'][0]
            if emb['embedding'] is None or len(emb['embedding']) == 0:
                raise RuntimeError(f"DashScope返回的embedding为空，text_index={emb.get('text_index', None)}")
            return emb['embedding']
        elif 'output' in rsp and 'embedding' in rsp['output']:
            # 兼容单条输入格式
            if rsp['output']['embedding'] is None or len(rsp['output']['embedding']) == 0:
                raise RuntimeError("DashScope返回的embedding为空")
            return rsp['output']['embedding']
        else:
            raise RuntimeError(f"DashScope embedding API返回格式异常: {rsp}")

    def _load_dbs(self):
        # 加载所有向量库和对应文档，建立映射
        all_dbs = []
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        for document_path in all_documents_paths:
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue
            # 用 metainfo['sha1'] 拼接 faiss 文件名
            sha1 = document.get('metainfo', {}).get('sha1', None)
            if not sha1:
                _log.warning(f"No sha1 found in metainfo for document {document_path.name}")
                continue
            faiss_path = self.vector_db_dir / f"{sha1}.faiss"
            if not faiss_path.exists():
                _log.warning(f"No matching vector DB found for document {document_path.name} (sha1={sha1})")
                continue
            try:
                # 原始方式：vector_db = faiss.read_index(str(faiss_path))
                # Windows 下 FAISS C++ 层 fopen 不识别含中文的路径，改为通过 Python open 读取再反序列化
                with open(str(faiss_path), 'rb') as f:
                    arr = np.frombuffer(f.read(), dtype=np.uint8)
                vector_db = faiss.deserialize_index(arr)
            except Exception as e:
                _log.error(f"Error reading vector DB for {document_path.name}: {e}")
                continue

            # FAISS 行号必须与 chunks 下标严格一一对应。重新分块后索引会过期，
            # 此时 faiss 只返回前 ntotal 个下标，检索结果与引用页码会整体错位
            # 且毫无报错，因此在加载阶段就明确失败。
            chunks = document.get("content", {}).get("chunks", [])
            if vector_db.ntotal != len(chunks):
                _log.error(
                    f"向量库已过期: {document_path.name} 有 {len(chunks)} 个 chunk，"
                    f"但 FAISS 只有 {vector_db.ntotal} 条向量。请重新运行 "
                    f"`pipeline.chunk_reports()` + `pipeline.create_vector_dbs()`。"
                )
                continue

            report = {
                "name": sha1,
                "vector_db": vector_db,
                "document": document
            }
            all_dbs.append(report)
        return all_dbs

    def _reports_for_company(self, company_name: str) -> List[Dict]:
        """返回该公司的**全部**已加载报告。

        历史实现在首个命中后 ``break``。本项目语料里 9 份文档的
        ``company_name`` 全是"中芯国际"（年报 / 深度报告 / 调研纪要 / 盈利预测 /
        行业对比……），于是"命中哪一份"完全取决于目录遍历顺序 —— 检索质量变成
        偶然结果，且另外 8 份文档永远检索不到。

        扩展到多公司语料时的已知陷阱
        --------------------------
        匹配用「``company_name`` 全等 **或** 公司名是 ``file_name`` 的子串」。
        全等没有问题，但子串那一支在**公司名互为前缀/子串**时会误命中：例如
        语料里同时有「中芯国际」与「中芯国际华虹」，查「中芯国际」会把后者的
        文档一并捞进来，答案会跨公司串联，且**不会报错**（页码、sha1 都合法，
        只是指错了公司）。

        当前语料只有一家公司，因此不影响正确性。扩到多公司时必须改成
        「先按 ``company_name`` 全等筛选；只有该公司在语料中完全不存在时，
        才退到 ``file_name`` 子串匹配」，并对退到子串命中的情况打日志。
        """
        hits: List[Dict] = []
        for report in self.all_dbs:
            metainfo = report.get("document", {}).get("metainfo", {}) or {}
            if (metainfo.get("company_name") == company_name
                    or company_name in metainfo.get("file_name", "")):
                hits.append(report)
        return hits

    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        """跨该公司**全部**文档检索，按相似度合并后返回全局 top_n。

        跨索引分数可比的前提：所有 FAISS 索引都由**同一个 embedding 模型**建出
        （见 :func:`src.ingestion.embedding_model`），且模型输出已归一化，
        因此 ``IndexFlatIP`` 的内积即余弦相似度，取值范围一致、可直接比较。
        若换模型重建过部分索引，这个前提就不成立 —— 此时合并排序会失真。
        """
        targets = self._reports_for_company(company_name)
        if not targets:
            # all_dbs 为空通常不是"没有这家公司"，而是所有索引都因过期被跳过，
            # 因此把两种原因都点明，避免排查时被误导。
            hint = ""
            if not self.all_dbs:
                hint = ("（已加载 0 个可用向量库：可能是分块报告与 FAISS 索引不匹配，"
                        "请重新执行 chunk_reports() 与 create_vector_dbs()）")
            _log.error(f"No report found with '{company_name}' company name. {hint}")
            raise ValueError(f"No report found with '{company_name}' company name.{hint}")

        # 查询向量只算一次，供全部索引复用
        embedding = self._get_embedding(query)
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)

        pooled: List[Dict] = []
        for report in targets:
            document = report["document"]
            vector_db = report["vector_db"]
            chunks = document["content"]["chunks"]
            pages = document.get("content", {}).get("pages", []) or []
            metainfo = document.get("metainfo", {}) or {}
            sha1 = metainfo.get("sha1")
            if not sha1:
                _log.warning(f"报告 {metainfo.get('file_name')} 缺少 sha1，跳过")
                continue

            k = min(top_n, len(chunks))
            if k <= 0:
                continue
            distances, indices = vector_db.search(x=embedding_array, k=k)
            for distance, index in zip(distances[0], indices[0]):
                if index < 0:      # FAISS 在结果不足时用 -1 填充
                    continue
                chunk = chunks[index]
                pooled.append({
                    "distance": round(float(distance), 4),
                    "page": chunk.get("page", 0),
                    "text": chunk["text"],
                    "pdf_sha1": sha1,
                    "file_name": metainfo.get("file_name", ""),
                    "_pages": pages,
                })

        pooled.sort(key=lambda r: r["distance"], reverse=True)
        if len(targets) > 1:
            hit_docs = {r["pdf_sha1"] for r in pooled[:top_n]}
            _log.info(f"[多文档检索] {company_name} 共 {len(targets)} 份文档参与检索，"
                      f"top_{top_n} 命中 {len(hit_docs)} 份")

        retrieval_results: List[Dict] = []
        # 去重键必须是 (sha1, page)：多份文档都存在"第 5 页"，只按页码去重
        # 会把不同 PDF 的同一页误判为重复。
        seen: set = set()
        for row in pooled:
            page = row["page"]
            if return_parent_pages:
                parent = next((p for p in row["_pages"] if p["page"] == page), None)
                if parent is None:
                    continue
                key = (row["pdf_sha1"], parent["page"])
                if key in seen:
                    continue
                seen.add(key)
                retrieval_results.append({
                    "distance": row["distance"],
                    "page": parent["page"],
                    "text": parent["text"],
                    "pdf_sha1": row["pdf_sha1"],
                    "file_name": row["file_name"],
                })
            else:
                key = (row["pdf_sha1"], page)
                if key in seen:
                    continue
                seen.add(key)
                retrieval_results.append({
                    "distance": row["distance"],
                    "page": page,
                    "text": row["text"],
                    "pdf_sha1": row["pdf_sha1"],
                    "file_name": row["file_name"],
                })
            if len(retrieval_results) >= top_n:
                break

        if not retrieval_results:
            _log.error(f"{company_name} 的 {len(targets)} 份文档均未检索到任何内容。")
        return retrieval_results

    def document_page_counts(self, company_name: str) -> Dict[str, int]:
        """返回该公司每份文档的页数，供引用校验做**逐文档**页码范围检查。

        多文档场景下单个 ``n_pages`` 已无意义：年报 222 页而调研纪要 22 页，
        "页码 150 是否越界"取决于它出自哪份 PDF。

        范围检查必须基于文档页数而不是检索结果：``full_context`` 模式下
        ``retrieve_all`` 会返回全部页，此时"页码是否在检索结果里"这一判据
        对 1..N 的任意页码都为真，等于没有校验。
        """
        counts: Dict[str, int] = {}
        for report in self._reports_for_company(company_name):
            pages = report.get("document", {}).get("content", {}).get("pages", []) or []
            sha1 = report.get("document", {}).get("metainfo", {}).get("sha1")
            if sha1 and pages:
                counts[sha1] = len(pages)
        return counts

    def retrieve_all(self, company_name: str) -> List[Dict]:
        """返回该公司全部文档的全部页面（``full_context`` 模式）。"""
        targets = self._reports_for_company(company_name)
        if not targets:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")

        all_pages: List[Dict] = []
        for report in targets:
            document = report["document"]
            pages = document["content"].get("pages", []) or []
            metainfo = document.get("metainfo", {}) or {}
            sha1 = metainfo.get("sha1")
            for page in sorted(pages, key=lambda p: p["page"]):
                all_pages.append({
                    "distance": 0.5,
                    "page": page["page"],
                    "text": page["text"],
                    "pdf_sha1": sha1,
                    "file_name": metainfo.get("file_name", ""),
                })
        return all_pages


class HybridRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.reranker = LLMReranker()

    def document_page_counts(self, company_name: str) -> Dict[str, int]:
        return self.vector_retriever.document_page_counts(company_name)
        
    def retrieve_by_company_name(
        self, 
        company_name: str, 
        query: str, 
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 10,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False
    ) -> List[Dict]:
        """
        使用混合检索方法进行检索和重排。
        
        参数：
            company_name: 需要检索的公司名称（会检索该公司的**全部**文档）
            query: 检索查询语句
            llm_reranking_sample_size: 首轮向量检索返回的候选数量
            documents_batch_size: 每次送入LLM重排的文档数
            top_n: 最终返回的重排结果数量
            llm_weight: LLM分数权重（0-1）
            return_parent_pages: 是否返回完整页面（而非分块）
        
        返回：
            经过重排的文档字典列表，包含分数、page 与 pdf_sha1
        """
        t0 = time.time()
        # 首轮向量检索已跨该公司的全部文档合并取 top_n（见 VectorRetriever）
        print("[计时] [HybridRetriever] 开始向量检索 ...")
        vector_results = self.vector_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages
        )
        t1 = time.time()
        print(f"[计时] [HybridRetriever] 向量检索耗时: {t1-t0:.2f} 秒")
        if not vector_results:
            return []
        # 跨文档来源分布：重排只看文本，看不出候选来自哪几份 PDF，
        # 全部集中在同一份文档往往意味着多文档检索没生效。
        src: Dict[str, Dict] = {}
        for r in vector_results:
            key = r.get("pdf_sha1", "")
            entry = src.setdefault(key, {"name": r.get("file_name") or key, "n": 0})
            entry["n"] += 1
        dist = ", ".join(f"{v['name'][:24]}×{v['n']}" for v in src.values())
        print(f"[多文档检索] 首轮 {len(vector_results)} 个候选来自 {len(src)} 份文档: {dist}")
        # 使用LLM对结果进行重排
        print("[计时] [HybridRetriever] 开始LLM重排 ...")
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )
        t2 = time.time()
        print(f"[计时] [HybridRetriever] LLM重排耗时: {t2-t1:.2f} 秒")
        print(f"[计时] [HybridRetriever] 总耗时: {t2-t0:.2f} 秒")
        return reranked_results[:top_n]
