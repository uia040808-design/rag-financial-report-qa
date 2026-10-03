import json
from typing import Union, Dict, List, Optional
import re
from pathlib import Path
from src.retrieval import VectorRetriever, HybridRetriever
from src.citation_resolver import resolve_quotes_to_pages, Citation
from src.api_requests import APIProcessor
from src.env_loader import generation_model
from tqdm import tqdm
import pandas as pd
import threading
import concurrent.futures
import time


def _coerce_page(page) -> Optional[int]:
    """把模型给出的页码归一化为 int，无法解释时返回 None。

    模型可能把页码输出成字符串（"6"），而 `'6' != 6`，不归一化会让真实引用
    被当成幻觉静默丢弃。同时拒绝 bool 与非整数 float，避免 int() 的隐式截断。
    """
    if isinstance(page, bool):
        return None
    if isinstance(page, int):
        return page
    if isinstance(page, float):
        return int(page) if page.is_integer() else None
    if isinstance(page, str):
        try:
            return int(page.strip())
        except ValueError:
            return None
    return None


class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Union[str, Path] = './vector_dbs',
        documents_dir: Union[str, Path] = './documents',
        questions_file_path: Optional[Union[str, Path]] = None,
        new_challenge_pipeline: bool = False,
        subset_path: Optional[Union[str, Path]] = None,
        parent_document_retrieval: bool = False,  # 是否启用父文档检索
        llm_reranking: bool = False,              # 是否启用LLM重排
        llm_reranking_sample_size: int = 5,
        top_n_retrieval: int = 10,
        parallel_requests: int = 10,
        api_provider: str = "dashscope", # openai
        answering_model: Optional[str] = None, # None -> env_loader.generation_model()
        full_context: bool = False
    ):
        # 初始化问题处理器，配置检索、模型、并发等参数
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.subset_path = Path(subset_path) if subset_path else None
        
        self.new_challenge_pipeline = new_challenge_pipeline
        self.return_parent_pages = parent_document_retrieval
        self.llm_reranking = llm_reranking
        self.llm_reranking_sample_size = llm_reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        # answering_model=None 时用项目默认模型（env_loader.generation_model）。
        # 原先硬编码 "qwen-turbo"，该模型额度耗尽后只有运行时才报 403。
        self.answering_model = answering_model or generation_model()
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.openai_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context

        self.answer_details = []
        self.detail_counter = 0
        self._lock = threading.Lock()

    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        # 加载问题文件，返回问题列表
        if questions_file_path is None:
            return []
        with open(questions_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results) -> str:
        """将检索结果格式化为RAG上下文字符串。

        每段都带上**来源文档名**：多文档检索下同一个页码在多份 PDF 里都存在，
        模型若不知道某段文字出自哪份文档，就无法在多份都提到"营业收入"时
        指出正确的那个；引用解析也失去了可用的消歧线索。
        """
        if not retrieval_results:
            return ""
        
        context_parts = []
        for result in retrieval_results:
            page_number = result['page']
            text = result['text']
            file_name = result.get('file_name') or result.get('pdf_sha1')
            if file_name:
                label = f"{file_name} — page {page_number}"
            else:
                label = f"page {page_number}"
            context_parts.append(f'Text retrieved from {label}: \n"""\n{text}\n"""')
            
        return "\n\n---\n\n".join(context_parts)

    def _extract_references(self, citations: list) -> list:
        """把 Citation 列表转成提交格式的 references。

        ``pdf_sha1`` 直接取自 Citation —— 不再查 subset.csv。历史实现用
        ``matching_rows.iloc[0]['sha1']``：subset.csv 里同一家公司有 9 行，
        于是**所有**引用都被标成同一份 PDF 的 sha1，即使内容实际来自研报。
        这类错误不报错、页码也在合法范围内，只是把引用指到了错误的文件上，
        比缺引用更难被发现。
        """
        refs = []
        for citation in citations:
            sha1 = getattr(citation, "sha1", None)
            page = getattr(citation, "page", citation)
            if not sha1:
                print(f"Warning: citation {citation} has no pdf_sha1 and was dropped "
                      f"— the submission format requires (pdf_sha1, page_index) and "
                      f"guessing a document would point the citation at the wrong file.")
                continue
            # page 为 None 的引用不能进提交物：换算 0-based 时 None - 1 会抛
            # TypeError；而若放行 page_index=None，评审侧也无法定位。
            # 正常路径下校验层已剔除这类引用，这里是最后一道闸。
            if page is None:
                print(f"Warning: citation {citation} has no page number and was dropped.")
                continue
            refs.append({"pdf_sha1": sha1, "page_index": page})
        return refs

    @staticmethod
    def _retrieval_key(result: Dict) -> Citation:
        """检索结果 -> Citation。

        对 ``page`` 缺失做显式处理：缺字段的检索结果返回 ``page=None`` 的
        Citation，它不会与任何正常引用相等，因此会在上下文检查里被剔除。
        原实现直接 ``result["page"]`` 抛 ``KeyError`` —— 而同函数里其它所有
        畸形输入（None 检索结果、page 为 None、缺 sha1）都被容错处理了，
        只有这一条会崩，是不一致。
        """
        return Citation(result.get("pdf_sha1"), result.get("page"))

    def _to_citations(
        self,
        claimed: list,
        retrieval_results: list,
    ) -> tuple:
        """把模型声明的页码归一化为 Citation 列表。

        三种输入形态：

        1. ``Citation`` —— 直接采用；
        2. ``{"pdf_sha1": ..., "page_index"/"page": ...}`` —— 从 dict 提取；
        3. 裸页码（int / 字符串）—— **无文档身份**，需靠检索结果反查。

        第 3 种是模型只给 ``relevant_pages`` 时的形态。多文档下裸页码本质上是
        歧义的：若该页码在检索面里出现于**多份**文档，就无法判断模型指的是哪
        一份。此时**丢弃并记日志**，不做猜测 —— 猜错会产出一条带着合法 sha1 与
        合法页码、却指向另一份 PDF 的假引用，比缺引用更难被发现。
        """
        if retrieval_results is None:
            retrieval_results = []
        candidates: List[Citation] = [
            self._retrieval_key(r) for r in retrieval_results
        ]

        # 缺 page 的检索结果不参与"裸页码反查"：它没有页码可贡献，
        # 留着会让 page=None 这个键混进映射，把"未给出的页码"和
        # "页码恰好是 None"混为一谈。
        page_to_shas: dict = {}
        for citation in candidates:
            if citation.page is None:
                continue
            page_to_shas.setdefault(citation.page, set()).add(citation.sha1)

        out: List[Citation] = []
        non_numeric: List = []
        ambiguous: List = []

        for item in claimed or []:
            if isinstance(item, Citation):
                page = _coerce_page(item.page)
                if page is None:
                    non_numeric.append(item)
                else:
                    out.append(Citation(item.sha1, page))
                continue

            if isinstance(item, dict):
                page = _coerce_page(item.get("page_index", item.get("page")))
                if page is None:
                    non_numeric.append(item)
                else:
                    out.append(Citation(item.get("pdf_sha1"), page))
                continue

            page = _coerce_page(item)
            if page is None:
                non_numeric.append(item)
                continue
            owners = page_to_shas.get(page, set())
            if len(owners) == 1:
                out.append(Citation(next(iter(owners)), page))
            elif len(owners) == 0:
                # 不在检索面里，交给上下文检查剔除（保留 None 以便走同一条日志）
                out.append(Citation(None, page))
            else:
                ambiguous.append(page)

        return out, non_numeric, ambiguous

    def _validate_page_references(
        self,
        claimed_pages: list,
        retrieval_results: list,
        min_pages: int = 0,
        max_pages: int = 8,
        n_pages: Optional[Dict[str, int]] = None,
    ) -> List[Citation]:
        """校验模型声称的引用，返回可信的 Citation 列表。

        四道过滤，彼此独立：

        1. **类型归一化**：字符串页码（"6"）转 int；无法解释的丢弃。
        2. **范围检查**：剔除超出**该文档**实际页数的页码。这道检查基于文档页数
           而非检索结果，因此在 ``full_context`` 模式（检索面 = 全部页）下依然
           有效。多文档下页码范围逐份判定：年报 222 页、调研纪要 22 页，
           "页码 150 是否越界"取决于它出自哪份 PDF。
        3. **上下文检查**：剔除未出现在检索结果里的引用。
        4. **截断**：超量时按检索得分保留，而非按模型给出的顺序。

        关于兜底（``min_pages``）
        ----------------------
        历史实现会在模型没给页码、或给的页码全被剔除时，塞入检索 Top 页凑够
        2 条。实测（eval/score_citations.py）：每条标注平均被塞 2.86 页，其中
        仅 30% 与答案相关，把引用精确率从 89.7% 拉低到 59.1%，并让日志里的
        "hallucinated" 措辞把召回不足误报成模型幻觉。

        因此兜底**默认关闭**（``min_pages=0``）。宁可少一条引用，也不要一条
        带着合法 pdf_sha1 与合法 page_index、却与答案无关的假引用 —— 后者更难
        被发现。需要旧行为时可显式传 ``min_pages=2``。

        参数 ``n_pages`` 由原来的单个页数改为 ``{sha1: 页数}`` 映射以支持多文档；
        传入单个 int 仍按单文档处理（所有引用共用该页数）。
        """
        if claimed_pages is None:
            claimed_pages = []

        # 兼容旧的单文档调用：传 int 时视为「所有引用共用这一个文档」
        page_counts: Dict[Optional[str], int] = {}
        if isinstance(n_pages, int):
            page_counts[None] = n_pages
        elif n_pages:
            page_counts = dict(n_pages)

        normalized, non_numeric, ambiguous = self._to_citations(
            claimed_pages, retrieval_results
        )

        retrieved_keys = [self._retrieval_key(r) for r in (retrieval_results or [])]
        retrieved_set = set(retrieved_keys)
        rank = {key: i for i, key in enumerate(retrieved_keys)}

        # ---- 1b. 歧义页码（多文档下无法确定出自哪份 PDF）----
        if ambiguous:
            print(f"Warning: Dropped {len(ambiguous)} ambiguous page references: "
                  f"{sorted(set(ambiguous))} — these page numbers occur in more than "
                  f"one retrieved document, so the source PDF cannot be determined "
                  f"without guessing.")

        # ---- 2. 范围检查（依赖各文档页数，不依赖检索面）----
        out_of_range: List[Citation] = []
        no_page: List[Citation] = []
        in_range: List[Citation] = []
        for citation in normalized:
            if citation.page is None:
                # 引文没解析出页码（或检索结果缺 page）—— 无从校验，
                # 交给上下文检查剔除，但单独记日志以便与越界区分。
                no_page.append(citation)
                continue
            limit = page_counts.get(citation.sha1)
            if limit is None:
                in_range.append(citation)
            elif 1 <= citation.page <= limit:
                in_range.append(citation)
            else:
                out_of_range.append(citation)

        # ---- 3. 上下文检查 ----
        # 未知文档身份（sha1=None）的引用按裸页码匹配，多文档下必然落空 ——
        # 这正是期望行为：无法确定来源就不引用。
        out_of_context = [c for c in in_range if c not in retrieved_set]
        validated = [c for c in in_range if c in retrieved_set]

        # 去重并保持模型给出的顺序
        seen = set()
        deduped: List[Citation] = []
        for citation in validated:
            if citation not in seen:
                seen.add(citation)
                deduped.append(citation)
        validated = deduped

        # ---- 日志：三类分开，不再一律叫 "hallucinated" ----
        if non_numeric:
            print(f"Warning: Dropped {len(non_numeric)} non-numeric page references: {non_numeric}")
        if no_page:
            print(f"Warning: Dropped {len(no_page)} citation(s) with no resolvable page "
                  f"number — the quote was not located in the retrieved context, or the "
                  f"retrieval result lacked a `page` field.")
        if out_of_range:
            detail = ", ".join(
                f"{c.sha1[:8] if c.sha1 else '<unknown>'}:p{c.page}"
                f"(上限{page_counts.get(c.sha1)})" for c in out_of_range
            )
            print(f"Warning: Dropped {len(out_of_range)} out-of-range page references: {detail}")
        if out_of_context:
            print(f"Warning: Dropped {len(out_of_context)} page references absent from the "
                  f"retrieved context — these may be recall misses rather than hallucinations: "
                  f"{sorted({(c.sha1 or '<unknown>')[:8] + ':p' + str(c.page) for c in out_of_context})}")

        # ---- 超量截断：按检索得分排序后保留，而非按模型给出的顺序 ----
        if max_pages and len(validated) > max_pages:
            validated.sort(key=lambda c: rank.get(c, len(rank)))
            print(f"Trimming references from {len(validated)} to {max_pages} pages "
                  f"(keeping the highest-ranked)")
            validated = validated[:max_pages]

        # ---- 兜底：默认关闭 ----
        if min_pages and len(validated) < min_pages and retrieval_results:
            existing = set(validated)
            for result in retrieval_results:
                citation = self._retrieval_key(result)
                if citation not in existing:
                    validated.append(citation)
                    existing.add(citation)
                    if len(validated) >= min_pages:
                        break

        return validated

    def _resolve_citations(
        self,
        answer_dict: dict,
        retrieval_results: list,
        n_pages: Optional[Dict[str, int]] = None,
    ) -> List[Citation]:
        """把模型给出的引用解析为可信的 Citation 列表。

        优先走**引文解析**：模型返回 ``relevant_quotes``（原文片段），页码与
        来源文档由字符串匹配得出。这样模型不参与任何页码运算，整类"页码差一"
        错误从源头消失（原先实测：落入检索面的差一引用 17/17 全部漏放）。

        若模型没给引文（提示词未生效、旧缓存答案、其他 provider），回退到直接
        使用 ``relevant_pages``，并给出明确告警 —— 此时仍存在差一风险，且
        多文档下页码归属也需靠检索结果反查。

        无论走哪条路径，最终引用都要过 :meth:`_validate_page_references` 的
        范围 / 上下文 / 类型 / 截断四道检查。
        """
        quotes = answer_dict.get("relevant_quotes")

        if isinstance(quotes, list) and quotes:
            citations, unresolved = resolve_quotes_to_pages(quotes, retrieval_results)
            if unresolved:
                print(f"Warning: {len(unresolved)} quote(s) could not be located in the "
                      f"retrieved context and were dropped: "
                      f"{[q[:40] for q in unresolved]}")
            if not citations:
                print("Warning: No quote resolved to a citation; "
                      "references will be empty rather than guessed.")
        else:
            legacy = answer_dict.get("relevant_pages") or []
            print(f"Warning: Model returned no `relevant_quotes`; falling back to raw "
                  f"`relevant_pages` ({legacy}). Page numbers are then taken at face "
                  f"value and off-by-one errors are no longer detectable.")
            citations, _, _ = self._to_citations(legacy, retrieval_results)

        return self._validate_page_references(
            citations, retrieval_results, n_pages=n_pages
        )

    def get_answer_for_company(self, company_name: str, question: str, schema: str) -> dict:
        # 针对单个公司，检索上下文并调用LLM生成答案
        t0 = time.time()
        if self.llm_reranking:
            retriever = HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        else:
            retriever = VectorRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        t1 = time.time()
        print(f"[计时] [get_answer_for_company] 检索器初始化耗时: {t1-t0:.2f} 秒")
        if self.full_context:
            retrieval_results = retriever.retrieve_all(company_name)
        else:           
            t2 = time.time()
            retrieval_results = retriever.retrieve_by_company_name(
                company_name=company_name,
                query=question,
                llm_reranking_sample_size=self.llm_reranking_sample_size,
                top_n=self.top_n_retrieval,
                return_parent_pages=self.return_parent_pages
            )
            t3 = time.time()
            print(f"[计时] [get_answer_for_company] 检索耗时: {t3-t2:.2f} 秒")
        if not retrieval_results:
            raise ValueError("No relevant context found")
        t4 = time.time()
        rag_context = self._format_retrieval_results(retrieval_results)
        t5 = time.time()
        print(f"[计时] [get_answer_for_company] 构建rag_context耗时: {t5-t4:.2f} 秒")
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model
        )
        t6 = time.time()
        print(f"[计时] [get_answer_for_company] LLM调用耗时: {t6-t5:.2f} 秒")
        self.response_data = self.openai_processor.response_data
        if self.new_challenge_pipeline:
            # 每份文档的页数用于**逐文档**页码范围检查；检索器不支持时退化为不做范围检查
            page_counts = None
            if hasattr(retriever, "document_page_counts"):
                try:
                    page_counts = retriever.document_page_counts(company_name)
                except Exception:  # noqa: BLE001 - 范围检查是增强项，不应拖垮问答
                    page_counts = None
            citations = self._resolve_citations(
                answer_dict, retrieval_results, n_pages=page_counts
            )
            # relevant_pages 保留裸页码供调试与 answer_details 展示；
            # 提交用的 references 走 citations，保留文档身份。
            answer_dict["relevant_pages"] = [c.page for c in citations]
            answer_dict["citations"] = citations
            answer_dict["references"] = self._extract_references(citations)
        print(f"[计时] [get_answer_for_company] 总耗时: {t6-t0:.2f} 秒")
        return answer_dict

    def _extract_companies_from_subset(self, question_text: str) -> list[str]:
        """从问题文本中提取公司名，匹配subset文件中的公司"""
        if not hasattr(self, 'companies_df'):
            if self.subset_path is None:
                raise ValueError("subset_path must be provided to use subset extraction")
            # 优先尝试 utf-8，失败则尝试 gbk
            try:
                self.companies_df = pd.read_csv(self.subset_path, encoding='utf-8')
            except UnicodeDecodeError:
                print('警告：subset.csv 不是 utf-8 编码，自动尝试 gbk 编码...')
                self.companies_df = pd.read_csv(self.subset_path, encoding='gbk')
        
        found_companies = []
        company_names = sorted(self.companies_df['company_name'].unique(), key=len, reverse=True)
        
        for company in company_names:
            # 只要公司名在问题文本中出现就算匹配（包含关系）
            if company in question_text:
                found_companies.append(company)
                question_text = question_text.replace(company, '')
        
        return found_companies

    def process_question(self, question: str, schema: str):
        # 处理单个问题，支持多公司比较
        if self.new_challenge_pipeline:
            extracted_companies = self._extract_companies_from_subset(question)
        else:
            extracted_companies = re.findall(r'"([^"]*)"', question)
        
        if len(extracted_companies) == 0:
            raise ValueError("No company name found in the question.")
        
        if len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            answer_dict = self.get_answer_for_company(company_name=company_name, question=question, schema=schema)
            return answer_dict
        else:
            return self.process_comparative_question(question, extracted_companies, schema)
    
    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        """创建答案详情的引用ID，并存储详细内容"""
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict['step_by_step_analysis'],
                "reasoning_summary": answer_dict['reasoning_summary'],
                "relevant_quotes": answer_dict.get('relevant_quotes', []),
                "relevant_pages": answer_dict['relevant_pages'],
                "response_data": self.response_data,
                "self": ref_id
            }
        return ref_id

    def _calculate_statistics(self, processed_questions: List[dict], print_stats: bool = False) -> dict:
        """统计处理结果，包括总数、错误数、N/A数、成功数"""
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(1 for q in processed_questions if (q.get("value") if "value" in q else q.get("answer")) == "N/A")
        success_count = total_questions - error_count - na_count
        if print_stats:
            print(f"\nFinal Processing Statistics:")
            print(f"Total questions: {total_questions}")
            print(f"Errors: {error_count} ({(error_count/total_questions)*100:.1f}%)")
            print(f"N/A answers: {na_count} ({(na_count/total_questions)*100:.1f}%)")
            print(f"Successfully answered: {success_count} ({(success_count/total_questions)*100:.1f}%)\n")
        
        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count
        }

    def process_questions_list(self, questions_list: List[dict], output_path: str = None, submission_file: bool = False, pipeline_details: str = "") -> dict:
        # 批量处理问题列表，支持并行与断点保存，返回处理结果和统计信息
        total_questions = len(questions_list)
        # 给每个问题加索引，便于后续答案详情定位
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]
        self.answer_details = [None] * total_questions  # 预分配答案详情列表
        processed_questions = []
        parallel_threads = self.parallel_requests

        if parallel_threads <= 1:
            # 单线程顺序处理
            for question_data in tqdm(questions_with_index, desc="Processing questions"):
                processed_question = self._process_single_question(question_data)
                processed_questions.append(processed_question)
                if output_path:
                    self._save_progress(processed_questions, output_path, submission_file=submission_file, pipeline_details=pipeline_details)
        else:
            # 多线程并行处理
            with tqdm(total=total_questions, desc="Processing questions") as pbar:
                for i in range(0, total_questions, parallel_threads):
                    batch = questions_with_index[i : i + parallel_threads]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                        # executor.map 保证结果顺序与输入一致
                        batch_results = list(executor.map(self._process_single_question, batch))
                    processed_questions.extend(batch_results)
                    
                    if output_path:
                        self._save_progress(processed_questions, output_path, submission_file=submission_file, pipeline_details=pipeline_details)
                    pbar.update(len(batch_results))
        
        statistics = self._calculate_statistics(processed_questions, print_stats = True)
        
        return {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics
        }

    def _process_single_question(self, question_data: dict) -> dict:
        question_index = question_data.get("_question_index", 0)
        
        if self.new_challenge_pipeline:
            question_text = question_data.get("text")
            schema = question_data.get("kind")
        else:
            question_text = question_data.get("question")
            schema = question_data.get("schema")
        try:
            answer_dict = self.process_question(question_text, schema)
            
            if "error" in answer_dict:
                detail_ref = self._create_answer_detail_ref({
                    "step_by_step_analysis": None,
                    "reasoning_summary": None,
                    "relevant_pages": None
                }, question_index)
                if self.new_challenge_pipeline:
                    return {
                        "question_text": question_text,
                        "kind": schema,
                        "value": None,
                        "references": [],
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref}
                    }
                else:
                    return {
                        "question": question_text,
                        "schema": schema,
                        "answer": None,
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref},
                    }
            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            if self.new_challenge_pipeline:
                return {
                    "question_text": question_text,
                    "kind": schema,
                    "value": answer_dict.get("final_answer"),
                    "references": answer_dict.get("references", []),
                    "answer_details": {"$ref": detail_ref}
                }
            else:
                return {
                    "question": question_text,
                    "schema": schema,
                    "answer": answer_dict.get("final_answer"),
                    "answer_details": {"$ref": detail_ref},
                }
        except Exception as err:
            return self._handle_processing_error(question_text, schema, err, question_index)

    def _handle_processing_error(self, question_text: str, schema: str, err: Exception, question_index: int) -> dict:
        """
        处理问题处理过程中的异常。
        记录错误详情并返回包含错误信息的字典。
        """
        import traceback
        error_message = str(err)
        tb = traceback.format_exc()
        error_ref = f"#/answer_details/{question_index}"
        error_detail = {
            "error_traceback": tb,
            "self": error_ref
        }
        
        with self._lock:
            self.answer_details[question_index] = error_detail
        
        print(f"Error encountered processing question: {question_text}")
        print(f"Error type: {type(err).__name__}")
        print(f"Error message: {error_message}")
        print(f"Full traceback:\n{tb}\n")
        
        if self.new_challenge_pipeline:
            return {
                "question_text": question_text,
                "kind": schema,
                "value": None,
                "references": [],
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref}
            }
        else:
            return {
                "question": question_text,
                "schema": schema,
                "answer": None,
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref},
            }

    def _post_process_submission_answers(self, processed_questions: List[dict]) -> List[dict]:
        """
        提交格式后处理：
        1. 页码从1-based转为0-based
        2. N/A答案清空引用
        3. 格式化为比赛提交schema
        4. 包含step_by_step_analysis
        """
        submission_answers = []
        
        for q in processed_questions:
            question_text = q.get("question_text") or q.get("question")
            kind = q.get("kind") or q.get("schema")
            value = "N/A" if "error" in q else (q.get("value") if "value" in q else q.get("answer"))
            references = q.get("references", [])
            
            answer_details_ref = q.get("answer_details", {}).get("$ref", "")
            step_by_step_analysis = None
            if answer_details_ref and answer_details_ref.startswith("#/answer_details/"):
                try:
                    index = int(answer_details_ref.split("/")[-1])
                    if 0 <= index < len(self.answer_details) and self.answer_details[index]:
                        step_by_step_analysis = self.answer_details[index].get("step_by_step_analysis")
                except (ValueError, IndexError):
                    pass
            
            # Clear references if value is N/A
            if value == "N/A":
                references = []
            else:
                # Convert page indices from one-based to zero-based (competition requires 0-based page indices, but for debugging it is easier to use 1-based)
                references = [
                    {
                        "pdf_sha1": ref["pdf_sha1"],
                        "page_index": ref["page_index"] - 1
                    }
                    for ref in references
                ]
            
            submission_answer = {
                "question_text": question_text,
                "kind": kind,
                "value": value,
                "references": references,
            }
            
            if step_by_step_analysis:
                submission_answer["reasoning_process"] = step_by_step_analysis
            
            submission_answers.append(submission_answer)
        
        return submission_answers

    def _save_progress(self, processed_questions: List[dict], output_path: Optional[str], submission_file: bool = False, pipeline_details: str = ""):
        if output_path:
            statistics = self._calculate_statistics(processed_questions)
            
            # Prepare debug content
            result = {
                "questions": processed_questions,
                "answer_details": self.answer_details,
                "statistics": statistics
            }
            output_file = Path(output_path)
            debug_file = output_file.with_name(output_file.stem + "_debug" + output_file.suffix)
            with open(debug_file, 'w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
            
            if submission_file:
                # Post-process answers for submission
                submission_answers = self._post_process_submission_answers(processed_questions)
                submission = {
                    "answers": submission_answers,
                    "details": pipeline_details
                }
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(submission, file, ensure_ascii=False, indent=2)

    def process_all_questions(self, output_path: str = 'questions_with_answers.json', submission_file: bool = False, pipeline_details: str = ""):
        result = self.process_questions_list(
            self.questions,
            output_path,
            submission_file=submission_file,
            pipeline_details=pipeline_details
        )
        return result

    def process_comparative_question(self, question: str, companies: List[str], schema: str) -> dict:
        """
        处理多公司比较类问题：
        1. 先将比较问题重写为单公司问题
        2. 并行处理每个公司
        3. 汇总结果并生成最终比较答案
        """
        # Step 1: Rephrase the comparative question
        rephrased_questions = self.openai_processor.get_rephrased_questions(
            original_question=question,
            companies=companies
        )
        
        individual_answers = {}
        aggregated_references = []
        
        # Step 2: Process each individual question in parallel
        def process_company_question(company: str) -> tuple[str, dict]:
            """Helper function to process one company's question and return (company, answer)"""
            sub_question = rephrased_questions.get(company)
            if not sub_question:
                raise ValueError(f"Could not generate sub-question for company: {company}")
            
            answer_dict = self.get_answer_for_company(
                company_name=company, 
                question=sub_question, 
                schema="number"
            )
            return company, answer_dict

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_company = {
                executor.submit(process_company_question, company): company 
                for company in companies
            }
            
            for future in concurrent.futures.as_completed(future_to_company):
                try:
                    company, answer_dict = future.result()
                    individual_answers[company] = answer_dict
                    
                    company_references = answer_dict.get("references", [])
                    aggregated_references.extend(company_references)
                except Exception as e:
                    company = future_to_company[future]
                    print(f"Error processing company {company}: {str(e)}")
                    raise
        
        # Remove duplicate references
        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("pdf_sha1"), ref.get("page_index"))
            unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())
        
        # Step 3: Get the comparative answer using all individual answers
        comparative_answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model
        )
        self.response_data = self.openai_processor.response_data
        
        comparative_answer["references"] = aggregated_references
        return comparative_answer

    def process_single_question(self, question: str, kind: str = "string"):
        """
        单条问题推理，返回结构化答案。
        kind: 支持 'string'、'number'、'boolean'、'names' 等
        """
        t0 = time.time()
        print("[计时] [单问] 开始公司名抽取 ...")
        # 公司名抽取
        if self.new_challenge_pipeline:
            extracted_companies = self._extract_companies_from_subset(question)
        else:
            extracted_companies = re.findall(r'"([^"]*)"', question)
        t1 = time.time()
        print(f"[计时] [单问] 公司名抽取耗时: {t1-t0:.2f} 秒")
        if len(extracted_companies) == 0:
            raise ValueError("No company name found in the question.")
        if len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            print("[计时] [单问] 开始检索与LLM推理 ...")
            t2 = time.time()
            answer_dict = self.get_answer_for_company(company_name=company_name, question=question, schema=kind)
            t3 = time.time()
            print(f"[计时] [单问] 检索+LLM推理耗时: {t3-t2:.2f} 秒")
            print(f"[计时] [单问] 总耗时: {t3-t0:.2f} 秒")
            return answer_dict
        else:
            print("[计时] [单问] 开始多公司比较 ...")
            t2 = time.time()
            answer_dict = self.process_comparative_question(question, extracted_companies, kind)
            t3 = time.time()
            print(f"[计时] [单问] 多公司比较耗时: {t3-t2:.2f} 秒")
            print(f"[计时] [单问] 总耗时: {t3-t0:.2f} 秒")
            return answer_dict
    