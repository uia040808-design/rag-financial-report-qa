import os
import re
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
import requests
import src.prompts as prompts
from concurrent.futures import ThreadPoolExecutor
from src.env_loader import load_project_env, generation_model
from src.dashscope_errors import (
    DashScopeError,
    DashScopeThrottled,
    raise_if_api_error,
)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# JinaReranker：基于Jina API的重排器，适用于多语言场景
class JinaReranker:
    def __init__(self):
        # 初始化Jina重排API地址和请求头
        self.url = 'https://api.jina.ai/v1/rerank'
        self.headers = self.get_headers()
        
    def get_headers(self):
        # 加载Jina API密钥，组装请求头
        load_project_env()
        jina_api_key = os.getenv("JINA_API_KEY")    
        headers = {'Content-Type': 'application/json',
                   'Authorization': f'Bearer {jina_api_key}'}
        return headers
    
    def rerank(self, query, documents, top_n = 10):
        # 调用Jina API进行重排，返回top_n相关文档
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": documents
        }

        response = requests.post(url=self.url, headers=self.headers, json=data)

        return response.json()

# LLMReranker：基于大模型的重排器，支持单条和批量重排
# 优先匹配小数（0.85 / .9 / 1.0），再考虑独立的整数 0 或 1
_SCORE_DEC_RE = re.compile(r"(?<![\d.])\d*\.\d+")
# 整数分数必须"独立"：前面不能是数字/小数/左方括号（排除 [1] 这类块号），
# 后面不能是数字/小数/右括号/冒号/等号（排除 "Block 1:" 这类标签）
_SCORE_INT_RE = re.compile(r"(?<![\d.\[])[01](?![\d.\]):=])")


def _parse_relevance_scores(text: str, expected: int) -> Optional[List[float]]:
    """从模型输出里解析相关性分数。

    历史实现直接返回 ``relevance_score: 0.0`` 硬编码值，于是
    ``combined_score = llm_weight*0.0 + vector_weight*distance``，
    **LLM 重排退化成恒等变换** —— README 里"显著提升检索相关性"是虚的，
    而且每次重排都在白付 token。

    模型输出通常形如 ``[1] 0.8 理由...`` 或 ``Block 1: 0.7``。这里按**出现
    位置**收集小数与独立整数，再取前 ``expected`` 个。块号（``[1]``、
    ``Block 1:``）会被正则排除，不会被误当成 0 分或满分。

    数量不足时返回 None，由调用方降级为纯向量排序 —— 宁可不用 LLM 分数，
    也不能用伪造的 0.0 冒充一个"LLM 认为不相关"的判断。
    """
    if expected <= 0:
        return []

    candidates: List[Tuple[int, float]] = []
    for m in _SCORE_DEC_RE.finditer(text):
        try:
            v = float(m.group(0))
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            candidates.append((m.start(), v))
    for m in _SCORE_INT_RE.finditer(text):
        candidates.append((m.start(), float(m.group(0))))

    candidates.sort(key=lambda x: x[0])
    values = [v for _, v in candidates]
    if len(values) < expected:
        return None
    return values[:expected]


def _parse_single_score(text: str) -> Optional[float]:
    values = _parse_relevance_scores(text, 1)
    return values[0] if values else None


class LLMReranker:
    def __init__(self, provider: str = "dashscope"):
        # 支持 openai/dashscope，默认 dashscope
        self.provider = provider.lower()
        self.llm = self.set_up_llm()
        self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks
      
    def set_up_llm(self):
        # 根据 provider 初始化 LLM 客户端
        load_project_env()
        if self.provider == "openai":
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "dashscope":
            import dashscope
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            return dashscope
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    @staticmethod
    @retry(wait=wait_exponential(multiplier=3, min=3, max=30),
           stop=stop_after_attempt(4),
           retry=retry_if_exception_type(DashScopeThrottled),
           reraise=True)
    def _call_dashscope_generation(**kwargs):
        """统一的重排调用入口，只对限流退避重试。

        重排是并发问答路径上最密集的 API 调用（每题 top_n/documents_batch_size
        次），实测会触发 RateQuota。这类错误重试即可恢复，参数错误则重试无益
        —— 由 :class:`DashScopeThrottled` 与 :class:`DashScopeError` 的区分决定。
        """
        import dashscope
        return dashscope.Generation.call(**kwargs)
    
    def get_rank_for_single_block(self, query, retrieved_document):
        # 针对单个文本块，调用LLM进行相关性评分
        user_prompt = f'/nHere is the query:/n"{query}"/n/nHere is the retrieved text block:/n"""/n{retrieved_document}/n"""/n'
        if self.provider == "openai":
            completion = self.llm.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt_rerank_single_block},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=self.schema_for_single_block
            )
            response = completion.choices[0].message.parsed
            response_dict = response.model_dump()
            return response_dict
        elif self.provider == "dashscope":
            messages = [
                {"role": "system", "content": self.system_prompt_rerank_single_block},
                {"role": "user", "content": user_prompt},
            ]
            rsp = self._call_dashscope_generation(
                model=generation_model(),
                messages=messages,
                temperature=0,
                result_format='message'
            )
            # 失败时 rsp 是 dict 但 output 为 None（限流 / 额度 / 参数）。
            # 不做这层检查就会在下一行炸出 `argument of type 'NoneType' is
            # not iterable`，把限流误报成代码 bug —— 实测并发问答时 5/5 全挂
            # 都是这一个原因。
            raise_if_api_error(rsp, "单块重排（DashScope）")
            if 'output' in rsp and 'choices' in rsp['output']:
                content = rsp['output']['choices'][0]['message']['content']
                score = _parse_single_score(content)
                if score is None:
                    print(f"[Rerank] 未能从模型输出解析相关性分数，降级为纯向量排序: "
                          f"{content[:100]!r}")
                    return {"relevance_score": None, "reasoning": content}
                return {"relevance_score": score, "reasoning": content}
            else:
                raise RuntimeError(f"DashScope返回格式异常: {rsp}")
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        # 针对多个文本块，批量调用LLM进行相关性评分
        formatted_blocks = "\n\n---\n\n".join([f'Block {i+1}:\n\n"""\n{text}\n"""' for i, text in enumerate(retrieved_documents)])
        user_prompt = (
            f"Here is the query: \"{query}\"\n\n"
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order."
        )
        if self.provider == "openai":
            completion = self.llm.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=self.schema_for_multiple_blocks
            )
            response = completion.choices[0].message.parsed
            response_dict = response.model_dump()
            return response_dict
        elif self.provider == "dashscope":
            messages = [
                {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                {"role": "user", "content": user_prompt},
            ]
            rsp = self._call_dashscope_generation(
                model=generation_model(),
                messages=messages,
                temperature=0,
                result_format='message'
            )
            # 健壮性检查，防止 rsp 为 None 或非 dict
            if not rsp or not isinstance(rsp, dict):
                raise DashScopeError(f"DashScope重排调用返回 None 或非 dict：{rsp}")
            # 失败时 rsp 是 dict 但 output 为 None（限流 / 额度 / 参数）。不做这层
            # 检查就会在下一行炸出 `argument of type 'NoneType' is not iterable`，
            # 把限流误报成代码 bug —— 实测并发问答 5/5 全挂都是这一个原因。
            raise_if_api_error(rsp, f"{len(retrieved_documents)} 块批量重排（DashScope）")
            if 'output' in rsp and 'choices' in rsp['output']:
                content = rsp['output']['choices'][0]['message']['content']
                n = len(retrieved_documents)
                scores = _parse_relevance_scores(content, n)
                if scores is None:
                    # 解析失败不返回伪造的 0.0 —— 那会让重排静默变成恒等变换。
                    # 返回 None 让 rerank_documents 明确降级并打日志。
                    print(f"[Rerank] 未能解析 {n} 个相关性分数，降级为纯向量排序: "
                          f"{content[:100]!r}")
                    return {"block_rankings": None, "raw": content}
                return {"block_rankings": [
                    {"relevance_score": s, "reasoning": content} for s in scores
                ]}
            else:
                raise RuntimeError(f"DashScope返回格式异常: {rsp}")
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        """
        使用多线程并行方式对多个文档进行重排。
        结合向量相似度和LLM相关性分数，采用加权平均融合。
        参数：
            query: 查询语句
            documents: 待重排的文档列表，每个元素需包含'text'和'distance'
            documents_batch_size: 每批送入LLM的文档数
            llm_weight: LLM分数权重（0-1），其余为向量分数权重
        返回：
            按融合分数降序排序的文档列表
        """
        # 按batch分组
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight
        llm_used = True

        if documents_batch_size == 1:
            def process_single_doc(doc):
                # 单文档重排
                ranking = self.get_rank_for_single_block(query, doc['text'])
                score = ranking.get("relevance_score")

                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = score
                if score is None:
                    # 解析失败：只用向量分，绝不用 0.0 冒充一个 LLM 分数
                    doc_with_score["combined_score"] = doc['distance']
                else:
                    doc_with_score["combined_score"] = round(
                        llm_weight * score + vector_weight * doc['distance'], 4)
                return doc_with_score

            # 多线程并行处理，max_workers=1 保证 dashscope LLM 串行调用，避免 QPS 超限
            with ThreadPoolExecutor(max_workers=1) as executor:
                all_results = list(executor.map(process_single_doc, documents))
                llm_used = all(d.get("relevance_score") is not None for d in all_results)

        else:
            def process_batch(batch):
                # 批量重排
                texts = [doc['text'] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                results = []
                block_rankings = rankings.get('block_rankings')

                if block_rankings is None:
                    # LLM 侧解析失败 -> 整批降级为纯向量排序
                    for doc in batch:
                        d = doc.copy()
                        d["relevance_score"] = None
                        d["combined_score"] = doc['distance']
                        results.append(d)
                    return results, False

                missing = 0
                if block_rankings is not None and len(block_rankings) < len(batch):
                    # 只在前 len(block_rankings) 个分数有对应的前提下按位置 zip；
                    # 多出来的块一律按"未给出分数"处理，见下方。
                    missing = len(batch) - len(block_rankings)
                    print(f"\nWarning: Expected {len(batch)} rankings but got "
                          f"{len(block_rankings)}; {missing} block(s) will fall back "
                          f"to pure-vector scoring.")
                    for i in range(len(block_rankings), len(batch)):
                        doc = batch[i]
                        print(f"  Missing ranking: page {doc.get('page', 'unknown')} "
                              f"(sha1={str(doc.get('pdf_sha1'))[:8]})")
                        print(f"  Text preview: {doc['text'][:100]}...\n")

                for i, doc in enumerate(batch):
                    doc_with_score = doc.copy()
                    rank = block_rankings[i] if (
                        block_rankings is not None and i < len(block_rankings)) else None
                    score = rank.get("relevance_score") if rank else None
                    doc_with_score["relevance_score"] = score
                    if score is None:
                        # 未给出分数 -> 只用向量分。
                        #
                        # 原实现在这里补 `relevance_score: 0.0`。而 0.0 在重排提示词
                        # 里的含义是"完全无关"，于是 combined_score = 0.3*distance，
                        # 真实得分 0.9 的块是 0.63+ —— **缺分数的块几乎必然被挤出
                        # top_k**，且日志只有一行 Warning。这与之前修掉的"硬编码 0.0
                        # 让重排退化成恒等变换"是同一类错误，只是方向相反：
                        # 那次是所有块都得 0.0，这次是漏判的块得 0.0。
                        doc_with_score["combined_score"] = doc['distance']
                    else:
                        doc_with_score["combined_score"] = round(
                            llm_weight * score + vector_weight * doc['distance'], 4)
                    results.append(doc_with_score)
                return results, True

            # 多线程并行处理，max_workers=1 保证 dashscope LLM 串行调用，避免 QPS 超限
            with ThreadPoolExecutor(max_workers=1) as executor:
                batch_results = list(executor.map(process_batch, doc_batches))

            # 扁平化结果
            all_results = []
            for results, used in batch_results:
                all_results.extend(results)
                llm_used = llm_used and used
        
        # 排序前明确告知调用方 LLM 分数是否真的参与了排序
        if not llm_used:
            print("[Rerank] 本次未取到任何 LLM 相关性分数，排序结果等价于纯向量排序。")
        else:
            scored = sum(1 for d in all_results if d.get("relevance_score") is not None)
            print(f"[Rerank] LLM 分数参与了 {scored}/{len(all_results)} 个文档的排序。")

        # 按融合分数降序排序
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results
