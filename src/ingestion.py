import os
import json
import time
from typing import List, Optional, Union
from pathlib import Path
from tqdm import tqdm

import faiss
import numpy as np
from tenacity import (retry, wait_fixed, stop_after_attempt,
                        retry_if_exception_type)
import dashscope
from dashscope import TextEmbedding

from src.env_loader import load_project_env, require_env

# VectorDBIngestor：向量库构建与保存工具
DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding-flash"

# 单次 embedding 请求的最大条数。**模型之间上限不同**：曾用25 时，
# qwen3.7-text-embedding-flash 直接返回
#   400 InvalidParameter: batch size is invalid, it should not be larger than 20
# 因此这里取各已知模型中最严格的那个值。可用环境变量覆盖。
MAX_BATCH_SIZE_DEFAULT = 20


def max_batch_size() -> int:
    load_project_env()
    try:
        return max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", MAX_BATCH_SIZE_DEFAULT)))
    except ValueError:
        return MAX_BATCH_SIZE_DEFAULT


def batch_delay() -> float:
    """批间间隔秒数。用于缓解 TPM 限流，可用 EMBEDDING_BATCH_DELAY 覆盖。"""
    load_project_env()
    try:
        return max(0.0, float(os.getenv("EMBEDDING_BATCH_DELAY", "1.5")))
    except ValueError:
        return 1.5


def embedding_model() -> str:
    """当前使用的 embedding 模型名。

    **文档侧与查询侧必须用同一个模型** —— 两边模型不一致时向量空间不同，
    内积结果毫无意义，而且不会报错。因此这里只保留一处定义，
    由 ``ingestion`` 与 ``retrieval`` 共同引用。

    可用环境变量 ``EMBEDDING_MODEL`` 覆盖。切换模型后**必须重建向量库**，
    否则新查询会去搜索旧模型产出的索引。
    """
    load_project_env()
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()


# 错误分类与透出的实现已提取到 src/dashscope_errors.py（ingestion / reranking /
# api_requests 三处共用）。此处转出以保持既有 import 路径可用。
from src.dashscope_errors import (  # noqa: E402
    DashScopeError,
    DashScopeThrottled,
    raise_if_api_error as _raise_if_api_error,
)


class VectorDBIngestor:
    def __init__(self):
        # 必须先加载环境变量：项目里没有任何模块在 import 阶段加载 .env，
        # 此前依赖其它模块恰好先调用过 load_dotenv()，属于顺序依赖的隐性故障。
        # 且仓库模板名为 `env`（无点号），标准 load_dotenv() 找不到它，
        # 会让 key 读成 None，最终报出与真实原因无关的 NoneType 错误。
        load_project_env()
        dashscope.api_key = require_env(
            "DASHSCOPE_API_KEY",
            hint="用于 DashScope text-embedding-v2 向量化。",
        )

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(6),
           retry=retry_if_exception_type(DashScopeThrottled), reraise=True)
    def _get_embeddings(self, text: Union[str, List[str]], model: Optional[str] = None) -> List[float]:
        # 获取文本或文本块的嵌入向量，支持重试（使用阿里云DashScope，分批处理）
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")
        
        # 保证 input 为一维字符串列表或单个字符串
        if isinstance(text, list):
            text_chunks = text
        else:
            text_chunks = [text]

        # 类型检查，确保每一项都是字符串
        if not all(isinstance(x, str) for x in text_chunks):
            raise ValueError("所有待嵌入文本必须为字符串类型！实际类型: {}".format([type(x) for x in text_chunks]))

        # 过滤空字符串
        text_chunks = [x for x in text_chunks if x.strip()]
        if not text_chunks:
            raise ValueError("所有待嵌入文本均为空字符串！")
        print('start embedding ================================')
        print('start embedding ================================')
        embeddings = []
        MAX_BATCH_SIZE = max_batch_size()
        LOG_FILE = 'embedding_error.log'
        model = model or embedding_model()
        delay = batch_delay()
        print(f'embedding model = {model}, batch_size = {MAX_BATCH_SIZE}, '
              f'{len(text_chunks)} 条, 批间间隔 {delay}s')
        for i in range(0, len(text_chunks), MAX_BATCH_SIZE):
            batch = text_chunks[i:i+MAX_BATCH_SIZE]
            if i > 0 and delay > 0:
                # 批间留出间隔，避免连续打满 TPM 限流。年报有 336 个 chunk，
                # 不做节流必然在 78% 处撞限流并从头重来。
                time.sleep(delay)
            resp = TextEmbedding.call(
                model=model,
                input=batch
            )
            _raise_if_api_error(resp, f"{len(batch)} 条文本的 embedding")
            if resp is None:
                # 未认证时 DashScope 返回 None。原先直接 `'output' in resp`
                # 会抛 "argument of type 'NoneType' is not iterable"，
                # 与真实原因（缺 key / 额度 / 参数）毫无关系。
                raise DashScopeError(
                    f"DashScope TextEmbedding 返回 None（未认证或请求被拒）。"
                    f"请检查 DASHSCOPE_API_KEY 是否有效、是否有剩余额度。"
                    f"当前 api_key={'已设置' if dashscope.api_key else '未设置'}。"
                )
            if 'output' in resp and 'embeddings' in resp['output']:
                for emb in resp['output']['embeddings']:
                    if emb['embedding'] is None or len(emb['embedding']) == 0:
                        error_text = batch[emb.text_index] if hasattr(emb, 'text_index') else None
                        with open(LOG_FILE, 'a', encoding='utf-8') as f:
                            f.write(f"DashScope返回的embedding为空，text_index={getattr(emb, 'text_index', None)}，文本内容如下：\n{error_text}\n{'-'*60}\n")
                        raise DashScopeError(f"DashScope返回的embedding为空，text_index={getattr(emb, 'text_index', None)}，文本内容已写入 {LOG_FILE}")
                    embeddings.append(emb['embedding'])
            elif 'output' in resp and 'embedding' in resp['output']:
                if resp['output']['embedding'] is None or len(resp['output']['embedding']) == 0:
                    with open(LOG_FILE, 'a', encoding='utf-8') as f:
                        f.write("DashScope返回的embedding为空，文本内容如下：\n{}\n{}\n".format(batch[0] if batch else None, '-'*60))
                    raise DashScopeError("DashScope返回的embedding为空，文本内容已写入 {}".format(LOG_FILE))
                embeddings.append(resp.output.embedding)
            else:
                raise DashScopeError(
                    f"DashScope embedding API 返回格式异常：{str(resp)[:200]}"
                )
        return embeddings

    def _create_vector_db(self, embeddings: List[float]):
        # 用faiss构建向量库，采用内积（余弦距离）
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # Cosine distance
        index.add(embeddings_array)
        return index
    
    def _process_report(self, report: dict):
        # 针对单份报告，提取文本块并生成向量库
        text_chunks = [chunk['text'] for chunk in report['content']['chunks']]

        # 绝不静默丢弃空块：FAISS 的行号必须与 chunks 下标严格一一对应，
        # 否则 chunks[index] 会取到错误的块，检索结果与引用页码全部错位。
        # 空白块应由分块阶段负责剔除（见 text_splitter.split_markdown_file）。
        empty_indices = [i for i, t in enumerate(text_chunks) if not t.strip()]
        if empty_indices:
            raise ValueError(
                f"存在 {len(empty_indices)} 个空白 chunk（首个下标 {empty_indices[0]}），"
                f"无法保证 FAISS 行号与 chunks 下标对齐；请在分块阶段剔除空白块。"
            )

        # 超长内容截断到 2048 字符
        max_len = 2048
        text_chunks = [t[:max_len] for t in text_chunks]
        embeddings = self._get_embeddings(text_chunks)
        index = self._create_vector_db(embeddings)

        # 交叉校验：向量条数必须与 chunk 数严格相等
        if index.ntotal != len(report['content']['chunks']):
            raise ValueError(
                f"向量条数({index.ntotal})与 chunk 数({len(report['content']['chunks'])})不一致，"
                f"检索结果与引用页码会错位。"
            )
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path, skip_existing: bool = True):
        # 批量处理所有报告，生成并保存faiss向量库
        all_report_paths = sorted(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        built = skipped = 0
        for report_path in tqdm(all_report_paths, desc="Processing reports for FAISS"):
            # 加载报告
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)

            # 用 metainfo['sha1'] 作为 faiss 文件名，避免中文和特殊字符
            sha1 = report_data["metainfo"].get("sha1", "")
            if not sha1:
                raise ValueError(f"分块报告 {report_path} 缺少 sha1 字段，无法保存 faiss 文件！")
            faiss_file_path = output_dir / f"{sha1}.faiss"

            # 断点续跑：已存在且条数与 chunk 数一致的索引直接跳过。
            # 建库耗时且会碰限流，中途失败后重跑不应把已完成的报告重来一遍。
            if skip_existing and faiss_file_path.exists():
                try:
                    with open(str(faiss_file_path), 'rb') as f:
                        existing = faiss.deserialize_index(np.frombuffer(f.read(), dtype=np.uint8))
                    n_chunks = len(report_data["content"]["chunks"])
                    if existing.ntotal == n_chunks and existing.d > 0:
                        skipped += 1
                        continue
                    print(f"\n索引已过期（{existing.ntotal} 向量 / {existing.d} 维 "
                          f"vs {n_chunks} chunks），重建 {report_path.name}")
                except Exception as e:  # noqa: BLE001 - 损坏的索引应当重建
                    print(f"\n索引损坏，将重建 {report_path.name}: {e}")

            index = self._process_report(report_data)
            # 原始方式：faiss.write_index(index, str(faiss_file_path))
            # Windows 下 FAISS C++ 层 fopen 不识别含中文的路径，改为先序列化再通过 Python open 写入
            arr = faiss.serialize_index(index)
            with open(str(faiss_file_path), 'wb') as f:
                f.write(arr.tobytes())
            built += 1

        print(f"\n向量库完成：新建 {built} 份，复用 {skipped} 份，共 {len(all_report_paths)} 份")