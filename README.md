# 中文金融报告 RAG 智能问答系统

基于检索增强生成（RAG）技术的中文金融报告智能问答系统。系统会对中芯国际的年报、券商研究报告、机构调研纪要等 PDF 文档进行解析、向量化与检索，并结合大模型生成带有引用出处的结构化答案。

本项目基于 [RAG Challenge 2](https://github.com/IlyaRice/RAG-Challenge-2) 竞赛获奖方案进行二次开发与优化，将原方案的英文竞赛场景改造为中文金融报告问答场景，并引入了阿里云百炼（DashScope）系列模型作为核心推理与向量化能力。

**项目背景参考：**
- 原方案介绍（俄语）：https://habr.com/ru/articles/893356/
- 原方案介绍（英语）：https://abdullin.com/ilya/how-to-build-best-rag/

## 功能特性

- **PDF 智能解析**：支持 Docling 结构化解析，也支持 MinerU 将 PDF 转换为高质量 Markdown
- **语义向量检索**：基于 FAISS 向量库 + DashScope `qwen3.7-text-embedding-flash` 向量模型，支持中文语义检索
- **关键词检索**：内置 BM25 传统检索，可进行混合检索
- **LLM 重排序**：使用大模型对初步检索结果进行二次排序，显著提升检索相关性
- **父文档检索**：检索到相关文本块后，向上回溯返回完整页面内容，保留上下文
- **结构化输出 + 思维链推理**：答案包含分步分析、推理摘要、相关页面、引用来源、最终答案等结构化字段
- **结构化输出**：DashScope 走 JSON 模式 + 提示词注入 schema + 五级解析阶梯（裸 JSON / 剥围栏 / 括号平衡扫描 / json_repair / 嵌套下钻），全程 Pydantic 校验；解析失败时返回**显式标记的降级记录**而非把原文伪装成答案
- **引用溯源**：MinerU 产出的 Markdown 不含页码，由 `src/pdf_page_map.py` 用字符 n-gram 顺序对齐把它映射回源 PDF 的真实页码（用 PDF 自带页脚交叉校验，一致率 100%）
- **引文式引用**：模型返回**原文片段**而非页码，系统用字符串匹配解析出页码，模型不参与页码运算，从源头消除"页码差一"错误
- **引用校验**：四道独立检查（页码类型归一化、文档页数范围、检索上下文、按检索排名截断），并按三类分别记录日志
- **引用评测**：`eval/` 提供带页码标注的固定集与评分脚本，量化引用精确率与召回率
- **Web 交互界面**：基于 Streamlit 的可视化问答界面，支持单问题即时问答
- **命令行工具**：基于 click 的 CLI，可独立运行流程中的每个阶段

## 技术栈

| 环节 | 技术 |
| --- | --- |
| PDF 解析 | Docling、MinerU |
| 文本分块 | 自研文本分割器（按 Token 数切分，含表格特殊处理） |
| 向量化 | 阿里云百炼 `qwen3.7-text-embedding-flash` |
| 向量检索 | FAISS（余弦内积） |
| 关键词检索 | BM25（rank-bm25） |
| 重排序 | LLM 重排（默认 qwen，可选 Jina Reranker） |
| 问答模型 | 通义千问（默认 `qwen-plus`，`GENERATION_MODEL` 可改；也支持 GPT-4o 等） |
| 交互界面 | Streamlit |

## 项目结构

```
.
├── main.py                    # CLI 入口（点击命令，支持流程各阶段独立运行）
├── app_streamlit.py           # Streamlit Web 问答界面
├── setup.py                   # Python 包配置
├── requirements.txt           # 依赖列表
├── env                        # 环境变量模板（需重命名为 .env 并填入密钥）
├── data/
│   └── stock_data/            # 中芯国际数据目录（PDF 报告、问题集、元数据）
└── src/
    ├── pipeline.py            # 主流程调度（分块、建库、问答），内置多种配置
    ├── pdf_parsing.py         # Docling PDF 结构化解析
    ├── pdf_mineru.py          # MinerU PDF 转 Markdown
    ├── parsed_reports_merging.py  # 解析结果规整为页文本
    ├── pdf_page_map.py        # Markdown 行 -> 源 PDF 真实页码的对齐
    ├── text_splitter.py       # 文本分块（页边界内切分，产出 chunks + pages）
    ├── ingestion.py           # 构建 FAISS 向量库 / BM25 索引
    ├── retrieval.py           # 向量检索 / BM25 检索 / 混合检索
    ├── reranking.py           # 检索结果重排序（LLM / Jina）
    ├── questions_processing.py # 问答主逻辑（检索、RAG 上下文、生成、引用校验）
    ├── citation_resolver.py   # 引文 -> 页码解析
    ├── structured_output.py   # 结构化输出解析、校验与降级标记
    ├── prompts.py             # 所有提示词与结构化输出 Schema
    ├── tables_serialization.py # 表格序列化（可选）
    ├── api_requests.py        # 大模型 API 调用封装
    └── api_request_parallel_processor.py  # 并发限流的批量 API 请求处理
└── eval/                      # 引用评测（标注集、评分脚本、回归门禁）
```

各模块的详细说明见 [docs/src_modules_overview.md](docs/src_modules_overview.md)。

## 评测

```bash
python -m eval.build_annotations          # 生成带页码标注的固定集
python -m eval.score_citations            # 引用精确率 / 召回率 / 兜底污染
python -m eval.score_citations --detail
python -m eval.score_structured           # 结构化输出解析与降级
```

两个脚本都支持 `--save-baseline` 建立基线，之后每次运行会与基线对比，
检出劣化时退出码非 0，可直接接入 CI。详见 [eval/README.md](eval/README.md)。

## 环境准备

### 1. 安装依赖

```bash
git clone <你的仓库地址>
cd <项目目录>
python -m venv venv
venv\Scripts\Activate.ps1   # Windows PowerShell 激活虚拟环境
pip install -r requirements.txt
```

### 2. 配置 API 密钥

将项目根目录下的 `env` 文件重命名为 `.env`，填入你自己的密钥：

```ini
# 必填：阿里云百炼（DashScope）密钥，用于向量化和问答模型
DASHSCOPE_API_KEY=sk-xxxxx

# 可选：MinerU token，仅标准 API（/api/v4）需要。
# 默认走 Agent 轻量 API（/api/v1/agent），免 Token、仅按 IP 限频，
# 因此不配置也能跑通 PDF -> Markdown。申请地址：https://mineru.net
MINERU_API_KEY=sk-xxxxx

# 可选：OpenAI 密钥，用于切换 GPT 系列模型
OPENAI_API_KEY=sk-xxxxx

# 可选：Gemini 密钥
GEMINI_API_KEY=AIzaSxxxxx

# 可选：Jina 重排密钥
JINA_API_KEY=jina_xxxxx
```

> 注：只有 `DASHSCOPE_API_KEY` 是必填（向量化与问答）。
> PDF 转 Markdown 走 MinerU 的 Agent 轻量 API，**免 Token**、仅按 IP 限频，
> 所以 `MINERU_API_KEY` 只在你显式改用标准 API 时才需要。
> 其余密钥仅在切换到对应模型时才需要。
>
> ⚠️ **请勿把真实密钥提交到 Git。** 本仓库根目录的 `env` 文件是模板，
> 但 `.gitignore` 里写的是 `env/`（带斜杠，只匹配目录），因此**这个文件本身
> 目前是被 git 跟踪的**。请把它改名为 `.env`（`.env` 已在 ignore 列表中），
> 并在模板仓库里只保留占位符。

### 2.1 PDF 转 Markdown（需要 MinerU token）

```bash
# 全量转换（跳过已存在的 .md）
python -m src.pdf_mineru

# 先用单个文件验证链路
python -m src.pdf_mineru --only 中原证券

# 强制重跑
python -m src.pdf_mineru --pdf-dir data/stock_data/pdf_reports --out data/stock_data/debug_data/03_reports_markdown
```

MinerU 走官方 batch 上传接口：申请预签名 URL → PUT 上传 → 轮询 batch 结果。
限额为单文件 ≤200MB、**≤200 页**、单次 ≤50 个文件；超页数的文件会在本地被
提前拦下并给出提示，不会浪费配额。产物除 `full.md` 外还会保存
`*.content_list.json`，其中带权威的 `page_idx`，用于页码对齐的交叉校验。

> 注意：中芯国际 2024 年年报有 **222 页**，超过单文件 200 页上限。若尚未转换，
> 需分页处理或改用其它解析路径。

### 3. 下载 Docling 模型（首次运行）

```bash
python main.py download_models
```

## 快速开始

### 方式一：一键运行完整流程

直接运行 `src/pipeline.py`，会自动执行「报告分块 -> 构建向量库 -> 处理问题」整个流程：

```bash
python -m src.pipeline
```

### 方式二：命令行分步运行

```bash
# 1. 解析 PDF 报告（支持并行）
python main.py parse-pdfs

# 2. 处理报告：分块并构建向量数据库
python main.py process-reports --config no_ser_tab

# 3. 处理问题，生成答案（max 为推荐最佳配置）
python main.py process-questions --config max
```

### 方式三：Streamlit Web 界面

```bash
streamlit run app_streamlit.py
```

打开浏览器即可进入问答界面，输入问题后点击「生成答案」，界面会展示分步推理、推理摘要、相关页面、引用来源和最终答案。

> 提示：当前知识库仅收录「中芯国际」，提问时需包含公司名才能检索，例如「中芯国际 2024 年营收情况如何？」

## 运行配置说明

系统内置了多套预置配置（见 `src/pipeline.py`），可通过 `--config` 参数切换：

| 配置名 | 说明 |
| --- | --- |
| `base` | 基础配置：多文档向量检索 + 父文档检索 |
| `pdr` | 启用父文档检索 |
| `max` | 推荐最佳配置：多文档向量检索 + 父文档检索 + LLM 重排 |

另有表格序列化配置：`ser_tab`（使用 LLM 序列化表格）/ `no_ser_tab`（不使用）。

### 模型配置

模型名不再硬编码，可用环境变量覆盖：

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `qwen3.7-text-embedding-flash` | 建库与查询的向量化（**两边必须一致**） |
| `GENERATION_MODEL` | `qwen-plus` | 问答与 LLM 重排 |
| `EMBEDDING_BATCH_SIZE` | `20` | 单次 embedding 请求条数（受模型上限约束） |
| `EMBEDDING_BATCH_DELAY` | `1.5` | 批间间隔秒数，缓解 TPM 限流 |

> 百炼的免费额度**按模型分别计算**。实测同一账号下十余个模型返回
> `403 AllocationQuota.FreeTierOnly`，但 `qwen-plus` 与
> `qwen3.7-text-embedding-flash` 仍有额度 —— 换模型往往是不需要充值就能继续的路径。
>
> 切换 `EMBEDDING_MODEL` 后**必须重建向量库**（`python -m src.pipeline`），
> 否则新查询会去搜索旧模型产出的索引，两边向量空间不一致且不报错。

## 数据集说明

当前数据目录为 `data/stock_data`，包含 9 份中芯国际相关的 PDF 文档（共 313 页）：

- **财报**：中芯国际 2024 年年度报告
- **券商研报**：上海证券、东方证券、中原证券、光大证券、兴证国际、华泰证券、国信证券对中芯国际的研究报告
- **调研纪要**：中芯国际机构调研纪要

主要数据文件：

| 文件 | 说明 |
| --- | --- |
| `pdf_reports/` | PDF 源报告 |
| `questions.json` | 测试问题集 |
| `subset.csv` | 报告元数据（sha1、文件名、公司名） |
| `databases/` | 运行后生成：分块报告（chunked_reports）与向量库（vector_dbs） |
| `debug_data/` | 运行后生成：PDF 转换的 Markdown 中间产物 |
| `answers_*.json` | 运行后生成：历史答案输出文件 |

其中 `databases/`、`debug_data/`、`answers_*.json` 均为运行产物，可通过 pipeline 重新生成，无需手动维护。

### 多文档检索

9 份文档的 `company_name` 都是「中芯国际」，因此**检索不能按公司名路由到单个文档**。
`retrieve_by_company_name` 会遍历该公司的全部索引、按相似度合并后取全局 top_n，
每个结果携带 `pdf_sha1` 与 `file_name`。

实测（`eval/score_citations.py`，46 条标注 / 63 个真值页，同为 `top_n=10` 预算）：

| 策略 | 召回 |
| --- | --- |
| 首个命中即 `break`（改造前） | **9.7%**（区间 1.6%~39.7%，取决于目录遍历顺序） |
| 多文档合并排序（当前） | **73.0%** |
| 单文档 oracle（已知答案所属 PDF） | 92.1% |

多文档距 oracle 差 19 个百分点，这部分要用更大候选预算换回来（`top_n=80` 时
93.7%）。生产配置 `llm_reranking_sample_size=30` → LLM 重排 → `top_n=6` 正落在这个区间。

> 注意对照组：oracle 假设已经知道答案在哪份 PDF 里，而原实现恰恰不具备该能力。
> 拿多文档去比 oracle 会得出「多文档反而更差」的错误结论。

### 跨文档引用

多文档下页码不唯一 —— 年报和调研纪要都有「第 5 页」。因此引用一律是
`(pdf_sha1, page_index)` 成对出现，页码由**引文解析**（原文片段字符串匹配）
得出，模型不参与任何页码运算。`eval/` 的 D 族 5 项指标全部 100%。

裸页码在多文档下若同时命中多份文档，会被**丢弃并记日志**而不是猜一份：猜错会
产出一条带着合法 sha1 与合法页码、却指向另一份 PDF 的假引用，比缺引用更难被发现。

## 免责声明

- 本项目需要自行准备各模型的 API 密钥，密钥费用由用户自行承担
- 系统生成的答案来源于对公开报告的检索与分析，仅供学习研究使用，不构成任何投资建议
- 本项目为研究性质代码，可能包含粗糙之处，请勿直接用于生产环境

## 许可证

本项目基于原 RAG Challenge 2 竞赛方案改造，遵循 [MIT License](LICENSE)。
