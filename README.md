# 中文金融报告 RAG 智能问答系统

基于检索增强生成（RAG）技术的中文金融报告智能问答系统。系统会对中芯国际的年报、券商研究报告、机构调研纪要等 PDF 文档进行解析、向量化与检索，并结合大模型生成带有引用出处的结构化答案。

本项目基于 [RAG Challenge 2](https://github.com/IlyaRice/RAG-Challenge-2) 竞赛获奖方案进行二次开发与优化，将原方案的英文竞赛场景改造为中文金融报告问答场景，并引入了阿里云百炼（DashScope）系列模型作为核心推理与向量化能力。

**项目背景参考：**
- 原方案介绍（俄语）：https://habr.com/ru/articles/893356/
- 原方案介绍（英语）：https://abdullin.com/ilya/how-to-build-best-rag/

## 功能特性

- **PDF 智能解析**：支持 Docling 结构化解析，也支持 MinerU 将 PDF 转换为高质量 Markdown
- **语义向量检索**：基于 FAISS 向量库 + DashScope text-embedding-v2 向量模型，支持中文语义检索
- **关键词检索**：内置 BM25 传统检索，可进行混合检索
- **LLM 重排序**：使用大模型对初步检索结果进行二次排序，显著提升检索相关性
- **父文档检索**：检索到相关文本块后，向上回溯返回完整页面内容，保留上下文
- **结构化输出 + 思维链推理**：答案包含分步分析、推理摘要、相关页面、引用来源、最终答案等结构化字段
- **引用校验**：自动校验并过滤大模型虚构的页码引用，只保留真实存在的出处
- **Web 交互界面**：基于 Streamlit 的可视化问答界面，支持单问题即时问答
- **命令行工具**：基于 click 的 CLI，可独立运行流程中的每个阶段

## 技术栈

| 环节 | 技术 |
| --- | --- |
| PDF 解析 | Docling、MinerU |
| 文本分块 | 自研文本分割器（按 Token 数切分，含表格特殊处理） |
| 向量化 | 阿里云百炼 text-embedding-v2 |
| 向量检索 | FAISS（余弦内积） |
| 关键词检索 | BM25（rank-bm25） |
| 重排序 | LLM 重排（默认 qwen，可选 Jina Reranker） |
| 问答模型 | 通义千问 qwen-turbo（可切换 GPT-4o 等） |
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
    ├── text_splitter.py       # 文本分块
    ├── ingestion.py           # 构建 FAISS 向量库 / BM25 索引
    ├── retrieval.py           # 向量检索 / BM25 检索 / 混合检索
    ├── reranking.py           # 检索结果重排序（LLM / Jina）
    ├── questions_processing.py # 问答主逻辑（检索、RAG 上下文、生成、引用校验）
    ├── prompts.py             # 所有提示词与结构化输出 Schema
    ├── tables_serialization.py # 表格序列化（可选）
    ├── api_requests.py        # 大模型 API 调用封装
    └── api_request_parallel_processor.py  # 并发限流的批量 API 请求处理
```

各模块的详细说明见 [docs/src_modules_overview.md](docs/src_modules_overview.md)。

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

# 可选：OpenAI 密钥，用于切换 GPT 系列模型
OPENAI_API_KEY=sk-xxxxx

# 可选：Gemini 密钥
GEMINI_API_KEY=AIzaSxxxxx

# 可选：Jina 重排密钥
JINA_API_KEY=jina_xxxxx
```

> 注：DASHSCOPE_API_KEY 是核心配置，必须填写。其余密钥仅在切换到对应模型时才需要。请勿将真实的密钥文件提交到 Git 仓库。

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
| `base` | 基础配置：向量检索 + 通义千问（qwen-turbo） |
| `pdr` | 启用父文档检索（代码中模型名配置为 GPT-4o，需配合 OpenAI 提供商使用） |
| `max` | 推荐最佳配置：向量检索 + 父文档检索 + LLM 重排 + 通义千问（qwen-turbo） |

另有表格序列化配置：`ser_tab`（使用 LLM 序列化表格）/ `no_ser_tab`（不使用）。

## 数据集说明

当前数据目录为 `data/stock_data`，包含 8 份中芯国际相关的 PDF 文档：

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

## 免责声明

- 本项目需要自行准备各模型的 API 密钥，密钥费用由用户自行承担
- 系统生成的答案来源于对公开报告的检索与分析，仅供学习研究使用，不构成任何投资建议
- 本项目为研究性质代码，可能包含粗糙之处，请勿直接用于生产环境

## 许可证

本项目基于原 RAG Challenge 2 竞赛方案改造，遵循 [MIT License](LICENSE)。
