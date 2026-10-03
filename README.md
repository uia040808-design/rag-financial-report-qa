# 中文金融报告 RAG 智能问答系统

基于检索增强生成（RAG）技术的中文金融报告智能问答系统。系统会对中芯国际的年报、券商研究报告、机构调研纪要等 PDF 文档进行解析、向量化与检索，并结合大模型生成带有引用出处的结构化答案。

本项目基于 [RAG Challenge 2](https://github.com/IlyaRice/RAG-Challenge-2) 竞赛获奖方案进行二次开发与优化，将原方案的英文竞赛场景改造为中文金融报告问答场景，并引入了阿里云百炼（DashScope）系列模型作为核心推理与向量化能力。

**项目背景参考：**
- 原方案介绍（俄语）：https://habr.com/ru/articles/893356/
- 原方案介绍（英语）：https://abdullin.com/ilya/how-to-build-best-rag/

## 功能特性

- **PDF 智能解析**：MinerU 转 Markdown（默认走 Agent 免 Token 接口）。Docling 解析代码保留但**不在主流程上** —— `Pipeline.parse_pdf_reports` 引用的 `parsed_reports_path` 属性在 `PipelineConfig` 里被注释掉，调用即 `AttributeError`；当前可用入口只有 MinerU
- **语义向量检索**：基于 FAISS 向量库 + DashScope `qwen3.7-text-embedding-flash` 向量模型，支持中文语义检索
- **多文档检索**：同一公司的全部文档合并排序，而非按公司名路由到单个文档（见下文实测）
- **关键词检索**：BM25 组件已实现（`BM25Ingestor` / `BM25Retriever`）但**未启用** —— 默认链路不建索引（`databases/` 下无 `bm25_dbs` 产物）、`BM25Retriever` 无任何调用方，且分词是 `str.split()`（对中文无效，需 jieba）。**不构成本系统当前能力**，实际检索是「向量检索 + LLM 重排 + 父文档检索」
- **LLM 重排序**：用大模型对检索结果二次排序；模型返回的分数被真正解析与使用，解析失败时降级为纯向量排序
- **父文档检索**：检索到相关文本块后，向上回溯返回完整页面内容，保留上下文
- **思维链推理**：答案包含分步分析、推理摘要、引用原文、最终答案等结构化字段
- **结构化输出**：DashScope 走 JSON 模式 + 提示词注入 schema + 五级解析阶梯（裸 JSON / 剥围栏 / 括号平衡扫描 / json_repair / 嵌套下钻），全程 Pydantic 校验；解析失败时返回**显式标记的降级记录**而非把原文伪装成答案
- **引用溯源**：MinerU 产出的 Markdown 不含页码，由 `src/pdf_page_map.py` 用字符 n-gram 顺序对齐把它映射回源 PDF 的真实页码。9/9 文档对齐成功；带可识别页脚的 227 页共校验 1752 行，与页脚所印页码 100% 一致（年报 1714 行、兴证国际 38 行；调研纪要无页脚，该份无此旁证）
- **引文式引用**：模型返回**原文片段**而非页码，系统用字符串匹配解析出页码，模型不参与页码运算，从源头消除"页码差一"错误（实测：差一引用原先拦截率 0，现该类错误无法表示）
- **跨文档引用**：引用一律是 `(pdf_sha1, page_index)` 成对出现 —— 多文档下页码不唯一，年报和调研纪要都有「第 5 页」
- **引用校验**：五道独立检查（页码类型归一化、逐文档页数范围、检索上下文、按检索排名截断、无页码剔除），按四类分别记录日志；页码兜底默认关闭
- **引用评测**：`eval/` 提供带页码标注的固定集（46 条 / 覆盖 9 份文档）与评分脚本，带基线回归门禁
- **Web 交互界面**：基于 Streamlit 的可视化问答界面，支持单问题即时问答
- **命令行工具**：基于 click 的 CLI，可独立运行流程中的每个阶段

## 技术栈

| 环节 | 技术 |
| --- | --- |
| PDF 解析 | MinerU（主流程）。Docling 保留但未接入，见「功能特性」 |
| 文本分块 | 自研文本分割器，**按行**切分（30 行/块、5 行重叠），在页边界内切分保证块不跨页 —— 父文档回溯的前提。按 token 的 300/50 路径已实现，未启用 |
| 向量化 | 阿里云百炼 `qwen3.7-text-embedding-flash` |
| 向量检索 | FAISS（余弦内积） |
| 关键词检索 | BM25（rank-bm25）——已实现**未启用**，见「功能特性」 |
| 重排序 | LLM 重排（默认 qwen）。`JinaReranker` 类已实现但**从未实例化**，未接入 |
| 问答模型 | 通义千问（默认 `qwen-plus`，`GENERATION_MODEL` 可改；也支持 GPT-4o 等） |
| 交互界面 | Streamlit |

## 项目结构

```
.
├── main.py                    # CLI 入口（点击命令，支持流程各阶段独立运行）
├── app_streamlit.py           # Streamlit Web 问答界面
├── setup.py                   # Python 包配置
├── requirements.txt           # 依赖列表
├── env                        # 环境变量模板（已取消 git 跟踪；.env 与 env 两种命名都支持）
├── data/
│   └── stock_data/            # 中芯国际数据目录（PDF 报告、问题集、元数据）
├── src/
│   ├── pipeline.py            # 主流程调度（分块、建库、问答），内置多种配置
│   ├── pdf_parsing.py         # Docling PDF 结构化解析
│   ├── pdf_mineru.py          # MinerU PDF 转 Markdown
│   ├── parsed_reports_merging.py  # 解析结果规整为页文本
│   ├── pdf_page_map.py        # Markdown 行 -> 源 PDF 真实页码的对齐
│   ├── text_splitter.py       # 文本分块（页边界内切分，产出 chunks + pages）
│   ├── ingestion.py           # 构建 FAISS 向量库（BM25Ingestor 已实现，未启用）
│   ├── retrieval.py           # 向量检索 + LLM 重排 + 父文档回溯（BM25Retriever 未接入）
│   ├── reranking.py           # 检索结果 LLM 重排序（JinaReranker 已实现，未接入）
│   ├── questions_processing.py # 问答主逻辑（检索、RAG 上下文、生成、引用校验）
│   ├── citation_resolver.py   # 引文 -> 页码解析
│   ├── structured_output.py   # 结构化输出解析、校验与降级标记
│   ├── prompts.py             # 所有提示词与结构化输出 Schema
│   ├── tables_serialization.py # 表格序列化（TableSerializer 已实现，主流程未调用）
│   ├── api_requests.py        # 大模型 API 调用封装
│   └── api_request_parallel_processor.py  # 并发限流的批量 API 请求处理
└── eval/                      # 引用评测（标注集、评分脚本、回归门禁）
```

## 评测

```bash
python -m eval.build_annotations          # 生成带页码标注的固定集
python -m eval.score_citations            # 引用精确率 / 召回率 / 兜底污染
python -m eval.score_citations --detail
python -m eval.score_structured           # 结构化输出解析与降级
```

两个脚本都支持 `--save-baseline` 建立基线，之后每次运行会与基线对比，
检出劣化时退出码非 0，可直接接入 CI。详见 [eval/README.md](eval/README.md)。

标注集：46 条 / 覆盖全部 9 份文档 / 真值页合计 63 页。页码由锚串在原文里
检索得到，不经模型推断；锚串需通过**全局区分度筛查**（长度、纯数字锚串长度、
全语料命中页数上限）—— 只按单文档计数不够，种子 `'22.5%'` 归一化后是 `'225'`，
在 9 份文档里命中 31 页，逐文档检查却全部通过。详见 eval/README.md。

当前基线（摘要）：

| 指标 | 修复前 | 当前 |
| --- | --- | --- |
| 最终引用精确率 | 59.1% | **100%** |
| 兜底页/标注 | 2.86 | **0.00** |
| 多文档召回 | 9.7% | **73.0%** |
| 越界剔除（full_context 下） | 无此项（会放行） | **100%** |
| 结构化输出误放率 | — | **0.0%** |

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

编辑项目根目录下的 `env`（或 `.env`，两种命名都支持，`src/env_loader.py`
按「真实环境变量 > `.env` > `env`」的优先级合并），填入你自己的密钥：

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

# 可选：Jina 重排密钥（JinaReranker 未接入主链路，配了暂不会被使用）
JINA_API_KEY=jina_xxxxx
```

> 注：只有 `DASHSCOPE_API_KEY` 是必填（向量化与问答）。
> PDF 转 Markdown 走 MinerU 的 Agent 轻量 API，**免 Token**、仅按 IP 限频，
> 所以 `MINERU_API_KEY` 只在你显式改用标准 API 时才需要。
> 其余密钥仅在切换到对应模型时才需要。
>
> ⚠️ **请勿把真实密钥提交到 Git。** `env` 文件已从 git 跟踪中移除
> （`git rm --cached env`），并在 `.gitignore` 里同时覆盖了文件与目录两种形式
> —— 原先只写 `env/`（带斜杠，只匹配目录），导致这个文件本身仍被跟踪。
> 建议改名为 `.env`（`.env` 已在 ignore 列表中）。
>
> 历史上 `src/pdf_mineru.py` 曾硬编码过一个**真实** token，且代码注释声称那只是
> 占位符。该 token 已从代码与 git 历史中移除，但**仍应视为已泄漏**，建议去
> [mineru.net](https://mineru.net) 控制台吊销重发。`get_api_key()` 现在只读
> 环境变量、不再有回退到模块常量的路径 —— 那个回退本身就是泄漏成因。

### 2.1 PDF 转 Markdown（默认无需 token）

```bash
# 全量转换（跳过已存在的 .md）
python -m src.pdf_mineru

# 先用单个文件验证链路
python -m src.pdf_mineru --only 中原证券

# 强制重跑
python -m src.pdf_mineru --pdf-dir data/stock_data/pdf_reports --out data/stock_data/debug_data/03_reports_markdown

# 显式走标准 API（需 MINERU_API_KEY；标准 API 当前不可用，见下）
python -m src.pdf_mineru --api standard
```

默认走 **Agent 轻量接口**（`/api/v1/agent`）：免 Token、仅按 IP 限频。
超 200 页的文档（如 222 页的年报）会在本地切分后逐份提交。

> **标准 API（`/api/v4`）当前不可用**：实测任务能被受理（`code=0`、返回
> `task_id`），但 `state` 永远停在 `pending`、`err_msg` 为空，原因未知。
> 已排除 token 无效、上传失败、端点错误、参数与配额等因素。
> 因此默认路径是 Agent API，`convert_pdfs`（标准 API）保留但需显式指定。
> 四个待处理的 `batch_id` 可用 `--resume` 续跑。

MinerU 产物除 `full.md` 外还会保存 `*.content_list.json`，其中带权威的
`page_idx`，用于页码对齐的交叉校验。

### 3. 下载 Docling 模型（可选，非必需）

主流程走 MinerU，**不需要这一步**，可跳过。仅当你准备恢复已停用的 Docling 解析链路时才需要：

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

# 单问模式：直接回答一个问题并打印结果（用于调试与验证）
python main.py process-questions --question "中芯国际2024年的产能利用率是多少？"
python main.py process-questions --question "..." --kind number
```

> `--config` 的可选值由 `configs` 字典**动态生成**，避免声明与实现不一致。
> 注意 `process-reports` 与 `process-questions` 的选项集合不同：
> 前者是预处理配置（`no_ser_tab` / `ser_tab`），后者是问答配置（`base` / `max` / `pdr`）。

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
| `base` | 基础配置：多文档向量检索（父文档检索未启用，`parent_document_retrieval` 默认 False） |
| `pdr` | 启用父文档检索 |
| `max` | 推荐最佳配置：多文档向量检索 + 父文档检索 + LLM 重排 |

另有预处理配置 `ser_tab` / `no_ser_tab`。⚠️ **当前 `ser_tab` 是空转的**：它只把输出目录改名为 `databases_ser_tab`，`process_parsed_reports()` 并不读 `use_serialized_tables`，分块与建库逻辑完全相同 —— `TableSerializer` 已导入但从未实例化，`main.py serialize-tables` 命令则直接报错。不建议使用。

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

当前数据目录为 `data/stock_data`，包含 9 份中芯国际相关的 PDF 文档（共 315 页）：

- **财报**：中芯国际 2024 年年度报告
- **券商研报**：上海证券、东方证券、中原证券、光大证券、兴证国际、华泰证券、国信证券对中芯国际的研究报告
- **调研纪要**：中芯国际机构调研纪要

主要数据文件：

| 文件 | 说明 |
| --- | --- |
| `pdf_reports/` | PDF 源报告（9 份，315 页） |
| `questions.json` | 测试问题集 |
| `subset.csv` | 报告元数据（sha1、文件名、公司名、来源类型、PDF 页数），9 行真实 sha1 |
| `databases/` | 运行后生成：分块报告（chunked_reports）与向量库（vector_dbs） |
| `debug_data/` | 运行后生成：PDF 转换的 Markdown 中间产物 |
| `answers_*.json` | 运行后生成：历史答案输出文件 |

其中 `databases/`、`debug_data/`、`answers_*.json` 均为运行产物，可通过 pipeline 重新生成，无需手动维护。

### 已知限制

- **扫描页的 PDF 文本层不完整**：机构调研纪要 PDF 有 6/22 页文本层近乎空白
  （需 OCR）。页码对齐用的是 MinerU 的 OCR 结果，内容正确；但用文本层做
  独立核验时这些页无法验证（`eval/` 已把它们单列为「无法核验」而非记为失败）。
- **公司名匹配在多公司扩展时需改法**：`_reports_for_company` 用
  「`company_name` 全等 **或** 公司名是 `file_name` 的子串」。当前语料只有一家
  公司故无影响，但若同时存在「中芯国际」与「中芯国际华虹」这类互为子串的公司名，
  后者会被前者误命中，答案跨公司串联且不报错。改法见代码注释。
- **多文档比单文档 oracle 差 19pt**：这是多文档检索的固有代价 —— 它要先花
  候选名额证明「该看哪份文档」。生产用「30 个候选 → LLM 重排 → 10 页」吸收。

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
93.7%）。生产配置 `llm_reranking_sample_size=30` → LLM 重排 → `top_n=10`
（`max_config.top_n_retrieval`）正落在这个区间。

> **为什么不把候选池加大到 60？** 实测加倍候选的召回增益上界只有 3 个真值页
> （85.7% → 90.5%），且这 3 个页在 60 名候选里的排名分别是 **43 / 55 / 60**。
> 重排只能在候选池内挑选，把第 43 名提到前 10 名要越过前面 30 个更强候选，
> 实际增益大概率是 0。代价则是批次数 3 → 6、重排耗时从实测约 105 秒涨到
> 约 210 秒。

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
