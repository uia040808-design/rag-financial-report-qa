# Qwen-Turbo API的基础限流设置为每分钟不超过500次API调用（QPM）。同时，Token消耗限流为每分钟不超过500,000 Tokens
from dataclasses import dataclass
from pathlib import Path
from pyprojroot import here
import logging
import os
import json
import pandas as pd
import shutil
import time

from src.pdf_parsing import PDFParser
from src import pdf_mineru
from src.parsed_reports_merging import PageTextPreparation
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.ingestion import BM25Ingestor
from src.questions_processing import QuestionsProcessor
from src.tables_serialization import TableSerializer
from typing import Optional

@dataclass
class PipelineConfig:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", serialized: bool = False, config_suffix: str = ""):
        # 路径配置，支持不同流程和数据目录
        self.root_path = root_path
        suffix = "_ser_tab" if serialized else ""

        self.subset_path = root_path / subset_name
        self.questions_file_path = root_path / questions_file_name
        self.pdf_reports_dir = root_path / pdf_reports_dir_name
        
        self.answers_file_path = root_path / f"answers{config_suffix}.json"       
        self.debug_data_path = root_path / "debug_data"
        self.databases_path = root_path / f"databases{suffix}"
        
        self.vector_db_dir = self.databases_path / "vector_dbs"
        self.documents_dir = self.databases_path / "chunked_reports"
        self.bm25_db_path = self.databases_path / "bm25_dbs"

        # self.parsed_reports_dirname = "01_parsed_reports"
        # self.parsed_reports_debug_dirname = "01_parsed_reports_debug"
        # self.merged_reports_dirname = f"02_merged_reports{suffix}"
        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"

        #self.parsed_reports_path = self.debug_data_path / self.parsed_reports_dirname
        #self.parsed_reports_debug_path = self.debug_data_path / self.parsed_reports_debug_dirname
        #self.merged_reports_path = self.debug_data_path / self.merged_reports_dirname
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname

@dataclass
class RunConfig:
    # 运行流程参数配置
    use_serialized_tables: bool = False
    parent_document_retrieval: bool = False
    use_vector_dbs: bool = True
    use_bm25_db: bool = False
    llm_reranking: bool = False
    llm_reranking_sample_size: int = 30
    top_n_retrieval: int = 10
    parallel_requests: int = 1 # 并行的数量，需要限制，否则会超出 DashScope 频率阈值
    pipeline_details: str = ""
    submission_file: bool = True
    full_context: bool = False
    api_provider: str = "dashscope" #openai
    # None 表示"用项目默认的生成模型"（见 env_loader.generation_model），
    # 这样换模型只需改 GENERATION_MODEL 环境变量，不必改这里的默认值。
    answering_model: Optional[str] = None  # gpt-4o-mini-2024-07-18 or "gpt-4o-2024-08-06"
    config_suffix: str = ""

class Pipeline:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", run_config: RunConfig = RunConfig()):
        # 初始化主流程，加载路径和配置
        self.run_config = run_config
        self.paths = self._initialize_paths(root_path, subset_name, questions_file_name, pdf_reports_dir_name)
        self._convert_json_to_csv_if_needed()

    def _initialize_paths(self, root_path: Path, subset_name: str, questions_file_name: str, pdf_reports_dir_name: str) -> PipelineConfig:
        """根据配置初始化所有路径"""
        return PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,
            config_suffix=self.run_config.config_suffix
        )

    def _convert_json_to_csv_if_needed(self):
        """
        检查是否存在subset.json且无subset.csv，若是则自动转换为CSV。
        """
        json_path = self.paths.root_path / "subset.json"
        csv_path = self.paths.root_path / "subset.csv"
        
        if json_path.exists() and not csv_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data)
                
                df.to_csv(csv_path, index=False)
                
            except Exception as e:
                print(f"Error converting JSON to CSV: {str(e)}")

    @staticmethod
    def download_docling_models(): 
        # 下载Docling所需模型，避免首次运行时自动下载
        logging.basicConfig(level=logging.DEBUG)
        parser = PDFParser(output_dir=here())
        parser.parse_and_export(input_doc_paths=[here() / "src/dummy_report.pdf"])

    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        """多进程并行解析PDF报告，提升处理效率
        参数：
            chunk_size: 每个worker处理的PDF数
            num_workers: 并发worker数
        """
        logging.basicConfig(level=logging.DEBUG)
        
        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))
        
        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size
        )
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def export_reports_to_markdown(self, only=None, force=False):
        """
        用 MinerU 把 pdf_reports 下的 PDF 批量转换为 Markdown，输出到
        debug_data/03_reports_markdown。

        :param only: 只处理文件名含这些子串的 PDF；None 表示全部
        :param force: True 时忽略已存在的 md，全部重新转换
        :return: {PDF 文件名: 产出的 md 路径 or None}
        """
        pdf_paths = sorted(self.paths.pdf_reports_dir.glob("*.pdf"))
        if only:
            pdf_paths = [p for p in pdf_paths if any(k in p.name for k in only)]
        if not pdf_paths:
            print(f"未在 {self.paths.pdf_reports_dir} 找到 PDF")
            return {}
        if not force:
            todo = [
                p for p in pdf_paths
                if not (self.paths.reports_markdown_path / f"{p.stem}.md").exists()
            ]
            if not todo:
                print(f"{len(pdf_paths)} 份 PDF 的 Markdown 均已存在，"
                      f"跳过转换（force=True 可强制重跑）")
                return {}

        print(f"共 {len(pdf_paths)} 个 PDF 待处理")
        return pdf_mineru.convert_pdfs(pdf_paths, self.paths.reports_markdown_path)

    def chunk_reports(self, include_serialized_tables: bool = False):
        """
        将规整后 markdown 报告分块，便于后续向量化和检索。

        分块时会借助 src/pdf_page_map.py 把 markdown 的每一行对齐回源 PDF 的
        真实页码，从而为每个 chunk 打上 page 标签，并额外产出 content.pages
        （父文档）。没有页码，父文档检索与引用出处都会失效。
        """
        text_splitter = TextSplitter()
        # 只处理 markdown 文件，输入目录为 reports_markdown_path，输出目录为 documents_dir
        print(f"开始分割 {self.paths.reports_markdown_path} 目录下的 markdown 文件...")
        # 自动传入 subset.csv 路径，便于补充 company_name 字段
        text_splitter.split_markdown_reports(
            all_md_dir=self.paths.reports_markdown_path,
            output_dir=self.paths.documents_dir,
            subset_csv=self.paths.subset_path,
            pdf_reports_dir=self.paths.pdf_reports_dir
        )
        print(f"分割完成，结果已保存到 {self.paths.documents_dir}")

    def create_vector_dbs(self):
        """从分块报告创建向量数据库"""
        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir
        
        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")
    
    def create_bm25_db(self):
        """从分块报告创建BM25数据库"""
        input_dir = self.paths.documents_dir
        output_file = self.paths.bm25_db_path
        
        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_reports(input_dir, output_file)
        print(f"BM25 database created at {output_file}")
    
    def parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        # 解析PDF报告，支持并行处理
        if parallel:
            self.parse_pdf_reports_parallel(chunk_size=chunk_size, max_workers=max_workers)

    def process_parsed_reports(self):
        """
        处理已解析的PDF报告，主要流程：
        1. 对报告进行分块
        2. 创建向量数据库
        """
        print("开始处理报告流程...")
        
        print("步骤1：报告分块...")
        self.chunk_reports()
        
        print("步骤2：创建向量数据库...")
        self.create_vector_dbs()
        
        print("报告处理流程已成功完成！")
        
    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        获取下一个可用的文件名，如果文件已存在则自动添加编号后缀。
        例如：若answers.json已存在，则返回answers_01.json等。
        """
        if not base_path.exists():
            return base_path
            
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename
            
            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        # 处理所有问题，生成答案文件
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context            
        )
        
        output_path = self._get_next_available_filename(self.paths.answers_file_path)
        
        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=self.run_config.submission_file,
            pipeline_details=self.run_config.pipeline_details
        )
        print(f"Answers saved to {output_path}")

    def answer_single_question(self, question: str, kind: str = "string"):
        """
        单条问题即时推理，返回结构化答案（dict）。
        kind: 支持 'string'、'number'、'boolean'、'names' 等
        """
        t0 = time.time()
        print("[计时] 开始初始化 QuestionsProcessor ...")
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=None,  # 单问无需文件
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=1,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context
        )
        t1 = time.time()
        print(f"[计时] QuestionsProcessor 初始化耗时: {t1-t0:.2f} 秒")
        print("[计时] 开始调用 process_single_question ...")
        answer = processor.process_single_question(question, kind=kind)
        t2 = time.time()
        print(f"[计时] process_single_question 推理耗时: {t2-t1:.2f} 秒")
        print(f"[计时] answer_single_question 总耗时: {t2-t0:.2f} 秒")
        return answer

preprocess_configs = {"ser_tab": RunConfig(use_serialized_tables=True),
                      "no_ser_tab": RunConfig(use_serialized_tables=False)}

# answering_model 留空 -> 由 env_loader.generation_model() 决定（可用
# GENERATION_MODEL 环境变量覆盖）。原先三处都硬编码 "qwen-turbo"，换模型要改
# 三处；且该模型额度耗尽时 403 只在运行时暴露。
base_config = RunConfig(
    parallel_requests=10,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + multi-doc retrieval + SO CoT",
    config_suffix="_base"
)

parent_document_retrieval_config = RunConfig(
    parent_document_retrieval=True,
    parallel_requests=20,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + multi-doc retrieval + SO CoT",
    config_suffix="_pdr"
)

## 这里
max_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=4,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + multi-doc retrieval + reranking + SO CoT",
    config_suffix="_max"
)


configs = {"base": base_config,
           "pdr": parent_document_retrieval_config,
           "max": max_config}


# 你可以直接在本文件中运行任意方法：
# python .\src\pipeline.py
# 只需取消你想运行的方法的注释即可
# 你也可以修改 run_config 以尝试不同的配置
if __name__ == "__main__":
    # 设置数据集根目录（此处以 test_set 为例）
    root_path = here() / "data" / "stock_data"
    print('root_path:', root_path)
    #print(type(root_path))
    # 初始化主流程，使用推荐的最佳配置
    pipeline = Pipeline(root_path, run_config=max_config)
    
    print('4. 将pdf转化为纯markdown文本')
    #pipeline.export_reports_to_markdown(only=['中原证券'])   # 先小批量验证

    # 5. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports
    print('5. 将规整后报告分块（页边界内切分 + 页码对齐），便于后续向量化')
    pipeline.chunk_reports() 
    
    # 6. 从分块报告创建向量数据库，输出到 databases/vector_dbs
    print('6. 从分块报告创建向量数据库，输出到 databases/vector_dbs')
    pipeline.create_vector_dbs()     
    
    # 7. 处理问题并生成答案，具体逻辑取决于 run_config
    # 默认questions.json
    print('7. 处理问题并生成答案，具体逻辑取决于 run_config')
    pipeline.process_questions() 
    
    print('完成')
