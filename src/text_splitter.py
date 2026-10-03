import json
import tiktoken
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import os

from src.pdf_page_map import align_markdown_to_pages, resolve_pdf_for_markdown

# 文本分块工具类，支持按页分块、表格插入、token统计等
class TextSplitter():
    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """按页分组已序列化表格，便于后续插入到对应页面分块中"""
        tables_by_page = {}
        for table in tables:
            if 'serialized' not in table:
                continue
                
            page = table['page']
            if page not in tables_by_page:
                tables_by_page[page] = []
            
            table_text = "\n".join(
                block["information_block"] 
                for block in table["serialized"]["information_blocks"]
            )
            
            tables_by_page[page].append({
                "page": page,
                "text": table_text,
                "table_id": table["table_id"],
                "length_tokens": self.count_tokens(table_text)
            })
            
        return tables_by_page

    def _split_report(self, file_content: Dict[str, any], serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]:
        """将报告按页分块，保留markdown表格内容，可选插入序列化表格块。"""
        chunks = []
        chunk_id = 0
        
        tables_by_page = {}
        if serialized_tables_report_path is not None:
            # 加载序列化表格，按页分组
            with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(parsed_report.get('tables', []))
        
        for page in file_content['content']['pages']:
            # 普通文本分块
            page_chunks = self._split_page(page)
            for chunk in page_chunks:
                chunk['id'] = chunk_id
                chunk['type'] = 'content'
                chunk_id += 1
                chunks.append(chunk)
            
            # 插入序列化表格分块
            if tables_by_page and page['page'] in tables_by_page:
                for table in tables_by_page[page['page']]:
                    table['id'] = chunk_id
                    table['type'] = 'serialized_table'
                    chunk_id += 1
                    chunks.append(table)
        
        file_content['content']['chunks'] = chunks
        return file_content

    def count_tokens(self, string: str, encoding_name="o200k_base"):
        # 统计字符串的token数，支持自定义编码
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(string)
        token_count = len(tokens)
        return token_count

    def _split_page(self, page: Dict[str, any], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Dict[str, any]]:
        """将单页文本分块，保留原始markdown表格。"""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(page['text'])
        chunks_with_meta = []
        for chunk in chunks:
            chunks_with_meta.append({
                "page": page['page'],
                "length_tokens": self.count_tokens(chunk),
                "text": chunk
            })
        return chunks_with_meta

    #对 json 文件分块，输出还是 json
    def split_all_reports(self, all_report_dir: Path, output_dir: Path, serialized_tables_dir: Optional[Path] = None):
        """
        批量处理目录下所有报告（json文件），对每个报告进行文本分块，并输出到目标目录。
        如果提供了序列化表格目录，会尝试将表格内容插入到对应页面的分块中。
        主要用于后续向量化和检索的预处理。
        参数：
            all_report_dir: 存放待处理报告json的目录
            output_dir: 分块后输出的目标目录
            serialized_tables_dir: （可选）存放序列化表格的目录
        """
        # 获取所有报告文件路径
        all_report_paths = list(all_report_dir.glob("*.json"))
        
        # 遍历每个报告文件
        for report_path in all_report_paths:
            serialized_tables_path = None
            # 如果提供了表格序列化目录，查找对应表格文件
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                if not serialized_tables_path.exists():
                    print(f"警告：未找到 {report_path.name} 的序列化表格报告")
                
            # 读取报告内容
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
                
            # 分块处理，插入表格分块（如有）
            updated_report = self._split_report(report_data, serialized_tables_path)
            # 确保输出目录存在
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 写入分块后的报告到目标目录
            with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)
                
        # 输出处理文件数统计
        print(f"已分块处理 {len(all_report_paths)} 个文件")

    def split_markdown_file(self, md_path: Path, chunk_size: int = 30, chunk_overlap: int = 5,
                            line_pages: Optional[List[int]] = None):
        """
        按行分割 markdown 文件，每个分块记录起止行号、内容与所属页码。

        分块在**页边界内**进行，因此每个 chunk 只属于一个页面——这正是
        ``parent_document_retrieval`` 能够从 chunk 回溯到完整页面的前提。

        :param md_path: markdown 文件路径
        :param chunk_size: 每个分块的最大行数
        :param chunk_overlap: 分块重叠行数
        :param line_pages: 与 markdown 行一一对应的 1-based 页码；
            为 None 时退化为"无页码"模式（引用将不可用，日志会明确告警）
        :return: 分块列表
        """
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if line_pages is not None and len(line_pages) != len(lines):
            raise ValueError(
                f"line_pages 长度({len(line_pages)})与 markdown 行数({len(lines)})不一致"
            )

        chunks = []
        chunk_id = 0

        # 先按页把行分组，再在每组内部滑动切分
        groups: List[Tuple[Optional[int], List[int]]] = []
        for index, line in enumerate(lines):
            page = line_pages[index] if line_pages is not None else None
            if groups and groups[-1][0] == page:
                groups[-1][1].append(index)
            else:
                groups.append((page, [index]))

        for page, indices in groups:
            step = max(1, chunk_size - chunk_overlap)
            for start in range(0, len(indices), step):
                window = indices[start : start + chunk_size]
                chunk_text = ''.join(lines[i] for i in window)
                # 不产出空白块：空块在向量化阶段无法嵌入，若放行会被静默丢弃，
                # 导致 FAISS 行号与 chunks 下标整体错位，所有引用随之失效。
                if not chunk_text.strip():
                    continue
                chunks.append({
                    'lines': [window[0] + 1, window[-1] + 1],  # 行号从1开始
                    'page': page,
                    'text': chunk_text,
                    'id': chunk_id,
                    'type': 'content',
                })
                chunk_id += 1
                if start + chunk_size >= len(indices):
                    break  # 已覆盖到本页末尾，避免尾部重叠块

        return chunks

    def build_pages(self, md_path: Path, line_pages: Optional[List[int]]) -> List[Dict]:
        """把 markdown 按页聚合成父文档列表，供父文档检索回溯使用。"""
        if line_pages is None:
            return []

        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        buckets: Dict[int, List[str]] = {}
        for index, page in enumerate(line_pages):
            if page is None:
                continue
            buckets.setdefault(page, []).append(lines[index])

        return [
            {"page": page, "text": ''.join(buckets[page])}
            for page in sorted(buckets)
        ]

    def split_markdown_reports(self, all_md_dir: Path, output_dir: Path, chunk_size: int = 30,
                               chunk_overlap: int = 5, subset_csv: Path = None,
                               pdf_reports_dir: Path = None):
        """
        批量处理目录下所有 markdown 文件，分块并输出为 json 文件到目标目录。

        每个输出 json 的 ``content`` 同时包含 ``chunks`` 与 ``pages``：
        前者供向量检索，后者供父文档检索与引用溯源。

        :param all_md_dir: 存放 .md 文件的目录
        :param output_dir: 输出 .json 文件的目录
        :param chunk_size: 每个分块的最大行数
        :param chunk_overlap: 分块重叠行数
        :param subset_csv: subset.csv 路径，用于建立 file_name 到 company_name 的映射
        :param pdf_reports_dir: 源 PDF 目录，用于把 markdown 行对齐回真实页码
        """
        # 建立 file_name（去扩展名）到 company_name 的映射
        file2company = {}
        file2sha1 = {}
        if subset_csv is not None and os.path.exists(subset_csv):
            # 优先尝试 utf-8，失败则尝试 gbk
            try:
                df = pd.read_csv(subset_csv, encoding='utf-8')
            except UnicodeDecodeError:
                print('警告：subset.csv 不是 utf-8 编码，自动尝试 gbk 编码...')
                df = pd.read_csv(subset_csv, encoding='gbk')
            # 自动识别主键列
            if 'file_name' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = os.path.splitext(str(row['file_name']))[0]
                    file2company[file_no_ext] = row['company_name']
                    if 'sha1' in row:
                        file2sha1[file_no_ext] = row['sha1']
            elif 'sha1' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = str(row['sha1'])
                    file2company[file_no_ext] = row['company_name']
                    file2sha1[file_no_ext] = row['sha1']
            else:
                raise ValueError('subset.csv 缺少 file_name 或 sha1 列，无法建立文件名到公司名的映射')

        all_md_paths = list(all_md_dir.glob("*.md"))
        output_dir.mkdir(parents=True, exist_ok=True)
        page_stats = []
        for md_path in all_md_paths:
            # 关键：把 markdown 的每一行对齐回源 PDF 的真实页码
            pdf_path = resolve_pdf_for_markdown(md_path, pdf_reports_dir)
            line_pages, report = align_markdown_to_pages(md_path, pdf_path)
            page_stats.append((md_path.name, report))

            chunks = self.split_markdown_file(
                md_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                line_pages=line_pages if report.get("aligned") else None,
            )
            pages = self.build_pages(md_path, line_pages if report.get("aligned") else None)

            output_json_path = output_dir / (md_path.stem + ".json")
            # 查找 company_name 和 sha1
            file_no_ext = md_path.stem
            company_name = file2company.get(file_no_ext, "")
            sha1 = file2sha1.get(file_no_ext, "")
            # metainfo 只保留 sha1、company_name、file_name 字段
            metainfo = {"sha1": sha1, "company_name": company_name, "file_name": md_path.name}
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {"metainfo": metainfo, "content": {"chunks": chunks, "pages": pages}},
                    f, ensure_ascii=False, indent=2,
                )

            if report.get("aligned"):
                agreement = report["footer_check"].get("agreement")
                quality = (f"页脚校验一致率={agreement:.1%}" if agreement is not None
                           else "无页脚可校验(建议人工抽查)")
                print(f"已处理: {md_path.name} -> {output_json_path.name} "
                      f"(chunks={len(chunks)}, pages={len(pages)}, "
                      f"对齐命中={report['directly_matched']}/{report['total_lines']} 行, "
                      f"{quality})")
            else:
                print(f"警告: {md_path.name} 未获得页码（{report.get('reason')}），"
                      f"引用将不可用。chunks={len(chunks)}")

        self._report_page_mapping(page_stats)
        print(f"共分割 {len(all_md_paths)} 个 markdown 文件")

    @staticmethod
    def _report_page_mapping(page_stats):
        """汇总页码对齐情况，让页码缺失这类问题无法被静默忽略。"""
        aligned = [r for _, r in page_stats if r.get("aligned")]
        failed = [(n, r.get("reason")) for n, r in page_stats if not r.get("aligned")]
        if failed:
            print("以下报告未能建立页码映射（引用将失效）：")
            for name, reason in failed:
                print(f"  - {name}: {reason}")
        if not aligned:
            return
        agreements = [r["footer_check"]["agreement"] for r in aligned
                      if r["footer_check"].get("agreement") is not None]
        if agreements:
            avg = sum(agreements) / len(agreements)
            print(f"页码对齐完成：{len(aligned)}/{len(page_stats)} 份报告，"
                  f"平均页脚交叉校验一致率 {avg:.1%}")
        else:
            print(f"页码对齐完成：{len(aligned)}/{len(page_stats)} 份报告"
                  f"（PDF 无页脚，缺少独立交叉校验）")
