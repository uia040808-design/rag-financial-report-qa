"""命令行入口。

修复了三个会让 README 里的命令直接崩溃的问题：

1. ``parse-pdfs`` 读取 ``paths.parsed_reports_path`` / ``parsed_reports_debug_path``，
   而这两个属性在 PipelineConfig 里是注释掉的 —— AttributeError。
   更根本的是：Docling 链路早已不在主流程上（主流程是 MinerU 转 markdown），
   所以这个命令改成直接调 ``export_reports_to_markdown``。
2. ``serialize-tables`` 调用 ``pipeline.serialize_tables()`` —— 该方法不存在。
   表格序列化依赖 Docling 解析产物，与当前 MinerU 链路不兼容，改为明确报错
   并说明原因，而不是抛 AttributeError。
3. ``process-questions --config`` 声明了 9 个选项，``configs`` 只有 3 个，
   其余 6 个会 KeyError。改为从 ``configs`` 动态生成选项列表，
   保证声明与实现永远一致。
"""

import click
from pathlib import Path

from src.pipeline import Pipeline, configs, preprocess_configs

DEFAULT_DATA_DIR = "data/stock_data"


def _resolve_data_dir(explicit: str) -> Path:
    """定位数据目录。默认为项目根下的 data/stock_data。

    原实现用 ``Path.cwd()``，于是"在项目根运行"与"在数据目录运行"两种
    习惯会指向不同位置，表现为 subset.csv / pdf_reports 找不到。
    """
    if explicit:
        return Path(explicit)
    here = Path(__file__).resolve().parent
    candidate = here / DEFAULT_DATA_DIR
    return candidate if candidate.exists() else Path.cwd()


@click.group()
def cli():
    """Pipeline command line interface for processing PDF reports and questions."""
    pass


@cli.command()
def download_models():
    """Download required docling models."""
    click.echo("Downloading docling models...")
    Pipeline.download_docling_models()


@cli.command()
@click.option('--data-dir', default=None,
              help=f'数据目录，默认 {DEFAULT_DATA_DIR}')
@click.option('--api', type=click.Choice(['agent', 'standard']), default='agent',
              help='agent=免 Token 轻量接口(默认)；standard=需配额的 v4 精准接口')
@click.option('--only', nargs=1, help='只处理文件名含该子串的 PDF')
@click.option('--force', is_flag=True, help='忽略已存在的 md，强制重新转换')
def parse_pdfs(data_dir, api, only, force):
    """把 pdf_reports 下的 PDF 转换为 Markdown（MinerU）。"""
    root = _resolve_data_dir(data_dir)
    pipeline = Pipeline(root)

    if api == 'agent':
        from src import pdf_mineru
        pdfs = sorted(pipeline.paths.pdf_reports_dir.glob("*.pdf"))
        if only:
            pdfs = [p for p in pdfs if only in p.name]
        if not force:
            todo = [p for p in pdfs
                    if not (pipeline.paths.reports_markdown_path / f"{p.stem}.md").exists()]
            if not todo:
                click.echo(f"{len(pdfs)} 份 PDF 的 Markdown 均已存在，跳过（--force 可强制重跑）")
                return
        click.echo(f"使用 Agent 轻量 API，待处理 {len(pdfs)} 个 PDF")
        result = pdf_mineru.convert_via_agent(pdfs, pipeline.paths.reports_markdown_path)
        ok = sum(1 for v in result.values() if v is not None)
        click.echo(f"完成 {ok}/{len(result)}")
    else:
        click.echo("使用标准 v4 API（需 MinerU 账号配额）")
        pipeline.export_reports_to_markdown(only=[only] if only else None, force=force)


@cli.command()
def serialize_tables():
    """表格序列化（当前不可用）。"""
    raise click.ClickException(
        "表格序列化依赖 Docling 的解析产物（parsed reports JSON），"
        "而当前主流程是 MinerU 直接转 Markdown，不产出该中间格式，因此无法使用。\n"
        "如需该能力，需要先恢复 Docling 解析链路并保留其 JSON 产物。"
    )


@cli.command()
@click.option('--data-dir', default=None, help=f'数据目录，默认 {DEFAULT_DATA_DIR}')
@click.option('--config', type=click.Choice(sorted(preprocess_configs)), default='no_ser_tab',
              help='预处理配置')
def process_reports(data_dir, config):
    """Chunk reports and build the vector database."""
    root = _resolve_data_dir(data_dir)
    pipeline = Pipeline(root, run_config=preprocess_configs[config])
    click.echo(f"Processing reports (config={config}, data={root})...")
    pipeline.process_parsed_reports()


@cli.command()
@click.option('--data-dir', default=None, help=f'数据目录，默认 {DEFAULT_DATA_DIR}')
@click.option('--config', type=click.Choice(sorted(configs)), default='max',
              help='问答配置。选项由 configs 动态生成，避免声明与实现不一致')
@click.option('--question', default=None, help='单问模式：直接回答该问题并打印结果')
@click.option('--kind', default='string',
              type=click.Choice(['string', 'number', 'boolean', 'name', 'names']),
              help='单问模式的答案类型')
def process_questions(data_dir, config, question, kind):
    """Answer the question set, or a single question with --question."""
    root = _resolve_data_dir(data_dir)
    pipeline = Pipeline(root, run_config=configs[config])

    if question:
        click.echo(f"单问模式: {question}")
        answer = pipeline.answer_single_question(question, kind=kind)
        click.echo("")
        for field in ('step_by_step_analysis', 'reasoning_summary'):
            if answer.get(field):
                click.echo(f"--- {field} ---")
                click.echo(answer[field])
                click.echo("")
        if answer.get('relevant_quotes'):
            click.echo("--- relevant_quotes ---")
            for q in answer['relevant_quotes']:
                click.echo(f"  > {q}")
            click.echo("")
        click.echo(f"--- final_answer ---\n{answer.get('final_answer')}")
        click.echo(f"--- relevant_pages ---\n{answer.get('relevant_pages')}")
        refs = answer.get('references') or []
        if refs:
            click.echo("--- references ---")
            for r in refs:
                click.echo(f"  pdf_sha1={r.get('pdf_sha1')} page_index={r.get('page_index')}")
        if answer.get('_degraded'):
            click.echo("")
            click.echo("!! 该答案的结构化输出降级，不可作为有效模型输出")
        return

    click.echo(f"Processing questions (config={config}, data={root})...")
    pipeline.process_questions()


if __name__ == '__main__':
    cli()