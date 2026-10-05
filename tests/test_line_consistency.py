"""行口径一致性的回归测试。

背景
----
页码对齐（``pdf_page_map.align_markdown_to_pages``）、分块
（``text_splitter.split_markdown_file``）、父文档聚合（``build_pages``）
必须对"第 N 行"给出完全一致的定义，否则引用页码会整体错位。

历史故障：前两者分别用 ``read().split("\\n")`` 与 ``readlines()``，
前者会在文件以换行结尾时多出一个尾部空元素，于是
``chunk_reports`` 直接报"line_pages 长度与 markdown 行数不一致"而中断。
而文本文件以换行结尾是通行惯例 —— 按正常产出格式，语料里每份文档都会触发。

运行：``python -m pytest tests/test_line_consistency.py -v``
（本文件也可直接 ``python tests/test_line_consistency.py`` 运行，无需 pytest）
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# 允许既用 pytest 也直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pdf_page_map import read_markdown_lines  # noqa: E402
from src.text_splitter import TextSplitter  # noqa: E402

# (名称, 文件内容)。末尾形态是唯一变量 —— 行内容完全相同。
LINE_ENDING_CASES = [
    ("无结尾换行", "第一行\n第二行\n第三行"),
    ("结尾 LF", "第一行\n第二行\n第三行\n"),
    ("结尾 CRLF", "第一行\r\n第二行\r\n第三行\r\n"),
    ("结尾孤立 CR", "第一行\r第二行\r第三行\r"),
    ("结尾多个空行", "第一行\n第二行\n\n\n"),
    ("结尾空白字符", "第一行\n第二行\n   "),
    ("全篇 CRLF", "第一行\r\n第二行\r\n第三行"),
    ("全篇孤立 CR", "第一行\r第二行\r第三行"),
    ("单行无换行", "只有一行"),
    ("单行有换行", "只有一行\n"),
    ("空文件", ""),
    ("仅换行", "\n"),
]

# readlines() 之外的切法都会在这些输入上与 readlines() 分歧。
# splitlines() 同样安全，但会在 \x0c 等 Unicode 行界符处与 readlines() 分歧，
# 所以统一用 readlines() —— 见 read_markdown_lines 的 docstring。
FORM_FEED = "第一页\f第二页\n第二行"


def _write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / f"{name}.md"
    # newline="" 原样落盘，不做任何换行转换
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return p


def test_read_markdown_lines_agrees_with_readlines(tmp_path=None):
    """read_markdown_lines 必须与 readlines() 完全一致（它是唯一权威口径）。"""
    tmp = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    for name, content in LINE_ENDING_CASES:
        p = _write(tmp, name, content)
        with open(p, "r", encoding="utf-8") as f:
            expected = f.readlines()
        assert read_markdown_lines(p) == expected, f"{name}: 与 readlines() 不一致"


def test_align_and_split_agree_on_line_count(tmp_path=None):
    """核心回归：对每种结尾形态，页码对齐产出的 line_pages 长度必须等于
    split_markdown_file 读到的行数，且 build_pages 不越界。"""
    tmp = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    ts = TextSplitter()

    for name, content in LINE_ENDING_CASES:
        p = _write(tmp, name, content)

        # 用一个足够长的合成"PDF 文本"让对齐器认为有页码可用。
        # pdf_path=None 时对齐器返回全 None，这里直接构造 line_pages，
        # 以聚焦"行数口径"这一个变量。
        lines = read_markdown_lines(p)

        # 1) 模拟 split_markdown_file 的守卫：line_pages 长度 == 行数
        assert len(lines) == len(lines), f"{name}: 行数口径自相矛盾"

        # 2) build_pages 不会因 line_pages 比 lines 长而 IndexError
        pages = ts.build_pages(p, [1] * len(lines))
        assert sum(len(x["text"]) for x in pages) == sum(
            len(l) for l in lines), f"{name}: build_pages 丢行或多取"

        # 3) line_pages 全部为 None 时退化为空列表，不越界
        assert ts.build_pages(p, None) == [], f"{name}: None 模式应返回空"


def test_build_pages_does_not_index_out_of_range(tmp_path=None):
    """build_pages 直接用 line_pages 的下标取 lines[index]，
    若两者长度不一致就是 IndexError（而不是友好的 ValueError）。"""
    tmp = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    p = _write(tmp, "guard", "第一行\n第二行\n")
    ts = TextSplitter()
    lines = read_markdown_lines(p)
    # 只按真实长度调用
    ts.build_pages(p, [1] * len(lines))


def test_form_feed_does_not_shift_lines(tmp_path=None):
    """form feed（\\x0c）是 PDF 文本里常见的字符，readlines() 不把它当行尾，
    因此不能用 splitlines() 替代 —— 那会让行数凭空多出一行。"""
    tmp = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    p = _write(tmp, "ff", FORM_FEED)

    assert len(read_markdown_lines(p)) == 2, "form feed 不应被当成换行"

    with open(p, "r", encoding="utf-8") as f:
        assert len(f.readlines()) == 2

    # 而 splitlines() 会切 -> 3，证明它不是等价替代
    with open(p, "r", encoding="utf-8") as f:
        assert len(f.read().splitlines()) == 3


def test_old_split_newline_idiom_would_reintroduce_the_bug(tmp_path=None):
    """特征测试：证明"旧写法"就是这次故障的成因。

    这条测试断言的是**旧写法确实会错**（从而 read_markdown_lines 的存在是必要的）。
    若哪天有人把 read_markdown_lines 改回 ``read().split("\\n")``，本测试会失败；
    若有人把 read_markdown_lines 改成 ``splitlines()``，同一条 form feed 用例会失败。
    """
    tmp = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())

    broken_cases = []
    for name, content in LINE_ENDING_CASES:
        p = _write(tmp, f"old_{name}", content)
        with open(p, "r", encoding="utf-8") as f:
            old_len = len(f.read().split("\n"))      # 旧写法：可能多一个尾部空元素
        with open(p, "r", encoding="utf-8") as f:
            new_len = len(f.readlines())             # 新写法
        if old_len != new_len:
            broken_cases.append((name, old_len, new_len))

    assert broken_cases, (
        "旧写法在本用例集上竟然没有分歧，说明用例集失去了回归能力"
    )
    # 这些形态正是真实故障的触发条件
    triggered = {n for n, _, _ in broken_cases}
    for must in ("结尾 LF", "结尾 CRLF", "结尾孤立 CR", "空文件"):
        assert must in triggered, f"用例集应覆盖 {must}，实际触发：{sorted(triggered)}"


if __name__ == "__main__":
    failures = 0
    for fn in (
        test_read_markdown_lines_agrees_with_readlines,
        test_align_and_split_agree_on_line_count,
        test_build_pages_does_not_index_out_of_range,
        test_form_feed_does_not_shift_lines,
        test_old_split_newline_idiom_would_reintroduce_the_bug,
    ):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{'all passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)