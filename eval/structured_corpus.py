"""结构化输出解析的回归语料。

语料有两个来源：
1. **真实历史产物** —— `data/stock_data/answers_qwen_turbo.json` 里 4 条 `value`
   是整坨 JSON 字符串的记录。那是 DashScope 分支剥围栏失败、原文被塞进
   `final_answer` 的现场。它们用的是**当时的契约**（`relevant_pages: List[int]`），
   所以必须用 LegacyStringSchema 回放，才能回答"新解析层能否把这些救回来"。
2. **人为构造的畸形** —— 覆盖其余解析阶梯，以及必须被拒的非法情况。

判定"应该失败"的用例，失败才是正确行为，因此分开统计。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field
from typing import List

ANSWERS = Path("data/stock_data/answers_qwen_turbo.json")


class LegacyStringSchema(BaseModel):
    """prompts 改为引文之前 string schema 的形状，仅用于回放历史语料。"""

    step_by_step_analysis: str = Field(default="")
    reasoning_summary: str = Field(default="")
    relevant_pages: List[int] = Field(default_factory=list)
    final_answer: str = Field(default="N/A")


def load_real_corpus() -> List[Tuple[str, str]]:
    """返回 [(问题, 当时未被解析出来的原文)]。"""
    if not ANSWERS.exists():
        return []
    out = []
    for item in json.loads(ANSWERS.read_text(encoding="utf-8")).get("answers", []):
        value = item.get("value")
        if isinstance(value, str) and value.lstrip().startswith("{"):
            out.append((item.get("question_text", ""), value))
    return out


def synthetic_cases(good: Dict) -> List[Tuple[str, str, bool]]:
    """返回 [(用例名, 原始文本, 是否应该解析成功)]。"""
    raw = json.dumps(good, ensure_ascii=False)
    cases = [
        # ---- 应成功 ----
        ("裸 JSON", raw, True),
        ("```json 围栏且位于开头", "```json\n" + raw + "\n```", True),
        ("★ 围栏前有引导语（原实现在此失败）",
         "根据年报信息，回答如下：\n```json\n" + raw + "\n```", True),
        ("★ JSON 前后都有说明文字（原实现在此失败）",
         "好的，以下是分析：\n" + raw + "\n以上。", True),
        ("★ 整个对象套进 final_answer（实测真实失败模式）",
         json.dumps({"final_answer": raw}, ensure_ascii=False), True),
        ("嵌套两层", json.dumps({"result": json.dumps({"data": raw}, ensure_ascii=False)},
                                ensure_ascii=False), True),
        ("★ 残缺 JSON（缺右括号）", raw[:-3], True),
        ("单引号非法 JSON", str(good).replace("'", '"'), True),
        ("围栏未闭合", "```json\n" + raw, True),
        ("relevant_quotes 为空列表",
         json.dumps({**good, "relevant_quotes": []}, ensure_ascii=False), True),
        # ---- 应失败：模型没给出可用结构 ----
        ("★ 空响应", "", False),
        ("纯自然语言无 JSON", "抱歉，我无法回答这个问题。", False),
        ("★ 缺 final_answer 字段",
         json.dumps({k: v for k, v in good.items() if k != "final_answer"},
                    ensure_ascii=False), False),
        ("relevant_quotes 类型错误（非字符串数组）",
         json.dumps({**good, "relevant_quotes": [1, 2]}, ensure_ascii=False), False),
    ]
    return cases