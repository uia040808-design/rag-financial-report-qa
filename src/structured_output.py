"""结构化输出解析：把大模型返回的文本变成经过校验的结构。

为什么需要这一层
----------------
原先只有 IBM / Gemini 两个 provider 接了 ``json_repair`` + Pydantic 校验 +
二次重解析，DashScope 分支是裸 ``json.loads``，且剥 markdown 围栏的判断写成
``content_str.startswith('```')`` —— 只要围栏前面有一句引导语（"根据年报信息，
回答如下："），围栏就不会被剥掉，``json.loads`` 随即失败，然后整段原文被塞进
``final_answer``。

这个失败在真实产物里留了痕：``answers_qwen_turbo.json`` 的 4 条 ``value`` 是整坨
JSON 字符串，``answers_qwen_turbo_debug.json`` 对应的 ``step_by_step_analysis`` 是
空字符串。也就是说，**结构化输出从未真正生效**，而退出码一直是 0。

解析阶梯
--------
按可靠性从高到低逐级尝试，每一级都记录用了哪条策略，便于事后审计：

    direct    整段文本直接就是 JSON
    fenced    剥掉 ``` 围栏（允许围栏前后有说明文字）
    embedded  文本中扫描出第一个括号平衡的 JSON 对象（字符串感知）
    repaired  以上都失败，用 json_repair 修补残缺 JSON
    nested    解析成功但字段嵌在某个字符串值里（模型把整个对象套进字段）

任何一级成功后都要过 Pydantic 校验；校验不过视为失败并记录原因。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

_FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+-]*)\s*\n(.*?)```", re.S)


@dataclass
class ParseOutcome:
    """解析结果。``ok`` 为 False 时 ``data`` 不可用于业务逻辑。"""

    ok: bool
    data: Optional[Dict[str, Any]] = None
    raw_text: str = ""
    strategy: str = "failed"
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return f"ok via {self.strategy}"
        return f"FAILED after {self.strategy}: {'; '.join(self.errors)[:200]}"


def _scan_balanced_object(text: str) -> Optional[str]:
    """扫描第一个括号平衡的 ``{...}``，字符串感知（正确跳过转义引号）。"""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _collect_candidates(obj: Any, required: List[str], depth: int = 0,
                        out: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """收集 obj 内所有"含全部必需字段"的对象，**越深越靠后**。

    失败形态有两种，都需要覆盖：
      纯嵌套   {"final_answer": "{...}"}                     —— 外壳字段不全
      空壳嵌套 {"step_by_step_analysis": "", ...,           —— 外壳字段齐全但值是壳，
                "final_answer": "{完整对象}"}                    真正的对象在某个值里

    后者不能靠"字段齐全就返回自身"解决：外壳的字段名对得上，但 ``relevant_quotes``
    是字符串而非数组，Pydantic 校验必然失败。因此把候选全部收集起来，交由调用方
    逐个校验；**优先尝试更深的**，因为嵌套越深越可能是真正的答案。
    """
    if out is None:
        out = []
    if depth > 5:
        return out

    if isinstance(obj, str):
        text = obj.strip()
        if text.startswith(("{", "[")):
            try:
                return _collect_candidates(json.loads(text), required, depth + 1, out)
            except json.JSONDecodeError:
                return out
        return out

    if isinstance(obj, list):
        for item in obj:
            _collect_candidates(item, required, depth + 1, out)
        return out

    if isinstance(obj, dict):
        if all(f in obj for f in required):
            out.append(obj)
        for value in obj.values():
            _collect_candidates(value, required, depth + 1, out)
    return out


def parse_structured_output(
    raw: str,
    response_format: Optional[Type[BaseModel]] = None,
) -> ParseOutcome:
    """把模型原始文本解析为经校验的结构化结果。"""
    outcome = ParseOutcome(ok=False, raw_text=raw or "")
    text = (raw or "").strip()
    if not text:
        outcome.errors.append("empty response")
        return outcome

    required: List[str] = []
    if response_format is not None:
        try:
            required = list(response_format.model_fields.keys())
        except AttributeError:  # pydantic v1 兼容
            required = list(getattr(response_format, "__fields__", {}).keys())

    def accept(candidate: Any, strategy: str) -> ParseOutcome:
        if response_format is None:
            if isinstance(candidate, dict):
                outcome.ok, outcome.data, outcome.strategy = True, candidate, strategy
            else:
                outcome.errors.append(f"{strategy}: not a JSON object")
            return outcome
        if not isinstance(candidate, dict):
            outcome.errors.append(f"{strategy}: not a JSON object")
            return outcome
        missing = [f for f in required if f not in candidate]
        if missing:
            outcome.errors.append(f"{strategy}: missing fields {missing}")
            return outcome
        try:
            validated = response_format.model_validate(candidate)
            outcome.ok = True
            outcome.data = validated.model_dump()
            outcome.strategy = strategy
        except Exception as exc:  # noqa: BLE001 - pydantic 校验错误类型很杂
            outcome.errors.append(f"{strategy}: schema validation failed: {exc}"[:300])
        return outcome

    def accept_any(parsed: Any, strategy: str) -> Optional[ParseOutcome]:
        """先校验整体；失败则在嵌套候选里逐个试（越深越优先）。"""
        got = accept(parsed, strategy)
        if got.ok:
            return got
        if not required or not isinstance(parsed, (dict, list, str)):
            return got
        for candidate in reversed(_collect_candidates(parsed, required)):
            if candidate is parsed:
                continue  # 整体已试过
            nested = accept(candidate, strategy + "+nested")
            if nested.ok:
                return nested
        return got

    # --- 1. direct ---
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        got = accept_any(parsed, "direct")
        if got.ok:
            return got

    # --- 2. fenced（允许围栏前后有说明文字）---
    match = _FENCE_RE.search(text)
    if match:
        inner = match.group(1).strip()
        try:
            got = accept_any(json.loads(inner), "fenced")
            if got.ok:
                return got
        except json.JSONDecodeError as exc:
            outcome.errors.append(f"fenced: {exc}"[:200])
    elif "```" in text:
        outcome.errors.append("fenced: opening fence without closing fence")

    # --- 3. embedded（括号平衡扫描）---
    scanned = _scan_balanced_object(text)
    if scanned:
        try:
            got = accept_any(json.loads(scanned), "embedded")
            if got.ok:
                return got
        except json.JSONDecodeError as exc:
            outcome.errors.append(f"embedded: {exc}"[:200])

    # --- 4. repaired ---
    try:
        from json_repair import repair_json

        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, list) and repaired:
            repaired = repaired[0]
        if isinstance(repaired, dict):
            got = accept_any(repaired, "repaired")
            if got.ok:
                return got
            outcome.errors.extend(got.errors[-1:])
        else:
            outcome.errors.append("repaired: result was not an object")
    except Exception as exc:  # noqa: BLE001
        outcome.errors.append(f"repaired: {exc}"[:200])

    return outcome


def build_schema_instruction(response_format: Type[BaseModel]) -> str:
    """把 Pydantic 模型转成简洁的 JSON 输出说明，用于注入 system prompt。

    DashScope 的 qwen-turbo 不支持 response_format 强约束，schema 只能靠提示词
    传达；原先 ``use_schema_prompt`` 把 dashscope 排除在外，模型连字段名都只能
    从示例里猜。这里显式列出字段名、类型与是否必填。
    """
    try:
        schema = response_format.model_json_schema()
    except AttributeError:  # pydantic v1
        schema = response_format.schema()

    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    def type_of(node: Dict[str, Any]) -> str:
        # Union 会被 json_schema 表达成 anyOf；逐个展开，别只报 "any"
        if "anyOf" in node:
            parts = [type_of(p) for p in node["anyOf"]]
            seen, uniq = set(), []
            for p in parts:
                if p not in seen:
                    seen.add(p)
                    uniq.append(p)
            return " | ".join(uniq)
        if "enum" in node:
            return " | ".join(str(x) for x in node["enum"])
        if node.get("type") == "array":
            inner = node.get("items", {})
            return f"array of {type_of(inner) if inner else 'any'}"
        return node.get("type", "any")

    lines = []
    for name, node in props.items():
        flag = "必填" if name in required else "可选"
        # 描述只取首行，且去掉行尾悬空的冒号/破折号（多行说明会在此截断）
        desc = str(node.get("description", "")).strip().split("\n")[0]
        desc = desc.rstrip("：:-— ").strip()
        if len(desc) > 110:
            desc = desc[:110].rstrip("：:-— ") + "…"
        lines.append(f'  - "{name}" ({type_of(node)}, {flag}){": " + desc if desc else ""}')

    body = "\n".join(lines)
    return (
        "你的回答必须是**一个 JSON 对象**，且只输出该 JSON 对象本身：\n"
        "  - 不要输出任何解释、前言或结语\n"
        "  - 不要用 markdown 代码块包裹\n"
        f"字段必须严格如下，顺序保持一致：\n{body}\n"
        "  - 数组字段用 JSON 数组；数值字段用 JSON 数字（不要加单位或千分位）\n"
        "  - 字符串字段里如需换行，转义为 \\n"
    )