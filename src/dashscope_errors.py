"""DashScope 业务错误的识别、分类与透出。

为什么单独成模块
----------------
服务端失败时会返回 ``status_code`` / ``code`` / ``message``（例如额度耗尽的
``403 AllocationQuota.FreeTierOnly``、批量超限的 ``400 InvalidParameter``），
而 ``output`` 为 ``None``。

原实现在三处都直接对 ``resp['output']`` 求值，于是把带明确处置建议的服务端提示
统统变成 ``TypeError: argument of type 'NoneType' is not iterable`` ——真实原因被
完全掩盖，且看起来像本地代码 bug。已实测发生三次：

  1. ``ingestion._get_embeddings``   批量 25 > 上限 20，报成 NoneType 错误
  2. ``reranking.get_rank_for_multiple_blocks`` 并发重排限流，报成 NoneType 错误
  3. ``api_requests`` 结构化输出解析失败路径

三处症状相同、根因相同，因此判定为**同一个系统性缺陷**而非三个孤立 bug，
故提取为单一入口，避免第四处再犯。

错误分类
--------
``DashScopeThrottled``（429 / ``Throttling.*``）与 ``DashScopeError`` 分开：
前者退避后重试有意义，后者重试只是浪费时间（参数非法、额度耗尽）。
把两者混为一谈的后果实测出现过 —— 建库在 78% 处撞限流后直接失败、
只能从头重来；反过来把限流当不可重试，则并发问答会整批报错。
"""

from __future__ import annotations

from typing import Any, Optional


class DashScopeError(RuntimeError):
    """DashScope 返回了业务错误。**重试无意义**（参数非法、额度耗尽等）。"""


class DashScopeThrottled(DashScopeError):
    """DashScope 触发限流（429 / ``Throttling.*``）。**应当退避后重试**。"""


# 已知错误码 -> 处置建议。服务端 message 通常已说明原因，这里补的是
# "在这个项目里具体该改哪个开关"。
_HINTS = {
    "AllocationQuota.FreeTierOnly":
        "账号处于\"仅使用免费额度\"模式且免费额度已耗尽；"
        "可在百炼控制台充值，或关闭该模式以使用付费额度；"
        "也可换用有配额的模型（EMBEDDING_MODEL 环境变量）。",
    "InvalidParameter":
        "请求参数不合法。注意各模型的批量上限不同，"
        "batch size 超限会报此项（可用 EMBEDDING_BATCH_SIZE 调小）。",
    "Throttling.AllocationQuota":
        "触发 TPM/并发限流。属可重试错误；反复失败请调小 EMBEDDING_BATCH_SIZE、"
        "调大 EMBEDDING_BATCH_DELAY，或在百炼控制台提升配额上限。",
    "Throttling.RateQuota":
        "触发请求频率限流。属可重试错误；并发问答时建议调低 parallel_requests。",
    "InvalidApiKey":
        "API Key 无效。请检查 DASHSCOPE_API_KEY（env 文件里可能是占位符，"
        "而真实 key 只存在于系统环境变量）。",
}


def _field(resp: Any, name: str) -> Optional[Any]:
    if isinstance(resp, dict):
        return resp.get(name)
    return getattr(resp, name, None)


def raise_if_api_error(resp: Any, context: str) -> None:
    """把 DashScope 的业务错误原样抛出。``resp`` 正常时不做任何事。

    同时兜住一种更隐蔽的形态：``resp`` 是 dict、``status_code`` 缺失或为 200，
    但 ``output`` 为 ``None``。此时不能放行 —— 否则调用方会在下游炸出与真实
    原因无关的 TypeError。这里给出明确异常并附上原始响应摘要。
    """
    if resp is None:
        return  # 未认证等情形交给调用方处理

    status = _field(resp, "status_code")
    code = _field(resp, "code")
    message = _field(resp, "message")

    if status is not None and int(status) != 200:
        code_str = str(code)
        hint = _HINTS.get(code_str, "请按服务端 message 的提示处理。")
        text = (
            f"DashScope 接口返回错误（{context}）：\n"
            f"  status_code = {status}\n"
            f"  code        = {code}\n"
            f"  message     = {message}\n"
            f"  处置建议    = {hint}"
        )
        if int(status) == 429 or code_str.startswith("Throttling"):
            raise DashScopeThrottled(text)
        raise DashScopeError(text)

    # 200 但 output 为空：不是业务错误，却同样无法使用。
    if isinstance(resp, dict) and resp.get("output", "missing") is None:
        raise DashScopeError(
            f"DashScope 接口返回 output=None（{context}），无内容可用。\n"
            f"  status_code = {status}\n"
            f"  code        = {code}\n"
            f"  message     = {message}\n"
            f"  原始响应    = {str(resp)[:300]}"
        )