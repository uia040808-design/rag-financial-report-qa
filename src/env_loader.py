"""项目环境变量加载。

为什么需要这一层
----------------
仓库自带的密钥模板文件名为 ``env``（无前导点），而 ``python-dotenv`` 的
``load_dotenv()`` 默认只找 ``.env``。于是 ``find_dotenv()`` 返回空字符串，
所有 ``os.getenv("DASHSCOPE_API_KEY")`` 读到 ``None``，DashScope 调用返回
``None``，最终报出与真实原因毫无关系的错误：

    if 'output' in resp and 'embeddings' in resp['output']:
    TypeError: argument of type 'NoneType' is not iterable

README 确实要求"把 env 重命名为 .env"，但只要有人没照做（或只有部分工具
读得到），错误就会伪装成别的问题。这里把两种命名都支持，并按优先级合并：
真实环境变量 > ``.env`` > ``env``。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

_log = logging.getLogger(__name__)

# 依次尝试的候选文件名。带点的是 python-dotenv 的惯例，不带点的是本仓库
# 历史模板的命名，两个都保留以免"改名之前完全不能用"。
CANDIDATES = (".env", "env")


def project_root() -> Path:
    """src/ 的父目录即项目根。"""
    return Path(__file__).resolve().parent.parent


def load_project_env(root: Path | None = None, override: bool = False) -> list[Path]:
    """加载项目环境变量，返回实际被读取的文件列表。

    ``override=False``（默认）时已有环境变量优先，因此 CI / shell 里显式
    导出的变量不会被文件覆盖。
    """
    base = Path(root) if root else project_root()
    loaded: list[Path] = []

    env_path = base / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=override)
        loaded.append(env_path)

    legacy = base / "env"
    if legacy.exists() and legacy != env_path:
        load_dotenv(legacy, override=override)
        loaded.append(legacy)

    if not loaded:
        _log.warning(
            "未找到环境变量文件（尝试过 %s）。相关 API 调用将因缺少密钥而失败。",
            ", ".join(str(base / n) for n in CANDIDATES),
        )
    return loaded


def require_env(name: str, hint: str = "") -> str:
    """取出必需的环境变量，缺失时给出可操作的报错。

    宁可在这里 fail fast，也不要把 None 传到 SDK 里，再收到一个
    ``NoneType is not iterable`` 这种与真实原因无关的错误。
    """
    load_project_env()
    value = os.getenv(name)
    if not value or value.strip() in ("", "your-api-key", "sk-REPLACE_WITH_YOUR_DASHSCOPE_KEY"):
        raise RuntimeError(
            f"缺少环境变量 {name}。{hint}\n"
            f"请在项目根目录放置 .env（或沿用现有的 env 文件）并填入真实值，"
            f"可参考 env 文件中的注释。"
        )
    return value.strip()


# 生成模型默认值。原先全项目硬编码 'qwen-turbo'，散落在 reranking /
# api_requests / pipeline / questions_processing 四处。硬编码的实际代价：
# 该模型的免费额度耗尽后，403 只在**运行时**才暴露，且要改四个文件。
DEFAULT_GENERATION_MODEL = "qwen-plus"


def generation_model() -> str:
    """当前使用的生成（问答 / 重排）模型名。

    与 :func:`src.ingestion.embedding_model` 同理，集中一处定义。
    可用环境变量 ``GENERATION_MODEL`` 覆盖。

    默认值选 ``qwen-plus`` 而非 ``qwen-turbo``：实测（2026-10）在同一账号下
    ``qwen-turbo`` / ``qwen-max`` / ``qwen-flash`` / ``deepseek-v3`` 等十余个
    模型全部返回 ``403 AllocationQuota.FreeTierOnly``，只有 ``qwen-plus``
    仍有可用额度。可见百炼的免费额度是**按模型分别计算**的 —— 换模型
    往往是唯一不需要充值就能继续的路径。
    """
    load_project_env()
    return os.getenv("GENERATION_MODEL", DEFAULT_GENERATION_MODEL).strip()