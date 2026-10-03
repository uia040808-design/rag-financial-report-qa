"""MinerU 云端 PDF -> Markdown 转换。

为什么重写
----------
原实现有三处会直接导致不可用：

1. **从不上传本地 PDF**。它把文件名拼在一个公网 OSS 地址后面
   （``https://vl-image.oss-cn-shanghai.aliyuncs.com/pdf/`` + file_name），
   任务因此指向一个 404 的 URL。实测 URL 提交路径还会被区域策略拒绝：
   ``{"code": -60023, "msg": "this URL is restricted by regional regulations"}``。
2. **剥 zip 用 ``rstrip('.zip')``**。``rstrip`` 剥的是*字符集*不是后缀，
   task_id 以 ``i`` / ``p`` / ``z`` 结尾就会被削掉，导致解压目录名 ≠ task_id，
   而调用方恰好依赖两者相等。
3. **``while True`` 无超时无重试**，接口持续报错就永久挂死。

现在的流程（官方 batch 上传接口）::

    POST /api/v4/file-urls/batch      -> batch_id + 预签名上传 URL
    PUT  <每个预签名 URL>              -> 上传 PDF 字节（不带 Content-Type）
    GET  /api/v4/extract-results/batch/{batch_id}  -> 轮询直到 state=done
    GET  <full_zip_url>               -> 下载并解压出 full.md + content_list.json

限额：单文件 ≤200MB、≤200 页、单次 ≤50 个文件。
"""

from __future__ import annotations

import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

_log = logging.getLogger(__name__)

BASE_URL = "https://mineru.net/api/v4"
MODEL_VERSION = "vlm"

# 原实现在此处硬编码占位符。现在优先读环境变量；保留模块常量作为回退，
# 以兼容把 token 直接写在本文件里的用法（但那不是好实践，见 README 安全提示）。
api_key = ""

CONNECT_TIMEOUT = 30
READ_TIMEOUT = 120
UPLOAD_TIMEOUT = 600
POLL_INTERVAL = 6
POLL_TIMEOUT = 40 * 60
# 连续这么多次仍是 pending 时提示"服务端排队"，便于区分排队与上传失败
STUCK_AFTER_POLLS = 10
MAX_PAGES_PER_DOC = 200


class MinerUError(RuntimeError):
    """MinerU 接口返回了非 0 的业务码，或响应结构不符合预期。"""


def get_api_key() -> str:
    """取 token：环境变量优先，其次模块常量。"""
    key = os.getenv("MINERU_API_KEY") or api_key
    if not key or key.strip() in ("", "你的api_key"):
        raise MinerUError(
            "未配置 MinerU token：请设置环境变量 MINERU_API_KEY，"
            "或填写 src/pdf_mineru.py 顶部的 api_key"
        )
    return key.strip()


def _headers() -> Dict[str, str]:
    return {"Content-Type": "application/json",
            "Authorization": f"Bearer {get_api_key()}"}


def _check(payload: dict, what: str) -> dict:
    """校验业务码。MinerU 用 HTTP 200 + code 表达错误，只看 HTTP 状态会漏掉。"""
    if not isinstance(payload, dict):
        raise MinerUError(f"{what}: 响应不是 JSON 对象 -> {str(payload)[:200]}")
    code = payload.get("code")
    if code != 0:
        raise MinerUError(f"{what}: code={code} msg={payload.get('msg')}")
    return payload.get("data") or {}


def check_pdf_page_count(pdf_path: Path) -> int:
    """读 PDF 页数，提前拦截超过 MinerU 单文件 200 页上限的文件。

    上限是在服务端强制的，被拒后任务根本不会建立，浪费一次配额。
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return -1
    try:
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            return len(doc)
        finally:
            doc.close()
    except Exception:  # noqa: BLE001 - 读不出页数就交给服务端判断
        return -1


# --------------------------------------------------------------------------- #
# 步骤 1：申请预签名上传 URL
# --------------------------------------------------------------------------- #
def check_quota() -> Optional[Dict[str, int]]:
    """查询剩余配额。

    **必须在提交前调用。** MinerU 在配额耗尽时的行为是**受理任务但不执行**：
    接口返回 ``code=0`` 并发出 task_id，任务永远停在 ``pending``，``err_msg``
    为空、不报任何错误。实测中因此空等了 50 分钟才定位到根因。

    返回 None 表示接口不可用（不阻断，让后续轮询去发现真实问题）。
    """
    try:
        resp = requests.get(f"{BASE_URL}/quota", headers=_headers(),
                            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        resp.raise_for_status()
        data = _check(resp.json(), "查询配额")
    except Exception as exc:  # noqa: BLE001 - 配额接口不可用不应阻断转换
        _log.warning("查询 MinerU 配额失败（将跳过前置检查）：%s", exc)
        return None

    left = data.get("user_left_quota")
    if left is None:
        left = data.get("total_left_quota")
    if left is None:
        return None

    quota = {"user_left_quota": data.get("user_left_quota"),
             "total_left_quota": data.get("total_left_quota")}
    if left <= 0:
        raise MinerUError(
            f"MinerU 剩余配额为 0（user_left_quota={quota['user_left_quota']}, "
            f"total_left_quota={quota['total_left_quota']}）。"
            f"此时提交的任务会被受理但永不执行，且不返回任何错误——"
            f"请先到 mineru.net 充值/开通额度，再重试。已提交的 batch 可用 "
            f"`--resume <batch_id>` 续等，配额到账后会自动继续。"
        )
    print(f"[MinerU] 剩余配额: user={quota['user_left_quota']} "
          f"total={quota['total_left_quota']}")
    return quota


def request_upload_urls(pdf_paths: List[Path]) -> Tuple[str, List[str]]:
    files = [{"name": p.name, "data_id": p.stem} for p in pdf_paths]
    if len(files) > 50:
        raise MinerUError(f"单次最多 50 个文件，收到 {len(files)}")
    body = {"files": files, "model_version": MODEL_VERSION, "enable_table": True}
    resp = requests.post(f"{BASE_URL}/file-urls/batch", headers=_headers(),
                         json=body, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    resp.raise_for_status()
    data = _check(resp.json(), "申请上传 URL")
    batch_id, urls = data.get("batch_id"), data.get("file_urls")
    if not batch_id or not urls or len(urls) != len(pdf_paths):
        raise MinerUError(f"申请上传 URL 返回异常: batch_id={batch_id}, "
                          f"urls={len(urls or [])} 期望 {len(pdf_paths)}")
    return batch_id, urls


# --------------------------------------------------------------------------- #
# 步骤 2：PUT 上传
# --------------------------------------------------------------------------- #
def upload_files(pdf_paths: List[Path], urls: List[str]) -> None:
    for path, url in zip(pdf_paths, urls):
        size_mb = path.stat().st_size / 1024 / 1024
        if size_mb > 200:
            raise MinerUError(f"{path.name} 大小 {size_mb:.1f}MB 超过 200MB 上限")
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                with open(path, "rb") as fh:
                    # 预签名 URL 已绑定签名，不能附带额外 header
                    res = requests.put(url, data=fh, timeout=UPLOAD_TIMEOUT)
                if res.status_code == 200:
                    print(f"    上传 {path.name} ({size_mb:.1f}MB) 完成")
                    break
                last_error = MinerUError(f"HTTP {res.status_code}: {res.text[:150]}")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            print(f"    上传 {path.name} 第 {attempt + 1} 次失败：{last_error}")
            time.sleep(3 * (attempt + 1))
        else:
            raise MinerUError(f"{path.name} 上传失败：{last_error}")


# --------------------------------------------------------------------------- #
# 步骤 3：轮询 batch 结果
# --------------------------------------------------------------------------- #
def poll_batch(batch_id: str, timeout: int = POLL_TIMEOUT) -> List[dict]:
    deadline = time.time() + timeout
    attempt = 0
    last_states: str = ""
    while time.time() < deadline:
        attempt += 1
        resp = requests.get(f"{BASE_URL}/extract-results/batch/{batch_id}",
                            headers=_headers(),
                            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        resp.raise_for_status()
        data = _check(resp.json(), "查询 batch 结果")
        results = data.get("extract_result") or []

        # 必须打印**具体状态名**。早前只打聚合计数，结果卡了 40 分钟都看不出
        # 是 pending（服务端排队）还是 waiting-file（上传没到）——两者的处置
        # 完全不同。
        state_str = ",".join(
            f"{r.get('file_name', '?')[:24]}={r.get('state')}"
            + (f"({r.get('err_msg')[:40]})" if r.get("err_msg") else "")
            for r in results
        )
        if state_str != last_states or attempt % 10 == 1:
            print(f"    [轮询 {attempt}] {state_str or '(无结果)'}")
            last_states = state_str

        done = [r for r in results if r.get("state") == "done"]
        failed = [r for r in results if r.get("state") in ("failed", "error")]

        if done or failed:
            return results

        # 长时间停在 pending 说明是服务端排队，不是上传问题 —— 提前给出可操作的提示
        if attempt == STUCK_AFTER_POLLS:
            still_pending = [r for r in results if r.get("state") == "pending"]
            if still_pending:
                print(f"    提示：{len(still_pending)} 个文件仍为 pending（服务端队列中，"
                      f"非上传失败）。state=waiting-file 才表示文件没收到。")

        time.sleep(POLL_INTERVAL)

    raise MinerUError(
        f"batch {batch_id} 超过 {timeout}s 仍未完成，已放弃（不再无限等待）。"
        f"最后状态: {last_states or '无'}"
    )


# --------------------------------------------------------------------------- #
# 步骤 4：下载并安全解压
# --------------------------------------------------------------------------- #
def _safe_members(zf: zipfile.ZipFile, dest: Path) -> List[zipfile.ZipInfo]:
    """过滤掉会写到 dest 之外的成员（zip-slip）。"""
    root = dest.resolve()
    safe = []
    for info in zf.infolist():
        target = (root / info.filename).resolve()
        if not str(target).startswith(str(root)):
            print(f"    跳过可疑成员（路径逃逸）: {info.filename}")
            continue
        safe.append(info)
    return safe


def download_and_extract(full_zip_url: str, dest_dir: Path) -> Dict[str, Path]:
    """下载 zip，抽出 ``full.md`` 与 ``*_content_list.json``。

    ``content_list.json`` 里每个 block 带权威的 ``page_idx``，是页码对齐的
    旁证来源 —— 纯扫描件页没有文本层、无法靠对齐得到页码，只能靠它。
    """
    resp = requests.get(full_zip_url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    resp.raise_for_status()

    dest_dir.mkdir(parents=True, exist_ok=True)
    import io
    found: Dict[str, Path] = {}
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        members = _safe_members(zf, dest_dir)
        zf.extractall(dest_dir, members=members)
        names = zf.namelist()

    for info in members:
        name = Path(info.filename).name
        if name == "full.md" or name.endswith("_content_list.json"):
            found[name] = dest_dir / info.filename
    return found


# --------------------------------------------------------------------------- #
# 高层入口
# --------------------------------------------------------------------------- #
def convert_pdfs(
    pdf_paths: List[Path],
    output_dir: Path,
    timeout: int = POLL_TIMEOUT,
) -> Dict[str, Optional[Path]]:
    """批量转换 PDF 为 Markdown，按 PDF 原名写入 ``output_dir``。

    返回 ``{pdf 名: 产出的 md 路径 or None}``。已存在的 md 会跳过，便于断点续跑。
    """
    pdf_paths = [Path(p) for p in pdf_paths]
    todo = []
    skipped = {}
    for p in pdf_paths:
        target = output_dir / f"{p.stem}.md"
        if target.exists():
            skipped[p.name] = target
        else:
            todo.append(p)

    if not todo:
        print(f"[MinerU] {len(skipped)} 个 Markdown 已存在，全部跳过")
        return skipped

    # 提前拦截超页数文件，避免浪费配额
    usable = []
    for p in todo:
        pages = check_pdf_page_count(p)
        if pages > MAX_PAGES_PER_DOC:
            print(f"[MinerU] 跳过 {p.name}：{pages} 页，超过单文件 "
                  f"{MAX_PAGES_PER_DOC} 页上限（服务端会直接拒绝）")
            skipped[p.name] = None
            continue
        usable.append(p)
    if not usable:
        return skipped

    print(f"[MinerU] 待转换 {len(usable)} 个文件（{len(skipped)} 个跳过）")

    # 配额为 0 时 MinerU 会受理任务但永不执行，必须前置拦截
    check_quota()

    batch_id, urls = request_upload_urls(usable)
    print(f"[MinerU] batch_id = {batch_id}")
    upload_files(usable, urls)
    print("[MinerU] 已全部上传，开始轮询解析结果…")
    results = poll_batch(batch_id, timeout=timeout)

    by_name = {r.get("file_name"): r for r in results}
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in usable:
        result = by_name.get(p.name, {})
        state = result.get("state")
        if state != "done":
            print(f"[MinerU] {p.name}: state={state} err={result.get('err_msg', '')}")
            skipped[p.name] = None
            continue

        work_dir = Path("_mineru_work") / str(batch_id) / p.stem
        try:
            files = download_and_extract(result["full_zip_url"], work_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[MinerU] {p.name}: 下载/解压失败 -> {exc}")
            skipped[p.name] = None
            continue

        md = next((v for k, v in files.items() if k == "full.md"), None)
        if md is None or not md.exists():
            print(f"[MinerU] {p.name}: zip 内未找到 full.md（成员: {list(files)}）")
            skipped[p.name] = None
            continue

        target = output_dir / f"{p.stem}.md"
        target.write_bytes(md.read_bytes())
        print(f"[MinerU] {p.name} -> {target.name}")

        # content_list.json 与 md 同名保存，供页码对齐交叉校验
        cl = next((v for k, v in files.items() if k.endswith("_content_list.json")), None)
        if cl is not None and cl.exists():
            (output_dir / f"{p.stem}.content_list.json").write_bytes(cl.read_bytes())
            print(f"[MinerU]   附带 content_list.json（{cl.stat().st_size} 字节）")

        skipped[p.name] = target

    return skipped


def resume_batch(
    batch_id: str,
    pdf_paths: List[Path],
    output_dir: Path,
    timeout: int = POLL_TIMEOUT,
) -> Dict[str, Optional[Path]]:
    """对已提交的 batch 继续轮询并落盘，不重新上传（重新上传会再消耗一次配额）。

    服务端排队（``state=pending``）时用这个入口续等，而不是整个流程重跑。
    """
    print(f"[MinerU] 恢复 batch {batch_id}，共 {len(pdf_paths)} 个文件")
    check_quota()
    results = poll_batch(batch_id, timeout=timeout)
    by_name = {r.get("file_name"): r for r in results}
    output_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Optional[Path]] = {}
    for p in pdf_paths:
        result = by_name.get(p.name, {})
        if result.get("state") != "done":
            print(f"[MinerU] {p.name}: state={result.get('state')} "
                  f"err={result.get('err_msg', '')}")
            out[p.name] = None
            continue
        work = Path("_mineru_work") / batch_id / p.stem
        files = download_and_extract(result["full_zip_url"], work)
        md = files.get("full.md")
        if md is None or not md.exists():
            out[p.name] = None
            continue
        target = output_dir / f"{p.stem}.md"
        target.write_bytes(md.read_bytes())
        print(f"[MinerU] {p.name} -> {target.name}")
        cl = next((v for k, v in files.items() if k.endswith("_content_list.json")), None)
        if cl is not None and cl.exists():
            (output_dir / f"{p.stem}.content_list.json").write_bytes(cl.read_bytes())
        out[p.name] = target
    return out


# --------------------------------------------------------------------------- #
# Agent 轻量解析 API（免 Token，按 IP 限频）
# --------------------------------------------------------------------------- #
# 官方文档：/api/v1/agent/parse/file -> PUT file_url -> GET /parse/{task_id}
# 限制：单文件 <=10MB、<=20 页、不支持批量。超出时本地切分后逐份提交。
#
# 为什么需要这条路径：标准 API（/api/v4）实测会受理任务但永不执行
# （state 恒为 pending、err_msg 为空、code=0），原因未知；且 /api/v4/quota
# 对个人专属 token 不适用（文档明确说明），返回 0/0 无参考意义。
# Agent 通道不依赖账号状态，实测可完整跑通。
AGENT_URL = "https://mineru.net/api/v1/agent"
AGENT_MAX_BYTES = 10 * 1024 * 1024
AGENT_MAX_PAGES = 20
AGENT_MAX_RETRIES = 3


def _agent_split_pdf(pdf_path: Path, out_dir: Path) -> List[Path]:
    """把超过 Agent 限制的 PDF 切成若干份，每份 <=AGENT_MAX_PAGES 页 / <=10MB。"""
    import pypdfium2 as pdfium

    src = pdfium.PdfDocument(str(pdf_path))
    try:
        total = len(src)
        out_dir.mkdir(parents=True, exist_ok=True)
        chunks: List[Path] = []
        for start in range(0, total, AGENT_MAX_PAGES):
            end = min(start + AGENT_MAX_PAGES, total)
            dst = pdfium.PdfDocument.new()
            try:
                dst.import_pages(src, list(range(start, end)))
                target = out_dir / f"{pdf_path.stem}__p{start + 1:04d}-{end:04d}.pdf"
                dst.save(str(target))
                # 单页可能超大（少见），超了就继续对半切
                while target.stat().st_size > AGENT_MAX_BYTES and \
                        (end - start) > 1:
                    dst.close()
                    mid = start + (end - start) // 2
                    dst = pdfium.PdfDocument.new()
                    dst.import_pages(src, list(range(start, mid)))
                    target.unlink(missing_ok=True)
                    target = out_dir / f"{pdf_path.stem}__p{start + 1:04d}-{mid:04d}.pdf"
                    dst.save(str(target))
                    end = mid
                chunks.append(target)
                print(f"    切分 {pdf_path.name}: 第 {start + 1}-{end} 页 -> "
                      f"{target.name} ({target.stat().st_size / 1024:.0f}KB)")
            finally:
                dst.close()
        return chunks
    finally:
        src.close()


def _agent_parse_one(pdf_path: Path, work_dir: Path) -> Optional[str]:
    """提交单个 PDF 给 Agent API，返回 markdown 文本。失败返回 None。"""
    blob = pdf_path.read_bytes()

    last_error: Optional[str] = None
    for attempt in range(1, AGENT_MAX_RETRIES + 1):
        try:
            r = requests.post(f"{AGENT_URL}/parse/file",
                              json={"file_name": pdf_path.name, "language": "ch"},
                              timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            j = r.json()
            if j.get("code") != 0:
                last_error = f"code={j.get('code')} msg={j.get('msg')}"
                # 参数类错误重试无意义
                if j.get("code") in (-30001, -30002, -30003, -30004):
                    print(f"    {pdf_path.name}: Agent 拒绝（{last_error}）")
                    return None
            else:
                data = j["data"]
                up = requests.put(data["file_url"], data=blob, timeout=UPLOAD_TIMEOUT)
                if up.status_code != 200:
                    last_error = f"上传 HTTP {up.status_code}"
                else:
                    return _agent_wait_and_fetch(data["task_id"], pdf_path, work_dir)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        print(f"    {pdf_path.name} 第 {attempt} 次失败：{last_error}")
        time.sleep(5 * attempt)
    print(f"    {pdf_path.name} 最终失败：{last_error}")
    return None


def _agent_wait_and_fetch(task_id: str, pdf_path: Path,
                          work_dir: Path, timeout: int = 15 * 60) -> Optional[str]:
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        r = requests.get(f"{AGENT_URL}/parse/{task_id}", timeout=30)
        j = r.json()
        if j.get("code") not in (0, None):
            print(f"    {pdf_path.name} 查询失败: code={j.get('code')} msg={j.get('msg')}")
            return None
        d = j.get("data") or {}
        state = d.get("state")
        if state == "done":
            url = d.get("markdown_url")
            if not url:
                return None
            md = requests.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            md.raise_for_status()
            return md.text
        if state in ("failed", "error"):
            print(f"    {pdf_path.name}: state={state} "
                  f"err={str(d.get('err_msg', ''))[:80]}")
            return None
        if attempt % 10 == 1:
            print(f"    {pdf_path.name}: {state} (+{attempt * 15}s)")
        time.sleep(15)
    print(f"    {pdf_path.name}: 超过 {timeout}s 仍未完成")
    return None


def convert_via_agent(
    pdf_paths: List[Path],
    output_dir: Path,
) -> Dict[str, Optional[Path]]:
    """用 Agent 轻量 API 批量转换。超限文档本地切分后逐份提交再拼接。"""
    pdf_paths = [Path(p) for p in pdf_paths]
    out: Dict[str, Optional[Path]] = {}
    work_root = Path("_mineru_work") / "agent"
    output_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for p in pdf_paths:
        target = output_dir / f"{p.stem}.md"
        if target.exists():
            print(f"[Agent] {p.name} 的 md 已存在，跳过")
            out[p.name] = target
        else:
            todo.append(p)

    for pdf in todo:
        size = pdf.stat().st_size
        if size <= AGENT_MAX_BYTES:
            pages = check_pdf_page_count(pdf)
            if 0 < pages <= AGENT_MAX_PAGES:
                pieces = [pdf]
                print(f"[Agent] {pdf.name} 直接提交（{pages} 页, {size/1024:.0f}KB）")
            else:
                print(f"[Agent] {pdf.name} {pages} 页 / {size/1024:.0f}KB 超出限制，切分")
                pieces = _agent_split_pdf(pdf, work_root / pdf.stem)
        else:
            print(f"[Agent] {pdf.name} {size/1024/1024:.1f}MB 超出 10MB，切分")
            pieces = _agent_split_pdf(pdf, work_root / pdf.stem)

        texts: List[str] = []
        ok = True
        for piece in pieces:
            md = _agent_parse_one(piece, work_root / pdf.stem)
            if md is None:
                ok = False
                break
            texts.append(md)

        if not ok or not texts:
            out[pdf.name] = None
            print(f"[Agent] {pdf.name} 未完整成功，跳过")
            continue

        target = output_dir / f"{pdf.stem}.md"
        target.write_text("\n\n".join(texts), encoding="utf-8")
        print(f"[Agent] {pdf.name} -> {target.name} "
              f"({len(pieces)} 段, {target.stat().st_size/1024:.0f}KB)")
        out[pdf.name] = target

    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="MinerU PDF -> Markdown")
    ap.add_argument("--pdf-dir", default="data/stock_data/pdf_reports")
    ap.add_argument("--out", default="data/stock_data/debug_data/03_reports_markdown")
    ap.add_argument("--only", nargs="*", help="只处理文件名含这些子串的 PDF")
    ap.add_argument("--resume", metavar="BATCH_ID",
                    help="继续轮询已提交的 batch，不重新上传（服务端排队时用）")
    ap.add_argument("--api", choices=["agent", "standard"], default="agent",
                    help="agent=免 Token 轻量接口(默认)；standard=需配额的 v4 精准接口")
    ap.add_argument("--timeout", type=int, default=POLL_TIMEOUT,
                    help=f"轮询上限秒数，默认 {POLL_TIMEOUT}")
    args = ap.parse_args()

    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if args.only:
        pdfs = [p for p in pdfs if any(k in p.name for k in args.only)]

    if args.resume:
        result = resume_batch(args.resume, pdfs, Path(args.out), timeout=args.timeout)
    elif args.api == "agent":
        print(f"使用 Agent 轻量 API（免 Token）。待处理 {len(pdfs)} 个 PDF")
        result = convert_via_agent(pdfs, Path(args.out))
    else:
        print(f"使用标准 v4 API。待处理 {len(pdfs)} 个 PDF")
        result = convert_pdfs(pdfs, Path(args.out), timeout=args.timeout)

    ok = sum(1 for v in result.values() if v is not None)
    print(f"\n完成 {ok}/{len(result)}")