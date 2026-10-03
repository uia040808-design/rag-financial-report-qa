"""结构化输出评分：回放真实历史产物 + 人为畸形语料。

回答两个问题：
1. 新解析层能否把当初没被解析出来的真实产物救回来（回归）？
2. 解析阶梯各级的实际命中率，以及"必须失败"的用例是否真的失败了
   （误放比误拒危险 —— 误放会让原文伪装成答案流进提交物）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from src.structured_output import parse_structured_output, build_schema_instruction

import src.prompts as P
from eval.structured_corpus import (
    LegacyStringSchema,
    load_real_corpus,
    synthetic_cases,
)

STR_SCHEMA = P.AnswerWithRAGContextStringPrompt.AnswerSchema
NUM_SCHEMA = P.AnswerWithRAGContextNumberPrompt.AnswerSchema
BASELINE = Path("eval/baseline_structured.json")

GOOD = {
    "step_by_step_analysis": "1. 分析问题\n2. 定位答案\n3. 得出结论",
    "reasoning_summary": "年报直接给出",
    "relevant_quotes": ["产能利用率85.6%"],
    "final_answer": "85.6%",
}


def score_real_corpus(verbose: bool = False) -> Dict:
    """用 legacy schema 回放历史产物。"""
    corpus = load_real_corpus()
    if not corpus:
        return {"count": 0, "recovered": 0, "rate": None}

    recovered = 0
    for question, raw in corpus:
        out = parse_structured_output(raw, LegacyStringSchema)
        recovered += 1 if out.ok else 0
        if verbose:
            status = "已恢复" if out.ok else "仍失败"
            print(f"  [{status}] 策略={out.strategy:10s} {question[:40]}")
            if out.ok:
                print(f"           final_answer: {str(out.data.get('final_answer'))[:56]}")
            else:
                print(f"           {out.errors[-1][:100]}")
    return {"count": len(corpus), "recovered": recovered,
            "rate": round(recovered / len(corpus), 4)}


def score_synthetic(verbose: bool = False) -> Dict:
    cases = synthetic_cases(GOOD)
    should_pass = [(n, r) for n, r, exp in cases if exp]
    should_fail = [(n, r) for n, r, exp in cases if not exp]

    passed = sum(1 for _, raw in should_pass
                 if parse_structured_output(raw, STR_SCHEMA).ok)
    correctly_rejected = sum(1 for _, raw in should_fail
                             if not parse_structured_output(raw, STR_SCHEMA).ok)

    strategies: Dict[str, int] = {}
    for _, raw in should_pass:
        st = parse_structured_output(raw, STR_SCHEMA).strategy
        strategies[st] = strategies.get(st, 0) + 1

    if verbose:
        print("  -- 应成功 --")
        for name, raw in should_pass:
            out = parse_structured_output(raw, STR_SCHEMA)
            print(f"    {name:44s} {'OK' if out.ok else 'FAIL':5s} 策略={out.strategy}")
        print("  -- 应失败（失败才是正确行为）--")
        for name, raw in should_fail:
            out = parse_structured_output(raw, STR_SCHEMA)
            flag = "OK" if not out.ok else "★误放"
            print(f"    {name:44s} {flag:5s} 策略={out.strategy}")

    return {
        "should_pass_total": len(should_pass),
        "should_pass_ok": passed,
        "should_pass_rate": round(passed / len(should_pass), 4) if should_pass else None,
        "should_fail_total": len(should_fail),
        "should_fail_rejected": correctly_rejected,
        "false_accept_rate": (
            round((len(should_fail) - correctly_rejected) / len(should_fail), 4)
            if should_fail else None
        ),
        "strategies": strategies,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="结构化输出解析评分")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--show-instruction", action="store_true")
    args = ap.parse_args()

    if args.show_instruction:
        print(build_schema_instruction(NUM_SCHEMA))
        return 0

    print("=" * 80)
    print("结构化输出解析评分")
    print("=" * 80)

    print("\n[1] 真实历史产物回放（用当时的契约 LegacyStringSchema）")
    real = score_real_corpus(args.detail)
    if real["count"]:
        print(f"  语料 {real['count']} 条 -> 恢复 {real['recovered']} 条  "
              f"恢复率 {real['rate']:.1%}")
    else:
        print("  未找到历史产物，跳过（先跑一次 process_questions 产出）")

    print("\n[2] 解析阶梯覆盖率")
    syn = score_synthetic(args.detail)
    print(f"  应成功用例 {syn['should_pass_ok']}/{syn['should_pass_total']}  "
          f"= {syn['should_pass_rate']:.1%}")
    print(f"  应失败用例被正确拒绝 {syn['should_fail_rejected']}/{syn['should_fail_total']}"
          f"   误放率 = {syn['false_accept_rate']:.1%}")
    print(f"  各阶梯命中: {syn['strategies']}")

    metrics = {**real, **{k: v for k, v in syn.items() if k != "strategies"}}

    if args.save_baseline:
        BASELINE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"\n基线已写入 {BASELINE}")
        return 0

    if BASELINE.exists():
        base = json.loads(BASELINE.read_text(encoding="utf-8"))
        print("\n与基线对比:")
        bad = []
        for key, cur in metrics.items():
            if key not in base or cur is None or base[key] is None:
                continue
            higher_better = key in ("recovered", "recovered_rate", "should_pass_rate",
                                     "should_fail_rejected")
            delta = cur - base[key]
            worse = (higher_better and delta < -0.001) or (not higher_better and delta > 0.001)
            if worse:
                bad.append(key)
            print(f"  {key:28s} {base[key]:>8.4f} -> {cur:>8.4f} ({delta:+.4f})"
                  f"{'  <-- 劣化' if worse else ''}")
        if bad:
            print(f"\n检出 {len(bad)} 项劣化：{bad}")
            return 1
        print("\n未检出劣化。")
    return 0


if __name__ == "__main__":
    sys.exit(main())