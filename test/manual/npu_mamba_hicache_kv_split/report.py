#!/usr/bin/env python3
"""读取 case_result.json 并生成 report.md。"""

import argparse
import json
import os
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--sglang-root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--npu-devices", required=True)
    ap.add_argument("--port", required=True)
    args = ap.parse_args()

    out = args.output
    case_path = os.path.join(out, "case_result.json")
    case = json.load(open(case_path)) if os.path.exists(case_path) else None

    try:
        commit = subprocess.run(
            ["git", "-C", args.sglang_root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    try:
        branch = subprocess.run(
            ["git", "-C", args.sglang_root, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        branch = "unknown"

    passed = bool(
        case
        and case.get("match")
        and not case.get("errors")
        and case.get("filler_done", 0) > 0
        and case.get("filler_done") == case.get("filler_requested")
        and case.get("mamba_backup_delta", 0) > 0
        and case.get("mamba_load_delta", 0) > 0
        and case.get("target_replay_host_tokens", 0) > 0
    )

    lines = []
    lines.append("# NPU Mamba HiCache state transfer 端到端测试报告\n")
    lines.append(f"- 结论: **{'通过' if passed else '失败'}**\n")
    reasons = []
    if case is None:
        reasons.append("case_result.json 缺失（用例未运行到写结果）")
    else:
        if case.get("errors"):
            reasons.append("请求阶段有错误: " + "; ".join(case["errors"]))
        if not case.get("match"):
            reasons.append("target 重放输出与首次不一致（或首次输出为空）")
        if case.get("filler_done", 0) != case.get("filler_requested"):
            reasons.append("filler 请求未全部完成，驱逐压力不足")
        if case.get("mamba_backup_delta", 0) <= 0:
            reasons.append("Mamba D2H 计数未增长")
        if case.get("mamba_load_delta", 0) <= 0:
            reasons.append("Mamba H2D 计数未增长")
        if case.get("target_replay_host_tokens", 0) <= 0:
            reasons.append("target 重放未命中 Host L2")
    if reasons:
        lines.append("- 失败/警告原因:\n")
        lines.extend(f"  - {r}\n" for r in reasons)
    lines.append("\n## 配置\n")
    lines.append(f"- 代码: {args.sglang_root} @ {branch} {commit}\n")
    lines.append(f"- 模型: {args.model}\n")
    lines.append(f"- NPU 卡: {args.npu_devices}\n")
    lines.append(f"- 端口: {args.port}\n")
    lines.append("- IO 后端: kernel_ascend（由 run_server.sh 配置）\n")
    if case:
        lines.append("\n## 指标汇总\n")
        lines.append("| 指标 | 值 |\n| --- | --- |\n")
        lines.append(f"| filler 完成数 | {case.get('filler_done', 0)} |\n")
        lines.append(f"| Mamba D2H tokens 增量 | {case.get('mamba_backup_delta', 0)} |\n")
        lines.append(f"| Mamba H2D tokens 增量 | {case.get('mamba_load_delta', 0)} |\n")
        lines.append(f"| target 重放 Host 命中 tokens | {case.get('target_replay_host_tokens', 0)} |\n")
        lines.append(f"| 首次输出长度 (chars) | {len(case.get('target_first') or '')} |\n")
        lines.append(f"| 重放输出长度 (chars) | {len(case.get('target_second') or '')} |\n")
        lines.append(f"| 输出一致 | {case.get('match')} |\n")
        lines.append("\n## 两次输出对比\n")
        lines.append("### 首次输出\n\n```\n"
                     + (case.get("target_first") or "(空)")[:2000] + "\n```\n")
        lines.append("### 重放输出\n\n```\n"
                     + (case.get("target_second") or "(空)")[:2000] + "\n```\n")

    report_path = os.path.join(out, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"report written: {report_path}")


if __name__ == "__main__":
    main()
