"""Summarize saved DSV4 NPU evidence without changing test results."""

import json
from pathlib import Path
import shlex
import sys


root = Path(sys.argv[1])
read = lambda path: json.loads(path.read_text(encoding="utf-8"))
launch = read(root / "audit/launch.json")
lines = [
    "# DeepSeek-V4-Flash DSV4 AIV D2H/H2D 验证",
    "",
    f"SGLang 提交：`{launch['git_commit']}`；sgl-kernel-npu 提交：`{launch['kernel_commit']}`。",
    f"SGLang 工作区状态：`{launch['git_status']}`。",
    "",
    "## 服务命令",
    "",
    "```bash",
    shlex.join(launch["argv"]),
    "```",
    "",
    "审计只统计显式 impl=aiv、算子返回且 NPU stream event 完成的记录。",
    "",
]

for result in sorted(root.glob("results_*")):
    summary_file = result / "summary.json"
    if not summary_file.exists():
        continue
    count = result.name.removeprefix("results_")
    summary = read(summary_file)
    plan = read(root / f"plan_{count}.json")
    info = read(result / "server_info.json")
    lines += [
        f"## {count} 题结果", "", "```json",
        json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
        f"few-shot 示例数：{plan['num_shots']}；输入长度：{min(len(q['input_ids']) for q in plan['questions'])}～{max(len(q['input_ids']) for q in plan['questions'])} token。",
        f"解析布局：`{info.get('hicache_mem_layout')}`；I/O：`{info.get('hicache_io_backend')}`。", "",
    ]
    for question in plan["questions"]:
        verdict_file = result / question["qid"] / "verdict.json"
        if not verdict_file.exists():
            continue
        verdict = read(verdict_file)
        tiers = verdict["cached_tokens_details"]
        lines.append(
            f"- {verdict['qid']}: A={verdict['baseline_answer']}, R={verdict['replay_answer']}, "
            f"真值={verdict['ground_truth']}, L1={tiers['device']}, L2={tiers['host']}, "
            f"答案一致={verdict['answer_match']}, 双向路径={verdict['path_covered']}"
        )
    lines.append("")
    first = result / plan["questions"][0]["qid"] / "verdict.json"
    if first.exists():
        verdict = read(first)
        lines += ["首题各池审计覆盖：", ""]
        for pool in verdict["d2h"]:
            d2h = verdict["d2h"][pool]
            h2d = verdict["h2d"][pool]
            lines.append(
                f"- {pool}: D2H {len(d2h['layers'])}/{d2h['expected_layers'][0]} 层，"
                f"H2D {len(h2d['layers'])}/{h2d['expected_layers'][0]} 层，"
                f"完成调用 {d2h['calls']}/{h2d['calls']}"
            )
        lines.append("")

model_variant = (
    "W8A8 MTP（modelslim）"
    if "modelslim" in launch["argv"]
    else "FP8 权重（FP4 experts 解码为 FP8）"
)
lines += [
    "## 边界", "",
    f"本次模型为 {model_variant}：`{launch['argv'][launch['argv'].index('--model-path') + 1]}`。",
    "NPU event 证明算子所在 stream 完成；未对 KV/state 字节逐一比对。",
    "A/R 答案与真值用于发现可见的回放错误；测试不评价并发或 TTFT。", "",
    f"证据目录：`{root}`。", "",
]
(root / "report.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:5]))
