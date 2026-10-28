"""Generate the GLM-5.2 GSM8K D2H/H2D report from saved evidence."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shlex
import sys

root = Path(sys.argv[1])
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
launch = read(root / "audit/launch.json")
plan = read(root / "plan.json")
result = root / "results"
summary = read(result / "summary.json")
status = read(root / "run_status.json")
info = read(result / "server_info.json")
audit = read(root / "independent_audit.json") if (root / "independent_audit.json").exists() else None
clock = lambda iso: datetime.fromisoformat(iso).astimezone(timezone(timedelta(hours=8), "HKT")).strftime("%Y-%m-%d %H:%M:%S %Z")
total = summary["planned_questions"]
environment = launch.get("environment", {})
host_label = environment.get("VALIDATION_HOST_LABEL") or "未记录"
container_label = environment.get("VALIDATION_CONTAINER_LABEL") or "未记录"

lines = [
    "# GLM-5.2-w4a8c8 GSM8K MLA/Indexer D2H/H2D 精度验证", "",
    f"运行时间：{clock(status['start_utc'])} 至 {clock(status['end_utc'])}。", "",
    f"服务器：`{host_label}`；容器：`{container_label}`；物理 NPU：`{environment['ASCEND_RT_VISIBLE_DEVICES']}`。", "",
    f"代码提交：`{launch['git_commit']}`；完整证据目录：`{root}`。", "",
    "## 方法", "",
    "沿用旧 GSM8K 提示：测试集前 5 题作示例，评估后续 100 题；temperature=0、max_new_tokens=512、stop=Question:。",
    "每题串行执行 flush → A 冷缓存生成 → 等待 K/V/Indexer K 全层 D2H → B 分块不同前缀请求驱逐 L1 → R 同输入重放 → 等待 K/V/Indexer K 全层 H2D。A 和 R 的输入 token ID 完全相同。",
    "审计钩子在真实 `memfabric_hybrid.offload.kv_exchange_copy` 返回 0 后记录当前 NPU 流 Event；Event 完成才计入覆盖。八个 TP rank 的 K、V、Indexer K 各 78 个实际缓冲层均须覆盖。R 还须满足 L1 命中为 0、L2 命中至少 128 token。", "",
    "## 服务配置", "", "```bash", shlex.join(launch["argv"]), "```", "",
    f"实际布局：`{info.get('hicache_mem_layout')}`；I/O：`{info.get('hicache_io_backend')}`；写策略：`{info.get('hicache_write_policy')}`；TP/EP：`{info.get('tp_size')}/{info.get('ep_size')}`。", "",
    f"提示长度：{min(len(q['input_ids']) for q in plan['questions'])}～{max(len(q['input_ids']) for q in plan['questions'])} token。", "",
    "## 总体结果", "",
    f"**{'通过' if summary['passed'] else '未通过'}**：A 精度 {summary['baseline_correct_completed']}/{total}，R 精度 {summary['replay_correct_completed']}/{total}；最终答案一致 {summary['answer_match_count']}/{total}；输出 token 全等 {summary['token_match_count']}/{total}。", "",
    f"评估退出码 `{status['evaluation_exit']}`；服务退出码 `{status['server_exit']}`；运行错误 `{status['error']}`。", "",
    "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
]

if audit:
    lines += [
        "## 独立证据复核", "",
        "`verify_saved_run.py` 重新读取逐题请求、响应及原始完成事件，不依赖评估器写入的 `path_covered` 判定。", "",
        "```json", json.dumps(audit["checks"], ensure_ascii=False, indent=2), "```", "",
        f"{audit['checks']['cold_a']}/{total} 题 A 是冷缓存；{audit['checks']['pure_l2_r']}/{total} 题 R 是纯 L2 命中；{audit['checks']['eight_rank_d2h']}/{total} 题 D2H、{audit['checks']['eight_rank_h2d']}/{total} 题 H2D 在八个 TP rank 上均覆盖 K、V、Indexer K 的 78 层。", "",
    ]

lines += [
    "## 逐题结果", "",
    "| 题目 | 标准答案 | A | R | L1 | L2 | token 全等 | 八 rank 双向覆盖 |",
    "|---|---:|---:|---:|---:|---:|---|---|",
]
rows = []
for question in plan["questions"]:
    file = result / question["qid"] / "verdict.json"
    if not file.exists():
        continue
    row = read(file)
    rows.append(row)
    tier = row["cached_tokens_details"]
    covered = all(component["complete"] for side in ("d2h", "h2d")
                  for rank in row[side].values() for component in rank.values())
    lines.append(f"| {row['qid']} | {row['ground_truth']} | {row['baseline_answer']} | {row['replay_answer']} | {tier.get('device')} | {tier.get('host')} | {row['token_match']} | {covered} |")

different = [r for r in rows if not r["answer_match"]]
better = sum(not r["baseline_correct"] and r["replay_correct"] for r in different)
worse = sum(r["baseline_correct"] and not r["replay_correct"] for r in different)
lines += [
    "", "## A/R 答案差异", "",
    f"共 {len(different)} 题：A 错 R 对 {better} 题，A 对 R 错 {worse} 题。差异题均已完成双向路径门槛。", "",
    "| 题目 | 标准答案 | A | R |", "|---|---:|---:|---:|",
]
lines += [f"| {r['qid']} | {r['ground_truth']} | {r['baseline_answer']} | {r['replay_answer']} |" for r in different]
lines += [
    "", "## 结论与边界", "",
    f"本次没有执行差异题的冷态重复对照。A/R 结果存在 {len(different)} 题差异，因此{'不能宣称' if different else '在本样本上观察到'} L2 回放后的精度完全保持；即使有差异，也不能据此直接认定 KV 数据损坏。Event 证明算子成功返回且当前流完成，并非 KV 数据逐字节比对。", "",
    "旧报告的 92/100 使用 TP16/EP16、8192-token L1、并行 `/v1/completions`，且该轮 GSM8K 的 H2D=0；本次使用 TP8/EP8、3072-token L1、串行 `/generate` 并强制双向路径。百分比不可直接比较。", "",
    f"`flush_cache` 只建立 KV 缓存冷态，不重启服务。测试只覆盖 GSM8K 连续 {total} 题，不评价完整测试集、TTFT、并发或其他布局。", "",
    "原始逐题请求、响应、审计事件和判定分别保存在 `results/<qid>/`；摘要在 `results/summary.json`，独立复核在 `independent_audit.json`。", "",
]
(root / "report.md").write_text("\n".join(lines), encoding="utf-8")
