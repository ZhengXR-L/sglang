"""Generate a human-readable report from saved Coder MLA evidence."""
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
info = read(result / "server_info.json")
lines = ["# DeepSeek-Coder-V2-Lite-Instruct GSM8K MLA D2H/H2D 精度验证", "",
         f"代码提交：`{launch['git_commit']}`；执行目录：`{root}`。", "",
         "## 方法", "",
         "按旧验证脚本使用测试集前 5 题作示例，之后的题目作评估；temperature=0，max_new_tokens=512，stop=Question:。",
         "每题执行 flush → A 冷态生成 → 等待 K/V 全层 D2H → B 分块不同前缀请求驱逐 L1 → R 同输入重放 → 等待 K/V 全层 H2D。",
         "每次 kv_exchange_copy 返回 0 后，在当前 NPU 流记录 Event；Event 完成才算传输完成。校验 R 的 L1 命中为 0、L2 命中达到阈值。",
         "模型回答准确率和 A/R 输出 token 全等分别统计。", "",
         "## 启动配置", "", "```bash", shlex.join(launch["argv"]), "```", "",
         f"实际 HiCache 布局：`{info.get('hicache_mem_layout')}`；I/O：`{info.get('hicache_io_backend')}`。", "",
         "## 结果", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
         f"提示长度：{min(len(q['input_ids']) for q in plan['questions'])}～{max(len(q['input_ids']) for q in plan['questions'])} token。", "",
         "| 题目 | 标准答案 | A | R | L1 | L2 | token 全等 | K/V 双向全层完成 |", "|---|---:|---:|---:|---:|---:|---|---|"]
for q in plan["questions"]:
    file = result / q["qid"] / "verdict.json"
    if not file.exists():
        continue
    row = read(file)
    tier = row["cached_tokens_details"]
    path = all(x["complete"] for side in ("d2h", "h2d") for x in row[side].values())
    lines.append(f"| {row['qid']} | {row['ground_truth']} | {row['baseline_answer']} | {row['replay_answer']} | {tier.get('device')} | {tier.get('host')} | {row['token_match']} | {path} |")
lines += ["", "每题原始请求、响应、AIV 完成事件及判定分别保存于 results/<qid>/。",
          "`flush_cache` 仅清空 KV 缓存；不重启服务。审计 Event 反映当前流完成，评测不测量 TTFT。",
          "旧 86/100 的服务、并发和接口可能不同；本结果仅与其题目和提示模板口径对齐。", ""]
mismatches = [read(result / q["qid"] / "verdict.json") for q in plan["questions"]
              if (result / q["qid"] / "verdict.json").exists()]
mismatches = [r for r in mismatches if not r["answer_match"]]
lines += ["## A/R 答案差异", "",
          f"差异题数：{len(mismatches)}。只要存在差异，不能声称回传后精度完全保持；原因需结合冷态重复对照判断。", "",
          f"其中 A 错 R 对 {sum(not r['baseline_correct'] and r['replay_correct'] for r in mismatches)} 题，A 对 R 错 {sum(r['baseline_correct'] and not r['replay_correct'] for r in mismatches)} 题，两者都错但答案不同 {sum(not r['baseline_correct'] and not r['replay_correct'] for r in mismatches)} 题。", "",
          "| 题目 | 标准答案 | A | R |", "|---|---:|---:|---:|"]
lines += [f"| {r['qid']} | {r['ground_truth']} | {r['baseline_answer']} | {r['replay_answer']} |" for r in mismatches]
if len(sys.argv) > 2:
    control = Path(sys.argv[2]) / "summary.json"
    if control.exists():
        c = read(control)
        lines += ["", "## 冷态重复对照", "",
                  f"对照目录：`{control.parent}`；选中 {c['selected']} 题，完成 {c['completed']} 题。",
                  f"两次冷态答案相同 {c['cold_answer_match_count']} 题；输出 token 全等 {c['cold_token_match_count']} 题。", "",
                  f"冷态首次答案与原 A 相同 {sum(r['cold_1'] == r['prior_a'] for r in c['results'])} 题。", "",
                  "| 题目 | 原 A | 原 R | 冷态 1 | 冷态 2 | 冷态一致 |", "|---|---:|---:|---:|---:|---|"]
        lines += [f"| {r['qid']} | {r['prior_a']} | {r['prior_r']} | {r['cold_1']} | {r['cold_2']} | {r['cold_answer_match']} |" for r in c["results"]]
        lines += ["", "冷态重复只检验同输入在无 H2D 条件下的稳定性。原 R 差异与 L2 回放路径相关，但现有证据无法区分 KV 传输数据错误和缓存命中后的计算路径差异。"]
lines += [""]
(root / "report.md").write_text("\n".join(lines), encoding="utf-8")
