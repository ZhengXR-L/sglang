# GLM-5.2-w4a8c8 GSM8K MLA/Indexer D2H/H2D 精度验证

运行时间：2026-10-28 08:58:00 HKT 至 2026-10-28 11:24:44 HKT。

服务器：`113.46.15.88`；容器：`sglang-lzx-B103`；物理 NPU：`4,5,6,7,8,9,14,15`。

代码提交：`183512c7f180befd45203d7783f3fd55c02fee47`；完整证据目录：`/home/l00889655/sglang/glm52_aiv_100_20261028_085800`。

## 方法

沿用旧 GSM8K 提示：测试集前 5 题作示例，评估后续 100 题；temperature=0、max_new_tokens=512、stop=Question:。
每题串行执行 flush → A 冷缓存生成 → 等待 K/V/Indexer K 全层 D2H → B 分块不同前缀请求驱逐 L1 → R 同输入重放 → 等待 K/V/Indexer K 全层 H2D。A 和 R 的输入 token ID 完全相同。
审计钩子在真实 `memfabric_hybrid.offload.kv_exchange_copy` 返回 0 后记录当前 NPU 流 Event；Event 完成才计入覆盖。八个 TP rank 的 K、V、Indexer K 各 78 个实际缓冲层均须覆盖。R 还须满足 L1 命中为 0、L2 命中至少 128 token。

## 服务配置

```bash
/usr/local/python3.12.13/bin/python3 -m sglang.launch_server --model-path /mnt/paas/weights/GLM-5.2-w4a8c8 --host 127.0.0.1 --port 32348 --tp-size 8 --ep 8 --moe-a2a-backend deepep --deepep-mode normal --deepep-dispatcher-output-dtype bf16 --enable-hicache --hicache-mem-layout layer_first --hicache-io-backend kernel_ascend --hicache-write-policy write_through --hicache-ratio 2.5 --mem-fraction-static 0.86 --max-total-tokens 3072 --device npu --attention-backend ascend --disable-cuda-graph --log-level info
```

实际布局：`layer_first_kv_split`；I/O：`kernel_ascend`；写策略：`write_through`；TP/EP：`8/8`。

提示长度：696～797 token。

## 总体结果

**未通过**：A 精度 97/100，R 精度 94/100；最终答案一致 93/100；输出 token 全等 28/100。

评估退出码 `1`；服务退出码 `0`；运行错误 `None`。

```json
{
  "planned_questions": 100,
  "covered_questions": 100,
  "path_coverage": 1.0,
  "all_questions_complete": true,
  "baseline_accuracy": 0.97,
  "replay_accuracy": 0.94,
  "baseline_correct_completed": 97,
  "replay_correct_completed": 94,
  "answer_match_count": 93,
  "token_match_count": 28,
  "absolute_accuracy_gate": null,
  "error": null,
  "passed": false,
  "protocol": "original first-five test-row shots; serial A/B/R with per-TP-rank K/V/Indexer D2H and H2D completion",
  "cold_definition": "flush establishes KV-cache cold, not process/state-memory cold"
}
```

## 独立证据复核

`verify_saved_run.py` 重新读取逐题请求、响应及原始完成事件，不依赖评估器写入的 `path_covered` 判定。

```json
{
  "questions": 100,
  "cold_a": 100,
  "pure_l2_r": 100,
  "eight_rank_d2h": 100,
  "eight_rank_h2d": 100,
  "a_correct": 97,
  "r_correct": 94,
  "answers_equal": 93,
  "tokens_equal": 28,
  "a_right_r_wrong": 5,
  "a_wrong_r_right": 2,
  "both_wrong_different": 0
}
```

100/100 题 A 是冷缓存；100/100 题 R 是纯 L2 命中；100/100 题在八个 TP rank 上的 K、V、Indexer K 均有完整的 78 层 D2H 与 H2D 完成事件。

## 逐题结果

| 题目 | 标准答案 | A | R | L1 | L2 | token 全等 | 八 rank 双向覆盖 |
|---|---:|---:|---:|---:|---:|---|---|
| test-5 | 64 | 64 | 64 | 0 | 640 | False | True |
| test-6 | 2.6E+2 | 2.6E+2 | 2.6E+2 | 0 | 640 | False | True |
| test-7 | 1.6E+2 | 1.6E+2 | 1.6E+2 | 0 | 640 | False | True |
| test-8 | 45 | 45 | 45 | 0 | 768 | False | True |
| test-9 | 4.6E+2 | 4.6E+2 | 4.6E+2 | 0 | 640 | False | True |
| test-10 | 366 | 366 | 366 | 0 | 640 | False | True |
| test-11 | 694 | 694 | 694 | 0 | 640 | False | True |
| test-12 | 13 | 12 | 12 | 0 | 640 | False | True |
| test-13 | 18 | 18 | 18 | 0 | 640 | True | True |
| test-14 | 6E+1 | 6E+1 | 6E+1 | 0 | 640 | False | True |
| test-15 | 125 | 125 | 125 | 0 | 640 | False | True |
| test-16 | 2.3E+2 | 2.3E+2 | 2.3E+2 | 0 | 640 | False | True |
| test-17 | 5.75E+4 | 5.75E+4 | 5.75E+4 | 0 | 640 | False | True |
| test-18 | 7 | 7 | 7 | 0 | 640 | False | True |
| test-19 | 6 | 6 | 6 | 0 | 640 | False | True |
| test-20 | 15 | 15.67 | 15 | 0 | 640 | False | True |
| test-21 | 14 | 14 | 14 | 0 | 640 | True | True |
| test-22 | 7 | 7 | 7 | 0 | 640 | False | True |
| test-23 | 8 | 8 | 8 | 0 | 640 | True | True |
| test-24 | 26 | 26 | 26 | 0 | 640 | False | True |
| test-25 | 2 | 2 | 2 | 0 | 640 | False | True |
| test-26 | 243 | 243 | 243 | 0 | 640 | True | True |
| test-27 | 16 | 16 | 16 | 0 | 640 | True | True |
| test-28 | 25 | 25 | 25 | 0 | 640 | True | True |
| test-29 | 104 | 104 | 104 | 0 | 640 | True | True |
| test-30 | 109 | 109 | 109 | 0 | 640 | False | True |
| test-31 | 8E+1 | 8E+1 | 8E+1 | 0 | 640 | False | True |
| test-32 | 35 | 35 | 35 | 0 | 640 | False | True |
| test-33 | 7E+1 | 7E+1 | 7E+1 | 0 | 640 | True | True |
| test-34 | 23 | 23 | 23 | 0 | 640 | True | True |
| test-35 | 9 | 9 | 9 | 0 | 640 | False | True |
| test-36 | 75 | 75 | 75 | 0 | 640 | False | True |
| test-37 | 2 | 0 | 2 | 0 | 640 | False | True |
| test-38 | 1E+1 | 1E+1 | 1E+1 | 0 | 640 | True | True |
| test-39 | 18 | 18 | 18 | 0 | 640 | False | True |
| test-40 | 8 | 8 | 8 | 0 | 640 | False | True |
| test-41 | 2E+2 | 2E+2 | 2E+2 | 0 | 768 | True | True |
| test-42 | 26 | 26 | 26 | 0 | 640 | False | True |
| test-43 | 48 | 48 | 2.4E+2 | 0 | 640 | False | True |
| test-44 | 2E+1 | 2E+1 | 2E+1 | 0 | 640 | False | True |
| test-45 | 104 | 104 | 104 | 0 | 640 | False | True |
| test-46 | 163 | 163 | 163 | 0 | 640 | False | True |
| test-47 | 8E+2 | 8E+2 | 8E+2 | 0 | 640 | True | True |
| test-48 | 8 | 8 | 8 | 0 | 640 | False | True |
| test-49 | 3E+1 | 3E+1 | 3E+1 | 0 | 640 | False | True |
| test-50 | 294 | 294 | 294 | 0 | 640 | False | True |
| test-51 | 5 | 5 | 5 | 0 | 640 | False | True |
| test-52 | 15 | 15 | 15 | 0 | 640 | False | True |
| test-53 | 4E+1 | 4E+1 | 4E+1 | 0 | 768 | False | True |
| test-54 | 4E+1 | 4E+1 | 4E+1 | 0 | 640 | False | True |
| test-55 | 14 | 14 | 14 | 0 | 640 | True | True |
| test-56 | 3 | 3 | 3 | 0 | 640 | False | True |
| test-57 | 83 | 83 | 83 | 0 | 640 | False | True |
| test-58 | 57 | 57 | 57 | 0 | 640 | False | True |
| test-59 | 187 | 187 | 187 | 0 | 640 | True | True |
| test-60 | 17 | 17 | 17 | 0 | 640 | False | True |
| test-61 | 1.43E+3 | 1.43E+3 | 1.43E+3 | 0 | 640 | True | True |
| test-62 | 2.5E+4 | 2.5E+4 | 7.5E+4 | 0 | 640 | False | True |
| test-63 | 1596 | 1596 | 1596 | 0 | 640 | False | True |
| test-64 | 3E+2 | 3E+2 | 3E+2 | 0 | 640 | False | True |
| test-65 | 36 | 36 | 36 | 0 | 640 | False | True |
| test-66 | 48 | 48 | 48 | 0 | 640 | True | True |
| test-67 | 595 | 595 | 595 | 0 | 640 | True | True |
| test-68 | 36 | 36 | 36 | 0 | 640 | True | True |
| test-69 | 6E+1 | 6E+1 | 6E+1 | 0 | 640 | False | True |
| test-70 | 7425 | 7425 | 7425 | 0 | 640 | False | True |
| test-71 | 6E+1 | 6E+1 | 6E+1 | 0 | 640 | False | True |
| test-72 | 221 | 221 | 221 | 0 | 640 | True | True |
| test-73 | 255 | 255 | 255 | 0 | 640 | False | True |
| test-74 | 88 | 88 | 88 | 0 | 768 | True | True |
| test-75 | 6E+1 | 6E+1 | 6E+1 | 0 | 640 | False | True |
| test-76 | 5 | 5 | 5 | 0 | 640 | False | True |
| test-77 | 1E+2 | 1E+2 | 1E+2 | 0 | 640 | False | True |
| test-78 | 6 | 6 | 5.94 | 0 | 640 | False | True |
| test-79 | 7E+1 | 7E+1 | 7E+1 | 0 | 640 | True | True |
| test-80 | 1E+1 | 1E+1 | 1E+1 | 0 | 640 | False | True |
| test-81 | 17 | 17 | 17 | 0 | 640 | False | True |
| test-82 | 623 | 623 | 623 | 0 | 640 | False | True |
| test-83 | 6E+2 | 6E+2 | 6E+2 | 0 | 640 | True | True |
| test-84 | 15 | 15 | 15 | 0 | 640 | True | True |
| test-85 | 44 | 44 | 44 | 0 | 640 | False | True |
| test-86 | 22 | 22 | 22 | 0 | 640 | False | True |
| test-87 | 9.36E+3 | 9.36E+3 | 1.008E+4 | 0 | 640 | False | True |
| test-88 | 8E+3 | 8E+3 | 8E+3 | 0 | 640 | False | True |
| test-89 | 24 | 24 | 24 | 0 | 640 | False | True |
| test-90 | 225 | 225 | 225 | 0 | 640 | False | True |
| test-91 | 28 | 28 | 28 | 0 | 640 | False | True |
| test-92 | 4 | 4 | 4 | 0 | 640 | True | True |
| test-93 | 36 | 36 | 36 | 0 | 640 | True | True |
| test-94 | 348 | 348 | 348 | 0 | 640 | False | True |
| test-95 | 4E+1 | 4E+1 | 4E+1 | 0 | 640 | True | True |
| test-96 | 3 | 3 | 3 | 0 | 640 | True | True |
| test-97 | 12 | 12 | 6 | 0 | 640 | False | True |
| test-98 | 5 | 5 | 5 | 0 | 640 | False | True |
| test-99 | 58 | 58 | 58 | 0 | 640 | False | True |
| test-100 | 175 | 175 | 175 | 0 | 640 | False | True |
| test-101 | 6 | 6 | 6 | 0 | 640 | False | True |
| test-102 | 26 | 26 | 26 | 0 | 640 | True | True |
| test-103 | 1.4E+2 | 1.4E+2 | 1.4E+2 | 0 | 640 | False | True |
| test-104 | 5E+2 | 5E+2 | 5E+2 | 0 | 640 | False | True |

## A/R 答案差异

共 7 题：A 错 R 对 2 题，A 对 R 错 5 题。差异题均已完成双向路径门槛。

| 题目 | 标准答案 | A | R |
|---|---:|---:|---:|
| test-20 | 15 | 15.67 | 15 |
| test-37 | 2 | 0 | 2 |
| test-43 | 48 | 48 | 2.4E+2 |
| test-62 | 2.5E+4 | 2.5E+4 | 7.5E+4 |
| test-78 | 6 | 6 | 5.94 |
| test-87 | 9.36E+3 | 9.36E+3 | 1.008E+4 |
| test-97 | 12 | 12 | 6 |

## 结论与边界

本次没有执行差异题的冷态重复对照。A/R 结果存在 7 题差异，因此不能宣称 L2 回放后的精度完全保持，也不能据此直接认定 KV 数据损坏。Event 证明算子成功返回且当前流完成，并非 KV 数据逐字节比对。

旧报告的 92/100 使用 TP16/EP16、8192-token L1、并行 `/v1/completions`，且该轮 GSM8K 的 H2D=0；本次使用 TP8/EP8、3072-token L1、串行 `/generate` 并强制双向路径。百分比不可直接比较。

`flush_cache` 只建立 KV 缓存冷态，不重启服务。测试只覆盖 GSM8K 连续 100 题，不评价完整测试集、TTFT、并发或其他布局。

原始逐题请求、响应、审计事件和判定分别保存在 `results/<qid>/`；摘要在 `results/summary.json`，独立复核在 `independent_audit.json`。
