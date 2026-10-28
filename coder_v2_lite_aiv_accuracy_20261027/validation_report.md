# DeepSeek-Coder-V2-Lite-Instruct GSM8K MLA D2H/H2D 精度验证

代码提交：`183512c7f180befd45203d7783f3fd55c02fee47`；执行目录：`/home/l00889655/sglang/coder_mla_aiv_100_20261027_211414`。

## 方法

按旧验证脚本使用测试集前 5 题作示例，之后的题目作评估；temperature=0，max_new_tokens=512，stop=Question:。
每题执行 flush → A 冷态生成 → 等待 K/V 全层 D2H → B 分块不同前缀请求驱逐 L1 → R 同输入重放 → 等待 K/V 全层 H2D。
每次 kv_exchange_copy 返回 0 后，在当前 NPU 流记录 Event；Event 完成才算传输完成。校验 R 的 L1 命中为 0、L2 命中达到阈值。
模型回答准确率和 A/R 输出 token 全等分别统计。

## 启动配置

```bash
/usr/local/python3.12.13/bin/python3 -m sglang.launch_server --model-path /mnt/paas/weights/DeepSeek-Coder-V2-Lite-Instruct --host 127.0.0.1 --port 32345 --trust-remote-code --tp 1 --device npu --attention-backend ascend --enable-hicache --hicache-mem-layout layer_first --hicache-io-backend kernel_ascend --hicache-write-policy write_through --hicache-ratio 2.0 --mem-fraction-static 0.6 --max-total-tokens 16384 --log-level info
```

实际 HiCache 布局：`layer_first_kv_split`；I/O：`kernel_ascend`。

## 结果

```json
{
  "planned_questions": 100,
  "covered_questions": 100,
  "path_coverage": 1.0,
  "all_questions_complete": true,
  "baseline_accuracy": 0.84,
  "replay_accuracy": 0.81,
  "baseline_correct_completed": 84,
  "replay_correct_completed": 81,
  "answer_match_count": 89,
  "token_match_count": 46,
  "absolute_accuracy_gate": null,
  "error": null,
  "passed": false,
  "protocol": "original first-five test-row shots; serial A/B/R with per-question K/V D2H and H2D completion",
  "cold_definition": "flush establishes KV-cache cold, not process/state-memory cold"
}
```

提示长度：833～937 token。

| 题目 | 标准答案 | A | R | L1 | L2 | token 全等 | K/V 双向全层完成 |
|---|---:|---:|---:|---:|---:|---|---|
| test-5 | 64 | 64 | 64 | 0 | 768 | False | True |
| test-6 | 2.6E+2 | 2.6E+2 | 2.6E+2 | 0 | 768 | True | True |
| test-7 | 1.6E+2 | 1.2E+2 | 1.6E+2 | 0 | 768 | False | True |
| test-8 | 45 | 135 | 475 | 0 | 896 | False | True |
| test-9 | 4.6E+2 | 4.6E+2 | 4.6E+2 | 0 | 768 | True | True |
| test-10 | 366 | 366 | 366 | 0 | 768 | True | True |
| test-11 | 694 | 694 | 694 | 0 | 768 | False | True |
| test-12 | 13 | 12 | 12 | 0 | 768 | True | True |
| test-13 | 18 | 9 | 9 | 0 | 768 | True | True |
| test-14 | 6E+1 | 6E+1 | 6E+1 | 0 | 768 | False | True |
| test-15 | 125 | 125 | 125 | 0 | 896 | False | True |
| test-16 | 2.3E+2 | 2.3E+2 | 2.3E+2 | 0 | 768 | True | True |
| test-17 | 5.75E+4 | 5.75E+4 | 5.75E+4 | 0 | 768 | False | True |
| test-18 | 7 | 7 | 7 | 0 | 768 | True | True |
| test-19 | 6 | 4 | 4 | 0 | 768 | True | True |
| test-20 | 15 | 14.67 | 14.67 | 0 | 768 | True | True |
| test-21 | 14 | 17 | 14 | 0 | 768 | False | True |
| test-22 | 7 | 7 | 7 | 0 | 768 | True | True |
| test-23 | 8 | 8 | 8 | 0 | 768 | True | True |
| test-24 | 26 | 26 | 26 | 0 | 768 | False | True |
| test-25 | 2 | 2 | 2 | 0 | 768 | False | True |
| test-26 | 243 | 243 | 243 | 0 | 768 | True | True |
| test-27 | 16 | 16 | 16 | 0 | 768 | False | True |
| test-28 | 25 | 25 | 25 | 0 | 768 | True | True |
| test-29 | 104 | 104 | 104 | 0 | 768 | True | True |
| test-30 | 109 | 109 | 109 | 0 | 768 | True | True |
| test-31 | 8E+1 | 8E+1 | 8E+1 | 0 | 768 | False | True |
| test-32 | 35 | 35 | 35 | 0 | 768 | False | True |
| test-33 | 7E+1 | 7E+1 | 7E+1 | 0 | 768 | False | True |
| test-34 | 23 | 23 | 23 | 0 | 768 | False | True |
| test-35 | 9 | 9 | 9 | 0 | 768 | True | True |
| test-36 | 75 | 75 | 75 | 0 | 768 | False | True |
| test-37 | 2 | 2 | 2 | 0 | 768 | False | True |
| test-38 | 1E+1 | 0.16666666666666666 | 0.16666666666666666 | 0 | 768 | False | True |
| test-39 | 18 | 18 | 18 | 0 | 768 | False | True |
| test-40 | 8 | 8 | 8 | 0 | 768 | True | True |
| test-41 | 2E+2 | 2E+2 | 2E+2 | 0 | 896 | False | True |
| test-42 | 26 | 26 | 26 | 0 | 768 | True | True |
| test-43 | 48 | 6.6E+2 | 6.6E+2 | 0 | 768 | True | True |
| test-44 | 2E+1 | 2E+1 | 2E+1 | 0 | 768 | False | True |
| test-45 | 104 | 104 | 104 | 0 | 896 | True | True |
| test-46 | 163 | 277 | 277 | 0 | 896 | True | True |
| test-47 | 8E+2 | 8E+2 | 8E+2 | 0 | 768 | True | True |
| test-48 | 8 | 8 | 8 | 0 | 768 | True | True |
| test-49 | 3E+1 | 3E+1 | 3E+1 | 0 | 768 | True | True |
| test-50 | 294 | 294 | 294 | 0 | 768 | False | True |
| test-51 | 5 | 5 | 5 | 0 | 768 | False | True |
| test-52 | 15 | 15 | 15 | 0 | 768 | False | True |
| test-53 | 4E+1 | 4E+1 | 4E+1 | 0 | 896 | True | True |
| test-54 | 4E+1 | 4E+1 | 4E+1 | 0 | 768 | True | True |
| test-55 | 14 | 14 | 14 | 0 | 768 | True | True |
| test-56 | 3 | 3 | 3 | 0 | 768 | True | True |
| test-57 | 83 | 83 | 83 | 0 | 768 | False | True |
| test-58 | 57 | 57 | 57 | 0 | 768 | False | True |
| test-59 | 187 | 187 | 187 | 0 | 768 | True | True |
| test-60 | 17 | 17 | 17 | 0 | 768 | False | True |
| test-61 | 1.43E+3 | 1.43E+3 | 1.43E+3 | 0 | 768 | True | True |
| test-62 | 2.5E+4 | 1.025E+6 | 1.025E+6 | 0 | 768 | False | True |
| test-63 | 1596 | 1596 | 1596 | 0 | 768 | False | True |
| test-64 | 3E+2 | 1E+2 | 2E+2 | 0 | 896 | False | True |
| test-65 | 36 | 36 | 12 | 0 | 768 | False | True |
| test-66 | 48 | 48 | 44 | 0 | 768 | False | True |
| test-67 | 595 | 595 | 595 | 0 | 768 | True | True |
| test-68 | 36 | 36 | 36 | 0 | 768 | True | True |
| test-69 | 6E+1 | 6E+1 | 6E+1 | 0 | 768 | True | True |
| test-70 | 7425 | 7425 | 7425 | 0 | 768 | True | True |
| test-71 | 6E+1 | 6E+1 | 6E+1 | 0 | 768 | False | True |
| test-72 | 221 | 221 | 221 | 0 | 768 | False | True |
| test-73 | 255 | 255 | 255 | 0 | 768 | False | True |
| test-74 | 88 | 88 | 88 | 0 | 896 | True | True |
| test-75 | 6E+1 | 6E+1 | 7.5 | 0 | 768 | False | True |
| test-76 | 5 | 9 | 9 | 0 | 768 | False | True |
| test-77 | 1E+2 | 1E+2 | 1E+2 | 0 | 768 | True | True |
| test-78 | 6 | 6 | 6 | 0 | 768 | False | True |
| test-79 | 7E+1 | 7E+1 | 7E+1 | 0 | 768 | False | True |
| test-80 | 1E+1 | 1E+1 | 1E+1 | 0 | 768 | False | True |
| test-81 | 17 | 17 | 17 | 0 | 768 | False | True |
| test-82 | 623 | 623 | 623 | 0 | 768 | False | True |
| test-83 | 6E+2 | 6E+2 | 6E+2 | 0 | 768 | True | True |
| test-84 | 15 | 15 | 15 | 0 | 768 | False | True |
| test-85 | 44 | 44 | 48 | 0 | 768 | False | True |
| test-86 | 22 | 22 | 22 | 0 | 896 | False | True |
| test-87 | 9.36E+3 | 1.08E+3 | 1.08E+3 | 0 | 768 | True | True |
| test-88 | 8E+3 | 8E+3 | 8E+3 | 0 | 768 | False | True |
| test-89 | 24 | 18 | 32 | 0 | 768 | False | True |
| test-90 | 225 | 225 | 225 | 0 | 768 | True | True |
| test-91 | 28 | 28 | 28 | 0 | 768 | True | True |
| test-92 | 4 | 4 | 4 | 0 | 768 | True | True |
| test-93 | 36 | 36 | 36 | 0 | 896 | True | True |
| test-94 | 348 | 348 | 348 | 0 | 768 | False | True |
| test-95 | 4E+1 | 4E+1 | 4E+1 | 0 | 768 | True | True |
| test-96 | 3 | 3 | 3 | 0 | 768 | True | True |
| test-97 | 12 | 12 | 12 | 0 | 768 | False | True |
| test-98 | 5 | 5 | 5 | 0 | 896 | True | True |
| test-99 | 58 | 58 | 4E+1 | 0 | 768 | False | True |
| test-100 | 175 | 175 | 175 | 0 | 896 | False | True |
| test-101 | 6 | 6 | 6 | 0 | 768 | True | True |
| test-102 | 26 | 52 | 51 | 0 | 768 | False | True |
| test-103 | 1.4E+2 | 1.4E+2 | 1.4E+2 | 0 | 768 | False | True |
| test-104 | 5E+2 | 5E+2 | 5E+2 | 0 | 768 | False | True |

每题原始请求、响应、AIV 完成事件及判定分别保存于 results/<qid>/。
`flush_cache` 仅清空 KV 缓存；不重启服务。审计 Event 反映当前流完成，评测不测量 TTFT。
旧 86/100 的服务、并发和接口可能不同；本结果仅与其题目和提示模板口径对齐。

## A/R 答案差异

差异题数：11。只要存在差异，不能声称回传后精度完全保持；原因需结合冷态重复对照判断。

其中 A 错 R 对 2 题，A 对 R 错 5 题，两者都错但答案不同 4 题。

| 题目 | 标准答案 | A | R |
|---|---:|---:|---:|
| test-7 | 1.6E+2 | 1.2E+2 | 1.6E+2 |
| test-8 | 45 | 135 | 475 |
| test-21 | 14 | 17 | 14 |
| test-64 | 3E+2 | 1E+2 | 2E+2 |
| test-65 | 36 | 36 | 12 |
| test-66 | 48 | 48 | 44 |
| test-75 | 6E+1 | 6E+1 | 7.5 |
| test-85 | 44 | 44 | 48 |
| test-89 | 24 | 18 | 32 |
| test-99 | 58 | 58 | 4E+1 |
| test-102 | 26 | 52 | 51 |

## 冷态重复对照

对照目录：`/home/l00889655/sglang/coder_mla_cold_repeat_20261027_213010`；选中 11 题，完成 11 题。
两次冷态答案相同 11 题；输出 token 全等 11 题。

冷态首次答案与原 A 相同 11 题。

| 题目 | 原 A | 原 R | 冷态 1 | 冷态 2 | 冷态一致 |
|---|---:|---:|---:|---:|---|
| test-7 | 1.2E+2 | 1.6E+2 | 1.2E+2 | 1.2E+2 | True |
| test-8 | 135 | 475 | 135 | 135 | True |
| test-21 | 17 | 14 | 17 | 17 | True |
| test-64 | 1E+2 | 2E+2 | 1E+2 | 1E+2 | True |
| test-65 | 36 | 12 | 36 | 36 | True |
| test-66 | 48 | 44 | 48 | 48 | True |
| test-75 | 6E+1 | 7.5 | 6E+1 | 6E+1 | True |
| test-85 | 44 | 48 | 44 | 44 | True |
| test-89 | 18 | 32 | 18 | 18 | True |
| test-99 | 58 | 4E+1 | 58 | 58 | True |
| test-102 | 52 | 51 | 52 | 52 | True |

冷态重复只检验同输入在无 H2D 条件下的稳定性。原 R 差异与 L2 回放路径相关，但现有证据无法区分 KV 传输数据错误和缓存命中后的计算路径差异。
