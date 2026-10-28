# DeepSeek-Coder-V2-Lite-Instruct GSM8K MLA AIV 验证

在 `sglang-lzx-B103` 容器中运行，使用独占的空闲 NPU。默认评估 100 题，先运行 2 题可作预检：

```bash
ROOT=/home/l00889655/sglang/sglang/coder_v2_lite_aiv_accuracy_20261027
RUN=/home/l00889655/sglang/coder_mla_aiv_$(date +%Y%m%d_%H%M%S)
python3 -u "$ROOT/run_100.py" "$RUN" 4 2
# 预检通过后用新 RUN 执行 100 题：
RUN=/home/l00889655/sglang/coder_mla_aiv_$(date +%Y%m%d_%H%M%S)
python3 -u "$ROOT/run_100.py" "$RUN" 4 100
```

第三个参数是 1–100 的评估题数；第二个参数是单个物理 NPU 编号。同一 RUN 不可复用。脚本仅终止自己启动的服务进程组。

`plan.json` 复刻旧 `e2e_gsm8k_eval.py` 的提示：测试集前 5 题作示例，评估第 6–105 题；生成参数为 temperature=0、max_new_tokens=512、stop=`Question:`。输入 token ID 及 SHA256 被保存，A 和 R 使用同一输入。

每题先 flush，再执行 A（冷缓存）→ 等待 K/V 两组件、27 层 D2H 完成 → B（约 2K token 一块的不同前缀请求，累计填充 L1）→ R（原输入），要求 R 的 L1 命中为 0、L2 命中至少 128 token，并等待 K/V 两组件、27 层 H2D 完成。审计从真实 `memfabric_hybrid.offload.kv_exchange_copy` 返回码及当前 NPU 流 Event 取得完成证据，而非统计日志行数。每题保留请求、响应、事件和判定。评估单线程执行，以阶段标记绑定传输事件。

`summary.json` 分别报告 A 与 R 精度、答案一致数和输出 token 完全一致数。只有全部题目通过路径门槛且 A/R 最终答案一致，整体 `passed` 才为 true。输出 token 全等是额外诊断项；预检中曾有答案相同但推理文字不同的样本。单题预测错误会降低绝对精度，但不直接说明回传造成损坏。可用 `--require-token-match` 增加严格文本一致门槛，或用 `--min-accuracy` 增加绝对精度门槛。

旧 86/100 的评估题目与提示模板相同，但旧测试使用 `/v1/completions` 并行调用及 `hicache-ratio=0.5`；本脚本使用 `/generate`、单线程、`hicache-ratio=2.0`（保留 A 的 L2 副本）且强制走 D2H/H2D。因此旧数字只作参考，不作为新测试的绝对通过标准。

若 A/R 答案不一致，`run_cold_repeat.py <ABR_RUN> <NEW_RUN> <NPU_ID>` 会在同样的服务配置下对这些题各执行两次 `flush_cache → generate`，保存原始响应，用来判断冷态生成本身的重复性。该对照只用于归因，不能替代 A/B/R 的 D2H/H2D 证据。

## 迁移到另一台服务器

需要 Linux、可用的 Ascend NPU/驱动与工具链、当前测试兼容的 SGLang 分支、`sgl_kernel_npu`、`memfabric_hybrid`、模型权重及 GSM8K test JSONL。数据每行必须有 `question`、`answer`。先检查 `npu-smi info`、模型路径和测试端口，并用一张空闲物理卡做 2 题预检。`run_server.sh` 固定读取 `/usr/local/Ascend/...` 环境脚本，若安装位置不同需调整。

自动入口 `run_100.py` 中的仓库外测试集、模型权重、端口 `32345` 均为写死值；换机器时先修改这些值，同时使 `run_server.sh` 的 `MODEL_PATH`、`SGLANG_ROOT`、`PORT` 与计划和评估地址一致。`run_100.py` 会创建全新的运行目录，拷贝当时脚本，启动并只清理自己创建的服务。不要复用旧 RUN，确保端口空闲且没有其他客户端流量。可通过 `SGLANG_ROOT` 指定仓库，但必须是包含 `python/sglang` 的根目录。服务配置为 TP1、`layer_first`、`kernel_ascend`、write-through、HiCache ratio 2.0；如改这些配置，结果不能直接与原报告比较。

运行后检查 `run_status.json`、`results/summary.json`、`results/` 中逐题记录和 `audit/` 事件日志；`validation_report.md`、`validation_summary.json` 记录原运行结论。确认每题 A 的冷态、R 的纯 L2 命中、27 层 K/V D2H 与 H2D 完成，再解读 A/R 精度、答案及 token 一致率。历史报告是固定环境的证据，新服务器必须重新生成证据。
