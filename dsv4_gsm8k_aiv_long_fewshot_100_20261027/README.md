# 长 few-shot GSM8K + 逐题 AIV D2H/H2D 精度验证

适配目标：远程 `layer_first_ascendc_sparse_copy` 的 DSV4 helper ABI，已核对提交 `356c078865ae63511685837082e1f1a49d697b62`。使用独立测试服务；不修改仓库业务代码，不自动下载数据，仅清理本测试启动的服务。已完成 100 题真实 NPU 精度评测，结果见 `npu_report_100_20261027.md`。

## 文件

- `gsm8k_aiv.py`：`plan / serve / eval` 三个子命令。
- `aiv_audit_hook.py`：每个 worker 中记录真实 `offload.kv_exchange_copy` 调用及 NPU event 完成事件。
- `run_server.sh`：沿用已验证的单机 TP8/EP8/attention-DP8 参数和 Ascend 环境。
- `run_100.py`：串行执行 100 题计划、服务启动、评测、清理与报告生成。
- `report_npu.py`：依据运行产物生成 NPU 验证报告。
- `npu_report_100_20261027.md`：本次已完成的 100 题验证报告。

## 准备不可变测试计划

需要独立的 GSM8K **train.jsonl** 和 **test.jsonl**，每行含 `question`、`answer`。当前已有的 `/home/l00889655/sglang/gsm8k_test.jsonl` 只能用于测试题；不能拿相同测试题作为 few-shot 示例。

在容器 `sglang-lzx-B103` 内，设本目录为 `$HERE`。以下 train 路径需要替换为实际准备好的训练集：

```bash
HERE=/home/l00889655/sglang/sglang/dsv4_gsm8k_aiv_long_fewshot_100_20261027
RUN=/home/l00889655/sglang/gsm8k_aiv_run_001
mkdir -p "$RUN"
python3 "$HERE/gsm8k_aiv.py" plan \
  --train /path/to/gsm8k_train.jsonl \
  --test /home/l00889655/sglang/gsm8k_test.jsonl \
  --test-offset 5 --num-questions 100 \
  --model-path /mnt/paas/weights/DeepSeek-V4-Flash-0731-w8a8 \
  --min-prompt-tokens 8192 --max-shots 128 --seed 0 \
  --output "$RUN/plan.json"
```

固定测试索引 5..104，与原 100 题一致；不随实际 shot 数变化。训练示例按 seed 固定顺序增加，直到实际 tokenizer 计算的每题长度达到 8192。不会填随机文本、截断训练答案或用测试题补足长度。计划保存完整 prompt、token IDs、真值、shot 索引和数据哈希。

这是 long-few-shot 协议，不应直接与原 5-shot 的 96/100 比较。公平对照是同一份 plan 的 A 与 R、或两种布局在同一 plan 上的结果。

## 启动专用服务

先确认八张卡空闲，显式设置卡号。不要直接复制某组卡号后覆盖他人服务：

```bash
export AUDIT_DIR="$RUN/audit"
export NPU_DEVICES=替换为八张空闲卡的逗号列表
export PORT=32420
bash "$HERE/run_server.sh" > "$RUN/server.log" 2>&1
```

前台运行，退出时按 Ctrl-C 交给 SGLang 自身清理。`AUDIT_DIR` 必须全新；不要提前往其中放日志。`serve` 在其中生成私有 sitecustomize，并通过 PYTHONPATH 传给每个子进程。使用其他分支时若 helper 不包含 pool/layer 等字段，不能直接认为本工具兼容。

## 运行评测

在另一个容器终端中，等待 `/health` 返回 200，然后：

```bash
python3 "$HERE/gsm8k_aiv.py" eval \
  --plan "$RUN/plan.json" --audit-dir "$RUN/audit" \
  --base-url http://127.0.0.1:32420 --dp-rank 0 \
  --eviction-tokens 65536 --max-new-tokens 512 \
  --output "$RUN/results" --min-accuracy 0.90
```

`0.90` 只是示例门槛，应按项目标准调整；不设置 min-accuracy 时只判定传输覆盖、有效输出和相对答案保持，不会声称满足某个绝对精度 SLA。可加 `--require-token-match` 将原始输出 IDs 全等也设为硬门槛；默认仍报告 token 一致率，但精度保持以数值答案为硬门槛。

上述手工 `plan / serve / eval` 步骤不会自行启动测试；`run_100.py` 可按相同协议自动串行执行 100 题。首次部署时仍建议先用 `plan --num-questions 2` 验证环境。

## 每题代码流程

1. 写当前题/阶段标识，flush 逻辑缓存。A 和 R 共用计划中的同一份 input_ids。
2. 发 A，要求 cached_tokens=0；记录普通重算答案和原始响应。
3. 等 A 阶段七个池的所有物理层都出现成功 AIV **完成事件**，证明 D2H 真实发生。缺池、缺层、错误返回、超时都失败。
4. 同一 DP rank 发 65536-token 的不同首 token B，输出仅 8 token。B 不计入 GSM8K 精度。
5. 发 R，严格要求 device=0、host>=2048 且实际 dp_rank 确认一致。仍在 L1、无缓存分项或路由未确认都会失败。
6. 等 R 阶段七池/各层 AIV H2D 完成事件，再提取数值答案并对照真值和 A。

池集合：SWA、C4 KV、C4 Indexer K、Indexer scale、C4 state、Indexer state、C128。该列表针对当前模型，不会因缺日志而自动缩减。

## 如何保证观测的是 AIV

hook 包住 DSV4 helper，给当前线程附加 pool/layer 上下文；但记录完成事件的位置在**真实 Memfabric kv_exchange_copy 返回 0 后**。随后在当前 NPU stream 记录 event，后台线程 query event 完成后写 JSONL。不会额外调用 stream/device synchronize，不会把普通 memcpy 记成 AIV。它要求现有 offload 算子遵守当前 stream 的既有调用契约。

它替代不了内存内容校验，所以还要求输出有效、A/R 数值答案一致，支持额外的输出 token 一致门槛。数据通路正确性与绝对数学正确率分别报告。

阶段归属依赖串行、**单一测试客户端、专用服务**和共享 audit 目录；不支持将本客户端挂到有其他流量的服务上作逐题证明。该观测会增加 event/JSONL 开销，不用于 TTFT 性能评估。

## 输出与失败口径

- `plan.json`：完整不可变输入和真值。
- `audit/launch.json`、`calls-*.jsonl`：启动命令及各 worker 的真实调用/完成证据。
- `results/server_info.json`、`arguments.json`：实际服务配置、容量及评测参数。
- 每题 `requests.jsonl / responses.jsonl / verdict.json`：A/B/R、缓存分项、逐池逐层证据及答案。
- `results/summary.json`：覆盖率、A/R 准确率、数值/原始 token 一致率、绝对精度门槛和最终结果。

路径证据不足或请求失败会中止，并以全部计划题数作分母报告未完成，不会把未覆盖题悄悄排除后宣称通过。答案变化会记录并继续统计，最终失败。即使 A/R 都答错但相同，也只表示该题答案保持，绝对精度仍按真值计算。

若题集未全部完成，accuracy 字段为 null，并另列已完成题的正确数量，避免将部分样本冒充整套精度。

flush_cache 只建立 KV 缓存冷态，不能重置已观察到的 C4 运行历史效应。每种布局应独立重启服务、使用相同计划与顺序，并在任何额外长往返预热前开始第一题。不要拿独立 AIV smoke 加原短 GSM8K 的结果代替本评测。

## 迁移到另一台服务器

准备可用的 Ascend 八卡、驱动与工具链、兼容的 SGLang DSV4 分支、`sgl_kernel_npu`/Memfabric、相同模型权重，以及独立的 GSM8K train/test JSONL（每行 `question`、`answer`）。确认 `npu-smi info` 中八张物理卡空闲，端口 `32420` 可用，且测试期间没有其他客户端访问专用服务。`run_server.sh` 读取 `/usr/local/Ascend/...` 的环境脚本；安装路径不同需调整。

上面的手工 `plan → serve → eval` 更方便迁移：把 `$HERE`、train/test 路径、模型路径、八卡列表、端口和 `SGLANG_ROOT` 替换为新服务器实际值，并保持 plan 的模型路径与服务模型路径一致。原 `run_100.py` 则把 train/test、模型和端口写死在代码中；要使用一键入口，必须先改这些常量，再运行 `python3 -u "$HERE/run_100.py" "$RUN" 0,1,2,3,4,5,6,7`，其中 RUN 是尚不存在的新目录，卡号按实际空闲卡替换。建议先用手工 plan 的 `--num-questions 2` 完成小样本验证。

复现时保持 TP8/EP8/attention-DP8、`layer_first`、`kernel_ascend`、write-through、HiCache ratio 2.5 和相同计划/题序；更换模型、代码或池容量需作为新实验记录。查看新 RUN 的 `run_status.json`、`results_100/summary.json`、逐题请求响应和 `audit/calls-*.jsonl`，逐题核对七池各层的 AIV D2H/H2D 完成事件及 R 的纯 L2 命中。原结论见 `npu_report_100_20261027.md`。
