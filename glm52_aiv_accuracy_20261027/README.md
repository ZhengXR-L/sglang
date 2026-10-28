# GLM-5.2 GSM8K 双向 HiCache 测试复现指南

本目录用于验证 `GLM-5.2-w4a8c8` 在 Ascend NPU 上的 GSM8K 精度，以及每题真实的 D2H、H2D 路径。`validation_report.md` 是 2026-10-28 在原服务器完成的结果；新服务器必须生成自己的运行目录和报告，不能沿用该结果。

## 1. 准备环境

在目标服务器的 NPU 容器中准备：

- 一份包含本目录和 `python/sglang` 的 SGLang Git 工作区。本测试依赖 `sglang.srt.mem_cache.pool_host.mla.MLATokenToKVPoolHost._launch_ascendc_kv_exchange`，换代码版本时先核对该接口。
- 可读取的 `GLM-5.2-w4a8c8` 权重。脚本要求 `config.json` 的 `model_type=glm_moe_dsa`、78 层；本模型的 K/V/Indexer K 实际缓冲层均为 78 层。
- GSM8K 测试 JSONL，至少 105 行，每行有字符串 `question`、`answer`，标准答案位于 `answer` 的 `####` 后。前 5 行作 few-shot 示例，评估第 6～105 行；不需要训练集文件。
- 已安装且匹配的 Ascend CANN/ATB、`torch_npu`、SGLang 运行依赖、`sgl_kernel_npu`、`memfabric_hybrid` 和 `transformers`。需要八张**同时空闲且显存容量均衡**的物理 NPU；先用 `npu-smi info` 检查进程和 HBM 占用。

`run_server.sh` 的 Ascend/CANN 路径可通过 `ASCEND_TOOLKIT_ROOT`、`CANN_PYTHON_SITE_PACKAGES`、`ATB_ENV_SCRIPT` 覆盖；不设置时使用原服务器路径。新服务器若安装位置或 CANN 版本不同，先设置这三个变量；确保环境初始化后 `python3 -c 'import acl; import sglang.srt.models.glm4_moe'` 成功。服务固定采用 TP8/EP8、DeepEP、BF16 KV、`layer_first_kv_split`、`kernel_ascend`、`write_through`、`max-total-tokens=3072`。模型、硬件或这些参数改变后，需重新解释精度和缓存结果。

## 2. 在容器中运行

进入目标容器的 shell，按实际路径设置变量。`SGLANG_ROOT` 指向 Git 工作区，`RUN_BASE` 用于存放较大的逐题原始证据；端口只需在容器内可用。

```bash
export SGLANG_ROOT=/path/to/sglang
export MODEL_PATH=/path/to/GLM-5.2-w4a8c8
export DATA_PATH=/path/to/gsm8k_test.jsonl
export PORT=32348
export VALIDATION_HOST_LABEL=your-server-name
export VALIDATION_CONTAINER_LABEL=your-container-name
# 若 Ascend/CANN 安装路径不同，再设置以下三项：
# export ASCEND_TOOLKIT_ROOT=/actual/ascend-toolkit/latest
# export CANN_PYTHON_SITE_PACKAGES=/actual/cann/python/site-packages
# export ATB_ENV_SCRIPT=/actual/atb/set_env.sh
NPU_DEVICES=0,1,2,3,4,5,6,7    # 改成八张实际空闲的物理卡
RUN_BASE=/path/to/test-runs
SCRIPT_DIR="$SGLANG_ROOT/glm52_aiv_accuracy_20261027"
mkdir -p "$RUN_BASE"

# 先用新目录做 2 题预检
SMOKE="$RUN_BASE/glm52_smoke_$(date +%Y%m%d_%H%M%S)"
python3 -u "$SCRIPT_DIR/run_100.py" "$SMOKE" "$NPU_DEVICES" 2
cat "$SMOKE/run_status.json"
cat "$SMOKE/results/summary.json"

# 预检的服务与双向路径正常后，换新目录跑 100 题
RUN="$RUN_BASE/glm52_100_$(date +%Y%m%d_%H%M%S)"
python3 -u "$SCRIPT_DIR/run_100.py" "$RUN" "$NPU_DEVICES" 100
cat "$RUN/run_status.json"
cat "$RUN/results/summary.json"
```

也可以从宿主机用 `docker exec -it <容器名> bash` 进入容器后执行。每次必须使用**不存在的**运行目录；脚本会启动专用服务，结束时只清理它启动的服务进程组。`run_100.py` 支持上述 `MODEL_PATH`、`DATA_PATH`、`PORT`、`SGLANG_ROOT` 环境变量；未设置时才使用原服务器默认值。

## 3. 测试过程与通过条件

每题依次执行 `flush_cache → A → B → R`：

1. A 用 5-shot 提示在 KV 缓存冷态生成，要求 `cached_tokens=0`，并等待八个 TP rank 的 K、V、Indexer K 各 78 层 D2H 完成。
2. B 发送不同前缀、分块生成的驱逐请求，累计占用 L1；其输出不参与数学评分。
3. R 用与 A 完全相同的输入 token ID 重放，要求 L1 命中为 0、L2 命中至少 128 token，并等待八个 rank 的三组件各 78 层 H2D 完成。
4. 统计 A 与 R 的 GSM8K 正确题数、最终答案一致题数和输出 token 全等题数。整体 `passed=true` 要求每题路径完整且 A/R **最终答案一致**；token 全等仅作诊断指标。

审计钩子 `aiv_audit_hook.py` 包装真实 `kv_exchange_copy`，要求返回码为 0，并在当前 NPU 流 Event 完成后记录传输。它证明调用及流完成，**不等于** KV 数据逐字节一致。

## 4. 读取和归档结果

一键脚本自动保存：

| 文件 | 作用 |
|---|---|
| `run_status.json` | 运行、评估、独立复核及报告生成的退出状态 |
| `results/summary.json` | A/R 精度、答案及 token 一致率、路径覆盖率 |
| `independent_audit.json` | 100 题逐题原始事件的独立复核结果 |
| `report.md` | 逐题表和正式结论 |
| `results/<qid>/` | 原始请求、响应、AIV 事件及单题判定 |
| `audit/launch.json`、`sha256.json` | 启动参数、代码版本、环境和脚本/数据哈希 |

若 A/R 答案有差异，评估退出码可能为 `1`，但脚本仍会保存汇总和报告。先看 `run_status.json` 的 `error`、`independent_audit_exit`、`report_exit`，再解读 `passed`。100 题结束后可手动重跑复核和报告：

```bash
python3 "$SCRIPT_DIR/verify_saved_run.py" "$RUN"
python3 "$SCRIPT_DIR/report_npu.py" "$RUN"
cp "$RUN/report.md" "$SCRIPT_DIR/validation_report_$(date +%Y%m%d).md"
```

`verify_saved_run.py` 固定要求完整 100 题，不用于 2 题预检。原始证据保持在 `RUN_BASE`，仓库目录仅保存脚本和需要提交的报告。报告中的服务器/容器标签取自 `VALIDATION_HOST_LABEL`、`VALIDATION_CONTAINER_LABEL`；请在启动前设置。

## 5. 原服务器结果的解读边界

原服务器 100 题运行：A 97/100，R 94/100；最终答案一致 93/100；纯 L2 命中与三组件双向全层覆盖均为 100/100。因此精度保持门槛未通过。按照当次要求，没有对 7 道差异题做冷态重复对照。旧 92/100 报告使用 TP16/EP16、并行接口且 H2D=0，两次百分比不能直接比较。
