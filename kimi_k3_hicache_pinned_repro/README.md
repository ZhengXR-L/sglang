# Kimi K3 HiCache L2 pinned memory 复现

## 测试目的

捕获 Kimi K3 在 Ascend NPU 上通过 HiCache L2 写回 KDA 状态时抛出的异常。

```text
RuntimeError: host buffer must use pinned memory
```

## 测试原理

以 `kernel_ascend` 和 `write_through` 启动 HiCache，连续发送共享 2048 token 前缀、末尾 token 不同的文本请求。每次请求默认生成 8 个 token，覆盖 prefill 和 decode，并给缓存写回留出时间。脚本查找异常文本及 `backup_from_device_all_layer`、`transfer_mamba_state` 两个栈帧。

## 前置依赖

- 运行在 Linux Ascend 容器内，具备 Bash、`setsid`、curl、Python 3、Git、`sglang` CLI 和可用的 NPU。
- 使用原故障的 Kimi K3 checkpoint、SGLang checkout、已安装的 `sgl_kernel_npu`、PyTorch / torch_npu 和 CANN 镜像。默认量化是 ModelSlim。
- 空闲卡数由 checkpoint、减层配置和并行方式决定。默认 4 卡只是配置示例。多机、DP、EP 参数按原启动配置填入 `EXTRA_SERVER_ARGS`，此入口负责启动本机进程。
- 请求直接构造 token ID，无需外部数据集。

## 环境变量

配置集中在 [env.sh](./env.sh)。可以编辑其中的默认值，或用同名环境变量覆盖。

| 变量 | 示例或默认值 | 含义 | 必填 |
| --- | --- | --- | --- |
| `REPRO_K3_MODEL_FILE` | 空 | 可选的独立 K3 模型适配文件，Mamba 仍从 SGLANG_ROOT 导入 | 否 |
| `REPRO_BYPASS_MOE` | `0` | 设为 1 时将 KimiMoE 前向改为直接返回输入，仅用于执行路径验证 | 否 |
| `MODEL_PATH` | 私有目录 | 实际模型路径，另设 Kimi Linear 路径时跳过减层准备 | 否 |
| `CONTEXT_LENGTH` | `4096` | 上下文上限 | 否 |
| `MAX_TOTAL_TOKENS` | `4096` | token pool 上限 | 否 |
| `MAX_RUNNING_REQUESTS` | `2` | 最大并发请求 | 否 |
| `SOURCE_MODEL` | `/mnt/paas/weights/Kimi-K3-w4a8` | 公共 ModelSlim W4A8 checkpoint | 是 |
| `PRIVATE_MODEL` | `/home/l00889655/kimi_truncation` | 私有五层目录，缺失时自动创建 | 否 |
| `SGLANG_ROOT` | 脚本所在仓库 | 容器里的 SGLang checkout | 是 |
| `ASCEND_ENV` | `/usr/local/Ascend/ascend-toolkit/latest/set_env.sh` | CANN 环境脚本 | 是 |
| `NPU_DEVICES` | `0,1,2,3` | 可见卡列表，列表长度作为本机 TP 默认值 | 是 |
| `PORT` | `30000` | 本机服务端口 | 否 |
| `STARTUP_TIMEOUT` | `1800` | 等待健康检查的秒数 | 否 |
| `REQUEST_TIMEOUT` | `120` | 每次请求的 HTTP 超时秒数 | 否 |
| `OBSERVE_SECONDS` | `10` | 请求阶段结束后继续观察日志的秒数 | 否 |
| `PREFIX_TOKENS` | `2048` | 公共前缀长度，应超过原配置的一页或一个状态跟踪间隔，并留出生成空间 | 否 |
| `REQUEST_COUNT` | `4` | 请求数，范围 2–100 | 否 |
| `QUANTIZATION` | `modelslim` | checkpoint 的量化方式 | 否 |
| `MEM_FRACTION_STATIC` | `0.75` | 设备静态内存比例 | 否 |
| `MAX_MAMBA_CACHE_SIZE` | `32` | 每个服务进程的 Mamba 状态池配置 | 否 |
| `HICACHE_RATIO` | `1.2` | Host / device cache 容量比例 | 否 |
| `HICACHE_MEM_LAYOUT` | `page_first` | HiCache 内存布局，可按原容器的配置覆盖 | 否 |
| `SGLANG_HICACHE_HOST_MEM_BACKEND` | 空 | 保留原环境的 allocator 选择，支持时可设 `acl` 或 `memfabric` | 否 |
| `SGLANG_NPU_HICACHE_MAMBA_IO` | `sync` | 选择包含目标 D2H 算子调用的同步分支 | 否 |
| `SGLANG_SET_CPU_AFFINITY` | `1` | 保留原用例 CPU affinity 设置 | 否 |
| `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS` | `1` | 每个 worker 限制为一张可见卡 | 否 |
| `SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE` | `1` | 原 NPU prefix cache 路径 | 否 |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` | `1` | 原 overlap plan stream 配置 | 否 |
| `SGLANG_RAGGED_VERIFY_MODE` | `static` | 保留原 ragged verify 配置 | 否 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | NPU allocator 配置 | 否 |
| `STREAMS_PER_DEVICE` | `32` | 每卡 stream 数 | 否 |
| `HCCL_BUFFSIZE` | `2000` | HCCL buffer 配置 | 否 |
| `HCCL_OP_EXPANSION_MODE` | `AIV` | HCCL 通信模式 | 否 |
| `EXTRA_SERVER_ARGS` | 空数组 | 在 env.sh 中设置原启动所需的额外并行、MoE 等参数 | 否 |

`env.sh` 还会派生 `ASCEND_RT_VISIBLE_DEVICES`、`PYTHONPATH` 和 `TP_SIZE`。

## 原始 dts_test.py 的单机改造

在本目录编辑 env.sh 后执行以下入口，它调用同目录 run_test.sh 完成服务起停、请求和报告生成。

```bash
python3 dts_test.py ./out/kimi-k3-5layers-l2
```

默认使用四张卡和私有五层目录。SGLANG_ROOT 应指向要验证的代码，例如镜像自带 checkout。kernel 使用已安装的 wheel。执行时核对 server.log 的 MAMBA_SOURCE。原 checkpoint 在 text_config 中记录 93 层，prepare_model.py 会在私有目录生成五层配置，保留第 4 层 full attention 和第 1、2、3、5 层 KDA。若使用外部已裁剪目录，设置 MODEL_PATH。

本地入口移除了多机 CI 基类及其集群节点、ConfigMap、Mooncake master 依赖。部署为单机 TP=4，attention DP 为默认 1。原脚本实际配置为 nnodes=4、tp-size=64、attn-dp-size=4。

L2 保留 kernel_ascend、write_through 和 hicache_ratio=1.2。移除 Mooncake L3 的存储参数、TENT/MemFabric 环境变量与库路径注入、storage 命中率断言。请在干净的 shell 中运行，避免继承原测试的 MC_*、MF_* 等环境变量。

减层后改为执行路径判定，移除了 GPQA/GSM8K 精度门槛。连续请求使用 2048 token 的共享前缀并各生成 8 token，替代原来的 64K 输入、1024 token 输出及 32 并发。默认上下文与 token pool 上限为 4096，运行请求上限为 2。

DSPARK 需要独立 draft checkpoint 及其减层/缓存配置，这版移除了 speculative 参数。MoE 仍默认执行，REPRO_BYPASS_MOE=1 可用于绕过此前的 renorm 算子兼容错误。单机使用默认 TP MoE，移除了 DeepEP 专用启动参数及其环境变量。图模式暂时关闭，以便直接观察调用栈。

分片只下载了前五层所需文件时，还需准备匹配的 checkpoint 索引及量化描述。prepare_model.py 当前只调整私有 config.json 并链接其他文件，不能替代离线 checkpoint 裁剪。

## 执行命令

目标环境为 `113.46.15.88` 上的 `sglang-lzx-B103` 容器。在容器内进入仓库的复现目录，使用独立输出目录保存每次结果。

```bash
bash run_test.sh ./out/kimi-k3-pinned
```

启动前会读取公共模型的 `config.json`，检查原模型层数，以及第 1 到第 5 层里是否同时包含 full attention 和 KDA。检查通过后，脚本创建或复用私有目录，把 `text_config.num_hidden_layers` 设为 5，并把 checkpoint 的其余条目软链接到私有目录。公共目录只用于读取。重复运行会复用指向同一来源的软链接。

若原环境用 `acl` 或 `memfabric`，请将原值填入 `env.sh` 或通过环境变量传入。若怀疑 allocator 差异，可固定其他参数，用两个不同的输出目录比较原值和空值。映射内存后端需要容器具备相应的运行时支持。

## 服务启动命令

需要单独启动服务时，在复现目录运行：

```bash
HERE=$(pwd)
source ./env.sh
if [[ "$MODEL_PATH" == "$PRIVATE_MODEL" ]]; then
  python3 -u "$HERE/prepare_model.py"
fi

# 使用已裁剪 checkpoint 时也检查层数和类型覆盖。
python3 - "$MODEL_PATH" "$PREFIX_TOKENS" "$CONTEXT_LENGTH" <<'CHECK'
import json
import sys
from pathlib import Path

config = json.loads((Path(sys.argv[1]) / "config.json").read_text())
text = config.get("text_config", config)
if text["num_hidden_layers"] != 5:
    raise SystemExit("本用例要求五层 checkpoint，请先准备私有减层目录")
attention = text["linear_attn_config"]
for name in ("kda_layers", "full_attn_layers"):
    kept = [layer for layer in attention[name] if 1 <= layer <= 5]
    if not kept:
        raise SystemExit(f"前五层没有覆盖 {name}")
    print(f"{name}={kept}", flush=True)
if int(sys.argv[2]) + 1 + 8 > int(sys.argv[3]):
    raise SystemExit("CONTEXT_LENGTH 至少需要覆盖 PREFIX_TOKENS + 1 + 8")
CHECK

QUANT_ARGS=()
if [[ "$QUANTIZATION" != "none" ]]; then
  QUANT_ARGS=(--quantization "$QUANTIZATION")
fi

sglang serve \
  --model-path "$MODEL_PATH" --tokenizer-path "$MODEL_PATH" \
  --host 127.0.0.1 --port "$PORT" --trust-remote-code \
  --device npu --attention-backend ascend --dtype bfloat16 \
  "${QUANT_ARGS[@]}" --tp-size "$TP_SIZE" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" \
  --enable-hierarchical-cache --enable-cache-report --hicache-io-backend kernel_ascend \
  --hicache-mem-layout "$HICACHE_MEM_LAYOUT" \
  --hicache-write-policy write_through --hicache-ratio "$HICACHE_RATIO" \
  --disable-decode-cuda-graph --disable-prefill-cuda-graph --context-length "$CONTEXT_LENGTH" \
  --max-total-tokens "$MAX_TOTAL_TOKENS" --max-running-requests "$MAX_RUNNING_REQUESTS" \
  "${EXTRA_SERVER_ARGS[@]}"
```

## 输出物与判定

| 文件 | 内容 |
| --- | --- |
| `server.log` | 私有模型准备、层类型覆盖、服务运行日志及 traceback |
| `eval.log` | 健康检查与请求日志，包括返回的生成 token 数和缓存元数据 |
| `report.log` | 报告生成日志 |
| `report.md` | 结论、退出码和关键配置 |

| 退出码 | 含义 |
| --- | --- |
| 0 | 复现目标异常，或请求完成且记录到 MAMBA_OP_OK |
| 1 | 请求完成，日志中未观察到目标异常 |
| 2 | 端口被占用、服务启动失败、请求失败或服务退出等测试错误 |
| 130 | 测试被中断 |

## 已知限制

- 异常是否出现取决于实际 allocator、CANN / torch_npu / kernel 版本和缓存路径。请求完成只代表本轮没有观察到目标报错，无法单独确认 Mamba 写回实际发生。需要确认时，可在容器的算子调用前记录目标张量的 `dst.is_pinned()`。
- 私有模型复用原分片和权重索引，加载器仍可能读取完整分片。
- 本用例使用串行文本请求，覆盖执行路径和同一异常的复现，不测精度、吞吐或所有 rank。

完整 Kimi Linear 可用 MODEL_PATH=/mnt/paas/weights/Kimi-Linear-48B-A3B-Instruct QUANTIZATION=none 运行。调用日志记录 MAMBA_SOURCE、MAMBA_OP_BEGIN、MAMBA_OP_OK。空的 host backend 会取消导出，让模型的 setdefault 生效。

MoE 旁路仍会加载原权重，仅跳过 MoE 前向计算。启用后输出文本用于执行路径检查。

## B050 混合量化的投影适配

若 B050 构建 K3 时提示 fused_qkvg_proj 的 Q/K/V 是 W8A8_DYNAMIC、G 是 FLOAT，可用 b050_split_qkvg.patch 生成独立模型文件。补丁复用 dts_kimi 已有的拆分逻辑，保持 QKV 融合、G 独立，以及对应的权重加载映射。应用前先核对源文件版本，patch 会在上下文不匹配时退出。

```bash
HERE=$(pwd)
source ./env.sh
mkdir -p ./b050_compat
cp "$SGLANG_ROOT/python/sglang/srt/models/kimi_k3.py" ./b050_compat/kimi_k3.py
patch --directory ./b050_compat -p1 < ./b050_split_qkvg.patch
REPRO_K3_MODEL_FILE="$PWD/b050_compat/kimi_k3.py" python3 dts_test.py ./out/b050-five-layer
```

测试钩子只替换 K3 模型模块，Mamba host allocator 和 transfer_mamba_state 继续使用指定环境的原实现。此适配与 pinned memory 异常的根因修复分开记录。

## 2026-11-26 B050 实测

在 113.46.15.88 的 sglang-lzx-B050 中，使用物理卡 8、9、10、11，TP=4，真实的五层 Kimi K3 W4A8 checkpoint，MoE 旁路关闭，只有 L2。应用独立的 split QKV/G 模型适配后成功启动。Mamba host allocator 和 transfer_mamba_state 保持镜像原实现，kernel 从安装的 wheel 导入。

基础四请求测试通过。随后以四组不同的 2048-token 前缀交替发送 36 个请求，每次生成 8 token，触发 L1 淘汰与 L2 回读。累计 host 命中 34816 token，记录到 432 次状态传输调用及 432 次成功，调用前 pinned=False 的记录为 0。此场景未复现 host buffer must use pinned memory。

服务器记录位于 /home/l00889655/report/b050_kimi_k3_5layer_cards8_11 和 /home/l00889655/report/b050_kimi_k3_5layer_churn。运行脚本副本位于 /home/l00889655/sglang/kimi_k3_l2_run_20261126，独立适配模型文件为该目录的 patched_kimi_k3.py。
