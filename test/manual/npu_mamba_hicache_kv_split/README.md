# NPU Mamba HiCache kv_split 端到端测试

## 测试目的

验证 `python/sglang/srt/mem_cache/pool_host/mamba.py` 的改动：kernel_ascend IO 后端下，MambaPoolHost 的 mamba 状态 H2D/D2H 传输全部改走 sgl-kernel-npu 的 split-KV 算子（`transfer_kv_per_layer_kv_split_pf_lf` / `transfer_kv_all_layer_kv_split_lf_pf`）。预期现象：驱逐压力下 mamba 状态正确备份到 host 并回读，同一 prompt 两次 greedy 生成输出完全一致，服务无崩溃。

## 测试原理

mamba 池按 request 存槽：一个请求只占一个状态槽，槽在请求结束后进入 LRU，device 槽满时被逐出到 host（kv_split D2H），下次命中 host L2 时恢复（kv_split H2D）。若传输路径写错层、错 slot 或视图不连续，恢复出的状态即是脏数据，greedy 续写会立刻发散。因此用例为：

1. target prompt（长且不与其他请求共享前缀）greedy 生成，记录输出；
2. 发送 5 个互不共享前缀的 filler 请求，使 device KV 池驱逐 target，同时避免把 target 从 Host L2 再次逐出；
3. 重放同一 target prompt：device 前缀缓存已逐出、命中 host L2，mamba 状态经 H2D 恢复后续写；
4. 从 `/metrics` 读取 `pool="mamba"` 的 D2H、H2D token 计数，分别要求 filler 后和 target 重放后增长；两次输出还须逐字符一致。报告中的计数增量为所有 TP rank 之和。

同时从 server.log 断言 `transfer mode: kv_split` 日志行存在，确认新路径（而非旧路径）被启用。

## 前置依赖

- 昇腾 NPU（A3/A5），CANN 版本与日常 sglang NPU 开发环境一致
- sgl-kernel-npu：包含 `operators/mem_cache/memfabric/kv_transfer` 模块（提供四个 split-KV 算子）
- sglang：本仓库该改动之后的工作区
- 模型：任一 hybrid（linear attention / mamba 结构）模型，如 Qwen3-Next 系列或 Falcon-H1 系列，需已下载到本地
- host 内存需容纳 `MAX_TOTAL_TOKENS × HICACHE_RATIO` 的层级缓存

## 环境变量

| 变量 | 示例值 | 含义 | 必填 |
| --- | --- | --- | --- |
| SGLANG_ROOT | /root/sglang | sglang 仓库路径 | 是 |
| KERNEL_ROOT | /root/sgl-kernel-npu | sgl-kernel-npu 仓库路径 | 是 |
| MODEL_PATH | /data/models/Qwen3-Next-80B | hybrid(mamba) 模型路径 | 是 |
| NPU_DEVICES | 0,1 | 空闲物理卡 ID 列表，卡数决定 tp-size | 是 |
| PORT | 30000 | 服务端口 | 否 |
| TARGET_PROMPT_CHARS | 20000 | target prompt 字符数 | 否 |
| TARGET_NEW_TOKENS | 64 | target 每次生成 token 数 | 否 |
| NUM_FILLER | 5 | filler 请求数（超过 device 容量，但总前缀应低于 Host 容量） | 否 |
| FILLER_PROMPT_CHARS | 12000 | 每个 filler 的 prompt 字符数 | 否 |
| FILLER_NEW_TOKENS | 8 | filler 生成 token 数 | 否 |
| STARTUP_TIMEOUT | 1800 | 服务启动等待秒数 | 否 |
| REQUEST_TIMEOUT | 600 | 单请求超时秒数 | 否 |
| MAX_TOTAL_TOKENS | 8192 | device KV 池容量（刻意调小制造驱逐） | 否 |
| CONTEXT_LENGTH | 32768 | 模型上下文长度 | 否 |
| HICACHE_RATIO | 4 | host 池 / device 池容量比 | 否 |
| SGLANG_HICACHE_HOST_MEM_BACKEND | acl | host 内存后端（acl 或 memfabric；本分支 hybrid 模型 MHA 池未做 memfabric 预留，实测用 acl） | 否 |

## 实测环境（2026-11-16 通过）

- 8 卡 Ascend910B 容器；最终 D2H/H2D 验证使用 NPU_DEVICES=0,1,2,3（tp=4），CANN 9.1.0，torch_npu 2.10.0
- 模型 Qwen3.5-4B（24 linear attention + 8 full attention 层），`--mamba-ssm-dtype bfloat16` 覆盖（Ascend910B 不支持 fp32 initial_state）
- 算子注册需 sgl-kernel-npu 源码树 + kv_exchange 私有编译包并用：
  - `PYTHONPATH` 指向源码树 `python/sgl_kernel_npu`（注册 `torch.ops.npu.causal_conv1d` 等前向算子）
  - `kv_exchange` 算子仅存在于私有包（如 `qwen3_kv_exchange_pkg_YYYYMMDD`），需 `torch.ops.load_library` 私有包的 `libsgl_kernel_npu.so`，并将其 `_cann/vendors/sgl_kernel_npu` 追加到 `ASCEND_CUSTOM_OPP_PATH`
  - **桥接方式（推荐）**：在 PYTHONPATH 前置目录放 `sitecustomize.py`，Python 进程启动时自动执行：

    ```python
    import os
    _VENDOR = "/path/to/qwen3_kv_exchange_pkg/python/sgl_kernel_npu/sgl_kernel_npu/_cann/vendors/sgl_kernel_npu"
    _paths = [p for p in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if p]
    if _VENDOR not in _paths:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join([_VENDOR] + _paths)
    _PRIV_LIB = "/path/to/qwen3_kv_exchange_pkg/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
    if os.path.exists(_PRIV_LIB):
        import torch
        torch.ops.load_library(_PRIV_LIB)
    ```

  - 二者缺一即崩溃：缺私有包 → D2H 崩 `kv_exchange` 不存在；缺源码树 → prefill/decode 崩 `causal_conv1d` 不存在
- 多容器共享宿主机时，若 HCCL 默认端口冲突，可在启动前按机器实际空闲端口设置 HCCL 端口变量。
- 最终验证：5 个 filler，Mamba D2H 计数增量 24、H2D 增量 8；重放命中 3584 个缓存 token，输出一致。`/tmp/mamba_e2e_out_metrics_v6/report.md`。

## 执行命令

```bash
bash run_test.sh /path/to/output
```

## 服务启动命令

单卡（`NPU_DEVICES=0`）默认形式，多卡时 `--tp-size` 由 `NPU_DEVICES` 的卡数推导：

```bash
sglang serve \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port 30000 --trust-remote-code \
  --device npu \
  --tp-size 1 \
  --enable-hierarchical-cache \
  --enable-metrics \
  --hicache-ratio 4 \
  --hicache-io-backend kernel_ascend \
  --max-total-tokens 8192 \
  --mamba-ssm-dtype bfloat16 \
  --context-length 32768
```

## 输出物与判定

| 文件 | 内容 |
| --- | --- |
| `<output>/server.log` | 服务完整日志 |
| `<output>/eval.log` | 用例执行日志 |
| `<output>/case_result.json` | 两次输出、filler 完成数、Mamba D2H/H2D 计数增量、错误列表 |
| `<output>/report.md` | 汇总报告（失败也生成） |
| `<output>/report.log` | 报告生成日志 |

通过需同时满足：

1. target 两次 greedy 输出完全一致（`case_result.json` 的 `match == true`）；
2. filler 全部成功（`filler_done == NUM_FILLER`），驱逐压力真实形成；
3. server.log 中存在 `transfer mode: kv_split`，确认新路径启用；
4. 无请求阶段错误。
5. `pool="mamba"` 的 D2H/H2D token 计数增量均大于 0。

任一不满足即失败，失败原因写入 report.md。

## 已知限制

- 通过 Mamba 池计数器断言发生了 L2 备份和回读；计数器不记录具体算子调用次数。
- 只做正确性判定，不做性能评价。
- 只覆盖 greedy 单并发串行场景，不覆盖 PD 分离、多 rank TP>1 下的传输正确性。
- draft（MTP）层是否被包含取决于所选模型是否启用 speculative decoding，本用例不强制。
