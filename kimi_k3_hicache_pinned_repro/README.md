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
- 空闲卡数由 checkpoint、减层配置和并行方式决定。默认 8 卡只是配置示例。多机、DP、EP 参数按原启动配置填入 `EXTRA_SERVER_ARGS`，此入口负责启动本机进程。
- 请求直接构造 token ID，无需外部数据集。

## 环境变量

配置集中在 [env.sh](./env.sh)。可以编辑其中的默认值，或用同名环境变量覆盖。

| 变量 | 示例或默认值 | 含义 | 必填 |
| --- | --- | --- | --- |
| `SOURCE_MODEL` | `/mnt/paas/weights/Kimi-K3-w4a8` | 公共 ModelSlim W4A8 checkpoint | 是 |
| `PRIVATE_MODEL` | `/home/l00889655/kimi_truncation` | 私有五层目录，缺失时自动创建 | 否 |
| `SGLANG_ROOT` | `/sgl-workspace/sglang` | 容器里的 SGLang checkout | 是 |
| `ASCEND_ENV` | `/usr/local/Ascend/ascend-toolkit/latest/set_env.sh` | CANN 环境脚本 | 是 |
| `NPU_DEVICES` | `0,1,2,3,4,5,6,7` | 可见卡列表，列表长度作为本机 TP 默认值 | 是 |
| `PORT` | `30000` | 本机服务端口 | 否 |
| `STARTUP_TIMEOUT` | `1800` | 等待健康检查的秒数 | 否 |
| `REQUEST_TIMEOUT` | `120` | 每次请求的 HTTP 超时秒数 | 否 |
| `OBSERVE_SECONDS` | `10` | 请求阶段结束后继续观察日志的秒数 | 否 |
| `PREFIX_TOKENS` | `2048` | 公共前缀长度，应超过原配置的一页或一个状态跟踪间隔，并留出生成空间 | 否 |
| `REQUEST_COUNT` | `4` | 请求数，范围 2–100 | 否 |
| `QUANTIZATION` | `modelslim` | checkpoint 的量化方式 | 否 |
| `MEM_FRACTION_STATIC` | `0.75` | 设备静态内存比例 | 否 |
| `MAX_MAMBA_CACHE_SIZE` | `32` | 每个服务进程的 Mamba 状态池配置 | 否 |
| `HICACHE_RATIO` | `2.0` | Host / device cache 容量比例 | 否 |
| `HICACHE_MEM_LAYOUT` | `page_first_kv_split` | HiCache 内存布局，可按原容器的配置覆盖 | 否 |
| `SGLANG_HICACHE_HOST_MEM_BACKEND` | 空 | 保留原环境的 allocator 选择，支持时可设 `acl` 或 `memfabric` | 否 |
| `SGLANG_NPU_HICACHE_MAMBA_IO` | `sync` | 选择包含目标 D2H 算子调用的同步分支 | 否 |
| `EXTRA_SERVER_ARGS` | 空数组 | 在 env.sh 中设置原启动所需的额外并行、MoE 等参数 | 否 |

`env.sh` 还会派生 `ASCEND_RT_VISIBLE_DEVICES`、`PYTHONPATH` 和 `TP_SIZE`。

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
source ./env.sh
python3 -u ./prepare_model.py
sglang serve \
  --model-path "$PRIVATE_MODEL" \
  --host 127.0.0.1 --port "$PORT" --trust-remote-code \
  --device npu --attention-backend ascend --dtype bfloat16 \
  --quantization "$QUANTIZATION" --tp-size "$TP_SIZE" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" \
  --enable-hierarchical-cache --hicache-io-backend kernel_ascend \
  --hicache-mem-layout "$HICACHE_MEM_LAYOUT" \
  --hicache-write-policy write_through --hicache-ratio "$HICACHE_RATIO" \
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
| 0 | 日志包含目标错误和两个目标栈帧，复现成功 |
| 1 | 请求完成，日志中未观察到目标异常 |
| 2 | 端口被占用、服务启动失败、请求失败或服务退出等测试错误 |
| 130 | 测试被中断 |

## 已知限制

- 异常是否出现取决于实际 allocator、CANN / torch_npu / kernel 版本和缓存路径。请求完成只代表本轮没有观察到目标报错，无法单独确认 Mamba 写回实际发生。需要确认时，可在容器的算子调用前记录目标张量的 `dst.is_pinned()`。
- 私有模型复用原分片和权重索引，加载器仍可能读取完整分片。
- 本用例使用串行文本请求，覆盖执行路径和同一异常的复现，不测精度、吞吐或所有 rank。
