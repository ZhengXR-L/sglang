# Qwen3-8B MHA HiCache KV split 端到端测试

## 测试目的

在真实 Qwen3-8B 推理服务中，分别验证 NPU `kernel_ascend` 的 `layer_first` 和 `page_first` 布局能完成 KV 的 D2H 备份与 H2D 回载。两种布局均须出现 Host 缓存命中，重放请求的输出 token 与首次请求一致。

## 测试原理

入口脚本依次启动两次 Qwen3-8B 服务。每次先核对本地 `config.json` 的 Qwen3-8B 结构，以及 `/model_info` 和 `/server_info` 返回的模型路径、`kernel_ascend`、布局和 HiCache 写策略。随后发送一个基准长提示和多个不同前缀的长提示，使较小的设备 KV 池发生驱逐；最后重放基准提示。

`write_through` 将 KV 写入 Host Pool；设备驱逐后，重放请求从 Host Pool 回载。用例要求 `/metrics` 中 KV Pool 的 `sglang:hicache_backup_tokens_total` 与 `sglang:load_back_tokens_total` 增加，且重放响应的 `cached_tokens_details.host` 大于零。当前 MHA Host Pool 在 `kernel_ascend` 下按布局分派到如下新算子：

| 布局 | D2H | H2D |
| --- | --- | --- |
| `layer_first` | `transfer_kv_all_layer_kv_split` | `transfer_kv_per_layer_kv_split` |
| `page_first` | `transfer_kv_all_layer_kv_split_lf_pf` | `transfer_kv_per_layer_kv_split_pf_lf` |

## 前置依赖

- Ascend NPU，至少 1 张空闲卡；已安装与设备匹配的驱动、CANN、`torch_npu`。
- 包含本次 MHA KV split 适配的 SGLang 源码、对应的 `sgl-kernel-npu`，以及其 Python 依赖（含 `requests`、`transformers`）。
- 本地 Qwen3-8B 模型目录，含 `config.json`、权重和 tokenizer。
- 可用的 `sglang serve` 命令；Host 内存通过 `acl` 注册。

## 环境变量

只修改 `env.sh`，或在执行前覆盖其中的变量。

| 变量 | 示例值 | 含义 | 必填 |
| --- | --- | --- | --- |
| `MODEL_PATH` | `/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B` | Qwen3-8B 模型目录 | 是 |
| `SGLANG_ROOT` | `/path/to/sglang` | SGLang 仓库根目录 | 是 |
| `KERNEL_ROOT` | `/path/to/sgl-kernel-npu` | 算子仓库根目录 | 是 |
| `ASCEND_ENV_SCRIPT` | `/usr/local/Ascend/ascend-toolkit/latest/set_env.sh` | Ascend 环境脚本 | 是 |
| `NPU_DEVICES` | `0` | 物理卡 ID，逗号分隔 | 是 |
| `PORT` | `30000` | 服务端口 | 否 |
| `STARTUP_TIMEOUT` | `1800` | 服务启动等待秒数 | 否 |
| `REQUEST_TIMEOUT` | `300` | 单请求和指标等待秒数 | 否 |
| `MAX_TOTAL_TOKENS` | `4096` | 设备 KV 池大小，用于触发驱逐 | 否 |
| `PAGE_SIZE` | `128` | HiCache 页大小 | 否 |
| `PROMPT_TOKENS` | `1024` | 每个提示的 token 数 | 否 |
| `OUTPUT_TOKENS` | `8` | 每次请求生成 token 上限 | 否 |
| `FILLER_REQUESTS` | `8` | 驱逐用长提示数量 | 否 |
| `HICACHE_RATIO` | `4` | Host 与设备缓存容量比 | 否 |
| `MEM_FRACTION_STATIC` | `0.8` | 静态 NPU 内存占比 | 否 |
| `SGLANG_HICACHE_HOST_MEM_BACKEND` | `acl` | Host 内存注册方式 | 否 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True,pin_memory_expandable_segments:True` | PyTorch NPU 分配配置 | 否 |
| `ASCEND_RT_VISIBLE_DEVICES` | `0` | 从 `NPU_DEVICES` 派生 | 自动 |
| `TP_SIZE` | `1` | 从 `NPU_DEVICES` 的卡数派生 | 自动 |
| `PYTHONPATH` | `<SGLANG_ROOT>/python:<KERNEL_ROOT>/python/sgl_kernel_npu` | 从源码路径派生 | 自动 |

## 执行命令

在配置好 `env.sh` 后运行：

```bash
bash run_test.sh /path/to/qwen3_8b_kv_split_result
```

脚本依次启动、测试并停止两个布局的服务，退出码为总体判定。

## 服务启动命令

`run_test.sh` 对 `LAYOUT=layer_first` 和 `LAYOUT=page_first` 各执行一次以下命令；变量均来自 `env.sh`：

```bash
sglang serve \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port "$PORT" --trust-remote-code \
  --device npu --attention-backend ascend \
  --tp-size "$TP_SIZE" --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-total-tokens "$MAX_TOTAL_TOKENS" --page-size "$PAGE_SIZE" \
  --enable-hierarchical-cache --hicache-ratio "$HICACHE_RATIO" \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel_ascend --hicache-mem-layout "$LAYOUT" \
  --enable-metrics --disable-cuda-graph
```

## 输出物与判定

输出目录包含 `server.layer_first.log`、`server.page_first.log`、`eval.layer_first.log`、`eval.page_first.log`、`layer_first.json`、`page_first.json`、两个 `*.exit_code`、`report.log` 和 `report.md`。入口脚本在成功、失败或中断时打印这些路径。

每个布局必须同时满足：服务解析参数正确；D2H/H2D 的 KV token 指标增量均大于零；重放命中 Host；生成 token ID 和文本与首次请求一致。任一条件失败则该布局失败，总体失败。`report.md` 在失败时也会生成，并列出已取得的证据和原因。

## 已知限制

- 本用例用请求、Host 命中和服务指标验证端到端路径，不直接逐字节读取 Host KV 内存，也不逐次统计算子调用。
- 默认池大小和提示数量适用于触发设备 KV 驱逐；如环境中的缓存行为不同，可在 `env.sh` 调整 `MAX_TOTAL_TOKENS`、`PROMPT_TOKENS`、`FILLER_REQUESTS` 和 `HICACHE_RATIO`。
- 只覆盖单机 TP，顺序请求，不评价性能或生成质量。
