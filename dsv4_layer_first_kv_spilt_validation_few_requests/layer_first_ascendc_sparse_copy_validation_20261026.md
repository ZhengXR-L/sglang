# layer_first_ascendc_sparse_copy 分支 DSV4 layer_first_kv_split 验证报告

- 验证日期：2026-10-26
- 服务器：113.46.15.88
- 容器：`sglang-lzx-B103`
- 仓库：`/home/l00889655/sglang/sglang`
- 目标分支：`layer_first_ascendc_sparse_copy`
- 目标提交：`356c078865ae63511685837082e1f1a49d697b62`（`feat(hicache): enable DSV4 AIV transfers for NPU layer_first_kv_split`）
- 模型：`/mnt/paas/weights/DeepSeek-V4-Flash-0731-w8a8`（量化 `modelslim`，KV `bfloat16`）
- 统一证据目录（BASE）：`/home/l00889655/sglang/sgl_notebook_l2_2n_20261026/`

> 说明：本轮未修改任何 SGLang 业务代码或内核包，仅只读检查 + 运行测试脚本。两轮测试均在**同一目标提交**上执行，仅改变 `--hicache-mem-layout`（页面布局）与端口，其余参数完全一致。

---

## 1. 源码与内核来源

| 项目 | 值 | 证据文件 |
|---|---|---|
| sglang 源码 | `/home/l00889655/sglang/sglang/python/sglang/__init__.py` | `runtime_environment.json` |
| 分支 | `layer_first_ascendc_sparse_copy`（`...lzx/layer_first_ascendc_sparse_copy`） | `sglang_status.txt` |
| commit | `356c078865ae63511685837082e1f1a49d697b62` | `sglang_commit` |
| 内核包 | `/usr/local/python3.12.13/lib/python3.12/site-packages/sgl_kernel_npu/__init__.py` | `runtime_environment.json` |
| 内核 SHA-256 | `7e14ee6cec76874bf92e1b837935b054eb45da69b01ecb2f19ea626e6d330c1a`（系统 `sgl_kernel_npu 2026.9.0`） | `runtime_environment.json` |
| AIV 稀疏拷贝算子 | `memfabric_hybrid.offload.kv_exchange_copy` | `memory_pool_host.py` 源码 |

内核默认使用系统安装目录，未使用任何 dsv4_test 专用安装目录。

---

## 2. 启动命令

两轮启动命令均由入口脚本 `run_layer_first_verify.sh` 生成，仅 `--hicache-mem-layout` 与 `--port` 不同（见 `launch_command.txt`）。

### 2.1 page_first 对照轮（`target_page_control_001`，端口 32370）

```bash
python3 -m sglang.launch_server --model-path /mnt/paas/weights/DeepSeek-V4-Flash-0731-w8a8 --page-size 128 --tp-size 8 --ep-size 8 --trust-remote-code --attention-backend dsv4 --device npu --watchdog-timeout 9000 --host 0.0.0.0 --port 32370 --mem-fraction-static 0.8 --max-running-requests 16 --chunked-prefill-size 32768 --max-total-tokens 66000 --quantization modelslim --reasoning-parser deepseek-v4 --kv-cache-dtype bfloat16 --random-seed 0 --context-length 1048576 --dp-size 8 --enable-hierarchical-cache --hicache-io-backend kernel_ascend --hicache-ratio 2.0 --hicache-write-policy write_through --hicache-mem-layout page_first --enable-dp-attention --enable-dp-lm-head --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled --moe-a2a-backend deepep --deepep-mode auto
```

### 2.2 layer_first 轮（`target_layer_first_001`，端口 32371）

```bash
python3 -m sglang.launch_server --model-path /mnt/paas/weights/DeepSeek-V4-Flash-0731-w8a8 --page-size 128 --tp-size 8 --ep-size 8 --trust-remote-code --attention-backend dsv4 --device npu --watchdog-timeout 9000 --host 0.0.0.0 --port 32371 --mem-fraction-static 0.8 --max-running-requests 16 --chunked-prefill-size 32768 --max-total-tokens 66000 --quantization modelslim --reasoning-parser deepseek-v4 --kv-cache-dtype bfloat16 --random-seed 0 --context-length 1048576 --dp-size 8 --enable-hierarchical-cache --hicache-io-backend kernel_ascend --hicache-ratio 2.0 --hicache-write-policy write_through --hicache-mem-layout layer_first --enable-dp-attention --enable-dp-lm-head --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled --moe-a2a-backend deepep --deepep-mode auto
```

### 2.3 关键环境变量（两轮一致，见 `runtime_environment.json`）

```
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7,8,9,10,11
SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
SGLANG_DSV4_AIV_TRACE=1                 # 仅 layer_first 轮新增，page_first 轮也保留但无 AIV 输出
HCCL_BUFFSIZE=2048, HCCL_INTRA_DMA_ENABLE=1, HCCL_SOCKET_IFNAME=lo
STREAMS_PER_DEVICE=32, PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
SGLANG_DSV4_FP4_EXPERTS=True
SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False, SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
PYTHONPATH=.../cann-9.1.0/python/site-packages:...
```

启动脚本同时执行了 swappiness=0、numa_balancing=0、CPU affinity 等配置（见 `run.log`）。

---

## 3. 参数解析确认（非仅看 CLI 字符串）

从两轮 `server.log` 的 `server_args=` 输出确认解析后的内部枚举：

| 参数 | page_first 对照轮 | layer_first 轮 |
|---|---|---|
| `hicache_mem_layout` | **`page_first_direct`** | **`layer_first_kv_split`** |
| `hicache_io_backend` | `kernel_ascend` | `kernel_ascend` |
| `hicache_write_policy` | `write_through` | `write_through` |
| `hicache_ratio` | `2.0` | `2.0` |
| `chunked_prefill_size` | `4096`（CLI 传 32768） | `4096` |
| `swa_full_tokens_ratio` | `0.1` | `0.1` |
| `tp_size` / `ep_size` / `attn_dp_size` | `8` / `8` / `8` | `8` / `8` / `8` |
| `page_size` | `128` | `128` |
| `max_total_tokens` | `66000` | `66000` |
| `mem_fraction_static` | `0.8` | `0.8` |
| `enable_dp_attention` / `enable_dp_lm_head` | `True` / `True` | `True` / `True` |
| `speculative_algorithm`（MTP） | `None`（关闭） | `None`（关闭） |
| `cuda_graph_backend_decode/prefill` | `disabled` | `disabled` |

结论：CLI `layer_first` 解析为内部枚举 `layer_first_kv_split`（走 AscendC/Memfabric AIV 路径）；CLI `page_first`（DSV4 非 MLA backend）解析为 `page_first_direct`（走旧传输算子，不用 AIV）。两轮仅 `hicache_mem_layout` 一项不同。

---

## 4. 逐轮请求与路由

三阶段请求（`bench_ids_dsv4.py`，`request_plan.json`）：

- populate A×8（输入 seed 60000–60007）→ evict B×8（输入 seed 60008–60015）→ replay A×8（输入 seed 60000–60007）
- 每条输入 65536 token，输出 8 token，temperature=0，ignore_eos=true
- roundrobin 路由：Ai / Bi / replay Ai 固定进入同一 DP rank i

程序化核验（对两轮分别执行）：

```
total reqs: 24   （8 pop + 8 evict + 8 replay）
input token len: 65536 / 65536 / 65536
A==R all 8: True        （replay 输入与 populate 完全相同）
A!=B all 8: True        （evict 输入与 populate 不同）
first tokens pop/replay:  [60000..60007]
first tokens evict:      [60008..60015]
```

路由核验（`responses.jsonl` 中 `meta_info.dp_rank`）：

| 请求 | dp_rank | 请求 | dp_rank |
|---|---|---|---|
| pop-0 / evict-0 / replay-0 | 0 | pop-4 / evict-4 / replay-4 | 4 |
| pop-1 / evict-1 / replay-1 | 1 | pop-5 / evict-5 / replay-5 | 5 |
| pop-2 / evict-2 / replay-2 | 2 | pop-6 / evict-6 / replay-6 | 6 |
| pop-3 / evict-3 / replay-3 | 3 | pop-7 / evict-7 / replay-7 | 7 |

即 `pop-i → rank i`、`evict-i → rank i`、`replay-i → rank i`，与 roundrobin 路由一致，满足 Ai/Bi/replayAi 同 DP rank。

---

## 5. 逐轮响应（输出 token 一致性与 L2 命中）

### 5.1 L2 命中（cache 分布）

所有 8 条 replay 请求均：

```
cached_tokens = 61440
cached_tokens_details = { "device": 0, "host": 61440 }
l2_proven = True
```

- `device=0` → L1 未命中；`host=61440` → L2 命中（脚本最小门槛 G=2048）。
- `61440 = 65536 - 4096`，即 4096 token 重新 prefill（SWA 前缀 + 分割头），非 100% 全命中，与 main 基线一致。

### 5.2 输出 token（8 个 input ID，两轮逐项相同）

A（populate）与 R（replay）输出完全一致，且 A 的输出在两轮（page_first 与 layer_first）**完全相同**，证明 layer_first 回迁后的正确性：

| rank | A/R 输出 ID（两轮一致，非全零） |
|---|---|
| 0 | `[607, 288, 733, 50073, 339, 128822, 223, 128822]` |
| 1 | `[81, 223, 21, 38, 223, 135, 249, 86]` |
| 2 | `[257, 223, 135, 249, 86, 714, 302, 1230]` |
| 3 | `[733, 20615, 379, 1230, 1230, 1230, 1230, 1230]` |
| 4 | `[679, 1230, 1886, 38998, 2525, 1230, 72394, 339]` |
| 5 | `[223, 23, 14, 1320, 223, 2115, 223, 5528]` |
| 6 | `[301, 223, 128822, 339, 1, 26, 16, 1]` |
| 7 | `[339, 223, 128822, 223, 128822, 223, 4256, 223]` |

B（evict）输出与 A 不同（不同前缀，用于驱逐 L2），例（rank 0–7）：`[733, 50073, 223, 135, 249, 75, 223, 135]`、`[339, 223, 128822, 223, 128822, 223, 4256, 223]`、`[733, 575, 339, 223, 128822, 201, 223, 128822]`、`[223, 128822, 201, 20759, 1, 26, 1, 26]`、`[339, 223, 128822, 223, 128822, 223, 4256, 223]`、`[367, 339, 223, 128822, 223, 128822, 223, 4256]`、`[60, 223, 22, 16, 18, 201, 128822, 65]`、`[339, 223, 128822, 223, 128822, 223, 4256, 223]`。

### 5.3 verdict

两轮 `verdict.json` 一致：`identical_count=8`、`l2_proven_count=8`、`passed=true`。`exit_code=0`。

---

## 6. 算子 trace（AIV kv_exchange_copy）

### 6.1 memfabric 分配与 DVA

`server.log` 确认 host 内存通过 memfabric 分配（8 个 DP rank 各一次）：

```
[HiCache] memfabric host memory initialized: reserve=1.46GB device=0..7
TRACE [OFFLOAD acc_offload.cpp:22 offload_init] offload_init ret: 0, deviceId: 0..7, reserveSize: 1570078592
```

- `offload_init ret: 0` 即底层 offload/memfabric 库初始化成功（返回 0，非 -1），`reserveSize=1570078592`（≈1.46GiB），DVA 由 `offload.empty` 分配且有效。
- `SGLANG_HICACHE_HOST_MEM_BACKEND = memfabric` 由 8 个 DP rank 各自打印确认。

### 6.2 AIV D2H/H2D 调用

layer_first 轮 `server.log` 含 **112 条** AIV trace（按 pool + direction 去重后统计）：

```
8 D2H swa                      8 H2D swa
8 D2H deepseek_v4_c4           8 H2D deepseek_v4_c4
8 D2H deepseek_v4_c4_indexer   8 H2D deepseek_v4_c4_indexer
8 D2H deepseek_v4_c4_indexer_scale   8 H2D deepseek_v4_c4_indexer_scale
8 D2H deepseek_v4_c4_state     8 H2D deepseek_v4_c4_state
8 D2H deepseek_v4_c4_indexer_state    8 H2D deepseek_v4_c4_indexer_state
8 D2H deepseek_v4_c128        8 H2D deepseek_v4_c128
```

= **7 个池 × 2 方向（D2H/H2D）× 8 rank = 112**。覆盖全部 7 个 DSV4 压缩池（SWA、C4 KV、C4 Indexer K、C4 Indexer scale、C4 state、C4 Indexer state、C128）。

典型 trace 行（DP0）：

```
[DP0 TP0 EP0] HiCache DSV4 AIV kv_exchange_copy D2H pool=deepseek_v4_c128    layout=layer_first_kv_split pages=2  layer=0 bytes_per_page=16384
[DP0 TP0 EP0] HiCache DSV4 AIV kv_exchange_copy D2H pool=deepseek_v4_c4      layout=layer_first_kv_split pages=32 layer=0 bytes_per_page=32768
[DP0 TP0 EP0] HiCache DSV4 AIV kv_exchange_copy D2H pool=deepseek_v4_c4_indexer       layout=layer_first_kv_split pages=32 layer=0 bytes_per_page=4096
[DP0 TP0 EP0] HiCache DSV4 AIV kv_exchange_copy D2H pool=deepseek_v4_c4_indexer_scale layout=layer_first_kv_split pages=32 layer=0 bytes_per_page=64
[DP0 TP0 EP0] HiCache DSV4 AIV kv_exchange_copy D2H pool=deepseek_v4_c4_indexer_state layout=layer_first_kv_split pages=2  layer=0 bytes_per_page=16384
[DP0 TP0 EP0] HiCache DSV4 AIV kv_exchange_copy D2H pool=deepseek_v4_c4_state          layout=layer_first_kv_split pages=2  layer=0 bytes_per_page=65536
[DP0 TP0 EP0] HiCache DSV4 AIV kv_exchange_copy D2H pool=swa                        layout=layer_first_kv_split pages=2  layer=0 bytes_per_page=131072
```

> 说明：trace 按 (pool, direction) 每进程只打首条（`_dsv4_aiv_logged` 去重），112 条是「8 rank × 7 pool × 2 方向」的确定性推演结果，不代表执行期间仅调用 112 次算子；实际 D2H/H2D 随每次回迁/驱逐按页多次调用。

### 6.3 page_first 对照轮无 AIV

page_first 对照轮 `server.log` 中 AIV trace 计数为 **0**（`grep -c "HiCache DSV4 AIV kv_exchange_copy"` 返回 0），确认 `page_first_direct` 走旧传输算子、未调用 AIV 稀疏拷贝，两轮形成正确对照。

---

## 7. 结论

在目标提交 `356c078...` 上，DSV4（modelslim W8A8、KV bf16、无 MTP、TP=8/DP=8/EP=8、DP Attention+DP LM head）配置下：

1. **布局与 IO 后端正确**：CLI `layer_first` → `layer_first_kv_split`；CLI `page_first` → `page_first_direct`；实际 IO 均为 `kernel_ascend`；host 内存经 memfabric 分配（`offload_init ret: 0`，DVA 有效）。
2. **L2 回迁成立**：8/8 replay 请求 `device=0, host=61440`（≥2048 门槛），L2 命中真实发生。
3. **输出 token 一致**：8/8 replay 输出与对应 A 输出逐项相同，且与 page_first 对照轮输出 ID 完全相同，无全零输出。
4. **AIV kv_exchange_copy 真实调用**：layer_first 轮 112 条 trace 覆盖 7 个池 × D2H/H2D 双向 × 8 rank；page_first 对照轮为 0 条，对照干净。
5. **exit code 均为 0**，无 lingering server/端口。

**验证通过**：目标分支 `layer_first_ascendc_sparse_copy` 的 DSV4 layer_first_kv_split + kernel_ascend + memfabric 路径在给定配置下正确，AIV 稀疏拷贝 D2H/H2D 与 L2 回迁语义正确。

### 结论边界（重要）

- 本验证**仅覆盖** DSV4、BF16 KV、无 MTP 这一配置下的 AIV L2 回迁与 8-token 一致性。
- **不代表**已验证：draft 备份/恢复、全部量化分支（FP8/FP4 等）、其他模型（非 DSV4 压缩池）、L3 storage、性能（layer_first vs page_first 的吞吐对比未做）、以及非 NPU 回归。
- 目标提交说明中未合入的实验性 C4 backend 修复未混入本轮验证，本轮仅验证原始目标提交。

---

## 8. 证据清单

| 证据 | page_first 对照轮 | layer_first 轮 |
|---|---|---|
| launch_command.txt | `target_page_control_001/` | `target_layer_first_001/` |
| runtime_environment.json | 同上 | 同上 |
| sglang_commit / sglang_status.txt | 同上 | 同上 |
| server.log | 同上 | 同上 |
| run.log | 同上 | 同上 |
| exit_code（=0） | 同上 | 同上 |
| results/requests.jsonl（24 条） | 同上 | 同上 |
| results/responses.jsonl（24 条） | 同上 | 同上 |
| results/request_plan.json | 同上 | 同上 |
| results/first_out.json | 同上 | 同上 |
| results/arguments.json | 同上 | 同上 |
| results/verdict.json | 同上 | 同上 |

两轮日志目录见前文 2.1 / 2.2。