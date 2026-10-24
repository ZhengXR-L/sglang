# DSV4 在 NPU 上验证 `layer_first_kv_split`：问题、证据与 C4 修复方案

记录日期：2026-10-24。验证环境：服务器 `113.46.15.88`、容器 `sglang-lzx-B103`、仓库 `/home/l00889655/sglang/sglang`、模型 `/mnt/paas/weights/DeepSeek-V4-Flash-0731-w8a8`，使用 8 张 Ascend NPU，`tp=8`、`ep=8`、`--attention-backend dsv4`、`--quantization modelslim`、`--kv-cache-dtype bfloat16`、`--page-size 128`。本文记录的是这一配置下的结果，不推断其他芯片、模型或图模式已通过。

## 1. 当前结论与代码状态

1. `layer_first_kv_split` 的 DSV4 host 池在 memfabric 上可以执行 AIV `kv_exchange_copy`，分页池和状态池的 D2H/H2D 数据往返测试通过。对应 HiCache 路由单测 8 项通过。
2. **原始 DSV4 后端的完整模型往返仍未通过。**W/E/R 测试中，W 为写入、E 为驱逐、R 为重复 W 提示词并回载缓存。原始后端下，AIV D2H/H2D 各出现 56 次，但 R 输出为空；`page_first` 对照同样失败。关闭 radix 的模型推理 W/R 正常，但不会验证 H2D 回载。
3. 故障定位到 radix 前缀命中后的 C4 compressor 续算：命中 4096 token 前缀后，首个新 C4 compressed KV（逻辑索引 1024）出现非有限值，并从首个 C4 注意力层（layer 2）向后传播。这与 host 内存布局无关；未开启 HiCache、但允许 radix 前缀复用时也能复现。
4. 曾试验修改 A3 C4 compressor 的状态表构造和 `cache_mode`，在相同模型及提示词下得到 W/R 相同、AIV D2H/H2D 各 56 次的完整通过结果。**该 C4 后端改动及配套单测随后已撤回**，当前仓库中的 `ascend_dsv4_backend.py` 与 Git 基线一致。因此该试验结果是修复方向的证据，不是当前工作区已完成的端到端验收。

本期保留的功能改动主要位于 `python/sglang/srt/mem_cache/memory_pool_host.py` 和 `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`。C4 故障应单独跟踪。

## 2. 实际启动路径与测试方法

在 NPU + memfabric + `kernel_ascend` 场景，用户侧启动参数应使用 `--hicache-mem-layout layer_first`。NPU 的 `set_default_server_args` 将它解析为内部布局 `layer_first_kv_split`；AIV 分支还要求 `ascendc_io_enabled()` 为真。**不要在 CLI 直接传内部值**：本次曾误把它加入参数 choices，导致 `set_default_server_args` 落入默认分支，将实际布局变成 `page_first_direct`；误改已撤回。判断实际走向应看服务日志中的最终 `hicache_mem_layout` 以及 `HiCache DSV4 AIV kv_exchange_copy ... layout=layer_first_kv_split` 标记。

测试脚本：`/home/l00889655/sglang/sglang/test/manual/hicache/e2e_layer_first_dsv4_roundtrip.sh`、同目录的 `.py` 文件。脚本启动时加载 CANN vendor `set_env.bash`，设置 `SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric`、`PYTHONPATH`、DSV4/DeepEP 环境变量；服务使用 `--enable-hicache --hicache-io-backend kernel_ascend --hicache-write-policy write_through --hicache-ratio 2.5`，并关闭 prefill/decode NPU graph。测试以约 4121-token 的 W/R 提示词和约 4633-token 的 E 提示词触发备份、L1 驱逐和 L2 回载；通过条件包括 W/R 非空且相同，以及 D2H、H2D 标记均大于零。只看生成接口返回成功或标记计数，不能证明模型输出正确。

## 3. 遇到的问题、定位与处理

| 问题 | 证据与原因 | 处理或验证边界 |
| --- | --- | --- |
| CLI 拒绝 `layer_first_kv_split`；临时加入 choices 后实际运行成 `page_first_direct` | `layer_first_kv_split` 是 NPU 内部解析结果；默认分支会覆盖直接传入的内部值 | 撤回 CLI choices 改动，传 `--hicache-mem-layout layer_first`；检查最终参数和 AIV 日志 |
| W/E/R 中 R 输出为空，即使 AIV D2H/H2D 均执行 | 原始后端的 AIV 测试 D2H/H2D 各 56 次；`page_first` 对照也为空；无 HiCache 的 radix 复用同样复现 | 将问题与 host 布局分开定位；不能把“算子已调用”当作端到端通过 |
| 关闭 radix 后 W/R 正常，但无法覆盖 H2D | radix 缓存关闭后不发布可供前缀命中的内容，测试日志 D2H/H2D 均为 0 | 只作为模型冷路径对照；不能用它替代 L1→L2→L1 验收 |
| 池级 smoke 首次运行报缺少 memfabric 环境变量 | `DeepSeekV4PagedHostPool` 对 NPU `layer_first_kv_split` 明确要求 memfabric | 运行测试时设置 `SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric`，随后分页池和状态池往返均通过 |
| 旧的直接算子 smoke 脚本报缺少 `pool_name` | 测试脚本调用 `_dsv4_ascendc_page_copy` 的签名已落后于业务函数 | 此脚本不计为通过；采用覆盖真实 host 池调用链的 `dsv4_pool_smoke.py` 和现有单测 |

### C4 异常的证据链

- R 阶段命中 4096 token radix 前缀，剩余 token 进入续算。页表、SWA 映射及 Full KV 在 W/R 对照中没有发现缺失页。
- C4 indexer top-k 返回 512 个有效索引，Q 和权重均为有限值；但它选到的 C4 compressed KV 索引 `1024` 的 `cmp_each_finite` 为 `0.0`。同条 trace 中其他采样索引为 `1.0`，原始 KV 的 `ori_finite=1.0`。
- layer 0/1 的输出仍为有限值；首个 C4 层 layer 2 的 `out_finite=0.0`，后续层随之异常。C4 trace 见 `/home/l00889655/sglang/e2e_dsv4_c4_index_trace_20261024/roundtrip_dsv4_server.log`，可定位 `DSV4_TOPK_TRACE`、`DSV4_C4_READ_TRACE`、`DSV4_ATTN_TRACE layer=2`。
- 在不启用 HiCache 的 radix 命中对照中仍出现 R 为空；禁用 radix 后 W/R 相同。因此异常发生在模型的前缀续算链路，不能归因于 AIV 搬运或 `layer_first_kv_split` 的 host 存储格式。

## 4. C4 后端修复试验：怎么改、验证了什么

试验文件是 `/home/l00889655/sglang/sglang/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py`，本地保留的试验版本为 `D:\Workspace\sglang\remote_dsv4_work\diagnose\ascend_dsv4_backend_clean.py`。它已从远程工作区撤回，可作为后续独立修复的参考，**不要误认为当前仓库包含此修复**。

原始 A3 路径将 C4 与 C128 都送入 `_build_explicit_state_block_table`：该表按“history + 当前输入”生成逐 token 的 state location，调用 `torch.ops.npu.compressor` 时两者都使用 `cache_mode=2`。试验发现，这组 C4 状态表/模式在 radix 命中续算时产生了错误的首个新 compressed KV。

试验修改仅针对 **A3 且 ratio=4 的 C4**：

1. 新增 `_build_c4_continuous_state_block_table`。按原始 token 位置的 block 构造状态表，利用 `req_to_token → full loc → SWA loc → state loc → state block id` 映射到 SWA 所属的 C4 state 页；表元素是 block id，而不是逐 token 的 state location。C4 每个 state 页是 8 个原始 token 的环，`coff=2`、`ratio=4` 对应 8-token history；无效位置及已驱逐的旧 SWA 页映射为安全占位块 `0`。
2. A3 C4 调用 compressor 时改为 `cache_mode=1`，使用上述 continuous-mode block 表。C128 继续原有表与 `cache_mode=2`；A5（`is_npu_arch35()`）继续原有 cycle-mode `cache_mode=2`。这样不扩大到未验证的路径。
3. 为 graph 路径同步修改 C4 表的固定形状预分配及 replay 时的刷新逻辑。此次模型 E2E 启动时关闭了 graph，因此 **graph 只做了代码和单测检查，没有 graph E2E 证据**。
4. 临时增加单测：检查跨物理 SWA 页、双请求的 block 映射，以及 A3 C4 调用确实传 `cache_mode=1`。试验版的 NPU 后端单测文件运行结果为 `66 passed`；相关新增测试随后随修复一并撤回。

核心调用关系可概括为：A3 C4 先按原始 token block 建表，`state_block_id = state_loc // state_pool.page_size`，再调用 `compressor(..., state_block_table=table, cache_mode=1)`；C128 和 A5 保持各自原有的表与 `cache_mode=2`。正式合入前应再次核对目标 CANN/芯片版本的算子 ABI，避免把该试验规则扩展到其他平台。

试验结果：无 HiCache 的 radix 命中 W/R 由失败转为通过；开启 HiCache、`layer_first_kv_split`、memfabric、AIV 的完整 W/E/R 得到 `W==R=True`、W/R 非空、AIV D2H `x56`、H2D `x56`、`RESULT: PASS`。日志：`/home/l00889655/sglang/e2e_dsv4_c4_continuous_trial_freecards_20261024/run.log` 与 `/home/l00889655/sglang/e2e_dsv4_c4_fix_aiv_roundtrip3_20261024/run.log`。这是针对该工作负载的有效修复试验；后续正式修复仍需补 graph、混合 batch、多种前缀长度和 C128 边界回归。

## 5. 已验证内容与尚未覆盖内容

| 配置或测试 | 结果 | 能证明什么 / 不能证明什么 |
| --- | --- | --- |
| 原始后端，`page_first`，HiCache W/E/R | D2H `x112`、H2D `x1344`，R 为空，失败 | 证明故障不只出现在 `layer_first_kv_split`；不能据此判断两种布局性能 |
| 原始后端，`layer_first_kv_split`，AIV W/E/R | AIV D2H/H2D 各 `x56`，R 为空，失败 | 证明目标算子实际被调用；不能证明完整模型正确性 |
| 原始后端，关闭 HiCache 与 radix | W/R 相同，`RESULT: PASS`，无传输 | 证明模型冷路径正常；不覆盖缓存回载 |
| 原始后端，memfabric host 池 smoke | paged/state 两类池的 D2H/H2D 逐字节一致 | 证明这两类池的传输数据正确；不覆盖全部模型子池或推理续算 |
| 当前 `test_dsv4_hicache_l2.py` | `8 passed` | 覆盖 AIV 只在 `layer_first_kv_split` 路由、页面元数据等；不替代 NPU E2E |
| 临时 C4 修复版，`layer_first_kv_split`，AIV W/E/R | W/R 相同且非空；AIV D2H/H2D 各 `x56`；`RESULT: PASS` | 证明试验修复在该 E2E 负载有效；**该修复当前未保留** |

关键原始日志：

- 原始后端 AIV 失败：`/home/l00889655/sglang/e2e_dsv4_aiv_final_20261024/run.log`
- `page_first` 对照失败：`/home/l00889655/sglang/e2e_dsv4_page_first_control_20261024/run.log`
- 无 HiCache、无 radix 对照通过：`/home/l00889655/sglang/e2e_dsv4_no_hicache_control_20261024/run.log`
- C4 indexer、compressed KV、首个异常注意力层 trace：`/home/l00889655/sglang/e2e_dsv4_c4_index_trace_20261024/roundtrip_dsv4_server.log`
- 临时 C4 修复后的 AIV 完整通过：`/home/l00889655/sglang/e2e_dsv4_c4_fix_aiv_roundtrip3_20261024/run.log`，服务日志为同目录 `roundtrip_dsv4_server.log`

可复查的命令（在服务器上执行）：

```bash
# 池级 NPU 数据往返；/tmp/dsv4_pool_smoke.py 是本次验证保留的测试脚本。
docker exec \
  -e SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric \
  -e PYTHONPATH=/home/l00889655/sglang/sglang/python:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/cann-9.1.0/python/site-packages:/sgl-workspace/sglang/python \
  -w /home/l00889655/sglang/sglang \
  sglang-lzx-B103 python /tmp/dsv4_pool_smoke.py

# HiCache 路由及页面元数据单测。
docker exec \
  -e PYTHONPATH=/home/l00889655/sglang/sglang/python:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/cann-9.1.0/python/site-packages:/sgl-workspace/sglang/python \
  -w /home/l00889655/sglang/sglang \
  sglang-lzx-B103 python -m pytest test/registered/unit/mem_cache/test_dsv4_hicache_l2.py -q
```

## 6. 后续验收建议

目前可把 `layer_first_kv_split` 的 **AIV 路由与分页/状态池数据搬运**列为已验证，把 **DSV4 模型完整 L1→L2→L1 往返**列为待 C4 后端修复的阻塞项。正式修复时，应在独立提交中恢复上述 C4 方案并复核 vendor ABI，然后至少执行：A3 eager 的无 HiCache radix 对照、AIV/`layer_first_kv_split` W/E/R、`page_first` 回归、graph replay、多请求及不同前缀边界测试。验收必须同时检查缓存命中、新 C4 KV 有限性、AIV D2H/H2D、以及 W/R 语义一致；单独关闭 radix 或只统计算子调用次数不足以证明完整功能。
