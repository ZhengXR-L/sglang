# 需求记录：NPU HiCache L1→L2 传输路径支持 layer_first 布局

> 记录时间：2026-10-19
> 目标完成时间：2026-10-30（功能 + 最低标准性能验证 + 提 PR）

## 1. 需求概述

在 **NPU（昇腾）场景**下，实现 HiCache **L1→L2 传输路径**对 **`layer_first` 数据布局**的支持，与社区主流实现对齐。

- L1：设备侧（NPU）KV Cache 内存池
- L2：主机侧内存池（host pool）及其存储后端

验收要求：
1. `layer_first` 布局下 L1 + L2 前缀缓存（HiCache）功能正常可用；
2. 端到端性能不低于 `page_first` 布局；
3. 10 月 30 日前完成功能开发与最低标准的性能验证，并提交 PR。

## 2. 背景与关键概念

### 2.1 HiCache 分层

```
   L1: 设备内存池 (NPU KV Cache)
        │  backup (D2H 写回)  /  load (H2D 预取)
        ▼
   L2: 主机内存池 (host pool)  +  存储后端 (mooncake / npu_memcache / ...)
```

### 2.2 传输后端 (`hicache_io_backend`)

| backend | 说明 |
|---------|------|
| `direct` | 直接 copy（CPU 侧张量） |
| `kernel`  | CUDA/HIP/XPU 的 `sgl_kernel` 传输 kernel |
| `kernel_ascend` | NPU 专用；layer_first 走 Memfabric AscendC 零拷贝 kernel `kv_exchange_copy`（封装为 `_transfer_ascendc_sparse_copy`），替代 `transfer_kv_dim_exchange` |

### 2.3 主机内存布局 (`hicache_mem_layout`)

| 布局 | host pool 组织方式 | 说明 |
|------|--------------------|------|
| `layer_first` | `[layer][page][...]`，每层独立连续 buffer | 经典布局，社区默认之一 |
| `page_first` | `[page][layer][...]` | 单一大 tensor，页面优先 |
| `page_first_direct` | `[page][layer][1][...]` | direct IO 用 |
| `page_first_kv_split` | K/V 分离的 page-first | MLA NPU 专用 |

参数定义见 `arg_groups/fields/memory.py`（`hicache_mem_layout` 默认值为 `page_first`）。

## 3. 现状分析（Gap）

### 3.1 NPU 当前强制 page_first 系布局

`hardware_backend/npu/utils.py`（`set_default_server_args`）在启用
`enable_hierarchical_cache` 或 `disaggregation_decode_enable_offload_kvcache` 时：

- 强制 `hicache_io_backend = "kernel_ascend"`
- MLA 模型 → `hicache_mem_layout = "page_first_kv_split"`
- 非 MLA 模型 → `hicache_mem_layout = "page_first_direct"`

即 **NPU 场景下目前完全没有暴露 `layer_first` 布局的能力**。

### 3.2 `kernel_ascend` 传输分支只处理 page_first 系布局

通用 host pool 的 `backup_from_device_all_layer` / `load_to_device_per_layer` 中，
`io_backend == "kernel_ascend"` 分支仅实现 `page_first_direct`，未见 `layer_first`：

- `mem_cache/pool_host/mha.py`
  - `load_to_device_per_layer`：`kernel_ascend` 下 `if self.layout == "page_first_direct"`，否则抛 `Unsupported layout`
  - `backup_from_device_all_layer`：`kernel_ascend` 下 `if self.layout == "page_first_direct"`
- `mem_cache/pool_host/mla.py`：`kernel_ascend` 分支同样只走 page-first 逻辑
- `mem_cache/pool_host/unified.py`：`prepare_transfer_indices` 的 `kernel_ascend` 分支直接
  `return host_indices, device_indices.cpu()`，未区分 `layer_first`

### 3.3 存储后端对 layer_first 的限制

`arg_groups/hicache_hook.py` 的 `resolve_storage_layout_compatibility()`：

```
Mooncake/Ascend MemCache storage backend does not support layer_first layout ...
```

当 `hicache_storage_backend` 为 `mooncake` / `npu_memcache` 且布局为 `layer_first` 时，
实际改写行为（对照 `hicache_hook.py` L182-204）：
- `direct` 后端 → 改写为 `page_first_direct`；
- `kernel` 后端 → 改写为 `page_first`；
- `kernel_ascend` → 保持 `layer_first`（仅 declare 同值 + 打误导性 warning，实际未切换）。

即 **NPU 存储后端（npu_memcache / Ascend MemCache）当前声明不支持 layer_first**；但对
`kernel_ascend + layer_first + mooncake/npu_memcache` 组合，现有代码是「放行」而非「拦截」，
仍存隐患。且现有拦截范围只覆盖 `mooncake`/`npu_memcache`，`file`/`nixl`/`hf3fs` 等其余
`hicache_storage_backend` 会一路通过配置、运行时才失败（详见设计文档 §7.3）。本次设计改为：
**配置解析阶段对 `layer_first + 任意非空 L3 backend` 显式抛错**——该拦截**限定在 NPU /
`kernel_ascend` 路径**（`resolve_storage_layout_compatibility` 是跨设备通用函数，须避免误伤 GPU
的既有行为），本期 NPU 的 layer_first 仅限纯 L1/L2。

### 3.4 已有部分 layer_first 先例（可作为参照）

`mem_cache/memory_pool_host.py` 中的 DeepSeek V4 专用 pool 已实现 `kernel_ascend` +
`layer_first`/`page_first` 两种布局（通过 `_host_page_view()` 生成 `[pages,1,P,1,dim]`
视图，再调用 `transfer_kv_dim_exchange`）：

- `DeepSeekV4PagedHostPool.backup_from_device_all_layer`（`kernel_ascend` 分支）
- `DeepSeekV4PagedHostPool.load_to_device_per_layer`
- `DeepSeekV4StateHostPool` 同上

这证明底层传输算子已具备配合 layer_first 视图工作的能力。DSv4 专用池先例仍作为
「layer_first 视图 / shape 处理」的参照；但本方案不再复用 `transfer_kv_dim_exchange`
（其逐层 launch 开销放大问题见 §4 与设计文档 §9），改用 Memfabric AscendC 零拷贝路径。

## 4. 社区主流实现参考（对齐目标）

社区（sglang 上游开源项目）在 CUDA/HIP 侧对 `layer_first` 布局的 L1→L2 传输，
使用 `sgl_kernel.kvcacheio` 的以下 kernel（`mem_cache/pool_host/mha.py` 等）：

- 全层 D2H：`transfer_kv_all_layer`（layer_first，全层一次性传输）
- 单层 H2D：`transfer_kv_per_layer`（layer_first）
- MLA 变体：`transfer_kv_all_layer_mla` / `transfer_kv_per_layer_mla`
- JIT 变体：`jit_transfer_hicache_all_layer` / `jit_transfer_hicache_one_layer`

NPU 对齐思路：在 `kernel_ascend` 后端下，为通用 host pool 补齐 `layer_first` 分支，
复用 `_transfer_ascendc_sparse_copy`（Memfabric AscendC AIV 零拷贝 kernel `kv_exchange_copy`，
底层 `host_layout_mode=1` 支持 layer-first），而非 `transfer_kv_dim_exchange`
（SDMA `aclrtMemcpy2dAsync`）。后者逐层（N=1）调用在短序列/单 token 场景会放大
launch 开销（设计文档 §9），故改用「一次 compact meta 下发、device 侧衍址、不 sync」的
AscendC 路径，使 `layer_first` 与 `page_first` 在功能与性能上等价。

NPU 侧已有布局转换先例：`sgl_kernel_npu.kvcacheio.transfer_mamba_state` 注释明确展示了
device层-first ↔ host 页-first 的转换能力。

## 5. 涉及关键文件

### sglang 主仓库（当前分支 `layer_first_ascendc_sparse_copy`）

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/arg_groups/fields/memory.py` | `hicache_mem_layout` / `hicache_io_backend` 参数定义 |
| `python/sglang/srt/arg_groups/hicache_hook.py` | 布局与 IO/存储后端的兼容性归一化 |
| `python/sglang/srt/hardware_backend/npu/utils.py` | NPU 默认参数解析（当前强制 page_first） |
| `python/sglang/srt/mem_cache/pool_host/mha.py` | 通用 MHA host pool 传输（gap 所在） |
| `python/sglang/srt/mem_cache/pool_host/mla.py` | 通用 MLA host pool 传输（gap 所在） |
| `python/sglang/srt/mem_cache/pool_host/unified.py` | 统一 host pool（gap 所在） |
| `python/sglang/srt/mem_cache/pool_host/page_unified.py` | page 统一 host pool |
| `python/sglang/srt/mem_cache/pool_host/base.py` | host pool 基类 |
| `python/sglang/srt/mem_cache/memory_pool_host.py` | DSv4 专用 pool（已有 layer_first+kernel_ascend 先例） |
| `python/sglang/srt/mem_cache/storage/npu_memcache/npu_memcache_store.py` | NPU 存储后端（声明不支持 layer_first） |

### sgl-kernel-npu 仓库

本方案无改动（§10.4）。

## 6. 多模型适配范围

需覆盖以下模型的 NPU HiCache（含 layer_first）能力：

| 模型 | KV 结构（线索） | 对应 host pool | 当前 NPU 布局 | layer_first 适配点 |
|------|----------------|----------------|---------------|--------------------|
| GLM-5.x | MHA/GQA（标准 attention） | `pool_host/mha.py` | `page_first_direct` | 补齐 `kernel_ascend` + `layer_first` 分支 |
| GLM-5.3-Flash | MHA（Flash 变体） | `pool_host/mha.py` | `page_first_direct` | 同上 |
| Kimi-K3 | MLA + 线性注意力（`KimiK3MLAAttention`，含 linear/delta attention，`uses_ssm_state` 命中） | 双路径：MLA KV → `pool_host/mla.py`；SSM state → `pool_host/mamba.py`（`MambaPoolHost`） | `page_first_kv_split` | MLA 侧补齐 `kernel_ascend` + `layer_first` 分支；SSM 侧复用 `MambaPoolHost` 已有 layer_first+kernel_ascend，仅回归 |
| DeepSeek-V4-Flash | DSv4 paged（MHC / indexer） | `memory_pool_host.py` 的 DSv4 专用 pool | `page_first_*` | 已有 layer_first 先例，重点验证与统一 |
| DeepSeek-V4.1-Flash | DSv4 paged（MHC / indexer） | `memory_pool_host.py` 的 DSv4 专用 pool | `page_first_*` | 同上 |

适配注意点：

1. **MHA 与 MLA 分开实现**：两类 host pool 的 layer_first 布局组织不同
   （K/V 是否分离、`item_size`/`token_stride_size` stride 不同），需分别在
   `mha.py`、`mla.py` 补齐 `kernel_ascend` 分支并各自回归。
2. **DSv4 系列复用先例**：`DeepSeekV4PagedHostPool` / `DeepSeekV4StateHostPool`
   已实现 `kernel_ascend` + `layer_first`，重点是验证与通用 pool 保持行为一致，
   避免重复造轮子。
3. **附带缓存**：部分模型（如 DSv4 indexer）存在 `index_k` / `index_k_scale` 等
   附带缓存，需确认其在 layer_first 下的传输参数
   （`transfer_kv_dim_exchange` 的 `device_index_k` / `device_index_k_scale`）。
4. **模型归属需逐一确认**：各模型最终走 MLA 还是 MHA backend、单/多 token 粒度，
   需在实现前以各模型实际 config 为准确认，避免按名称臆断。
5. **Kimi-K3 的 SSM state 是独立路径**：Kimi-K3 同时保有线性注意力（Delta）的 recurrent/conv
   状态，由 `MambaPoolHost`（`pool_host/mamba.py`）承载（`kv_cache_builder.py::uses_ssm_state`
   通过 `kimi_linear_config` 命中），`layout` 默认 `layer_first` 且已有 kernel_ascend 逐层复制
   路径。验收须覆盖「L2 前缀命中后恢复 SSM state」，只看 MLA K/V 搬运无法证明 Kimi-K3 达标（§11.2）。

## 7. 验收标准

1. **功能**：`--hicache-mem-layout layer_first`（+ `kernel_ascend`）在 NPU 下可正常启动，
   L1→L2 写回（backup）与 L2→L1 预取（load）正确，前缀缓存命中/回退结果与 page_first 一致；
   覆盖 MHA（GLM-5.x / GLM-5.3-Flash）、MLA（Kimi-K3）、DSv4（V4-Flash / V4.1-Flash）三类模型路径；
   Kimi-K3 除 MLA K/V 外还须验证「L2 前缀命中后恢复线性注意力（SSM）状态」正确（§11.2）。
2. **性能**：`layer_first` 布局端到端性能不低于 `page_first` 布局（最低标准性能验证即可）。
3. **范围边界（本期）**：layer_first 仅限纯 L1/L2——不接任何 L3 存储后端（`--hicache-storage-backend`
   非空时启动即报错）。L1↔L2 传输在**本次改造的通用 MHA/MLA host pool** 上**强制走 Memfabric
   AscendC 零拷贝路径**（`SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric`），非 memfabric host 内存
   不再作为这两类 pool 的 layer_first 传输后端；**例外**：DeepSeek-V4 专用池（`memory_pool_host.py`）
   与 Kimi-K3 的 SSM 状态池（`MambaPoolHost`）保留其既有 `transfer_kv_dim_exchange` / pinned 路径，
   不在本期「强制 AscendC」范围内（设计文档 §4.1 / §10.2）。
4. **交付**：10 月 30 日前完成上述功能与最低标准性能验证，并提交 PR。

## 8. 时间节点

| 节点 | 内容 |
|------|------|
| 2026-10-30 前 | 完成功能开发 |
| 2026-10-30 前 | 完成最低标准性能验证 |
| 2026-10-30 前 | 提出 PR |

## 9. 待确认事项

- [ ] 社区"主流实现"是否特指某个上游 PR / 具体 branch（需进一步确认对齐基线）
- [x] L3 存储后端：**已定——本期仅覆盖 L1/L2 host pool（不含 L3）**。任意非空
      `--hicache-storage-backend`（不限于 npu_memcache/mooncake）在 NPU `kernel_ascend` 下与
      `layer_first` 组合，配置解析阶段即抛错（设计文档 §7.3）。
- [x] 各模型 backend 归属：**已定**——GLM-5.x / GLM-5.3-Flash 走 `mha.py`；Kimi-K3 走 MLA
      (`mla.py`) + SSM state (`MambaPoolHost`) 双路径；DeepSeek-V4-Flash / V4.1-Flash 走 DSv4 专用池。
- [x] DSv4 indexer 附带缓存：**已定——不在本次改动范围**。DSv4 走 `DeepSeekV4PagedHostPool`
      既有 layer_first+kernel_ascend 先例，仅做回归；通用 `mla.py` 的 indexer（`is_deepseek_dsa` 族）
      作为可选扩展点。
- [x] 性能验收基准：**已定**——对比模型与序列长度、预取命中、吞吐/时延指标见 §11.3 及设计文档 §10.1
      （含短序列 / 单 token 逐层 launch 开销场景）。

## 10. 变更清单

分三块改动（底层 `kv_exchange_copy` 的 `host_layout_mode=1` 已实现，无需任何 kernel 调整）。

### 10.1 布局选择 / 开关（让 NPU 能选 layer_first）

| 文件 | 位置 | 改动 |
|------|------|------|
| `python/sglang/srt/hardware_backend/npu/utils.py` | `set_default_server_args`（L173-194） | ① 仅当 `cfg.hicache_mem_layout != "layer_first"` 时才投影默认（MLA→`page_first_kv_split`、否则→`page_first_direct`）；显式 `layer_first` 跳过覆盖、放行。**memfabric 强制不在此函数做**（此阶段无 host pool 实例，无法区分通用 MHA/MLA pool 与 `MambaPoolHost`/DSv4 专用池），改由 host pool `init_kv_buffer` 按 pool class 判定（设计文档 §6.8） |
| `python/sglang/srt/arg_groups/hicache_hook.py` | `resolve_storage_layout_compatibility`（L180-204） | 配置解析阶段对 `layer_first + 任意非空 L3 + io_backend==kernel_ascend` 显式**抛错**（本期 NPU 仅纯 L1/L2）；纯 L1/L2 放行 layer_first；GPU 等其它设备行为不变 |

> 判定方式（已定）：只判 `cfg.hicache_mem_layout == "layer_first"`（默认是 `page_first`，能反推出用户显式指定），
> 不声称能识别其它显式 CLI 值；`declare_resolution` 的 stash 声明优先级高于 `_raw_input`（见设计文档 §7.2）。

### 10.2 传输实现（补 `kernel_ascend` + `layer_first` 分支，走 AscendC 零拷贝）

通用 MHA/MLA pool 的 `kernel_ascend` 分支目前都只认 page-first 系布局（`unified.py` 的
layer_first 属 page_unified 系，不在本期 MHA/MLA 必改范围，见 §6 表格）。本次在 layer_first
分支改用 `_transfer_ascendc_sparse_copy`（而非 `transfer_kv_dim_exchange`）：

| 文件 | 方法 | 改动 |
|------|------|------|
| `python/sglang/srt/mem_cache/pool_host/mha.py` | `backup_from_device_all_layer` / `load_to_device_per_layer` | 新增 `layer_first` 子分支，补充 `_transfer_ascendc_sparse_copy`（MHA 无此方法，需新增类似实现） |
| `python/sglang/srt/mem_cache/pool_host/mla.py` | `backup_from_device_all_layer` / `load_to_device_per_layer` | layer_first 分支从 `_transfer_one_layer`（`transfer_kv_dim_exchange`）改为 `_transfer_ascendc_sparse_copy`（`host_layout_mode=1`） |

> **并非「只补传输分支」**：
> - MHA：`init_kv_buffer` 虽已是 K/V 分离（`dims=(2, layer_num, size, head_num, head_dim)`，layer_first
>   drain shape 已就绪），但当前用 `ALLOC_MEMORY_FUNCS["npu"]`（普通 pinned，**且调用处恒传
>   `registration_granularity_bytes`**），**必须补齐 memfabric 分配 + `ensure_memfabric_capacity`**。
>   注意 `alloc_with_memfabric` 签名是 `(dims, dtype, device, pin_memory, allocator)`，**不接收
>   `registration_granularity_bytes`**，故不能只替换 `alloc_func`，须像 `mla.py` 那样单独写调用
>   （设计文档 §5.2(1)）；否则 `get_dva(host_ptr)` 失败或初始化 `TypeError`。另需为 MHA 的 host
>   buffer（`[layer, size, H, D]`，隐式 page）新增 component 元数据构建（stride 推导与 MLA 不同，
>   见设计文档 §5）。
> - MHA 的 V 宽度限定：本路径**仅支持 `head_dim == v_head_dim` 的对称 MHA**（`head_dim != v_head_dim`
>   的模型走 `AsymmetricMHATokenToKVPoolHost`，其 layer_first/kernel_ascend 均不支持，见设计文档 §5.4）。
> - MLA：现有 layer_first 是「合并记录」`[layer_num, size, 1, kv_cache_dim]`，不满足 AscendC
>   component 的 K/V 分离契约（设计文档 §6.3），**需改 `init_kv_buffer` 分开分配** `k_buffer`/`v_buffer`
>   （FP8 DSA 时 `v_buffer` 为 5-D 零元素空张量）。
> - D2H（`backup_from_device_all_layer`）按 **buffer_info 分段**调 `_transfer_ascendc_sparse_copy`
>   （target 一次、每个 draft 一次）：target 与 draft 的 device buffer 属于不同 `buffer_info`、
>   地址不连续，单一 component（一个 base + 一套 layer 区间）无法跨 info 表达。**每段必须把 device
>   base 与 host base 都对齐到「该段 layer 0」**（host 侧按全局层号切片，draft 的 device 本地层从 0
>   起、host 层在 `target_layer_num` 之后），再 `layer_start=0`、`layer_num=段层数`；不能把全局
>   `layer_id` 直接当 `layer_start`。H2D（`load_to_device_per_layer`）单层语义同样按此切片 device 与
>   host 到单层，且不 sync load stream（逐层流水）。映射细节与调用示意见设计文档 §3 / §5.3 / §6.4。

### 10.3 复用 / 参照（不新增轮子）

- `mla.py` 的 `_transfer_ascendc_sparse_copy` 是现成的 AscendC 零拷贝传输实现（`_ascendc_comp_meta`
  拼 9-int 元数据 + `_launch_ascendc_kv_exchange` 下发 `offload.kv_exchange_copy`），MHA 复用其
  「component 元数据 + host DVA 改写 + launch」骨架，仅需按 MHA 的 host buffer
  （`[layer, size, H, D]`，隐式 page）重写 stride 推导。
- `memory_pool_host.py` 的 `DeepSeekV4PagedHostPool._host_page_view()`（L351-372）仍可作为
  layer_first 视图处理的参照（仅 shape/view 层面，不再复用 `transfer_kv_dim_exchange`）。

### 10.4 底层 kernel（外部依赖，sglang / sgl-kernel-npu 均不改）

- `_transfer_ascendc_sparse_copy` 底层是外部包 `memfabric_hybrid` 的 `offload.kv_exchange_copy`
  （AIV 融合 kernel）。**已确认**：该 kernel 的 `host_layout_mode=1`（layer-first）已实现
  （`OffloadKvExchangeKernel::LayerFirstWalker`，layer-outer / page-inner + peek-ahead 合并
  连续 page），sglang / sgl-kernel-npu 侧**无需改动外部依赖**。
- **sglang 侧只改 Python**：`mla.py` 的 `_transfer_ascendc_sparse_copy` 传 `host_layout_mode=1`，
  `_ascendc_comp_meta` 按 layer-first 填 host stride。**无需再改 `kvcacheio.py` 的空 V 占位**
  （那是 `transfer_kv_dim_exchange` 特有路径；AscendC 用 component 元数据，不存在该问题）。
- sgl-kernel-npu 本仓库 **无改动**。

### 10.5 建议落地顺序

1. ~~确认外部 `kv_exchange_copy` 支持 `host_layout_mode=1`~~（**已确认**：`LayerFirstWalker` 已实现，见 §10.4）；
2. MLA：`_transfer_ascendc_sparse_copy` 支持 `host_layout_mode=1`，layer_first 分支接入（D2H 全量 / H2D per-layer）；
3. MHA：补充 `_transfer_ascendc_sparse_copy` 类似实现 + layer_first 分支接入 + 单测；
4. MLA FP8 DSA / indexer 在 AscendC component 路径下的闭合 + 单测；
5. 开关（10.1，强制 memfabric）+ 端到端精度（含 Kimi-K3 SSM state 恢复回归）；
6. 性能对比（长/短序列）+ CI 注册。

## 11. 验收测试用例设计

### 11.1 单元测试：布局搬运正确性（不依赖真实模型）

放 `test/registered/unit/npu/mem_cache/`，参照 `test_unified_hicache_transfer_lifecycle.py` /
`test_hicache_staged_write_back_dispatch.py` 的 dummy-pool 模式，对 MHA、MLA（含/不含 indexer）、
FP8 DSA 各构造 `layout="layer_first"` 的 host pool + 已知内容的 device buffer：

- 断言（通用）：
  1. `backup_from_device_all_layer`（D2H，AscendC 按 buffer_info 分段：target 一次、每个 draft 一次）后，
     host 的 `k_buffer[l]`（及 `v_buffer[l]`）逐层 == 预期值（MLA 为独立 `k_buffer`/`v_buffer`，
     非合并 `kv_buffer`）；
  2. `load_to_device_per_layer`（H2D，AscendC 将 device 与 host 都切单层后按段内
     `layer_start=0`、`layer_num=1` 调用）后，device buffer 正确复原；
  3. **layer_first 与 page_first 写回/加载结果完全一致**（等价性对拍）。
- MLA indexer 专测：构造带 `index_k`（及 `index_k_scale`）的子集层，验证
  ① `index_k`/`index_k_scale` 作为 AscendC 独立 component（`_build_ascendc_component_meta` 的第 3/4
  个 component）被正确搬运、device 层号→index slot 映射正确；② host scale 为 5-D
  `[num_indexer_layers, page_num, page_size, 1, 1]`；③ D2H/H2D 各跑一次，断言 component 数量
  ≤ 4、无越界（对应设计文档 §6.7）；④ **indexer 整块 + 绝对 slot 定位**：取一个
  `ik_start > 0` 的命中层（非首个 index slot），断言该层 index_k/scale 搬运到**正确的** slot、
  未发生「先切片再叠加绝对 slot」的双重偏移（即不以段内切片 + `index_k_layer_start=ik_start`
  同时传）；⑤ **None 场景**：普通 MLA（`native.index_k is None`）与 BF16 有 index_k 无
  `index_scales` 的 pool，断言 H2D/D2H 不生成对应 index/scale component、不因直接切片 None 报错。
- FP8 DSA 专测：`v_width==0` 时 `v_buffer` 为 5-D 零元素空张量（非 `None`、不占容量），
  `_build_ascendc_component_meta` 的 V component 因 `numel()==0` 被跳过、只保留 K component，
  断言容量闭合 + D2H/H2D 无越界（对应设计文档 §6.6）。

### 11.2 端到端功能 / 精度测试（NPU，覆盖多模型）

放 `test/registered/npu/basic_function/HiCache/`，参照现有 `test_npu_hicache_mha.py` /
`test_npu_hicache_mla.py` 的启动 + GSM8K 精度模式，新增 `--hicache-mem-layout layer_first`：

| 用例 | 模型 | 断言 |
|------|------|------|
| MHA-layer_first | Qwen2.5-7B（GLM-5.x / GLM-5.3-Flash 作必跑验收模型） | GSM8K 精度 ≥ 阈值，且 == page_first 精度 |
| MLA-layer_first | Kimi-K3（**必跑**；DeepSeek-V2-Lite 仅作无 SSM 的纯 MLA K/V CI 冒烟） | MLA K/V 精度 == page_first；**且前缀命中后恢复线性注意力（SSM）state 正确** |
| DSv4-layer_first | V4-Flash / V4.1-Flash | 同上，只做回归（DSv4 专用池，非本次改动） |

关键验证点：**同一模型下 `layer_first` 与 `page_first` 精度一致**（等价性），而非仅各自达标。

**Kimi-K3 SSM state 恢复专项（必跑）**：Kimi-K3 同时有 Delta 线性注意力的 recurrent/conv 状态，
由 `MambaPoolHost`（`pool_host/mamba.py`）承载，与 MLA K/V 是两条独立 host pool 分支。因此除上表
MLA K/V 精度外，须**单独增加一次**「L2 前缀命中后恢复 SSM state」端到端：跑「前缀命中 → H2D 恢复
temporal/conv state（+ side state）→ 续写解码」，校验续写 logits/精度与无缓存基准对齐。仅用
DeepSeek-V2-Lite（无 SSM state）过测无法覆盖 `MambaPoolHost` 这条分支。

### 11.3 性能验证（最低标准 = 端到端不低于 page_first）

- 用 server benchmark（参照 `test/registered/kernels/benchmark/kvcache/bench_hicache.py`
  或 server 压测）；
- 同模型、同前缀命中率下对比 `layer_first` vs `page_first` 的吞吐/时延，
  断言 **layer_first ≥ page_first**（或差异在允许阈值内，波动阈值 ±5%）；
- **必须覆盖两类输入长度**：①长序列（prefill 数千 token，验证数据拷贝主导时基本持平）；
  ②**短序列 / 单 token 续写**——旧方案（`transfer_kv_dim_exchange` 逐层 N=1）launch 开销放大
  `layer_num` 倍，是本方案改走 AscendC 的动机；改用 `_transfer_ascendc_sparse_copy`（一次 compact
  meta 下发、device 侧衍址、不 sync）后，需验证该开销确实消除（设计文档 §9）。短序列不达标时须
  记录并回到 launch 开销优化项，**不得以长序列通过掩盖短序列回退**。

### 11.4 CI 注册

- 为新用例加 `register_npu_ci(...)`，复用现有 suite 分片（`base-b-test-*` / `nightly-*`），
  PR 流水线走 `run_npu_pr_smoke`，nightly 走完整精度。

## 12. 开发计划

> 小步落地，每步可独立验证，并尽量配单测。本方案核心传输算子改为 AscendC 零拷贝
> `_transfer_ascendc_sparse_copy`（底层 `offload.kv_exchange_copy`，`host_layout_mode=1`），
> **不涉及 sgl-kernel-npu 改动**；sglang 侧路径相对 `sglang/` 包根（与 §11 一致）。
> 单测沿用 `test_hicache_staged_write_back_dispatch.py` 的 dummy-pool 模式。

### 阶段 A：前置依赖确认 + MLA 接入（有现成 AscendC 实现，先跑通）

**Step 1 — （已确认）`kv_exchange_copy` 支持 `host_layout_mode=1`（无代码改动）**
- 改动：无。已确认外部包 `memfabric_hybrid` 的 `offload.kv_exchange_copy` 已实现
  `host_layout_mode=1`（`LayerFirstWalker`，layer-outer / page-inner + peek-ahead 合并连续 page；
  header 第 7 个 int）。component 元数据 host stride 语义也已确认：`hostPagePitch`(o+4) /
  `hostLayerPitch`(o+5) 为独立字段、地址公式 mode 无关（设计文档 §2.4）。
- 交付：无（原「确认结论 + stride 语义记录」已完成，见设计文档 §2.4/§6.4(C)）。

**Step 2 — MLA `init_kv_buffer` 分开分配 k/v**
- 改动：`python/sglang/srt/mem_cache/pool_host/mla.py::init_kv_buffer` 在 `MLASplitBuffers` +
  layer_first 下，把「合并记录」拆成独立 `k_buffer`/`v_buffer`（`[layer_num, page_num, page_size, 1, width]`）；
  `_size_per_token` 容量闭合。
- 单测：`test/registered/unit/npu/mem_cache/test_hicache_layer_first_mla.py`（新建）
  —— 断言 k/v buffer 形状正确、token 容量与 page_first 一致。

**Step 3 — MLA `_transfer_ascendc_sparse_copy` 支持 `host_layout_mode=1` + layer_first D2H 接入**
- 改动：`mla.py` 的 `_transfer_ascendc_sparse_copy` / `_ascendc_comp_meta` / `_build_ascendc_component_meta`
  在 layer_first 下传 `host_layout_mode=1`、按 layer-first 填 host stride
  （`host_layer_stride = stride(0)*itemsize`、`host_page_stride = stride(1)*itemsize`）；
  `backup_from_device_all_layer` 的 layer_first 分支从 `_transfer_one_layer` 改为按 buffer_info
  分段调 `_transfer_ascendc_sparse_copy`（target 一次、每个 draft 一次，各自覆盖其全部层，
  host 层号经 `target_layer_num + 本地层号` 映射；draft 通过 device/host 分别切片、以段内层号
  0 调用，验证写入对应 host 层且不触发越界）。
- 单测：同 Step 2 文件 —— 含 draft 层的 pool，backup 后断言 `k_buffer[host_layer_id]` /
  `v_buffer[...]` 逐层正确、映射不错层；并对拍 page_first 结果一致。

**Step 4 — MLA layer_first H2D（per-layer）**
- 改动：`mla.py::load_to_device_per_layer` 走 `_transfer_ascendc_sparse_copy`，按
  `layer_start`/`layer_num` 只搬传入 `layer_id`，不 sync load stream（逐层流水）。
- 单测：同 Step 2 文件 —— backup+load 复原；多层只 load 一层，断言其它层 device 未被写
  （验证 per-layer 语义）。

### 阶段 B：MHA 补充 AscendC 实现 + 接入

**Step 5 — MHA 补 memfabric 分配 + 新增 `_transfer_ascendc_sparse_copy`（stride 推导重写）**
- 改动：`python/sglang/srt/mem_cache/pool_host/mha.py` 的 `init_kv_buffer` 在 `_is_npu and
  memfabric_host_memory_enabled()` 时改用 `alloc_with_memfabric` + 先 `ensure_memfabric_capacity`。
  **注意 `alloc_with_memfabric` 不接收 `registration_granularity_bytes`（签名
  `(dims, dtype, device, pin_memory, allocator)`），须单独写调用而非替换 `alloc_func`**（设计文档
  §5.2(1)）；并新增类似 `_transfer_ascendc_sparse_copy` 的实现，复用 MLA 的「component 元数据 +
  host DVA 改写 + `offload.kv_exchange_copy` launch」骨架，仅按 MHA host buffer（`[layer, size,
  head_num, head_dim]`，隐式 page）重写 stride 推导（设计文档 §5）。
- 单测：`test/registered/unit/npu/mem_cache/test_hicache_layer_first_mha.py`（新建）
  —— 构造 dummy layer_first pool，断言 component 元数据 stride / DVA 改写正确。

**Step 6 — MHA layer_first D2H/H2D 接入**
- 改动：`mha.py::backup_from_device_all_layer`（D2H 按 buffer_info 分段，target 一次、每个
  draft 一次，每段 device/host 对齐到段内 layer 0）与 `load_to_device_per_layer`（H2D per-layer，
  device/host 切单层）的 layer_first 子分支接入 Step 5 的 AscendC 路径（映射与示例见设计文档 §5.3）。
- 单测：同 Step 5 文件 —— backup 后 `k_buffer[l]`（及 `v_buffer[l]`）逐层 == 期望；backup+load 复原；单层语义；
  draft 层写入正确 host 层（`target_layer_num + i`）的越界/错层回归。

### 阶段 C：MLA 特殊路径（indexer / FP8 DSA）

**Step 7 — MLA indexer 作为独立 component**
- 改动：`mla.py` 的 `_build_ascendc_component_meta` 把 `index_k` / `index_k_scale` 作为第 3/4 个
  component（`_indexer_slot_range_for_layer` 做 device 层→slot 映射；host scale 取 5-D
  `[num_indexer_layers, page_num, page_size, 1, 1]`）。
- 单测：`test/registered/unit/npu/mem_cache/test_hicache_layer_first_mla_indexer.py`（新建）
  —— 带 index_k/scale 的子集层 pool，断言 index_k/scale 搬运正确、slot 映射正确、component 数量 ≤ 4；
  另覆盖 `ik_start > 0` 的绝对 slot 定位（不双重偏移）与 None 场景（无 index_k / 无 index_scales
  时跳过、不生成 component）。

**Step 8 — FP8 DSA 空 V component 跳过**
- 改动：`mla.py` 在 `v_width==0` 时 `v_buffer` 用 5-D 零元素空张量（非 `None`、不占容量），
  `_build_ascendc_component_meta` 的 V component 因 `numel()==0` 被跳过、只保留 K component；容量闭合。
- 单测：同 Step 7 文件 —— DSA 场景容量闭合 + backup/load 不抛错 + 空 V 分支被跳过。

### 阶段 D：开关 + 集成

**Step 9 — 布局开关 & 强制 memfabric & L3 拒绝**
- 改动：`python/sglang/srt/hardware_backend/npu/utils.py` 仅 `!= "layer_first"` 时投影默认（此函数
  **不做** memfabric 强制）；memfabric 强制改由通用 MHA/MLA pool 的 `init_kv_buffer` 按 **pool class**
  判定（`MHATokenToKVPoolHost`/`MLATokenToKVPoolHost` 且 `layer_first` → 要求 memfabric；
  `MambaPoolHost` 与 DSv4 专用池为独立 class、天然豁免，**不用** `is_deepseek_v4`/`uses_ssm_state`
  做模型级短路，因 Kimi 同时含 MLA 池与 SSM 池，见设计文档 §6.8）；`python/sglang/srt/arg_groups/hicache_hook.py`
  对 `layer_first + 任意非空 L3 + kernel_ascend` 配置阶段抛错。
- 单测：`test/registered/unit/npu/test_hicache_layer_first_config.py`（新建，参照
  `test_sparsity_driven_kv_offload_config.py`）—— layer_first 不被覆盖；通用 MHA/MLA pool 的
  layer_first 强制 memfabric、`MambaPoolHost`/DSv4 池豁免；layer_first+非空 L3 抛错；GPU 路径行为不变。

**Step 10 — 端到端精度（NPU 集成，非单测）**
- 测试：`test/registered/npu/basic_function/HiCache/` 新增 layer_first 用例（参照
  `test_npu_hicache_mha.py` / `test_npu_hicache_mla.py`）；Kimi-K3 追加「L2 前缀命中后恢复
  SSM state」专项（走 `MambaPoolHost` 分支）。

**Step 11 — 性能对比 + CI 注册**
- 测试：`test/registered/kernels/benchmark/kvcache/bench_hicache.py` 或 server 压测，长/短序列两类；
  `register_npu_ci(...)` 注册（§11.4）。

**提交节奏建议**：Step 1（结论记录，随文档）、Step 2-4、Step 5-6、Step 7-8、Step 9 各为一个可独立
review 的 commit（单测随改动一起提交）；Step 10-11 为集成门禁，不混入前面的功能 commit。