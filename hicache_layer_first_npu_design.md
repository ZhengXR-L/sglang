# 功能设计：NPU HiCache L1→L2 传输路径支持 layer_first 布局

> 关联需求：`hicache_layer_first_npu_requirements.md`（需求记录，独立文档）
> 本文档独立成篇，专注「如何改动实现 layer_first」的功能设计，不与需求文档合并。
> 日期：2026-10-21 ｜ 状态：设计稿 v3（核心传输算子由 `transfer_kv_dim_exchange` 切换为
> Memfabric AscendC 零拷贝 `_transfer_ascendc_sparse_copy`，`host_layout_mode=1`）

---

## 1. 目标与非目标

### 1.1 目标
在 NPU（Ascend）场景下，让 HiCache 的 host pool（L2）支持 `layer_first` 布局，使
`--hicache-mem-layout layer_first`（配合 `kernel_ascend`）下，L1↔L2 的 backup / load
传输功能正确、性能不低于 `page_first` 系布局。

### 1.2 非目标（本次不覆盖）
- L3 存储后端（**所有** `hicache_storage_backend`，含 `mooncake` / `npu_memcache` /
  `file` / `nixl` / `hf3fs` / `aibrix` / `tensorcast` / `seaweedfs`）对 layer_first 的支持：
  本期的 layer_first 仅限纯 L1/L2（`--hicache-storage-backend` 为空），NPU（`kernel_ascend`）下
  任何非空 L3 后端组合在**启动配置解析阶段**拒绝（见 §7.3）；
- **非 Memfabric host 内存下的 layer_first（仅限通用 MHA/MLA pool）**：本方案在**通用 MHA/MLA
  host pool** 的 L1↔L2 传输强制走 Memfabric AscendC 零拷贝路径，`layer_first` 仅支持
  `SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric`（§6.8）；非 memfabric host 内存不再作为这两类
  pool 的 layer_first 传输后端（启动配置阶段强制，§7.2）。**例外**：DeepSeek-V4 专用池与
  Kimi-K3 的 SSM 状态池（`MambaPoolHost`）保留既有 pinned + `transfer_kv_dim_exchange` 路径（§4.1）。
- `page_head` / `page_unified` 等其它布局在 NPU 下的补全；
- 底层 `kv_exchange_copy` 内核的 `host_layout_mode=1` 实现 —— 位于外部依赖 `memfabric_hybrid`，
  **已实现（`LayerFirstWalker`），本方案无需改动外部依赖**（§2.4）。

---

## 2. 设计基石：AscendC `kv_exchange_copy` 的传输契约

这是整个设计的核心约束，必须先从底层 kernel 理解清楚。本方案**不再使用**
`transfer_kv_dim_exchange`（SDMA `aclrtMemcpy2dAsync` 2D strided copy），改走 Memfabric
AscendC AIV 融合 kernel。

### 2.1 现有 Python 封装与元数据结构

现有 `mla.py::_transfer_ascendc_sparse_copy`（L549-618）已经是该 kernel 的完整 Python 封装，
结构如下（本次改动在其上做 **layer-first 扩展**，不重写骨架）。

**header（8 个 int64，`vals[0:8]`）**：

| 下标 | 含义 | 本次 layer_first 的值 |
|------|------|----------------------|
| 0 | component 数量 `len(comps)`（≤ 4） | K/V/index_k/index_k_scale |
| 1 | `num_pages = host_indices.numel() // page_size` | 同 |
| 2 | `page_size` | 同 |
| 3 | direction（D2H/H2D） | 同 |
| 4 | `device_indices.data_ptr()` | 同（device 常驻，不转 CPU） |
| 5 | `host_indices.data_ptr()` | 同（`to_device_no_sync` 上传，不 sync） |
| **6** | **`host_layout_mode`** | **0=page-first → 改传 1=layer-first** |
| 7 | 保留 `0` | 同 |

**每个 component 的元数据（9 个 int64）**（`_ascendc_comp_meta`，L621-641）：

```
(dev_ptr,              # device 数据首址（data_ptr）
 host_ptr,             # host 数据首址（data_ptr，之后被 _rewrite_host_base_to_dva 转 DVA）
 dev_stride0,          # device 层间字节 stride（device 恒 layer-first）
 dev_stride1,          # device token/page 间字节 stride
 host_page_stride,     # host page 间字节 stride
 host_layer_stride,    # host 层间字节 stride
 width*itemsize,       # 单行字节宽（shape[2:] 各维乘积 * itemsize）
 lo, hi)               # 该 component 的层区间 [lo, hi)
```

- `_build_ascendc_component_meta`（L643-698）拼出 K（+V，若 `numel()>0`）+ index_k/scale 的
  component 列表；`len(comps) > 4` 抛错。
- `_rewrite_host_base_to_dva`（L700-731）：用 `memfabric_hybrid.offload.get_dva()` 把每个
  component 的 `host_ptr`（VA）改写为 DVA，供 AIV 在 device 侧直接解引用 —— 这是该 kernel
  能「indices 不回传 CPU、不阻塞 load stream」的根因。
- `_launch_ascendc_kv_exchange`（L733-744）：把 `vals` 元数据 pin + 拷到 device，调
  `offload.kv_exchange_copy(meta, device)`，`ret != 0` 抛错。

### 2.2 关键语义（对比 `transfer_kv_dim_exchange`）

| 维度 | `transfer_kv_dim_exchange`（旧方案） | `kv_exchange_copy`（本方案） |
|------|-------------------------------------|------------------------------|
| 实现 | sgl-kernel-npu 内 SDMA `aclrtMemcpy2dAsync` 2D copy | 外部 `memfabric_hybrid` AIV 融合 kernel |
| 每层调用 | 逐层 `N=1`（host 跨层 pitch 被固定为「单 page 宽」） | **每个 buffer_info 一次 compact meta 下发其全部层**（跨层 stride 由 component 元数据显式表达） |
| indices | 转 CPU（`host_indices` 在 host 侧） | 常驻 device，不 round-trip |
| load stream | 每层 `on_layer_done` 下逐层发起 | 不 sync load stream，可与 compute overlap |
| host 内存 | 任意（甚至普通 pinned） | **必须 Memfabric-mapped**（kernel 直接解引用 host 指针，需 `get_dva`） |
| 短序 launch 开销 | 放大 `layer_num` 倍 | 每个 info 一次下发（消除逐层） |

### 2.3 host 布局与 `host_layout_mode`

`host_layout` 描述 host pool 里「层」与「page」的维度顺序，由 header 第 7 个 int
（`host_layout_mode`）传给 kernel：

- `0`（page-first，现有唯一支持）：host buffer 为 `[page, layer, page_size, 1, width]`，
  此时 `host_page_stride = stride(0)`、`host_layer_stride = stride(1)`（现有 `_ascendc_comp_meta`
  L628-629 正是此假设）。
- `1`（layer-first，本次扩展）：host buffer 为 `[layer, page, page_size, 1, width]`，
  `host_layer_stride = stride(0)`、`host_page_stride = stride(1)`。

**为什么 layer_first 必须走这条路径**：`transfer_kv_dim_exchange` 的 host 跨层 pitch 被硬编码
为「单 page 宽」（`k_host_pitch = P*heads*head_dim*itemsize`），无法表达 layer_first 的「跨层
stride == 整层大小」，故只能逐层（`N=1`）调用。而 `kv_exchange_copy` 用 component 元数据里的
`host_layer_stride` 显式携带跨层 stride，layer_first 下 `host_layer_stride = 整层字节数`，
即可**每个 buffer_info 一次下发其全部层**，消除逐层 launch 开销（这正是需求文档 §11.3 的短序列性能动机）。

### 2.4 底层 kernel 能力边界（已确认支持，无前置阻塞）

`memfabric_hybrid` 源码确认：`offload.kv_exchange_copy` 的 `host_layout_mode=1`（layer-first）
**已完整实现**，sglang / sgl-kernel-npu 侧**无需改动外部依赖**：

- kernel 枚举 `HostLayoutMode { HOST_LAYOUT_PAGE_FIRST=0, HOST_LAYOUT_LAYER_FIRST=1 }`；
  `Process()` 依 `meta[6]` 选择 `PageFirstWalker` 或 `LayerFirstWalker`。
- `LayerFirstWalker` 为 layer-outer / page-inner：page block-partition 到各 AIV core，并
  peek-ahead 合并「host 与 device 都连续」的 page run 为一次大拷贝，降低 DMA 请求数；找不到
  连续 run 时退化为逐 page 拷贝（同一套 UB 双缓冲）。
- **地址公式 mode 无关**：`MakeEntry` 里
  `hostRow = hostBase + hostPage*hostPagePitch + layer*hostLayerPitch`，`hostPagePitch`（o+4）与
  `hostLayerPitch`（o+5）是两个独立语义字段；`host_layout_mode` 只决定遍历顺序，不影响地址计算。
  故无需改 kernel，sglang 侧只需把这两个 pitch 按实际 host 布局填对。

sglang 侧改动收敛为：

1. `_transfer_ascendc_sparse_copy` 的 header 第 7 位传 `1`（layer-first）；
2. `_ascendc_comp_meta` 在 layer-first 下填 `host_layer_pitch = stride(0)`、`host_page_pitch =
   stride(1)`（§6.4(C)）。

---

## 3. 总体方案

对通用 MHA / MLA host pool，在 `backup_from_device_all_layer`（D2H）和
`load_to_device_per_layer`（H2D）的 `kernel_ascend` 分支新增 `layout == "layer_first"`
子分支，实现为 **Memfabric AscendC 零拷贝 `_transfer_ascendc_sparse_copy`
（`host_layout_mode=1`）**：

```
# 传输方法约定（layer_first）：_transfer_ascendc_sparse_copy 的 device 视图与 host 视图必须
# 「段内对齐到 layer 0」。kernel 的地址公式对 device 与 host 使用同一个 layer 计数器：
#     deviceRow = devBase + layer*devLayerPitch
#     hostRow   = hostBase + page*hostPagePitch + layer*hostLayerPitch
# 因此 lo/hi 只能描述「从两端 base 起步的相对层号」，不能沿用「device 全局层号 == host 全局
# 层号」的隐含假设——draft 的 device 本地层从 0 起，其 host 层却落在 target_layer_num 之后。

# D2H（backup）：无 layer_id 入参，按 buffer_info 分段（target 1 段 + 每个 draft 1 段）
host_offset = 0                                   # 该段在全局 host 的起始层号
for info in self._buffer_infos:                   # MLA 用 _native_infos[0..]，驱动语义相同
    dev_k, dev_v = info.transfer_buffers()        # [N, device_tokens, ...]，N == 该段 device 层数（MLA 分开取 native.k / native.v）
    n = dev_k.shape[0]
    host_k = self.k_buffer[host_offset : host_offset + n]   # host 切到该段（段内 layer 0 == host_offset）
    host_v = self.v_buffer[host_offset : host_offset + n]
    _transfer_ascendc_sparse_copy(dev_k, dev_v, host_k, host_v, host_indices, device_indices,
                                  direction=D2H, layer_start=0, layer_num=n)
    host_offset += n

# H2D（load）：只搬传入的 layer_id（per-layer，逐层流水，不 sync）。先反查所属 info 与全局
# host 层，把 device 与 host 都切到单层（段内 layer 0），再 lo=0、hi=1：
info_index = 0 if not is_draft else (layer_id - self._device_layer_num + 1)   # MLA；MHA 用 _buffer_infos
dev_k_full, dev_v_full = self._buffer_infos[info_index].transfer_buffers()    # 或 _native_infos[info_index]
local_layer = 0 if is_draft else layer_id                                     # draft 的 device 本地层 = 0
host_layer_id = self._host_layer_by_controller[layer_id]                      # draft → target_layer_num + i
dev_k  = dev_k_full[local_layer : local_layer + 1]                            # device 切单层（段内 layer 0）
dev_v  = dev_v_full[local_layer : local_layer + 1]
host_k = self.k_buffer[host_layer_id : host_layer_id + 1]
host_v = self.v_buffer[host_layer_id : host_layer_id + 1]
_transfer_ascendc_sparse_copy(dev_k, dev_v, host_k, host_v, host_indices, device_indices,
                              direction=H2D, layer_start=0, layer_num=1)
```

**为什么 D2H / H2D 必须分开写，且不能直接传全局 layer_id**：`l2_transfer.py::submit_host_to_device`
（L145-183）在引擎层 `for layer_id in range(transfer_layer_id_max)` 逐层调用
`load_to_device_per_layer_physical`，并在每层后触发 `on_layer_done(layer_id)`。因此 H2D 回调是
「单层」语义；D2H 无 `layer_id` 入参，按 buffer_info 分段、每段一次下发其全部层。

但**分段后不能再复用现有 `_build_ascendc_component_meta` 的参数形态**：它目前直接读
`self.k_buffer`（整池）与 `buffers.k`（该 info 的 device 张量），并把同一个 `lo/hi` 同时当作
device 与 host 的层区间。page-first 下 target 的 device 层号 == host 层号，这个假设才成立；一旦
引入 draft，draft 的 device 本地层从 0 起、host 层在 `target_layer_num` 之后，二者不再同号——若
沿用旧形态，要么 `k_hi` 越界（`host_layer_num != device_layer_num` 校验触发），要么读写错误的
host 层。因此 `_transfer_ascendc_sparse_copy` / `_build_ascendc_component_meta` 必须改为**接收
显式的 host 切片视图**（`host_k` / `host_v` / `host_index_k` / `host_index_k_scale`），并让每段
的 device base 与 host base 都对齐到「该段的 layer 0」，`lo=0`、`hi=段层数`（如上代码块）。

target 与 draft 的 device buffer 分属不同 `buffer_info`、地址不连续，单一 component（一个
base + 一套 layer 区间）无法跨 info 表达，必须分段调用；分段的 host 偏移用
`_host_layer_by_controller`（MLA）或 `layer_start` 累加（MHA）求得。

MHA 的 K/V 本来就是两块独立 buffer，此模式可直接套用（§5）；MLA 因 host 侧有「合并 vs 分离」
的布局选择，需专门展开（§6）。两者都在 layer_first 下把 host buffer 组织为 layer-first 五维
（或等价的 5-D 视图），使 `host_layer_stride = stride(0)` 能表达跨层 stride。

---

## 4. 参照实现：DeepSeek V4 专用 pool（仅作 shape/view 参照）

`memory_pool_host.py` 的 `DeepSeekV4PagedHostPool` / `DeepSeekV4StateHostPool` 已实现
layer_first + kernel_ascend（走 `transfer_kv_dim_exchange` 逐层）。本方案**不改其传输算子**
（DSv4 专用池仅回归验证），仅借用其 shape/view 层面的先例：

- `_host_page_view(l)`（L351-372）：把 layer_first 的 `kv_buffer[l]` 转成
  `[num_host_pages, 1, P, 1, dim]` 五维视图 —— 说明「layer_first 张量 → 显式五维视图」的
  处理思路仍适用，但本方案的目标视图是 **layer-first**（`[l, pages, P, ...]` 而非
  `[pages, 1, P, ...]`）。
- 其 D2H/H2D 的逐层循环结构（L528-554 / L625-647）仍是「按 `_native_infos` 分块、避免跨
  draft 索引越界」的驱动范本（§6.4 复用该驱动结构，仅把算子换成 AscendC）。

### 4.1 为什么不将 DSv4 专用池也切到 AscendC

DSv4 专用池（`DeepSeekV4PagedHostPool` / `DeepSeekV4StateHostPool`）**继续走
`transfer_kv_dim_exchange` 逐层，不改用 `_transfer_ascendc_sparse_copy`**，四方面理由：

1. **内存模型不匹配**：DSv4 的 `device_buffers` 与 `kv_buffer` 都是「**每层独立 alloc、地址不
   连续**」的 list（`kv_buffer[l]` 为独立的 `[num_host_pages, item_bytes]` uint8 张量）；
   `kv_exchange_copy` 的地址公式 `device_base + layer*device_layer_pitch` /
   `host_base + host_page*host_page_pitch + layer*host_layer_pitch` 要求所有层处于**单一连续
   地址空间**。要把 30+ 层拼进一个大连续张量才能喂给 AscendC，改动大且 DSv4 的 device
   native buffer 是框架按多个 sub-pool（C4 KV / FP4 indexer / unified_kv）独立分配，层间
   连续化几乎不可控。
2. **host 内存非 memfabric**：DSv4 host buffer 走 `alloc_with_pin_memory`（普通 pinned），
   **无 DVA**；`kv_exchange_copy` 依赖 `offload.get_dva` 把 host VA 转 device VA，对非
   memfabric 内存不可用。
3. **token 级路径覆盖不了**：DSv4 的 C4 fused row 是 token 粒度拷贝，走专用算子
   `transfer_cache_dsv4_mla`（`memory_pool_host.py` L473 / L580）；`kv_exchange_copy` 是 page
   行级 copy，**无法替代**这条路径，即使切了 AscendC 也仍须保留该专用算子。
4. **性能收益不确定、投入产出比低**：即便完成前两项改造，收益也仅来自「减少逐层 N=1 的
   launch 开销」，只在短序列 / 少量 page 命中时明显；且被「多 sub-pool 分次调用」与
   「token 级路径不参与」稀释。长序列大块搬移时两者都逼近 DMA 带宽上限，差异消失。

因此本次对 DSv4 专用池**只做回归验证**（确认「强制 memfabric + AscendC」的改动不误伤其
`layer_first + kernel_ascend + 普通 pinned` 的既有传输路径），不改造其传输算子。

---

## 5. MHA 详细设计（补充 `_transfer_ascendc_sparse_copy`）

### 5.1 现状 buffer 形状

`mha.py` 的 `init_kv_buffer` 已支持 layer_first：

```python
# layer_first: dims = (2, layer_num, size, head_num, head_dim)
# buffer[0] -> self.k_buffer : [layer_num, size, head_num, head_dim]
# buffer[1] -> self.v_buffer : [layer_num, size, head_num, head_dim]
```

- host：`k_buffer[l]` = `[size, head_num, head_dim]`，其中 `size == page_num * page_size`
  （**隐式 page**，4-D）；
- device：NPU 分支通过 `info.transfer_buffers()`（`device_pool_info.MHABufferInfo`）取到
  `[N, device_tokens, head_num, head_dim]` / `[N, device_tokens, head_num, v_head_dim]` 的多层
  连续张量（第 0 维是层，layer-first，K/V 天然分离）；`self._buffer_infos` 是一串
  `MHABufferInfo`（主 pool + 可能的 draft buffer infos），需要做「全局层号 ↔ 本地层号」映射。

### 5.2 设计改动：MHA 补 memfabric 分配 + 新增 `_transfer_ascendc_sparse_copy`

**（1）先补 memfabric 分配（注意签名差异，不能只替换 `alloc_func`）**：MHA 的 `init_kv_buffer`
当前用 `ALLOC_MEMORY_FUNCS["npu"]`（`alloc_with_pin_memory`），调用处**恒传入
`registration_granularity_bytes`**（`mha.py` L328-336）；而 `alloc_with_memfabric` 的签名是
`(dims, dtype, device, pin_memory, allocator)`（`npu_memfabric.py` L106-120），**不接收
`registration_granularity_bytes`**（该参数仅对 CUDA `alloc_with_host_register` 有意义，
`common.py` L353-359）。直接 `alloc_func = alloc_with_memfabric` 后沿用现有 kwargs 会在初始化时
抛 `TypeError`。因此 MHA 的 memfabric 分支须**单独写调用**（像 `mla.py` L379-385 那样不走
`ALLOC_MEMORY_FUNCS`）：

```python
if self.layout == "layer_first" and _is_npu:
    if not memfabric_host_memory_enabled():          # 与 §6.8 的强制校验一致：未开启即报错
        raise ValueError(
            "layer_first layout requires SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric "
            "(generic MHA/MLA host pool AscendC path)"
        )
    ensure_memfabric_capacity(total_bytes, torch.npu.current_device())
    buffer = alloc_with_memfabric(dims, dtype=self.dtype, device=self.device,
                                  pin_memory=self.pin_memory, allocator=self.allocator)
else:
    buffer = alloc_func(dims, dtype=..., ..., registration_granularity_bytes=...)
```

`total_bytes` 按 MHA layer_first 的两块 buffer 体积闭合（`2 * layer_num * size * head_num *
head_dim * itemsize`，`size = page_num * page_size`）；否则 `_rewrite_host_base_to_dva` 的
`offload.get_dva(host_ptr)` 对普通 pinned 指针会失败。

**（2）新增 `_transfer_ascendc_sparse_copy`**：MHA **没有**此方法，需新增类似实现。复用 MLA 的
「component 元数据（9-int）+ host DVA 改写 + `offload.kv_exchange_copy` launch」骨架
（即 §2.1 的 `_ascendc_comp_meta` / `_rewrite_host_base_to_dva` / `_launch_ascendc_kv_exchange`
三段逻辑），**重写 component stride 推导**：MHA 的 host buffer 是 4-D
（`[layer, size, head_num, head_dim]`，隐式 page），需先 `view` 成显式 page 的 5-D 再填元数据。

**MHA component 视图构造**（layer_first）：

```python
# device：info.transfer_buffers() -> [N, device_tokens, head_num, head_dim]
dev_k = device_k.view(N, device_pages, page_size, head_num, head_dim)   # 5-D，layer-first
# host：k_buffer -> [layer_num, page_num, page_size, head_num, head_dim]
host_k = self.k_buffer.view(layer_num, page_num, page_size, head_num, head_dim)  # 5-D，layer-first
```

- device 恒 layer-first：`dev_stride0 = dev_k.stride(0)*itemsize`（层间）、`dev_stride1 =
  dev_k.stride(1)*itemsize`（page 间）；
- host layer-first：`host_layer_stride = host_k.stride(0)*itemsize`、`host_page_stride =
  host_k.stride(1)*itemsize`；`width = head_num * head_dim * itemsize`（`v` 同理用 `v_head_dim`）。

**header 第 7 位传 `1`（`host_layout_mode=1`）**。

### 5.3 D2H / H2D 接入

在 `mha.py::backup_from_device_all_layer` / `load_to_device_per_layer` 的 `kernel_ascend` 分支
新增 `layer_first` 子分支。MHA 的 host `k_buffer` / `v_buffer` 为 layer-first 4-D
（`[layer_num, size, head_num, head_dim]`，dim 0 = 层，`k_buffer`==`kv_buffer[0]`、`v_buffer`==
`kv_buffer[1]`，见 mha.py L371-376）；device 侧 `info.transfer_buffers()` 返回该 info 的
`(device_k, device_v)`，`device_k.shape[0]` 为该 info 的层数（target = `target_layer_num`，
每个 draft = 1）。层号映射沿用现有 `layer_start` 累加（backup）与 `remaining` 定位（load）两套
已存在逻辑（mha.py L702-732 / L519-549），不新造 `_buffer_info_and_local_for_layer`：

```python
# D2H（backup）：按 _buffer_infos 分段，host 切到该段、device 用该段（两者段内 layer 0 对齐）
layer_start = 0
for info in self._buffer_infos:
    device_k, device_v = info.transfer_buffers()
    n_layer = device_k.shape[0]
    host_k = self.k_buffer[layer_start : layer_start + n_layer]   # [n_layer, size, H, D]
    host_v = self.v_buffer[layer_start : layer_start + n_layer]
    _transfer_ascendc_sparse_copy(device_k, device_v, host_k, host_v, host_indices,
                                  device_indices, direction=D2H, layer_start=0, layer_num=n_layer)
    layer_start += n_layer

# H2D（load）：layer_id 是全局层号（MHA：host_layer_id == layer_id），用 remaining 定位到 info 内的本地层
remaining = layer_id
for info in self._buffer_infos:
    device_k, device_v = info.transfer_buffers()
    n_layer = device_k.shape[0]
    if remaining < n_layer:
        _transfer_ascendc_sparse_copy(device_k[remaining : remaining + 1],
                                      device_v[remaining : remaining + 1],
                                      self.k_buffer[layer_id : layer_id + 1],
                                      self.v_buffer[layer_id : layer_id + 1],
                                      host_indices, device_indices,
                                      direction=H2D, layer_start=0, layer_num=1)
        break
    remaining -= n_layer
```

上面 `host_k` 切片就是「把该段 host base 对齐到段内 layer 0」的落地：draft 段的
`layer_start == target_layer_num`（第一个 draft），其 device `device_k.shape[0]==1`、本地层 0，
host 切到 `self.k_buffer[target_layer_num:]`，两者层号再重新对齐，因此 `layer_start=0`、
`layer_num=段层数` 即可，无需再传全局 `layer_id`。

### 5.4 关键点

- MHA 的 K/V 本就是两块独立 buffer，一次 `_transfer_ascendc_sparse_copy` 的 K + V 两个 component
  同时传输。
- `head_num` 在 device 与 host 侧一致（MHA 无 rank-replication），可直接对齐。
- host 的 `size` 必须是 `page_num * page_size` 的整倍数，才能 `view` 成 `[layer, page_num,
  page_size, head_num, head_dim]`（否则需 `_require_packed_device_rows` 或拒绝，见 §8）。
- **NHD vs MLA 扁平对 kernel 无区别**：`kv_exchange_copy` 是「字节行」级搬运，只看一个 page 的
  连续字节数 `width` 与三个 pitch（§2.4），**不解读 head 维语义**。MHA 的 NHD
  `[..., page_size, head_num, head_dim]` 与 MLA 的扁平 `[..., page_size, 1, k_width]` 等价——
  `width = page_size × head_num × head_dim × itemsize` 自动涵盖（`_ascendc_comp_meta` 取
  `prod(shape[2:])`）；`head_num × head_dim` 只是「单 token 宽度」的另一种表达。
- **本路径仅支持 `head_dim == v_head_dim` 的对称 MHA**：主类 `init_kv_buffer` 的 layer_first
  `dims=(2, layer_num, size, head_num, head_dim)` 对 K/V 都用 `head_dim`；`head_dim != v_head_dim`
  的模型走 `AsymmetricMHATokenToKVPoolHost`（L1343），其 `init_kv_buffer` 与 kernel_ascend 均
  不支持 layer_first。故 MHA AscendC layer_first 只覆盖对称 MHA。GQA（`head_group_num > 1`、
  `head_dim` 相同）仍属对称，可覆盖；`v_head_dim != head_dim` 的 MiMo 一类不在本期范围。

---

## 6. MLA 详细设计（扩展 `_transfer_ascendc_sparse_copy`，含 indexer 与 FP8 DSA）

### 6.1 逻辑顺序

MLA 的合并 KV 记录逻辑顺序为：

```
cache_k = [ latent K (长度 kv_lora_rank) | RoPE K (长度 qk_rope_head_dim) ]
```

即 **K 段（压缩 KV / latent）在前，V/RoPE 段在后**。NPU 设备池 `NPUMLATokenToKVPool.set_kv_buffer`
（`memory_pool_npu.py` L982-1024）收到合并 `cache_k` 时正是按此分割：

```python
cache_k, cache_v = cache_k.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
```

### 6.2 设备池物理布局（普通 BF16 vs FP8 DSA）

| 项 | 普通 BF16 MLA | FP8 DSA MLA（打包） |
|----|--------------|--------------------|
| K buffer 宽 | `kv_lora_rank` | `kv_cache_dim = kv_lora_rank + kv_lora_rank//128*4 + qk_rope_head_dim*2` |
| V buffer 宽 | `qk_rope_head_dim` | `0`（`kr_cache_dim = 0`，RoPE 已打包进 K） |
| 存储 dtype | `store_dtype` | `float8_e4m3fn`（打包段内含 BF16 rope / FP32 scale） |
| index_k buffer | `index_head_dim`（部分层） | `index_head_dim`（部分层）+ `index_k_scale_buffer`（FP32） |

- 普通 BF16：device 侧是**两块独立 buffer**（`k_buffer` 宽 `kv_lora_rank`、`v_buffer` 宽
  `qk_rope_head_dim`），**不是**一条拼好的 `kv_cache_dim` 记录。
- `MLASplitBuffers`（`device_pool_info.py`）:`k`/`v`/`index_k`/`index_scales`/
  `index_layer_indices`/`k_width`/`v_width`/`index_width`。其中 FP8 DSA 时
  `k_width == kv_cache_dim`（打包宽）、`v_width == 0`。

### 6.3 核心矛盾：layer_first 不能照搬「合并记录」

现有 GPU 语义下 MLA 的 host `layer_first` 分支（`mla.py` L288-294）是「合并记录」：

```python
dims = (layer_num, size, 1, kv_cache_dim)   # 每 token 一条 K|V 拼接记录
```

但 AscendC `kv_exchange_copy` 以 **component** 为单位搬运：每个 component 是「一种固定宽度的连续
buffer」（`width` = `shape[2:]` 各维乘积，见 §2.1）。合并记录里 token 的 `kv_cache_dim` 宽 = K 段
（`kv_lora_rank`）+ V 段（`qk_rope_head_dim`）拼接，无法用**单一** component 表达（K 与 V 宽度、
dtype 语义都不同）；即便把整条记录当一个 component 传，也会破坏「K/V 分离、逐类 GPA 解引用」的
kernel 语义。

**结论**：NPU layer_first 的 L1↔L2 传输必须采用「分开分配」两块 buffer —— K 一个 component、V
一个 component（FP8 DSA 下 V component 因 `numel()==0` 被跳过，见 §6.6）。「合并记录」只保留给
GPU 语义 / L3 存储字节序，不作为 L1↔L2 的 component 布局。

### 6.4 方案：layer_first 分开分配 + AscendC component

新增 host `init_kv_buffer` 的 layer_first 分配逻辑（复用 `page_first_kv_split` 的分段维度，把
`[page_num, layer_num, ...]` 换成 `[layer_num, page_num, ...]`）：

| host buffer | 形状（layer-first） | 宽度 |
|-------------|------|------|
| `k_buffer` | `[layer_num, page_num, page_size, 1, k_width]` | `k_width = kv_lora_rank`（BF16）/ `kv_cache_dim`（FP8 DSA） |
| `v_buffer` | `[layer_num, page_num, page_size, 1, qk_rope_head_dim]`（BF16）；FP8 DSA 为 **5-D 零元素空张量** `[layer_num, page_num, page_size, 1, 0]` | BF16 `qk_rope_head_dim`；FP8 DSA `v_width==0`（`numel()==0`，不占容量） |
| `index_k_buffer` | `[num_indexer_layers, page_num, page_size, 1, index_width]` | 部分层 |
| `index_k_scale_buffer` | `[num_indexer_layers, page_num, page_size, 1, 1]`（FP32，5-D） | 仅 FP8 DSA + indexer |

**传输驱动（D2H 全量 / H2D per-layer）与 AscendC 元数据落地：**

**(A) D2H（`backup_from_device_all_layer`）——非层分片路径进入 `kernel_ascend` 分支，新增
`layer_first` 子分支：** 按 `_native_infos` 分段（target 一次、每个 draft 一次），沿用 mla.py
L892-905 的分块驱动以避免跨 `_native_infos` 索引越界；每段调用 `_transfer_ascendc_sparse_copy`
时 device 与 host 都切到该段、以 `layer_start=0`、`layer_num=段层数` 调用（**不是** `layer_num=-1`），
映射与调用示例见下方「MLA 层号映射（target / draft）与 host 切片」。

**(B) H2D（`load_to_device_per_layer`）——** 复用 `_transfer_one_layer`（mla.py L763-766）的
`info_index` / `device_layer` / `host_layer_id` 反查，把 device 与 host 都切单层后以
`layer_start=0`、`layer_num=1` 调用（**不是** `layer_start=layer_id`），示意同下方代码块。

**(C) `_ascendc_comp_meta` 支持 layer-first host stride：**

现有实现（§2.1）假设 host 是 `[page, layer, ...]`（`host_page_stride=stride(0)`、
`host_layer_stride=stride(1)`）。layer_first 下 host buffer 是 `[layer, page, ...]`，需按
`host_layout_mode` 分支互换：

```python
@staticmethod
def _ascendc_comp_meta(dev_t, host_t, lo, hi, host_layout_mode):
    itemsize = dev_t.dtype.itemsize
    width = 1
    for dim in dev_t.shape[2:]:
        width *= dim
    if host_layout_mode == 0:                      # page-first: [page, layer, page_size, 1, w]
        host_page_stride = host_t.stride(0) * itemsize
        host_layer_stride = host_t.stride(1) * itemsize
    else:                                          # layer-first: [layer, page, page_size, 1, w]
        host_layer_stride = host_t.stride(0) * itemsize
        host_page_stride = host_t.stride(1) * itemsize
    return (
        dev_t.data_ptr(), host_t.data_ptr(),
        dev_t.stride(0) * itemsize, dev_t.stride(1) * itemsize,
        host_page_stride, host_layer_stride,
        width * itemsize, lo, hi,
    )
```

**(D) `_build_ascendc_component_meta` 的 host 层数来源随布局切换：**

现 `host_layer_num = self.k_buffer.shape[1]`（L657）是 page-first 下的层数取法（shape 第 1 维）；
layer-first 下应取 `self.k_buffer.shape[0]`（层在 0 维）。同理 indexer 的
`ik_lo/ik_hi` 默认区间（L676 的 `self.index_k_buffer.shape[1]`）layer-first 下改为 `shape[0]`。

**(E) `_transfer_ascendc_sparse_copy` 的 header 第 7 位传 `1`：**

`vals[6] = 1  # host_layout_mode: 1 = layer-first`（现为 `0 # page-first`，L612）。

**层分片（`_layer_shard_enabled`）不在本次范围**：与 `page_first` 现状一致。
`backup_from_device_all_layer` 在进入 `kernel_ascend` 分支前仍会先判断 `_layer_shard_enabled` 并
逐层调用 `_backup_from_device_per_layer`；该方法保留 `if io_backend not in ("kernel", "direct"):
raise ValueError`，即 layer_first + kernel_ascend + 层分片组合仍被明确拒绝（与
`page_first_kv_split` 行为一致，本次不实现）。

**MLA 层号映射（target / draft）与 host 切片**：device 层号 → host 层号用
`_host_layer_by_controller`（mla.py L167-175）反查；draft 层号在 `_device_layer_num` 之后、映射到
`target_layer_num + i`；native buffer 选 `_native_infos[info_index]`（target=0，draft 依
`layer_id` 回退，`info_index = layer_id - _device_layer_num + 1`）。

分段后 **`_build_ascendc_component_meta` 不能再读 `self.k_buffer` / `self.v_buffer` 整池**，必须
改为接收显式的 host 切片视图（与 §3 一致）。D2H 按 `_native_infos` 分段、`host_offset` 累加：

```python
# MLA D2H：target 用 _native_infos[0]，每个 draft 各一次 _native_infos[j]
for info_index, info in enumerate(self._native_infos):
    native = info.native_buffers()                       # .k / .v / .index_k / .index_scales
    n = native.k.shape[0]
    host_lo = 0 if info_index == 0 else self.target_layer_num + (info_index - 1)
    host_k = self.k_buffer[host_lo : host_lo + n]        # 段内 layer 0 对齐
    host_v = self.v_buffer[host_lo : host_lo + n]
    _transfer_ascendc_sparse_copy(native, host_k, host_v, host_indices, device_indices,
                                  direction=D2H, layer_start=0, layer_num=n)
# indexer 仅在 target 的 native 上存在（index_layer_indices），其 host index_k/scale 视图随
# target 段一起整体传入；per-layer 时用 _indexer_slot_range_for_layer(native, device_layer) 定 slot。

# MLA H2D：单层，复用 _transfer_one_layer 的反查（L763-766）已有逻辑
native = self._native_infos[info_index].native_buffers()
device_layer = 0 if is_draft else layer_id
host_layer_id = self._host_layer_by_controller[layer_id]
_transfer_ascendc_sparse_copy(native.k[device_layer : device_layer + 1],
                              native.v[device_layer : device_layer + 1],
                              self.k_buffer[host_layer_id : host_layer_id + 1],
                              self.v_buffer[host_layer_id : host_layer_id + 1],
                              host_indices, device_indices,
                              direction=H2D, layer_start=0, layer_num=1)
```

关键点同 §3：`lo/hi` 用同一个 layer 计数器同时索引 device 与 host 两端，因此每段必须让 device
base 与 host base 都对齐到「该段 layer 0」，再传 `lo=0`、`hi=段层数`；不能把 `layer_id`（全局）
直接当 `layer_start` 传进去（draft 的 device 本地层是 0，host 层却在 `target_layer_num` 之后，
会越界或读写错层）。

### 6.4.1 派生接口适配（不能只改分配与传输分支）

`pool_host/mla.py` 下多个派生接口当前按「合并 kv_buffer」工作，layer_first split 化后必须逐项
定义行为（或明确禁止），否则 L2 容量与 disagg 注册会出错：

| 接口 | 现状（合并语义） | layer_first split 下的处理 |
|------|------------------|---------------------------|
| `data_refs`（L248-255） | 指向合并 `kv_buffer` 逐层视图 | 仿 `page_first_kv_split` 先例：指向 `k_buffer` 逐层视图；`v_buffer` / `index` 单独持有，不进 `data_refs` |
| `get_contiguous_buf_infos`（L265-278） | 按 `kv_buffer[i].nbytes` 报合并长度 | 改为报 `k_buffer` + `v_buffer`（+ index）的指针与长度，不能再引用已不存在的合并 `kv_buffer` |
| `get_data_page` / `set_from_flat_data_page`（L1024-1070） | 合并记录按 layer_first 读写整层 | 二者是 L3 存储路径，layer_first 组合已在**配置解析阶段**对任意非空 L3 后端拒绝（§7.3），故此处作为防御性兜底显式 `raise NotImplementedError`（正常启动永远到不了） |
| `get_size_per_token` → `_size_per_token`（L176-189） | 基础值 `kv_cache_dim*itemsize*layer_num`，仅 `page_first_kv_split` 计入 indexer | ① indexer 同样计入：`if layout in ("page_first_kv_split","layer_first") and native.index_k is not None`，累加 `index_width*itemsize*index_layers`（FP8 再 + `4*index_layers`）；② 基础值须与实际分配的 k/v 宽度闭合（见下方「V buffer 容量闭合」） |

其中 `_size_per_token` 是数据正确性关键：漏算 indexer 会让 `--hicache-size` 分配不足，
在含 indexer 的 DSA / GLM5.x 模型上造成容量误判或越界。

**V buffer 容量闭合（P0，决策：FP8 DSA 不分配 v_buffer）**：

- 现有 `page_first_kv_split` 的 `init_kv_buffer`（mla.py L370-383）**无条件分配** `v_buffer`
  （宽 `qk_rope_head_dim`），即便 FP8 DSA 的 device `v_buffer` 宽 `kr_cache_dim==0`；而
  `_size_per_token` 基础值只按 `kv_cache_dim*itemsize` 计算。于是 FP8 DSA 下实际分配了
  `kv_cache_dim + qk_rope_head_dim` 宽，却只按 `kv_cache_dim` 规划容量，**二者不一致**（多占
  `qk_rope_head_dim` 一段不被计量的内存）。
- 新 layer_first 必须闭合：**FP8 DSA 不分配 `qk_rope_head_dim` 宽的 `v_buffer`，改为分配
  `[layer_num, page_num, page_size, 1, 0]` 的 5-D 零元素空张量（`numel()==0`，不占容量）**，
  `k_buffer` 宽取 `native.k_width`（=打包宽 `kv_cache_dim`）；普通 BF16 分配 `v_buffer`
  （宽 `qk_rope_head_dim`），`k_buffer` 宽取 `kv_lora_rank`。两种情况下：

  ```
  BF16:   kv_cache_dim = kv_lora_rank + qk_rope_head_dim = k_width + v_width  → 基础值闭合
  FP8:    kv_cache_dim = k_width（打包宽），v_buffer 为 5-D 零元素空张量（numel=0）  → 基础值闭合
  ```

  故 `_size_per_token` 基础值 `kv_cache_dim*itemsize*layer_num` 无需额外改动即可覆盖两者，
  无需再为 v_buffer 单独加分量；indexer 按 ① 额外累加。
- 接口/传输对 FP8 DSA 的 `v_buffer` 处理：`init_kv_buffer` 里 FP8 DSA 分支把 `self.v_buffer` 分配为
  上述 5-D 零元素空张量（**不是 `None`**，保证 component 构建时 `numel()>0` 判断安全）；
  `_build_ascendc_component_meta` 靠 `if buffers.v.numel() > 0 and self.v_buffer.numel() > 0`
  跳过 V component（现 L670 已具备）；`get_contiguous_buf_infos` / `data_refs` 只需按
  `numel()==0` 跳过该空 V（与 §6.4.1 一致）。

### 6.5 合并记录方案（方案 B 备选，不推荐）

若坚持 host 用合并 `kv_cache_dim` 记录（`[layer_num, size, 1, kv_cache_dim]`），字节布局
定义为 **K 段在前、V/RoPE 段在后**（与 §6.1 逻辑序一致）：

```
token 行 = [ latent(kv_lora_rank) | rope(qk_rope_head_dim) ]
```

但 AscendC 以 component（固定宽度单一类 buffer）为搬运单位，合并记录无法拆成干净的 K/V 两个
component（或需先 staging 拆块，额外内存 + 拷贝），违背「端到端性能不低于 page_first」目标。
因此记录其字节序约定即可，L1↔L2 传输不采此方案。

### 6.6 FP8 DSA 特殊路径（单列，不套用普通 BF16）

FP8 DSA 的打包由 `_pack_dsa_fp8_kv_cache`（`memory_pool_npu.py` L949-970）完成，字节布局
（`torch.cat` 顺序）为：

```
packed = [ latent FP8 (kv_lora_rank × 1B) | rope BF16 (qk_rope_head_dim × 2B) | scale FP32 (kv_lora_rank//128 × 4B) ]
```

- 总宽 `kv_cache_dim = kv_lora_rank + kv_lora_rank//128*4 + qk_rope_head_dim*2`；
- device `v_buffer` 宽 `kr_cache_dim == 0`（`numel()==0`，`_build_ascendc_component_meta` 因
  `buffers.v.numel() > 0` 为假而跳过 V component）；
- host 侧**不分配 `qk_rope_head_dim` 宽的 `v_buffer`**，只分配 `k_buffer`（宽 `kv_cache_dim`，取
  `native.k_width`）+ `index_k_buffer` + `index_k_scale_buffer`；`v_buffer` 以 **5-D 零元素空张量**
  存在（容量闭合见 §6.4.1「V buffer 容量闭合」）；
- component 元数据下，K component 的 `width = kv_cache_dim * itemsize`（`dev_t.shape[2:]` 乘积
  自动算对打包宽），仅一个 K component 参与传输；`dev_v`/`host_v` 因 `numel()==0` 被跳过，**无需
  传 `None`**（与 `transfer_kv_dim_exchange` 的「必传 5-D V」不同，AscendC 用 component 元数据，
  不存在 `device_v.sizes()[4]` 越界问题）；
- 不能把打包段当作「普通 BF16 的 K 段 + V 段」两段处理：打包段内部已含 rope 与 scale，顺序、
  宽度、dtype 均不同，必须单列分支。
- **验证（只传 K 路径）**：单测构造 `v_width==0` 的 DSA pool，断言 `_build_ascendc_component_meta`
  只返回 K component（`len(comps)==1`，若含 indexer 则 +index 分支）、容量闭合、D2H/H2D 无越界；
  端到端再用 FP8 DSA 模型跑 D2H/H2D 校验 K 段逐层搬运正确（§10.1）。

### 6.7 indexer 附带缓存（作为独立 component）

- `index_k` 仅存在于 `index_layer_indices` 指定的子集层（如 GLM5.x 为 21/78 层）；host 侧层数
  == `num_indexer_layers`。
- 现有 `_build_ascendc_component_meta`（L673-693）已把 `index_k` / `index_k_scale` 作为第 3/4 个
  component（`comp_meta(device_index_k, self.index_k_buffer, ik_lo, ik_hi)` + scale），本次只需：
  ① layer-first 下 `index_k_buffer` / `index_k_scale_buffer` 改为 layer-first 5-D
  `[num_indexer_layers, page_num, page_size, 1, width]`（§6.4 表）；② 默认 `ik_hi` 来源随布局
  `shape[0]`/`shape[1]` 切换（§6.4(D)）。
- **index 层区间**与 K component 的 `layer_start/layer_num` 相互独立：`_transfer_ascendc_sparse_copy`
  有独立 `index_k_layer_start` / `index_k_layer_num` 入参；per-layer H2D 时须按 device 层 → index
  slot 映射（复用 `_indexer_slot_range_for_layer`，L541-547）传对区间，不能对 model 层盲目逐
  indexer 层传。
- `len(comps) > 4` 抛错（L694-697）：含 indexer 的完整 K/V/index_k/index_k_scale 恰好 4 个
  component，在 FP8 DSA（无 V）+ indexer 下为 3 个（K + index_k + scale）。

### 6.8 Memfabric / AscendC 融合路径（本次强制，不再与 layer_first 互斥）

`_transfer_ascendc_sparse_copy` 的底层 kernel 直接解引用 host pool 指针（`get_dva`），
**host pool 必须 Memfabric-mapped**（`SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric`）。因此

> **本方案反转旧决策：通用 MHA/MLA pool 的 layer_first 一律强制 memfabric + AscendC，非
> memfabric host 内存不再支持这两类 pool 的 layer_first；DSv4 专用池与 Kimi SSM 状态池除外（§4.1）。**

落地方式——**在 host pool 分配阶段强制**（`MHATokenToKVPoolHost` / `MLATokenToKVPoolHost` 的
`init_kv_buffer`，见 §5.2(1) 与 mla.py 现有 memfabric 分支），fail-fast 仍在启动期、早于首次
L2 传输。**为何不在 `set_default_server_args` 配置阶段强制**：该阶段尚无任何 host pool 实例，
无法区分「Kimi 的 MLA KV 池（须强制 memfabric）」与「Kimi 的 SSM 状态池 `MambaPoolHost`（须豁免）」；
而 `uses_ssm_state` 是模型级布尔（`kv_cache_builder.py` L159，Kimi 同时命中 MLA 与 SSM 两套池），
若用它整模型短路，会把 Kimi 的 MLA KV 池也一并豁免，与需求文档 §7 第 3 条「通用 MLA 池强制
memfabric」冲突。故强制判定**落在能拿到 pool 具体 class 的分配点**：

```python
# mha.py / mla.py 的 init_kv_buffer 内（此时 self 已是具体 pool 实例）
if _is_npu and self.layout == "layer_first":      # 仅 NPU；GPU 的 layer_first 沿用原 pinned 路径
    if not memfabric_host_memory_enabled():
        raise ValueError(
            "layer_first layout requires SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric "
            "(generic MHA/MLA host pool AscendC path)"
        )
    alloc_func = alloc_with_memfabric   # 见 §5.2(1) 的签名差异说明
```

（`memfabric_host_memory_enabled()` 从 `pool_host/npu_memfabric.py` 导入，与
`ascendc_io_enabled()` 同源。）

- **强制范围按 pool class 判定（可执行）**：`isinstance(pool, (MHATokenToKVPoolHost,
  MLATokenToKVPoolHost))` 且 `layout == "layer_first"` → 强制 memfabric；`MambaPoolHost`
  （`pool_host/mamba.py`）与 `DeepSeekV4*HostPool`（`memory_pool_host.py`，默认 layer_first）
  是两个独立 class，各自 `init_kv_buffer` 不进入上面的 memfabric 分支，**天然豁免**——无需在配置
  阶段用 `is_deepseek_v4` / `uses_ssm_state` 做模型级短路（后者尤其错误，见上）。
- 默认（非 memfabric）host pool 走 `transfer_kv_dim_exchange`（page_first 系），**不受影响**；
  layer_first 子分支内不再单独判 `ascendc_io_enabled()`（已由 pool 分配的 memfabric 校验保证）。
- 这一范围限制须同步写进需求/验收说明：`--hicache-mem-layout layer_first` 在通用 MHA/MLA pool
  上强制 `SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric`（需求文档 §7 第 3 条）。
- 底层 `host_layout_mode=1` 的能力缺口属外部 `memfabric_hybrid`，非 sglang / sgl-kernel-npu
  可独立补齐（§2.4）——这是本方案对外部依赖的唯一前置条件。

---

## 7. 布局开关与归一化

### 7.1 目标语义

**用户显式指定 `--hicache-mem-layout layer_first` 时，NPU 尊重该值（不覆盖）；未显式指定时
维持现有默认（`page_first_kv_split` / `page_first_direct`）。**

参考 GPU 实现：GPU 侧不强制覆盖 layout —— `layer_first` 是合法值（`mha.py`/`mla.py` 均有
layer_first 分支），仅 NPU 的 `hardware_backend/npu/utils.py` 无条件把 layout 投影覆盖为
page_first 系，这是「NPU 无 layer_first」的根源。

### 7.2 改动 `hardware_backend/npu/utils.py`（L173-194）

当前：

```python
if cfg.enable_hierarchical_cache or cfg.disaggregation_decode_enable_offload_kvcache:
    declare_resolution(args, "set_default_server_args", hicache_io_backend="kernel_ascend")
    if use_mla_backend(args):
        declare_resolution(..., hicache_mem_layout="page_first_kv_split")
    else:
        declare_resolution(..., hicache_mem_layout="page_first_direct")
```

改为：**仅当 raw 值不是 `layer_first` 时才投影默认**；raw 值为 `layer_first` 则跳过覆盖、放行。
memfabric 的强制**不在此函数做**——本函数运行时尚未构建 host pool，无法区分通用 MHA/MLA pool 与
`MambaPoolHost` / DSv4 专用池（Kimi 同时含前两类），故交给 host pool 分配阶段按 pool class 判定
（§6.8）。

**判定方式的准确边界（不声称能识别「任意显式输入」）**：
- `server_args._raw_input`（`arg_groups/pipeline.py` L27-32）保存的是「所有字段的当前值快照」，
  **含默认值**，因此不能区分「用户显式传 `page_first`」与「默认 `page_first`」。
- `declare_resolution` 写入的 stash 在 `resolution_result`（`overrides.py` L178-190）里
  **优先级高于 `_raw_input`**（先查 stash 声明，后回退 raw input）。
- 所以本次只做一种判别：`cfg.hicache_mem_layout == "layer_first"`（默认值是 `page_first`，
  该值为 `layer_first` 即可反推为用户显式指定），即：

```python
if cfg.hicache_mem_layout != "layer_first":
    declare_resolution(
        args,
        "set_default_server_args",
        hicache_mem_layout=(
            "page_first_kv_split" if use_mla_backend(args) else "page_first_direct"
        ),
    )
else:
    # layer_first 放行；memfabric 强制改由 host pool 分配阶段按 pool class 判定（§6.8）
    pass
```

文档仅承诺「显式 `layer_first` 会被放行」，不宣称能识别其它显式 CLI 值。

**强制 memfabric 的落地方式（已统一为「显式校验」，与 §5.2(1)/§6.8 一致）**：通用 MHA/MLA
pool 的 `init_kv_buffer` 在 `_is_npu and layout == "layer_first"` 时**要求 memfabric 已开启，
否则抛 `ValueError`**（不自动设置环境变量）。理由：

1. **确定性**：不静默改写全局状态（`os.environ` 是进程级副作用，且 `memfabric_host_memory_enabled()`
   在 host pool 构造前就可能已被其它路径/专用池读取，中途置位会引入时序依赖）；
2. **与需求一致**：需求文档 §7 第 3 条明言「非 memfabric 不再作为这两类 pool 的 layer_first 传输
   后端」——即视为用户显式要求 memfabric，缺失即配置错误，报错信息应清晰可定位（指向
   `SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric`）。

不做「自动置位」选项，避免与上述「显式校验」形成两种不一致的启动行为。

> 此前的「layer_first + memfabric 互斥 → 启动报错」逻辑（旧 §7.2/§6.8）**撤销**，不再写回。

### 7.3 改动 `arg_groups/hicache_hook.py`（L180-204）

现状 `resolve_storage_layout_compatibility` 的真实行为（对照 L182-204）：

- `storage_backend` 不是 `mooncake`/`npu_memcache`（含 `None`，纯 L1/L2）→ 直接 return，不拦截；
- `direct` → 改写 `page_first_direct`；`kernel` → 改写 `page_first`；
- `kernel_ascend` → `new_layout = cfg.hicache_mem_layout`（**保持 layer_first**），仅 declare 一个
  同值声明并打一条误导性 warning（「switching to layer_first」实际未切换）。

即：现有代码只对 `mooncake`/`npu_memcache` 两个后端做 branch 处理，且对 `kernel_ascend` 静默
放行 layer_first；`file` / `nixl` / `hf3fs` / `aibrix` / `tensorcast` / `seaweedfs` 等其余 6 个
backend（`memory.py` L148-158）会一路通过配置，直到运行时 `get_data_page`/`set_from_flat_data_page`
才失败，构成「启动成功、运行时才炸」的隐患。

本次设计：把「layer_first + 非空 L3」的拒绝**前移到配置解析阶段**、覆盖**所有**非空
`hicache_storage_backend`（不只是 mooncake/npu_memcache），但**只限定到本期无此能力的 NPU /
`kernel_ascend` 路径**。行为统一为**抛错（raise）**——因为用户显式指定了 layer_first，不存在
「改写布局后仍尊重用户意图」的合法目标布局，改写反而静默改变语义；与 §1.2「本次不含 L3」对齐。

> `resolve_storage_layout_compatibility` 是**跨设备通用路径**：GPU 的 `direct`/`kernel` 后端对
> layer_first + L3 有各自的既有行为（`direct`→`page_first_direct`、`kernel`→`page_first` 改写），
> 其它 backend 在 GPU 下也不应被 NPU 的限制误伤。因此拦截条件必须同时判 `hicache_io_backend ==
> "kernel_ascend"`（NPU 专用 IO 后端），不得无条件拦截所有 layer_first + L3。

即把 `resolve_storage_layout_compatibility` 改为：

```python
def resolve_storage_layout_compatibility(server_args: Any):
    cfg = resolving_view(server_args)
    if (
        cfg.hicache_mem_layout == "layer_first"
        and cfg.hicache_storage_backend is not None
        and cfg.hicache_io_backend == "kernel_ascend"   # 仅 NPU 本期不支持 layer_first + L3
    ):
        raise ValueError(
            "layer_first layout does not support any HiCache storage backend on "
            "NPU in this release (got --hicache-storage-backend "
            f"{cfg.hicache_storage_backend!r}); use a page_first layout for L3."
        )
    # ... 其余（direct/kernel 改写、GPU 放行）逻辑不变 ...
```

- 纯 L1/L2（`hicache_storage_backend is None`）继续放行 layer_first；
- NPU + `kernel_ascend` + 任意非空 L3 → 启动即抛错；
- GPU 等其它设备的 layer_first + L3 保持既有行为不变；
- `get_data_page` / `set_from_flat_data_page`（§6.4.1）里的 `raise NotImplementedError` 降级为
  防御性兜底（正常启动到不了该分支）。

---

## 8. 边界情况与校验

1. **device/host packed 与 5-D view**：AscendC component 需要 5-D（layer-first）
   `[layer, page, page_size, 1, width]` 视图。MHA 的 host `[layer, size, H, D]` 须能整整除成
   `[layer, page_num, page_size, H, D]`（`size == page_num*page_size`）；若 device buffer 是
   strided（如 unified pool），需先 `_require_packed_device_rows` 或拒绝。MHA/MLA 标准 pool 为 packed。
2. **空层 / layer shard**：draft pool、层分片（`dsa`/`dcp`）下 `layer_num` 动态，需用
   `_owned_device_layer_ids()` / `_device_owned_layer_range()`（`base.py`）而非硬编码；
   层分片与 layer_first 组合在本期明确拒绝（§6.4，与 `page_first_kv_split` 一致）。
3. **indices 常驻 device**：AscendC 把 `host_indices`/`device_indices` `to_device_no_sync` 上传、
   `record_stream` 保活（L580-588），不转 CPU。`host_indices.numel() % page_size == 0` 由
   `num_pages = host_indices.numel() // self.page_size` 隐含约束。
4. **空 V（FP8 DSA）**：`v_buffer` 用 **5-D 零元素空张量**（最后维 0，`numel()==0`，非 `None`，
   见 §6.4.1/§6.6），`_build_ascendc_component_meta` 靠 `numel()>0` 跳过 V component，无
   `transfer_kv_dim_exchange` 的 `device_v.sizes()[4]` 越界问题。
5. **component 上限**：`len(comps) > 4` 抛错（L694-697）；layer_first 完整 K/V/index_k/index_k_scale
   恰为 4，FP8 DSA（无 V）+ indexer 为 3，均 ≤ 4。
6. **host DVA 派生**：Memfabric-mapped host 内存是 `get_dva` 的前提（§2.1/§6.8）；非 memfabric
   host 内存下 `get_dva` 会失败，已由通用 MHA/MLA pool 分配点按 pool class 强制 memfabric 排除
   （§6.8）。

---

## 9. 风险与待确认

- [x] layer_first 的 layout 命名：**已定——复用 `layer_first` 名，不新增 `layer_first_kv_split`**
      枚举项（与 GPU 对齐：GPU 只有一个 layer_first 名；K/V 是否分离是 NPU 设备池 native 属性，
      由 `isinstance(buffer_info.buffers, MLASplitBuffers)` 决定，不泄露进 layout 枚举）。
- [x] **外部 kernel 依赖**：`host_layout_mode=1` 已由外部 `memfabric_hybrid.offload.kv_exchange_copy`
      实现（`LayerFirstWalker`，§2.4），**前置阻塞消除**，sglang / sgl-kernel-npu 侧无需改动外部依赖。
- [x] host stride 语义已确认：`hostPagePitch`（o+4）/`hostLayerPitch`（o+5）是独立语义字段、地址
      公式 mode 无关（§2.4）；layer_first 下填 `hostLayerPitch = stride(0)`、`hostPagePitch =
      stride(1)`（§6.4(C)），无「Python 侧填对、kernel 侧读反」风险（kernel 按字段名取，不按 mode 推断）。
- [x] indexer 附带缓存是否纳入本次范围：**已定——纳入**，`_size_per_token` 的 indexer 计入条件
      （§6.4.1）与 AscendC index component（§6.7）已一并落定。
- [ ] FP8 DSA 打包段的 host `k_buffer` 宽度必须用 `native.k_width`（打包宽），不能误用
      `kv_lora_rank`；需单测覆盖 FP8 DSA 场景（含 5-D 零元素空 V 的「只传 K」验证，§6.6）。
- [x] 短序列 launch 开销：**已定——由方案本身解决**。旧方案逐层 `transfer_kv_dim_exchange`（N=1）
      的 launch 开销放大 `layer_num` 倍（短序列最易回退）；本方案 AscendC 按 buffer_info 分段、
      每个 info 一次 compact meta 下发其全部层、不 sync load stream，理论上消除该开销。仍需性能验证
      覆盖短序列确认（§10.1），但不再
      是「无法分组」的被动风险，而是「验证方案有效性」的主动项。

---

## 10. 落地顺序

1. ~~确认外部 `kv_exchange_copy` 支持 `host_layout_mode=1`~~（**已确认**：`LayerFirstWalker` 已实现，
   §2.4，无外部改动）。
2. MLA：`_ascendc_comp_meta` / `_build_ascendc_component_meta` / `_transfer_ascendc_sparse_copy`
   支持 `host_layout_mode=1`（§6.4(C)(D)(E)），`init_kv_buffer` layer_first 分开分配（§6.4 表 +
   §6.4.1 容量闭合）。
3. MLA layer_first 分支接入 D2H（全量）/ H2D（per-layer）+ 单元测试闭环（普通 BF16，含 target/draft）。
4. MHA 补充 `_transfer_ascendc_sparse_copy`（§5.2，component stride 重写）+ 分支接入 + 单元测试。
5. MLA FP8 DSA（§6.6，空 V component 跳过）+ indexer（§6.7，第 3/4 个 component）+ 单测。
6. 布局开关 + 强制 memfabric + L3 拒绝（§7）+ 端到端精度测试（含 Kimi-K3 SSM state 恢复回归）。
7. 性能对比（长/短序列）+ CI 注册（对应需求文档 §11）。

### 10.1 最小必跑矩阵与判定规则（需求要求 10-30 前做性能验证）

**必跑模型**（按需求 5 模型逐一归位）：

| 模型 | 架构判定 | 传输路径 | layer_first 归属 |
|------|----------|----------|------------------|
| GLM-5.x（MHA/GQA） | 标准 MHA | 通用 `mha.py` | 本次 §5 |
| GLM-5.3-Flash | MHA | 通用 `mha.py` | 本次 §5 |
| Kimi-K3 | MLA + 线性注意力（`kimi_linear` / `KimiK3`，`uses_ssm_state` 命中） | 双路径：MLA KV 走通用 `mla.py`（普通 BF16，无 indexer）；SSM state 走 `MambaPoolHost`（`pool_host/mamba.py`） | 本次 §6（MLA）+ 复用/回归 `mamba.py` |
| DeepSeek-V4-Flash | `is_deepseek_v4`（`DeepseekV4ForCausalLM`，model_type=`deepseek_v4`） | DSv4 专用 `DeepSeekV4PagedHostPool` | **已支持（§4），本次仅回归** |
| DeepSeek-V4.1-Flash | `is_deepseek_v4`（复用 V4 架构名）+ `model_type=deepseek_v41` | 同上 DSv4 专用池 | 同上 |

> DeepSeek-V4-Flash / V4.1-Flash 走 `is_deepseek_v4`（`DeepseekV4ForCausalLM` 系列）而非
> `is_deepseek_dsa`（DeepSeek V3.2 / GLM-5.2 等 DSA 族）；其 host 侧用专用
> `DeepSeekV4PagedHostPool`（已实现 layer_first + kernel_ascend，见 §4），**但该专用池走的是
> `transfer_kv_dim_exchange` 逐层路径，不在本次「强制 AscendC」的改动范围内**——本方案只改通用
> `mha.py` / `mla.py` 的 layer_first；DSv4 专用池是否也切到 AscendC 属二期，本次仅回归确认不被
> 本次改动影响。
>
> **回归方式**：这两个模型的 `DeepSeekV4PagedHostPool` 分支必须用 V4-Flash / V4.1-Flash 实际模型
> 配置单独跑通，**不能用 MLA（Kimi-K3 / GLM-5.x）测试代替**——三条路径对应的 pool/分支不同
> （DSv4 专用池 vs 通用 `mla.py`），用 MLA 过测无法覆盖 DSv4 专用池的回归。

**Kimi-K3 双路径与 SSM state 覆盖（P1）**：Kimi-K3 **不是纯 MLA**。`kv_cache_builder.py::uses_ssm_state`
（L157-167）通过 `kimi_linear_config(model_config) is not None`（`KimiK3Config` 含 `KimiLinearConfig`）
判定 Kimi-K3 同时保有**线性注意力（Delta）的 recurrent/conv 状态**，这部分 L2 host pool 由
`MambaPoolHost`（`pool_host/mamba.py`）承载，而 **不是** `mla.py`——只看 MLA K/V 搬运无法完整证明
Kimi-K3 的验收要求。

- `MambaPoolHost` 已支持 layer_first + kernel_ascend：`layout` 默认 `layer_first`（mamba.py L92）；
  D2H `backup_from_device_all_layer` 的 layer_first 分支逐层 `_copy_tensor`（L812-828），H2D
  `load_to_device_per_layer` 逐层 `_copy_tensor` + 首层 `_transfer_slot_states`（L697-701），其中
  `_copy_tensor` 的 `kernel_ascend` 分支（L487-493）已做逐层复制。故 **SSM state 侧为「复用 +
  回归验证」路径，不在本次新增改动范围内**（与 §4 DSv4 专用池同一性质）。
- **测试补充（必跑）**：除 MLA K/V 前缀命中外，须增加一次**「L2 前缀命中后恢复线性注意力状态」**
  端到端测试——用 Kimi-K3 实际模型跑「前缀命中 → H2D 恢复 temporal/conv state（+ side state）→
  续写解码」，校验续写 logits/精度与无缓存基准对齐。仅跑 MLA K/V 命中无法覆盖 SSM state 恢复这条
  `MambaPoolHost` 分支。

**通用 `mla.py` 的 FP8 DSA / DSA indexer（§6.6 / §6.7）** 属于 `is_deepseek_dsa` 族
（DeepSeek V3.2、GLM-5.2 等），**不在本次需求 5 模型清单内**，作为通用 MLA pool 的完整性支持
列为可选扩展点，不纳入最低必跑。

**性能指标与判定**：

- 同一 IO 后端配置下 A/B 对比（**双方都 memfabric 开**，因为 layer_first 强制 AscendC）：`layer_first`
  走 `_transfer_ascendc_sparse_copy` vs `page_first_kv_split`（MLA）/ `page_first_direct`（MHA）。
  若需对拍非 memfabric 的 SDMA 路径，另行单列，不计入本验收基准。
- 指标：端到端吞吐（token/s）、TTFT、TPOT；前缀命中率固定。
- 覆盖两类输入长度：**长序列**（prefill 数千 token，验证数据拷贝主导时基本持平）与
  **短序列 / 单 token 续写**（验证 AscendC 是否确实消除逐层 launch 开销，§9 风险项）。
- 判定规则：两类场景下 `layer_first` 指标均 **不低于 page_first**（吞吐 ≥ 1.0x、时延 ≤ 1.0x，
  波动阈值 ±5%）。若短序列不达标，须回到 §9 的 AscendC host stride / 元数据下发开销优化项，
  不得以长序列通过掩盖短序列回退。

**功能验收**：`--hicache-mem-layout layer_first` + `kernel_ascend`（不含 L3、强制 memfabric）正常
启动、前缀命中正确、精度回归通过（复用需求文档 §11 的 GSM8K few-shot 方式）。