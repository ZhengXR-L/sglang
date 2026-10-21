# 功能设计：NPU HiCache L1→L2 传输路径支持 layer_first 布局

> 关联需求：`hicache_layer_first_npu_requirements.md`（需求记录，独立文档）
> 本文档独立成篇，专注「如何改动实现 layer_first」的功能设计，不与需求文档合并。
> 日期：2026-10-20 ｜ 状态：设计稿 v2（已整合 MLA 布局澄清 + FP8 DSA 特殊路径 + 布局开关语义）

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
- `page_head` / `page_unified` 等其它布局在 NPU 下的补全。

---

## 2. 设计基石：`transfer_kv_dim_exchange` 的传输契约

这是整个设计的核心约束，必须先从底层算子理解清楚。

### 2.1 算子签名与硬性约束

Python 封装 `sgl_kernel_npu/sgl_kernel_npu/kvcacheio.py::transfer_kv_dim_exchange`，
C++ 实现 `csrc/transfer_kv_dim_exchange/op_host/transfer_kv_dim_exchange.cpp`。硬性校验：

```
device_k.dim() == host_k.dim() == 5
device_k.sizes()[0] == host_k.sizes()[1]      # 连续层数 N 相等
device_k.sizes()[2] == host_k.sizes()[2] == page_size
```

一个算子调用处理 **N 个连续层**：

| 张量 | 形状 | 布局语义 |
|------|------|----------|
| `device_k` | `[N, device_pages, page_size, heads, head_dim]` | **layer-first**（层在第 0 维） |
| `host_k`   | `[host_pages, N, page_size, heads, head_dim]` | **page-first**（层在第 1 维） |

K / V 各自独立校验与拷贝；`index_k` / `index_k_scale` 由 Python wrapper **拆成额外的
`transfer_kv_dim_exchange` 调用**处理（`kvcacheio.py` L94-122），并不在 C++ 主函数签名内。

### 2.2 底层 2D 拷贝语义（为什么叫 dim exchange）

C++ 用 `aclrtMemcpy2dAsync` 做 2D strided copy，逐 page 循环（L76-104）：

```
num_pages = len(device_indices) / page_size
for page i:
    device_page = device_indices[i*P] / page_size
    host_page   = host_indices[i*P]   / page_size
    device_k_ptr = device_k[0][device_page]     # 第 0 层、该 page 首址
    host_k_ptr   = host_k[host_page][0]         # 该 host page、第 0 层首址

    k_device_pitch = device_pages * P * heads * head_dim * itemsize   # 一整层字节数
    k_host_pitch   = P * heads * head_dim * itemsize                  # 单个 page 字节数
    k_width        = P * heads * head_dim * itemsize                  # 每行宽度
    height         = N                                                # 层数 = 行数

    aclrtMemcpy2dAsync(host, k_host_pitch, device, k_device_pitch, k_width, height, ...)
```

它把「N 个连续层」当作 N 行，每行拷一个 page 的 `k_width` 字节：
- 源（device）：行间间隔 `k_device_pitch`（= 整层），因为 device 是 layer-first，层间连续；
- 目标（host）：行间间隔 `k_host_pitch`（= 单 page），因为 host 是 page-first，一 page 内 N 层紧密排布。

### 2.3 由此推出的两条关键约束

**(A) host 页内必须连续**：每行的 `k_width = P * heads * head_dim * itemsize` 意味着 host 侧
同一个 page 内，「每层一块连续 `P * head_dim` 字节、token 间 stride == head_dim」。
任何使页内 token 间 stride ≠ head_dim 的布局都不能直接喂给算子。

**(B) layer_first host 只能逐层（N=1）**：通用 host pool 的 layer_first 是「**一个连续张量的逐层
视图**」（`kv_buffer[l]`，层间 stride == 整层大小），并非不连续独立 buffer。真正限制是：该算子的
host 跨层 pitch 被固定为「单 page 宽」（`k_host_pitch = P*heads*head_dim*itemsize`，见 §2.2），
而 layer_first 实际跨层 stride 是「整层大小」，两者无法放进同一个 5-D 视图参与多层（`N>1`）复制。
因此唯一做法是每层单独构造视图、每次算子只处理 `N=1` 层（此时 `height=1`，跨层 pitch 不参与）。
这正是 DeepSeek V4 专用 pool 已落地的方式（§4）。

---

## 3. 总体方案

对通用 MHA / MLA host pool，在 `backup_from_device_all_layer`（D2H）和
`load_to_device_per_layer`（H2D）的 `kernel_ascend` 分支新增 `layout == "layer_first"`
子分支，实现为**逐层五维视图 + `transfer_kv_dim_exchange`**（每次 `N=1`）：

```
# D2H（backup）：本方法无 layer_id 入参，内部遍历所有层
for l in range(layer_num):
    dev_view  = device层l.view(...).unsqueeze(0)       # [1, dev_pages, P, H, D]
    host_view = host层l.buffer.view(...).unsqueeze(1)  # [pages, 1, P, H, D]
    transfer_kv_dim_exchange(..., direction=TransferDirection.D2H)

# H2D（load）：只搬传入的 layer_id（单层），不再内部循环所有层
dev_view  = device层(layer_id).view(...).unsqueeze(0)
host_view = host层(layer_id).buffer.view(...).unsqueeze(1)
transfer_kv_dim_exchange(..., direction=TransferDirection.H2D)
```

每次调用满足契约（`N=1`：`dim0 == dim1 == 1`，`dim2 == P`）。

**为什么 D2H / H2D 必须分开写**：`l2_transfer.py::submit_host_to_device`（L145-183）在引擎层
`for layer_id in range(transfer_layer_id_max)` 逐层调用 `load_to_device_per_layer_physical`，
并在每层后触发 `on_layer_done(layer_id)`。因此 H2D 回调是「单层」语义——若它内部再循环所有层，
会重复搬运约 `层数²` 次，且「本层完成」事件不再对应本层。反过来 `backup_from_device_all_layer`
（D2H）无 `layer_id` 入参，由本方法内部遍历所有层。两者必须分别建模。

MHA 的 K/V 本来就是两块独立 buffer，此模式可直接套用（§5）；MLA 因 device 侧 K/V 分离、
host 侧有「合并 vs 分离」的布局选择，需专门展开（§6）。

---

## 4. 参照实现：DeepSeek V4 专用 pool（已支持 layer_first）

`memory_pool_host.py` 的 `DeepSeekV4PagedHostPool` / `DeepSeekV4StateHostPool` 已实现
layer_first + kernel_ascend，是本次改动的直接范本：

- `_host_page_view(l)`（L351-372）：把 layer_first 的 `kv_buffer[l]` 转成
  `[num_host_pages, 1, P, 1, dim]` 五维视图。
- `backup_from_device_all_layer` 的 kernel_ascend 分支（L528-554）：逐层循环，device 侧
  `device_buffers[l].unsqueeze(0)` → `[1, dev_pages, P, 1, dim]`，配合 `_host_page_view(l)`
  调算子，`direction=D2H`。
- `load_to_device_per_layer` 的 kernel_ascend 分支（L625-647）：同上，`direction=H2D`。

**可复用点**：把「layer_first → `[pages, 1, P, H, D]` 五维视图」抽象成公共方法
（放到 `pool_host/base.py` 或新 helper），MHA/MLA 复用，避免三处重复。

---

## 5. MHA 详细设计（先做，最简单）

### 5.1 现状 buffer 形状

`mha.py` 的 `init_kv_buffer` 已支持 layer_first：

```python
# layer_first: dims = (2, layer_num, size, head_num, head_dim)
# buffer[0] -> self.k_buffer : [layer_num, size, head_num, head_dim]
# buffer[1] -> self.v_buffer : [layer_num, size, head_num, head_dim]
```

- host：`k_buffer[l]` = `[size, head_num, head_dim]`，其中 `size == page_num * page_size`；
- device：NPU 分支通过 `info.transfer_buffers()`（`device_pool_info.MHABufferInfo`）取到
  `[N, device_tokens, head_num, head_dim]` / `[N, device_tokens, head_num, v_head_dim]` 的多层
  连续张量（第 0 维是层，layer-first，K/V 天然分离）；`self._buffer_infos` 是一串
  `MHABufferInfo`（主 pool + 可能的 draft buffer infos），需要做「全局层号 ↔ 本地层号」映射。

> `page_first_direct` 现状（mha.py L649-661）已验证 `transfer_kv_dim_exchange` 的 host 用法：
> `self.k_buffer[:, layer_start:layer_end]` 是 `[page_num, layer_num, P, H, D]`。

### 5.2 设计改动

在 `backup_from_device_all_layer` / `load_to_device_per_layer` 的 kernel_ascend 分支中分别新增
`layer_first` 子分支（§3 伪代码的 MHA 实例化）。**D2H 内部遍历所有层；H2D 只搬传入的 `layer_id`。**

**MHA 层号映射**：device 侧 `_buffer_infos` 是「主 pool + 可能的 draft buffer infos」序列，各自第 0 维
是本地层编号；host `k_buffer/v_buffer` 下标是「全局层号」。沿用现有 `_npu_transfer_buffers()`
（mha.py L520-531）的 `layer_start` 累加：第 `i` 个 `info` 内 `local_id` → `global_id = layer_start +
local_id`。MHA 无 rank-replication、无层分片，`host_layer_id == layer_id`（mha.py L390）。

**(A) D2H（`backup_from_device_all_layer`）——内部遍历所有层：**

```python
elif io_backend == "kernel_ascend":
    if self.layout == "page_first_direct":
        # ... 现有逻辑不变（用 _npu_transfer_buffers() 一次性多层）...
    elif self.layout == "layer_first":
        layer_start = 0
        for info in self._buffer_infos:                 # 主 pool + draft infos
            device_k, device_v = info.transfer_buffers() # 各自 [N, tokens, H, D(hd)]
            for local_id in range(device_k.shape[0]):
                global_id = layer_start + local_id       # 全局层号 → host k_buffer 下标
                device_pages = device_k.shape[1] // self.page_size
                dev_k = device_k[local_id].view(
                    device_pages, self.page_size, self.head_num, self.head_dim
                ).unsqueeze(0)                           # [1, dev_pages, P, H, D]
                dev_v = device_v[local_id].view(
                    device_pages, self.page_size, self.head_num, self.v_head_dim
                ).unsqueeze(0)
                host_k = self.k_buffer[global_id].view(
                    self.page_num, self.page_size, self.head_num, self.head_dim
                ).unsqueeze(1)                           # [pages, 1, P, H, D]
                host_v = self.v_buffer[global_id].view(
                    self.page_num, self.page_size, self.head_num, self.v_head_dim
                ).unsqueeze(1)
                transfer_kv_dim_exchange(
                    device_indices=device_indices, host_indices=host_indices,
                    device_k=dev_k, host_k=host_k, device_v=dev_v, host_v=host_v,
                    page_size=self.page_size, direction=TransferDirection.D2H)
            layer_start += device_k.shape[0]
    else:
        raise ValueError(...)
```

**(B) H2D（`load_to_device_per_layer`）——只搬传入的 `layer_id`，不再循环：**

```python
elif io_backend == "kernel_ascend":
    if self.layout == "page_first_direct":
        # ... 现有逻辑不变（`if host_layer_id == 0` 才一次性传所有层）...
    elif self.layout == "layer_first":
        info, local_id = self._buffer_info_and_local_for_layer(layer_id)  # target/draft 反查
        global_id = layer_id                     # MHA: host 下标 == layer_id
        device_k, device_v = info.transfer_buffers()  # 该 info 的多层张量
        device_pages = device_k.shape[1] // self.page_size
        dev_k = device_k[local_id].view(
            device_pages, self.page_size, self.head_num, self.head_dim
        ).unsqueeze(0)
        dev_v = device_v[local_id].view(
            device_pages, self.page_size, self.head_num, self.v_head_dim
        ).unsqueeze(0)
        host_k = self.k_buffer[global_id].view(
            self.page_num, self.page_size, self.head_num, self.head_dim
        ).unsqueeze(1)
        host_v = self.v_buffer[global_id].view(
            self.page_num, self.page_size, self.head_num, self.v_head_dim
        ).unsqueeze(1)
        transfer_kv_dim_exchange(
            device_indices=device_indices, host_indices=host_indices,
            device_k=dev_k, host_k=host_k, device_v=dev_v, host_v=host_v,
            page_size=self.page_size, direction=TransferDirection.H2D)
    else:
        raise ValueError(...)
```

`_buffer_info_and_local_for_layer(layer_id)` 复用 `layer_start` 累加逻辑反查 `layer_id` 落在哪个
`info`（target 主 pool vs draft），返回 `(info, local_id)`；该映射与 §10 第 1 步的视图 helper 一起
沉淀为公共方法，避免 D2H / H2D 两处重复。

### 5.3 关键点
- MHA 的 K/V 本就是两块独立 buffer，页内 token 连续（stride == head_dim），满足 §2.3(A)，
  一次调用即可同时传 K+V。
- `head_num` 在 device 与 host 侧一致（MHA 无 rank-replication），可直接对齐。
- indices 直接使用 token 级索引（由 `cache_controller.move_indices` 在 kernel_ascend 旧路径下
  转 CPU）；算子内部 `/page_size` 求 page index，无需 DSv4 那样的 native page 扩展。

---

## 6. MLA 详细设计（最复杂，含 indexer 与 FP8 DSA）

### 6.1 逻辑顺序（先定死）

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

但 §2.3(A) 指出，`transfer_kv_dim_exchange` 要求 host 页内 token 间 stride == head_dim。
合并记录里 token 间 stride == `kv_cache_dim`（K 紧跟 V），无法用单一 head_dim 表达，
必须拆成 K 段（head_dim=`kv_lora_rank`）与 V 段（head_dim=`qk_rope_head_dim`）两块
**各自连续**的 buffer，否则页内 pitch 错位。

**结论**：NPU layer_first 的 L1↔L2 传输必须采用「分开分配」两块 buffer（与现有
`page_first_kv_split` 同构，仅 page 维与 layer 维互换）；「合并记录」只保留给 GPU 语义 /
L3 存储字节序，不作为 L1↔L2 的直接 memcpy 布局。

### 6.4 方案：layer_first 分开分配（推荐）

新增 host `init_kv_buffer` 的 layer_first 分配逻辑（可复用 `page_first_kv_split` 的分段维度，
仅把 `[page_num, layer_num, ...]` 换成 `[layer_num, page_num, ...]`）：

| host buffer | 形状 | 宽度 |
|-------------|------|------|
| `k_buffer` | `[layer_num, page_num, page_size, 1, k_width]` | `k_width = kv_lora_rank`（BF16）/ `kv_cache_dim`（FP8 DSA） |
| `v_buffer` | `[layer_num, page_num, page_size, 1, qk_rope_head_dim]`（普通 BF16）；FP8 DSA 为 **5-D 零元素空张量** `[layer_num, page_num, page_size, 1, 0]` | 普通 BF16 参与传输；FP8 DSA `v_width==0`（`numel()==0`，不占容量，但仍以 5-D 空张量传 V） |
| `index_k_buffer` | `[num_indexer_layers, page_num, page_size, 1, index_width]` | 部分层 |
| `index_k_scale_buffer` | `[num_indexer_layers, page_num, page_size, 1, 1]`（FP32，5-D） | 仅 FP8 DSA + indexer |

**D2H 内部遍历所有层；H2D 只搬传入的 `layer_id`（单层）。** 两者共用同一个「单层搬运转译」
helper（§10 第 1 步沉淀），只在驱动方式上不同；V **始终传 5-D 张量，不传 `None`**（§2.1 契约 +
算子读取 `device_v.sizes()[4]`，见下方「V 参数必须为 5-D」）。

**(A) D2H（`backup_from_device_all_layer`）——内部遍历 target + draft 各层：**

非层分片路径进入 L994 的 `kernel_ascend` 分支，新增 `layer_first` 子分支：

```python
elif io_backend == "kernel_ascend":
    if self.layout == "page_first_kv_split":
        # ... 现有逻辑不变（一次性传所有层） ...
    elif self.layout == "layer_first":
        # 沿用现有逐层驱动（mla.py L892-905）：target 再 draft，避免跨 _native_infos 索引越界
        for target_id in self._target_layer_indices:
            self._transfer_one_layer(target_id, host_indices, device_indices,
                                     is_draft=False, direction=TransferDirection.D2H)
        for draft_i in range(self._draft_layer_num):
            self._transfer_one_layer(self._device_layer_num + draft_i, host_indices, device_indices,
                                     is_draft=True, direction=TransferDirection.D2H)
    else:
        raise ValueError(...)
```

> **层分片（`_layer_shard_enabled`）不在本次范围**：与 `page_first` 现状一致。`backup_from_device_all_layer`
> 在进入 `kernel_ascend` 分支之前，仍会先判断 `_layer_shard_enabled` 并逐层调用
> `_backup_from_device_per_layer`；该方法保留 `if io_backend not in ("kernel", "direct"): raise ValueError`，
> 即 layer_first + kernel_ascend + 层分片组合仍会被明确拒绝（与 `page_first_kv_split` 行为一致，本次不实现）。

**(B) H2D（`load_to_device_per_layer`）——只搬传入的 `layer_id`：**

```python
elif io_backend == "kernel_ascend":
    if self.layout == "page_first_kv_split":
        # ... 现有逻辑不变（`if device_layer_id != 0: return`，一次性传所有层） ...
    elif self.layout == "layer_first":
        self._transfer_one_layer(layer_id, host_indices, device_indices,
                                 is_draft=is_draft, direction=TransferDirection.H2D)
    else:
        raise ValueError(...)
```

**单层转译 helper（D2H/H2D 共用，含 K/V + indexer）：**

```python
def _transfer_one_layer(self, layer_id, host_indices, device_indices, *, is_draft, direction):
    info_index = layer_id - self._device_layer_num + 1 if is_draft else 0
    native = self._native_infos[info_index].native_buffers()      # target=0 / draft=[1..]
    device_layer_id = 0 if is_draft else layer_id
    host_layer_id = self._host_layer_by_controller[layer_id]      # draft → host 下标
    # K / V
    dev_k = native.k[device_layer_id].unsqueeze(0)                # [1, dev_pages, P, 1, k_width]
    host_k = self.k_buffer[host_layer_id].unsqueeze(1)            # [pages, 1, P, 1, k_width]
    dev_v = native.v[device_layer_id].unsqueeze(0)                # [1, dev_pages, P, 1, v_width]
    host_v = self.v_buffer[host_layer_id].unsqueeze(1)            # [pages, 1, P, 1, v_width]
    # indexer（若有）：device 层 → index slot 映射（§6.7，复用 _indexer_slot_range_for_layer）
    device_index_k = host_index_k = device_index_k_scale = host_index_k_scale = None
    ik_slot, ik_num = self._indexer_slot_range_for_layer(native, device_layer_id)
    if ik_num > 0:
        device_index_k = native.index_k[ik_slot].unsqueeze(0)     # [1, dev_pages, P, 1, index_width]
        host_index_k = self.index_k_buffer[ik_slot].unsqueeze(1)  # [pages, 1, P, 1, index_width]
        if native.index_scales is not None and self.index_k_scale_buffer is not None:
            device_index_k_scale = native.index_scales[ik_slot].unsqueeze(0)  # wrapper 自动 pad 到 5-D
            host_index_k_scale = self.index_k_scale_buffer[ik_slot].unsqueeze(1)  # [pages, 1, page_size, 1, 1]
    transfer_kv_dim_exchange(
        device_indices=device_indices, host_indices=host_indices,
        device_k=dev_k, host_k=host_k, device_v=dev_v, host_v=host_v,
        device_index_k=device_index_k, host_index_k=host_index_k,
        device_index_k_scale=device_index_k_scale, host_index_k_scale=host_index_k_scale,
        page_size=self.page_size, direction=direction)
```

> `host_indices` / `device_indices` 作为 helper **显式入参**（D2H/H2D 两侧都是调用点传入，不依赖
> 闭包），避免复用时的隐式状态。`device_index_k` / `device_index_k_scale` 交给 Python wrapper
> （`kvcacheio.py` L94-122）拆成额外 `transfer_kv_dim_exchange` 调用，**device** scale 会自动从
> 4-D pad 到 5-D（host scale 在上表已直接按 5-D 分配，无需补维）。

**MLA 层号映射（target / draft）**：
- device 层号 → host 层号用 `_host_layer_by_controller`（mla.py L167-175）反查；draft 层号在
  `_device_layer_num` 之后、映射到 `target_layer_num + i`。
- native buffer 选 `_native_infos[info_index]`（target=0，draft 按 `layer_id` 回退），与现有
  `page_first_kv_split` H2D 分支（mla.py L763-765）一致。
- D2H 不能对同一个 `native.k` 直接 `range(layer_num)` 跨 draft 索引：必须按 `_native_infos` 分块
  （target 用 `[0]`、draft 依次用 `[1..]`），与原 `_backup_from_device_per_layer`（mla.py L892-905）
  的 target/draft 划分一致。上面的 helper 通过 `info_index`/`device_layer_id` 已内建此映射。

> 说明：`native.k[l]` 形状 `[dev_pages, P, 1, k_width]`，`unsqueeze(0)` 后 `dim0=1`；
> `self.k_buffer[l]` 形状 `[page_num, P, 1, k_width]`，`unsqueeze(1)` 后 `dim1=1`；二者
> `dim0 == dim1 == 1`、`dim2 == P`，满足契约。K 与 V 的 head_dim 可不同（算子分别取
> `k_head_dim` / `v_head_dim`，heads_num 均取 device_k.sizes()[3]=1）。

**V 参数必须为 5-D 张量（不能传 `None`）**：Python 封装 `transfer_kv_dim_exchange`
（`kvcacheio.py` L50-93）把 `device_v`/`host_v` 作为**必选 `torch.Tensor`** 直接透传给
`torch.ops.npu.transfer_kv_dim_exchange`，且 C++（`transfer_kv_dim_exchange.cpp`）会无判读
`device_v.sizes()[4]`（L67）。因此 `None` 会直接报错。FP8 DSA（`v_width==0`）也必须传**五维、
零元素**的空 V 张量（`native.v` 与 `self.v_buffer` 天然就是 5-D、最后维为 0），由算子内部
`numel()!=0` 判断跳过 V 拷贝——**不要**在伪代码里把 V 短路成 `None`（§6.6 有补充与验证要求）。

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
  上述 5-D 零元素空张量（**不是 `None`**，因为 §6.4 已说明算子要求 5-D Tensor 入参）；传输分支
  直接把它与 `native.v`（同为 5-D、最后维 0）一起传入算子，靠 `numel()==0` 跳过 V 拷贝；
  `get_contiguous_buf_infos` / `data_refs` 只需按 `numel()==0` 跳过该空 V（与 §6.4.1 一致）。

### 6.5 合并记录方案（方案 B 备选，不推荐）

若坚持 host 用合并 `kv_cache_dim` 记录（`[layer_num, size, 1, kv_cache_dim]`），字节布局
定义为 **K 段在前、V/RoPE 段在后**（与 §6.1 逻辑序一致）：

```
token 行 = [ latent(kv_lora_rank) | rope(qk_rope_head_dim) ]
```

但传输时因 §6.3 的 pitch 约束，需要先 staging 把合并记录拆成 K/V 两块连续 buffer（或按
`page_size` 维度做中间重组），额外内存 + 拷贝，违背「端到端性能不低于 page_first」目标。
因此记录其字节序约定即可，L1↔L2 传输不采此方案。

### 6.6 FP8 DSA 特殊路径（单列，不套用普通 BF16）

FP8 DSA 的打包由 `_pack_dsa_fp8_kv_cache`（`memory_pool_npu.py` L949-970）完成，字节布局
（`torch.cat` 顺序）为：

```
packed = [ latent FP8 (kv_lora_rank × 1B) | rope BF16 (qk_rope_head_dim × 2B) | scale FP32 (kv_lora_rank//128 × 4B) ]
```

- 总宽 `kv_cache_dim = kv_lora_rank + kv_lora_rank//128*4 + qk_rope_head_dim*2`；
- device `v_buffer` 宽 `kr_cache_dim == 0`（`numel()==0`，算子 K/V 分支因 `numel!=0` 判断自动跳过 V）；
- host 侧**不分配 `qk_rope_head_dim` 宽的 `v_buffer`**，只分配 `k_buffer`（宽 `kv_cache_dim`，取
  `native.k_width`）+ `index_k_buffer` + `index_k_scale_buffer`；`v_buffer` 以 **5-D 零元素空张量**
  存在（容量闭合见 §6.4.1「V buffer 容量闭合」：基础值 `kv_cache_dim*itemsize` 正好等于 k_buffer
  实际宽，无多占）；
- 逐层传输时**只传 K（打包段）**，每 token 连续打包（页内 stride == `kv_cache_dim`，宽对齐
  正常）；`dev_v/host_v` 仍传 5-D 空张量（`native.v` / `self.v_buffer`，最后维 0），算子按
  `numel==0` 跳过 V——**不能传 `None`**（§6.4「V 参数必须为 5-D」）；
- 不能把打包段当作「普通 BF16 的 K 段 + V 段」两段处理：打包段内部已含 rope 与 scale，顺序、
  宽度、dtype 均不同，必须单列分支。
- **验证（只传 K 路径）**：单测构造 5-D 零元素 V（最后维 0）调算子，确认 `numel()==0` 分支不触发
  V 拷贝、`device_v.sizes()[4]` 读取安全；端到端再用 FP8 DSA 模型跑 D2H/H2D 校验 K 段逐层搬运正确、
  无 V 段越界（§10.2）。

### 6.7 indexer 附带缓存

- `index_k` 仅存在于 `index_layer_indices` 指定的子集层（如 GLM5.x 为 21/78 层）；host 侧
  index buffer 层数 == `num_indexer_layers`。
- `page_first_kv_split` 现状由 Python wrapper 把 `index_k`/`index_k_scale` 拆成独立
  `transfer_kv_dim_exchange` 调用（`kvcacheio.py` L94-122）；layer_first 下同样拆成独立调用，
  但需**逐 indexer 层（N=1）**，而不能照搬「一次性传所有 indexer 层」（因为 host index buffer
  也是 layer_first）。
- 逐层循环需按 device layer id → index slot 做映射，复用现有 `_indexer_slot_range_for_layer`
  （`mla.py` L466-472），不能对 model 层盲目逐 indexer 层传；该映射与逐层 indexer 传输已内建到
  `_transfer_one_layer`（§6.4），load/backup 两侧共用。
- **wrapper 的占位 V 改为 5-D 零元素张量（P1，方案：改 wrapper、不动 C++）**：`kvcacheio.py` 为
  `index_k`（L98-L99）与 `index_k_scale`（L115-L116）两段额外 `transfer_kv_dim_exchange` 调用，
  用 `torch.empty(0)`（**一维**）作 `device_v`/`host_v` 占位。但 C++ 在 `numel()!=0` 守卫之外
  **无条件读 `device_v.sizes()[4]`**（`transfer_kv_dim_exchange.cpp` L67），一维 `torch.empty(0)`
  是对该维的越界读取（未定义行为）。因此把这两处占位 V 改为**五维零元素张量**
  `torch.empty((*k.shape[:-1], 0), dtype=k.dtype, device=k.device)`（与对应 K 同页/层布局、
  最后维为 0），使 `sizes()[4]==0` 且 `numel()==0`，由算子 `numel!=0` 判断跳过 V 拷贝。此方案只改
  wrapper、不改 C++，对现有 `page_first_kv_split` 的 indexer 调用行为不变（`numel` 恒为 0）。
  **并补一条「indexer 实际调用」单测**：构造带 `index_k`（及 `index_k_scale`）的 MLA host pool，
  D2H/H2D 各跑一次，断言 wrapper 内两段 indexer 子调用能正常跳过空 V、不越界（§6.4 helper
  已验证 K/V 主路径，此处专测 indexer 子路径）。

### 6.8 Memfabric/AscendC 融合路径（本次不考虑）

`SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric`（`npu_memfabric.py` L31-46）同时开启两件事：
① host pool 改用 Memfabric-mapped 内存；② L1↔L2 改走 AscendC AIV 融合 kernel
`kv_exchange_copy`（`ascendc_io_enabled()`，`mla.py` L767/L997 分流）。该融合路径的元数据把
`host_layout_mode` 固定为 0（=page-first）（`mla.py` L536），且 `_ascendc_comp_meta`（L552-553）
按 page/layer stride 假设 page-first，**不支持 layer_first 的跨层 stride**。

**本次决策：不考虑支持 memfabric / AscendC 融合路径。** 落地方式——**前移到启动配置校验**
（`hardware_backend/npu/utils.py::set_default_server_args`，与 §7.2 的布局投影同处一个阶段），
在服务启动即失败，而不是等首次 L2 传输才失败：

```python
# set_default_server_args 内、hicache 分支中（§7.2 的 layout 判定之后）
if (
    cfg.hicache_mem_layout == "layer_first"
    and memfabric_host_memory_enabled()
):
    raise ValueError(
        "layer_first layout does not support "
        "SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric in this release"
    )
```

（`memfabric_host_memory_enabled()` 从 `pool_host/npu_memfabric.py` 导入；它等价于
`ascendc_io_enabled()`，二者都读同一个 `SGLANG_HICACHE_HOST_MEM_BACKEND`。）

- 默认（非 memfabric）host pool 走 `transfer_kv_dim_exchange` 逐层，是本次全部范围；layer_first
  分支内不再单独判 `ascendc_io_enabled()`（已由启动校验保证到不了）。
- 这一范围限制须同步写进需求/验收说明：`--hicache-mem-layout layer_first` 与
  `SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric` 二者互斥，二期支持前启动即报错。
- 后续（二期）支持 memfabric 需二选一：扩展 sgl-kernel-npu csrc 增加 `host_layout_mode=1`
  （layer-first striding），或回退旧算子（注意 memfabric host 指针对 `aclrtMemcpy2dAsync`
  的可访问性需另行验证，且旧算子把 indices 转 CPU 会破坏逐层流水与性能目标）。

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

改为：**仅当 raw 值不是 `layer_first` 时才投影默认**；raw 值为 `layer_first` 则跳过覆盖。

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
```

  文档仅承诺「显式 `layer_first` 会被放行」，不宣称能识别其它显式 CLI 值。

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

1. **device rows 是否 packed**：layer_first 的 kernel_ascend 路径直接按 `page_size` 连续取数，
   若 device buffer 是 strided（如 unified pool），需先 `_require_packed_device_rows` 或拒绝。
   MHA/MLA 标准 pool 为 packed。
2. **空层 / layer shard**：draft pool、层分片（`dsa`/`dcp`）下 `layer_num` 动态，逐层循环需用
   `_owned_device_layer_ids()` / `_device_owned_layer_range()`（`base.py`）而非硬编码。
3. **indices 对齐**：算子要求 `device_indices.numel() % page_size == 0`；逐层循环复用同一组
   indices，只搬 layer 维。
4. **空 V**：DSv4 用 `torch.empty(0)` 表示无 V；MHA 有 V，MLA 普通 BF16 有 V、FP8 DSA 的
   `v_buffer` 用 **5-D 零元素空张量**（最后维 0，`numel()==0`，非 `None`，见 §6.4.1/§6.6）。
5. **FP8 DSA 的 `native.v` 宽度为 0**：`native.v` 已是 5-D、最后维 0 的合法张量，直接
   `native.v[device_layer_id].unsqueeze(0)` 即可，**无需**对 `None` 短路；`self.v_buffer` 同
   （5-D 零元素空张量），传输时照常传入算子（§6.4「V 参数必须为 5-D」）。

---

## 9. 风险与待确认

- [x] layer_first 的 layout 命名：**已定——复用 `layer_first` 名，不新增 `layer_first_kv_split`**
      枚举项（与 GPU 对齐：GPU 只有一个 layer_first 名；K/V 是否分离是 NPU 设备池 native 属性，
      由 `isinstance(buffer_info.buffers, MLASplitBuffers)` 决定，不泄露进 layout 枚举）。
- [x] Memfabric / AscendC 融合路径：**已定——本次不考虑**，在**启动配置校验阶段**
      （`set_default_server_args`）拒绝 `layer_first + memfabric`（§6.8），仅支持默认（非 memfabric）
      host pool。
- [ ] 逐层循环（N=1）每层一次算子调用的 launch 开销 vs page_first 单次多层调用——**短序列 /
      decode 单 token 场景下 launch 开销放大 `layer_num` 倍，是最可能不达标的点**，性能验证必须
      覆盖短序列（§10.1）。layer_first 跨层 stride=整层大小、算子 host pitch=一页宽，无法分组，
      优化只能压低单次调用开销，而非减少调用次数。
- [x] indexer 附带缓存是否纳入本次范围：**已定——纳入**，`_size_per_token` 的 indexer 计入条件
      （§6.4.1）与逐层 indexer 传输（§6.7，含 wrapper 占位 V 改 5-D 与「indexer 实际调用」单测）
      已一并落定。
- [ ] FP8 DSA 打包段的 host `k_buffer` 宽度必须用 `native.k_width`（打包宽），不能误用
      `kv_lora_rank`；需单测覆盖 FP8 DSA 场景（含 5-D 零元素空 V 的「只传 K」验证，§6.6）。

---

## 10. 落地顺序

1. 抽象公共 helper：`layer_first → [pages,1,P,H,D]` 视图（仿 `_host_page_view`）+ 单层搬运转译
   `_transfer_one_layer`（§6.4，内置 info_index / device_layer_id / host_layer_id / index slot 映射，
   host_indices/device_indices 显式入参，D2H/H2D 共用，保证 V 始终为 5-D 张量、indexer 逐层搬运）。
2. MHA layer_first（§5）+ 单元测试闭环。
3. MLA layer_first（§6，普通 BF16，含 indexer）+ 单元测试闭环（含 §6.4.1 派生接口适配、
   `_size_per_token` 的 indexer 计入）。
4. FP8 DSA 特殊路径（§6.6）+ 单测覆盖打包宽度与空 V。
5. 布局开关（§7）+ 端到端精度测试。
6. 性能对比 + CI 注册（对应需求文档 §11）。

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
> `DeepSeekV4PagedHostPool`（已实现 layer_first + kernel_ascend，见 §4），**不在本次「通用 host
> pool 补 layer_first」的改动范围内**，仅回归确认不被本次改动影响。
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

- 同一 IO 后端配置下 A/B 对比（**双方都 memfabric 关**，排除 §6.8 影响）：`layer_first` vs
  `page_first_kv_split`（MLA）/ `page_first_direct`（MHA）。
- 指标：端到端吞吐（token/s）、TTFT、TPOT；前缀命中率固定。
- 覆盖两类输入长度：**长序列**（prefill 数千 token，验证数据拷贝主导时基本持平）与
  **短序列 / 单 token 续写**（验证逐层 launch 开销是否放大，§9 风险项）。
- 判定规则：两类场景下 `layer_first` 指标均 **不低于 page_first**（吞吐 ≥ 1.0x、时延 ≤ 1.0x，
  波动阈值 ±5%）。若短序列不达标，须记录并回到 §9 的 launch 开销优化项，不得以长序列通过掩盖
  短序列回退。

**功能验收**：`--hicache-mem-layout layer_first` + `kernel_ascend`（不含 L3、不含 memfabric）正常
启动、前缀命中正确、精度回归通过（复用需求文档 §11 的 GSM8K few-shot 方式）。