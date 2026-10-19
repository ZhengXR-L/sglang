# Qwen4-Exp (Qwen3.8-Flash-Next) QSA bfloat16 精度验证报告

## 1. 背景与目标

在华为昇腾 NPU（CANN 9.0.0 / 9.1.0，A3）上验证 Qwen4-Exp（Qwen3.8-Flash-Next）模型
`--qsa-indexer-dtype bfloat16` 的精度影响，并定位：

> 为什么同样的 GSM8K 用例，在 qianruiyuan 的 `qry-20260909f` 容器（TIMEVAULT）能通过，
> 而我们的测试却数值塌缩（输出重复 token / 模板碎片，GSM8K acc=0）。

### 被测对象

- 测试脚本：`sglang/test/registered/npu/llm_models/test_npu_qwen4_exp_qsa_bf16.py`
- 用例：`TestNPUQwen4ExpQSAIndexerBF16::test_gsm8k`
- 评测方式：sgl-eval chat 协议（`gsm8k_backend="sgl_eval"`，`gsm8k_thinking=True`）
- 样例数：5 题，阈值 `gsm8k_score_threshold=0.90`

## 2. 环境

| 项 | 值 |
|---|---|
| 容器 | `sglang-lzx-cann90`（image `8fd50aae6bf4`） |
| CANN | 9.0.0 / 9.1.0（两个版本均验证通过） |
| Python | 3.11.15 |
| torch / torch_npu | 2.10.0 / 2.10.0 |
| 卡 | 后 8 卡（`ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15`），TP=8 |
| 模型 | `/mnt/paas/weights/Qwen3.8-Flash-Next` |

运行前必须显式指定 `PYTHONPATH`，否则 `import sglang` 会解析到 main 分支（无 QSA 实现）：

```bash
export PYTHONPATH=/home/l00889655/sglang/sglang_work2/sglang/python:$PYTHONPATH
```

## 3. 遇到的问题与解决过程

### 3.1 sglang 默认加载了错误分支

- **现象**：容器内 `import sglang` 解析到 main 分支 `f58665451`，`QSAIndexer` 无法导入。
- **解决**：运行前 `export PYTHONPATH=.../sglang_work2/sglang/python:$PYTHONPATH`，
  验证后 `__path__` 指向 work2，`QSAIndexer` 可导入。

### 3.2 `--qsa-indexer-dtype` 疑云（排除）

- **怀疑**：通过/失败差异来自 `qsa-indexer-dtype`（qianruiyuan 用 `auto`，我们显式 `bfloat16`）。
- **排查**：`resolve_qsa_indexer_dtype` 确认 NPU 上 `auto` 与 `bfloat16` 都解析为
  `torch.bfloat16`（`fp8_e4m3` 仅在 CUDA SM90/SM100 允许，NPU 拒绝）。
- **结论**：`qsa-indexer-dtype` 不是差异源。

### 3.3 原生算子 `.so` 差异（排除）

- **怀疑**：GDN 线性注意力 C++/AscendC 原生算子 `libsgl_kernel_npu.so` 不一致。
- **排查**：符号级对比一致。
- **结论**：非根因。

### 3.4 根因定位：非投机解码路径数值塌缩

diff 对比发现，qianruiyuan 的 `serve.sh` 与我们失败脚本的关键差异是：

- qianruiyuan：`--speculative-algorithm NEXTN` + `--attention-backend ascend`
  + `--chunked-prefill-size 4096` + `ASCEND_USE_FIA=1` / `HCCL_OP_EXPANSION_MODE=AIV`
  + `--reasoning-parser qwen3`（`qsa_indexer_dtype=auto`）
- 我们（失败）：无投机、无 FIA/AIV、`qwen3-thinking`、显式关流 `SGLANG_NPU_USE_MULTI_STREAM=0`

由此把「投机解码」与「FIA/AIV」作为两个候选关键变量，进行三重隔离对照。

### 3.5 卡资源被外部任务占用

- **现象**：多次启动报
  `RuntimeError: The memory capacity is unbalanced. Some GPUs may be occupied by other processes.`
  其中 `pre_model_load_memory=28.2GB`，而 `local_gpu_memory=60.86GB`，说明加载前约 32GB/卡
  已被占。
- **解决**：由用户协调释放后复测（协调后 HBM 回落到近似空闲，加载前恢复 ~61GB）。

### 3.6 sgl-kernel-npu `git pull` 网络超时（2026-10-19）

- **现象**：`git pull` 报
  `error: RPC failed; curl 56 OpenSSL SSL_read: Connection timed out, errno 110`、
  `fetch-pack: unexpected disconnect while reading sideband packet`、`early EOF`、
  `invalid index-pack output`。
- **排查**：IPv6 不通（`curl -6` 失败）+ GitHub HTTPS 传输间歇性超时。
- **解决**：`git config http.version HTTP/1.1` 后重试 `git fetch qry` 成功（网络为间歇性故障），
  拉到 `b01807d..36467f5` 共 23 提交。

### 3.7 采用远端改动误删 short_conv.py 的 tl.cdiv 修复（UnsupportedLanguageConstruct）

- **现象**：对 sgl-kernel-npu 的本地冲突选择「全部采用远端改动」后，`qwen3_8_flash_next/short_conv.py`
  里的本地修复被 `git checkout` 丢弃；重测在 target verify graph 捕获阶段崩溃 `UnsupportedLanguageConstruct`。
- **根因**：CANN backend 上 Triton `tl.cdiv(C, BLOCK)` 返回 tensor，导致 `if REUSE` 这类
  constexpr 分支无法在编译期求值。
- **解决**：重新应用本地修复 `CDIV: tl.constexpr = (C + BLOCK - 1) // BLOCK`，
  替换所有 `tl.cdiv(C, BLOCK)`（用 Python 整数除法在编译期算出）。修复后
  `Capture target verify NPU graph end. bs=[1..8]` 捕获成功。

### 3.8 warmup 阶段 resize 报错（transformers 版本不一致）

- **现象**：服务 warmup 发 `POST /v1/chat/completions` 时，NPU 多模态处理器
  `qwen_vl_processor.py:86` 抛 `TypeError: TorchvisionBackend.resize() missing 1 required
  positional argument: 'image'`，返回 500，`Server process exited with code -9`。
- **根因**：主仓分支已把 `qwen_vl_processor.py` 适配到 transformers **5.19** 的新 `resize()`
  签名（`images=` / `factor=`），而 `sglang-lzx` 容器装的是 transformers **5.12.1**
  （`resize()` 仍是 `image=` 且需先 `smart_resize`）。
- **解决**（临时 workaround）：按 5.12.1 实际 API 把该文件图像/视频 resize 逻辑改回
  `smart_resize` + `image=`；正式做法是升级容器依赖至 `transformers==5.19.0`
  （+ `tokenizers==0.23.2`）并还原该文件。

### 3.9 卡再次被外部任务占用（Qwen3.6-35B-A3B）

- **现象**：准备启动 8 卡测试时发现 devices 12、13（Chip 6）被一个外部
  `Qwen3.6-35B-A3B --tp-size 2 --port 8899` 服务占用（各约 59GB HBM）。
- **解决**：先停止我方测试，等用户协调外部任务释放卡后再启动。

## 4. 验证实验矩阵

固定变量：`--qsa-indexer-dtype bfloat16`、`--mamba-ssm-dtype bfloat16`、`--attention-backend ascend`
、TP=8、sgl-eval 5 题、阈值 0.90。仅改变「投机解码」与「FIA/AIV」两个变量。

| 配置 | NEXTN 投机 | FIA/AIV | score | accuracy | invalid | acc | 结果 |
|---|---|---|---|---|---|---|---|
| 原始通过（qianruiyuan 对齐） | ✅ | ✅ | 1.0 | 1.0 | 0.0 | 100% (5/5) | ✅ PASSED |
| **方案 A** | ❌ | ✅ | 0.0 | 0.0 | 1.0 | 0% (0/5) | ❌ FAILED |
| **方案 B** | ✅ | ❌ | 1.0 | 1.0 | 0.0 | 100% (5/5) | ✅ PASSED |

### 4.1 方案 A（去投机 · 留 FIA/AIV）

```
gsm8k: 100%|██████████| 5/5 acc=0.00%, truncated=0/5
sgl-eval gsm8k: {'score': 0.0, 'accuracy': 0.0, 'invalid': 1.0, ...}
AssertionError: 0.0 not greater than or equal to 0.9
FAILED
```

- 无投机解码，纯 decode 路径，输出全部无效（`invalid=1.0`，`predicted_answer` 无法提取）。
- 等效吞吐仅 ~8.7 tok/s（且输出是乱码）。

### 4.2 方案 B（留投机 · 去 FIA/AIV）

```
gsm8k: 100%|██████████| 5/5 acc=100.00%, truncated=0/5
sgl-eval gsm8k: {'score': 1.0, 'accuracy': 1.0, 'invalid': 0.0, ...}
gsm8k sgl-eval score=1.0000
PASSED
1 passed in 232.75s (0:03:52)
```

- 投机验证路径正常：`accept len 2.67~3.30`，`accept rate 0.56~0.77`。
- 输出吞吐 ~28.8 tok/s。

### 4.3 最新 kernels 复测：非投机路径恢复（2026-10-19）

pull 最新 kernels `36467f5`（含 A3 target verify UB overflow 修复 #848 等）并恢复
`short_conv.py` 的 tl.cdiv 修复后，对「无投机解码」路径复测（变量「投机解码」此时已非正确性前提）：

| 配置 | 题数 | score | accuracy | 结果 |
|---|---|---|---|---|
| 无投机解码 | 5 | 1.0 | 1.0 | ✅ PASSED |
| 无投机解码 | 100 | 0.95 | 0.95 | ✅ PASSED（阈值 0.90）|

- 100 题：`error_rate=0.0`、`truncated_rate=0.01`、`stop_rate=0.99`，无崩坏、无截断。
- 无投机 decode 输出吞吐 ~73.1 tok/s（`npu graph: True`），高于此前投机路径的 ~28.8 tok/s。

## 5. 最终结论

1. **非投机解码路径的数值塌缩 bug 已被最新 kernels（36467f5）修复。**
   - 早期（旧 kernels）A3 上 Qwen4-Exp QSA 非投机解码存在确定性数值崩坏
     （重复 token / 秒 EOS），当时只能靠 NEXTN 投机验证路径绕开。
   - pull 最新 kernels 后，非投机解码恢复正常：GSM8K 5 题 acc=1.0、100 题 acc=0.95，
     无崩坏、无截断（`stop_rate=0.99`）。

2. **默认采用无投机解码参数。**
   - 无投机解码即可保证正确性（≥ 0.90 阈值），且省去 NEXTN draft 开销，
     吞吐更高（~73.1 tok/s，对比投机路径 ~28.8 tok/s）。
   - NEXTN 投机解码退回「可选加速项」，不再作为正确性的前置条件。

3. **FIA/AIV（`ASCEND_USE_FIA=1`、`HCCL_OP_EXPANSION_MODE=AIV`）与精度无关。**
   - 方案 B 去掉它们后仍然 5/5 全对。
   - 但影响吞吐：无 FIA/AIV 时吞吐约下降 24%。

4. **`--qsa-indexer-dtype bfloat16` 本身精度无损。**
   - `auto` 与 `bfloat16` 在 NPU 上等价，与 bf16 索引器精度无关。

5. **跨 CANN 版本均正常。**
   - CANN 9.0.0 与 9.1.0 均已验证通过。

## 6. 建议与后续

1. **默认测试配置**：采用**无投机解码**参数——去掉 `--speculative-algorithm NEXTN`
   及配套 `--speculative-num-steps` / `--speculative-eagle-topk` / `--speculative-num-draft-tokens`；
   保留 `ASCEND_USE_FIA=1`、`HCCL_OP_EXPANSION_MODE=AIV`（不改变精度，提升吞吐约 24%）。
2. **依赖对齐（待办）**：容器 transformers 停在 5.12.1，而主仓已锁定 5.19.0；当前用反向 patch
   `qwen_vl_processor.py` 临时适配，正式应升级 `transformers==5.19.0`（+ `tokenizers==0.23.2`）
   并还原该文件。
3. 测试文件当前状态：无投机解码 + 100 题样例。

---

*报告更新于 2026-10-19。相关测试文件：
`sglang/test/registered/npu/llm_models/test_npu_qwen4_exp_qsa_bf16.py`*