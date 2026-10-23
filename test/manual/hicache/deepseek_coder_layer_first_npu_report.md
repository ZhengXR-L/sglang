# DeepSeek-Coder NPU layer_first HiCache 测试报告

- 测试日期：2026-10-23
- 测试模型：DeepSeek-Coder-V2-Lite-Instruct（MLA 架构）
- 测试平台：Ascend910 NPU（sglang-lzx-B103 容器）

---

## 1. 测试目的

验证 NPU 平台下 HiCache `layer_first` KV 内存布局的 MLA 传输路径（AscendC `kv_exchange_copy` 稀疏拷贝算子）在 DeepSeek-Coder 模型上的正确性与性能，覆盖以下 4 个维度：

| 测试 | 目的 |
|------|------|
| e2e_layer_first_mla.sh | 验证服务器能以 `layer_first` + `kernel_ascend` + `memfabric` 正常启动，单请求输出正确（正确性 smoke） |
| e2e_layer_first_mla_roundtrip.sh | 验证 L1 设备池淘汰触发 D2H、前缀重放触发 H2D 的双向往返，数据无损恢复 |
| e2e_layer_first_mla_gsm8k.sh | 验证启用 layer_first 后模型生成精度不退化（GSM8K 100 题 5-shot） |
| bench_layer_vs_page_roundtrip.sh | 对比 `layer_first` 与 `page_first_kv_split` 两种布局的 TTFT / 吞吐性能 |

---

## 2. 测试环境

- 运行容器：`sglang-lzx-B103`
- 软件版本：Python 3.12.13，torch 2.10.0（torch_npu 可用），CANN 9.1.0
- 硬件：Ascend910，单卡（`NPU_DEVICES=8`，属芯片 4，空闲约 3GB / 64GB）
- 模型：`/mnt/paas/weights/DeepSeek-Coder-V2-Lite-Instruct`（`--tp 1` 单卡可装载）

统一关键启动参数：

```
--device npu
--attention-backend ascend
--tp 1
--enable-hicache
--hicache-io-backend kernel_ascend
--hicache-write-policy write_through
SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
# 布局：--hicache-mem-layout layer_first 或 page_first_kv_split
```

---

## 3. 脚本运行方式

所有测试均在容器内执行，并设置独立的日志目录 `/home/l00889655/sglang/e2e_layer_first_logs_20261023`：

```bash
docker exec -e PYTHONPATH="/usr/local/Ascend/cann-9.1.0/python/site-packages:/home/sgl-workspace/sglang/python" \
  sglang-lzx-B103 bash -c \
  'LOG_DIR=/home/l00889655/sglang/e2e_layer_first_logs_20261023 bash <脚本绝对路径>'
```

> **关键前提（PYTHONPATH 修复）**：容器默认 `PYTHONPATH` 不含 CANN 的 Python site-packages，导致 `import acl` 失败，进而 `sglang.srt.models.deepseek_v2` 原生实现导入失败，模型会回退到 Transformers 后端并在 remote `modeling_deepseek.py` 中因 `_attn_implementation='sglang'` 触发 `KeyError: 'sglang'` 崩溃。
> 修正后 `PYTHONPATH` 须同时包含本地 sglang checkout、CANN site-packages（提供 `acl`）、容器原有 `PYTHONPATH`：
> `/home/l00889655/sglang/sglang/python:/usr/local/Ascend/cann-9.1.0/python/site-packages:/home/sgl-workspace/sglang/python`

4 个脚本路径（`/home/l00889655/sglang/sglang/test/manual/hicache/`）：

1. `e2e_layer_first_mla.sh`
2. `e2e_layer_first_mla_roundtrip.sh`
3. `e2e_layer_first_mla_gsm8k.sh`
4. `bench_layer_vs_page_roundtrip.sh`（性能测试，同目录 `bench_layer_vs_page_roundtrip.py`）

---

## 4. 测试结果

### 4.1 smoke 端到端正确性 —— PASS

- 服务器健康就绪：约 59s
- 请求 `1 + 1`，生成文本 `" 2"`，与预期一致

```
PASS: layer_first MLA HiCache smoke test succeeded (completion: [ 2])
```

### 4.2 往返传输（D2H + H2D）—— PASS

三阶段 `write → evict → replay` 全部完成，needle `48219` 正确恢复：

```
W answer: 'The secret passcode is 48219.'  (last_num=48219)
E answer: '5'  (last_num=5)
R answer: 'The secret passcode is 48219.'  (last_num=48219)
server log: kv_exchange_copy D2H x4, kv_exchange_copy H2D x27
RESULT: PASS
```

即设备池淘汰触发 4 次 D2H 写入、前缀重放触发 27 次 H2D 恢复，且 needle 数据往返无损。

### 4.3 GSM8K 精度 —— 0.860 (86/100)

- 100 题，5-shot，temperature 0.0，max_tokens 512
- invalid（未抽取到答案）：0/100
- request errors：0/100
- 耗时：31.85s（3.14 q/s）

结论：启用 layer_first 后无请求异常、无答案抽取失败，86% 精度为模型自身能力水平，L2 传输路径未破坏生成。

### 4.4 性能对比 —— layer_first 与 page_first_kv_split 基本持平，replay 略优

配置：`num_prompts=10`、`prompt_tokens≈1500`、`max_tokens=64`、`warmup_rounds=1`、`measure_rounds=3`（3 轮平均）。

| 阶段 | layer_first | page_first_kv_split |
|------|-------------|---------------------|
| fill mean_ttft | 519.2 ms | 518.8 ms |
| fill throughput | 111.4 tok/s | 112.1 tok/s |
| evict mean_ttft | 500.7 ms | 503.4 ms |
| replay mean_ttft | 256.0 ms | 264.4 ms |
| replay median_ttft | 229.4 ms | 237.2 ms |
| replay throughput | 173.7 tok/s | 170.2 tok/s |

补充：两者 Round3 needle 匹配均为 30/30，`kv_exchange_copy` 传输计数均为 D2H x38 / H2D x324（场景一致）。

---

## 5. 结论

1. NPU 下 DeepSeek-Coder 的 layer_first MLA 传输路径（AscendC `kv_exchange_copy` + memfabric）可正常启动、正确推理，smoke / roundtrip / gsm8k 三项正确性测试全部通过。
2. 性能上 layer_first 与 page_first_kv_split 基本持平，其中 replay（H2D 前缀恢复）阶段 layer_first 略优约 3%（mean_ttft 256.0ms vs 264.4ms）。
3. 运行前提是 `PYTHONPATH` 必须包含 CANN site-packages（提供 `acl`），否则 `deepseek_v2` 原生实现导入失败并触发 Transformers 回退崩溃（`KeyError: 'sglang'`）。