# GLM-5.2 W4A8C8：NPU `layer_first` 验证与排障记录

日期：2026-10-23  
服务器：`113.46.15.88`  
容器：`sglang-lzx-B103`  
代码目录：`/home/l00889655/sglang/sglang`

> **模型名称核对**：本记录的实际模型路径是 `/mnt/paas/weights/GLM-5.2-w4a8c8`。先前讨论中出现的 `GLM-5.2-w8a8c8` 不是本次运行的模型，不能把以下结果当成 W8A8C8 的验证结论。这里的 `c8` 也不能单凭模型名称解释成 `--kv-cache-dtype fp8_e4m3`；本次脚本没有设置该参数，验证的是默认 BF16 KV cache 下的 `k/v/index_k` 路径，不含 FP8 `index_k_scale`。

## 1. 验证范围与结论

| 项目 | 条件 | 结果 | 可证明的范围 |
| --- | --- | --- | --- |
| 8 卡 D2H/H2D 往返 | 物理 NPU 8–15，TP8/EP8，L1 3072 tokens | PASS；D2H 32 条、H2D 624 条、`(k,v,index_k)` 656 条 | BF16 DSA Indexer 路径在该配置下完成写入、逐层恢复和目标数字复现 |
| 16 卡 D2H/H2D 往返 | 物理 NPU 0–15，TP16/EP16，L1 8192 tokens | PASS；D2H 32 条、H2D 1248 条、`(k,v,index_k)` 1280 条 | 同上；16 卡配置也能运行 |
| 16 卡 GSM8K | 100 题、5-shot、`temperature=0` | 92/100，0 个请求错误 | 该配置下的生成正确性抽样；本次 GSM8K 服务日志 H2D=0，因此它本身不验证 L2 恢复 |
| 16 卡 L2 恢复 TTFT | 两种布局；各轮每布局预热 1 次、测量 5 轮 × 2 个请求；正反顺序各跑一轮 | 两轮 `layer_first` 的 median 和 p95 均略慢于 `page_first_kv_split`；性能门槛 FAIL | 功能路径被触发，但当前短样本性能测试未达到“不能弱于 page_first”的目标 |

往返脚本用 W（写入目标数字 `48219`）、E（用额外提示词驱逐 L1）、R（重放原提示词）三个请求验证 L2。两次运行均记录到 W/R 输出目标数字 `48219`，E 请求完成。脚本同时断言 D2H、H2D 和 `k,v,index_k` 日志标记非零。完整运行脚本见第 5 节。

**证据解释边界**：`kv_exchange_copy` 日志标记位于内核调用之前，单条标记不是拷贝成功的独立证明。本次结合服务端无相关异常、请求返回 200、R 正确复现目标数字和双向标记，形成端到端证据；仍不是 KV 内容逐字节比对，也没有覆盖 FP8 scale 分支。TTFT 结果是当前测试负载下的观察，不是全场景性能结论。

## 2. 16 卡功能验证的关键日志

脚本：`test/manual/hicache/e2e_layer_first_glm52_indexer_roundtrip_16npu.sh`。关键参数：`--tp-size 16 --ep 16 --moe-a2a-backend deepep --deepep-mode normal --deepep-dispatcher-output-dtype bf16 --enable-hicache --hicache-mem-layout layer_first --hicache-io-backend kernel_ascend --hicache-write-policy write_through --hicache-ratio 2.5 --mem-fraction-static 0.85 --max-total-tokens 8192 --device npu --attention-backend ascend --disable-cuda-graph`；`HCCL_BUFFSIZE=1024`；文档提示约 6000 tokens，驱逐提示约 6500 tokens。

以下摘自归档的 `16npu/roundtrip_server.log`（行号属于该归档文件）：

```text
799:[2026-10-23 07:57:30 TP0 EP0] max_total_num_tokens=8192, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=2048, context_len=1048576, available_gpu_mem=23.70 GB
1120:[2026-10-23 07:58:20 TP14 EP14] HiCache AscendC kv_exchange_copy D2H: layout=layer_first_kv_split pages=47 layers=[0,78) components=3 (k,v,index_k)
1137:[2026-10-23 07:58:43] INFO:     127.0.0.1:45118 - "POST /v1/completions HTTP/1.1" 200 OK
1155:[2026-10-23 07:58:47] INFO:     127.0.0.1:38476 - "POST /v1/completions HTTP/1.1" 200 OK
1156:[2026-10-23 07:58:47 TP15 EP15] HiCache AscendC kv_exchange_copy H2D: layout=layer_first_kv_split pages=47 layers=[0,1) components=3 (k,v,index_k)
2407:[2026-10-23 07:59:13] INFO:     127.0.0.1:35478 - "POST /v1/completions HTTP/1.1" 200 OK
```

`layers=[0,78)` 的 D2H 对应全层备份；`layers=[0,1)` 的 H2D 是逐层恢复调用的一个样例。`components=3` 明确表示本次传输了 K、V、Indexer K；默认 BF16 KV cache 不触发 `index_k_scale`。本次日志计数是 D2H 32、H2D 1248、组件标记 1280，且没有 `kv_exchange_copy failed` 或 traceback。往返脚本最终输出 `PASS: layer_first GLM-5.2-w4a8c8 Indexer (index_k) D2H+H2D roundtrip OK`。

### GSM8K 补充验证

脚本：`test/manual/hicache/e2e_layer_first_glm52_indexer_gsm8k_16npu.sh`；评估日志摘录：

```text
GSM8K eval: num_questions=100 shots=5 base_url=http://127.0.0.1:32348 temperature=0.0 max_tokens=512
GSM8K accuracy: 0.920 (92/100)
invalid (no answer extracted): 0/100
request errors: 0/100
latency: 326.46s  (0.31 q/s)
```

对应服务日志中 D2H 为 1392 条、H2D 为 0 条，因此 92/100 不应被表述为“GSM8K 已验证从 L2 读回”。

## 3. 8 卡排障经过与最终方案

8 卡脚本是 `test/manual/hicache/e2e_layer_first_glm52_indexer_roundtrip.sh`，成功运行时通过 `NPU_DEVICES=8,9,10,11,12,13,14,15` 指定物理卡 8–15。下面的错误摘录均来自第 5 节归档，除第一项和 0.85 阈值尝试外。

| 阶段 | 关键证据 / 现象 | 处理 |
| --- | --- | --- |
| 模型导入 | 初次运行报 `No module named 'acl'`；回退到 Transformers 后报 `Unsupported TP style 'mla_kv_a_proj'`。初次失败日志后来被覆盖，只保留了当时命令输出记录。 | 把 Ascend toolkit 和 CANN 的 Python site-packages 加入 `PYTHONPATH`，启动前执行 `python -c "import acl; import sglang.srt.models.glm4_moe"`。 |
| 权重装载后 KV 池无空间 | `ValueError: Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.75. Raise --mem-fraction-static above 0.824 ...` | 将 `--mem-fraction-static` 调高；最终 8 卡使用 `0.86`，同时缩小 L1 token 上限。 |
| MoE 分发形状不匹配 | `When groupList is not null, size of groupList(tensor) 256 should be equal to weight dim 0 128 with groupType 0.` 不带 DeepEP 的配置在 CUDA graph/实际请求阶段失败。 | 使用 `--moe-a2a-backend deepep --deepep-mode normal --deepep-dispatcher-output-dtype bf16 --ep 8`，并关闭 CUDA graph。设置 `SGLANG_DEEPEP_BF16_DISPATCH=1` 不能单独替代这一组配置。 |
| 大 L1 内存不足 | 16K-token 配置装载 MLA KV 池时 `NPU out of memory. Tried to allocate 1.23 GiB ... 994.85 MiB free`；6K-token 配置可启动，但请求阶段报 `HcclAllreduce` 内存分配失败。0.85 也低于一次启动给出的 0.8524 阈值。 | 8 卡最终改为 `--max-total-tokens 3072 --mem-fraction-static 0.86`，并把 W/E 提示词压到约 2500/2800 tokens。 |
| HCCL 缓冲区不足 | 默认 `HCCL_BUFFSIZE=200MB` 时日志要求 `NEEDED_HCCL_BUFFSIZE ... = 213MB`；改为 256MB 后，较大 batch 又要求 `333MB`。 | 最终设置 `HCCL_BUFFSIZE=512`。 |

8 卡成功日志摘录：

```text
384:[2026-10-23 07:45:41 TP0 EP0] max_total_num_tokens=3072, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=1536, context_len=1048576, available_gpu_mem=2.96 GB
582:[2026-10-23 07:46:05 TP5 EP5] HiCache AscendC kv_exchange_copy D2H: layout=layer_first_kv_split pages=19 layers=[0,78) components=3 (k,v,index_k)
618:[2026-10-23 07:46:29 TP0 EP0] HiCache AscendC kv_exchange_copy H2D: layout=layer_first_kv_split pages=19 layers=[0,1) components=3 (k,v,index_k)
```

最终 8 卡运行记录：W/R 均得到 `48219`，E 完成，D2H 32、H2D 624、`(k,v,index_k)` 656，脚本输出 PASS。此结论只针对该缩小后的 L1 和提示长度；此前高 token 配置失败属于资源约束，不等价于功能逻辑失败。

## 4. TTFT 性能结果：当前未达目标

脚本：`bench_glm52_layout_ttft_16npu.sh` 与 `.py`。每个布局使用相同的 16 卡 TP/EP、模型、L1=8192、L2 ratio=2.5；每轮 2 条约 3000-token 提示词并发，预热 1 轮，测量 5 轮，共 10 条测量请求。TTFT 是服务端流式响应中的第一个**非空文本 token**到达时间。每个测量循环都记录 H2D 标记，并断言 2/2 精确前缀重放及目标数字存在。正序运行 `layer_first` 后 `page_first_kv_split`；反序反过来。

| 顺序 | 布局 | mean | median | p95 |
| --- | --- | ---: | ---: | ---: |
| 正序 | `layer_first` | 1146.6 ms | 1159.0 ms | 1407.2 ms |
| 正序 | `page_first_kv_split` | 1141.5 ms | 1144.4 ms | 1352.7 ms |
| 反序 | `layer_first` | 1164.6 ms | 1165.0 ms | 1435.6 ms |
| 反序 | `page_first_kv_split` | 1141.7 ms | 1150.7 ms | 1354.2 ms |

两份 `comparison.log` 的关键输出：

```text
# 正序
layer_first vs page_first mean: +0.4%
layer_first vs page_first median: +1.3%
layer_first vs page_first p95: +4.0%
TTFT non-regression (median and p95): FAIL

# 反序
layer_first vs page_first mean: +2.0%
layer_first vs page_first median: +1.2%
layer_first vs page_first p95: +6.0%
TTFT non-regression (median and p95): FAIL
```

每布局每轮服务日志 D2H=384、测量阶段 H2D=12480；测量循环内 2/2 精确重放。预热循环曾出现 1/2 精确重放，未计入统计。两轮样本量较小，当前可以判定**这两轮没有满足既定 TTFT 不回退门槛**，暂不能判断慢在哪里或推算生产负载的差距。后续应在固定并发、提示长度、缓存命中率和机器负载下扩大样本，结合内核耗时/等待时间剖析原因。

## 5. 复现与证据位置

所有命令均在服务器的 `sglang-lzx-B103` 容器中执行；宿主机上用 `docker exec` 启动：

```bash
# 8 卡往返（物理 8–15）
docker exec -e NPU_DEVICES=8,9,10,11,12,13,14,15 -e PORT=32346 sglang-lzx-B103 \
  bash /home/l00889655/sglang/sglang/test/manual/hicache/e2e_layer_first_glm52_indexer_roundtrip.sh

# 16 卡往返
docker exec sglang-lzx-B103 \
  bash /home/l00889655/sglang/sglang/test/manual/hicache/e2e_layer_first_glm52_indexer_roundtrip_16npu.sh

# 16 卡 GSM8K
docker exec sglang-lzx-B103 \
  bash /home/l00889655/sglang/sglang/test/manual/hicache/e2e_layer_first_glm52_indexer_gsm8k_16npu.sh

# TTFT 正序；反序添加 -e RUN_ORDER=page_first_first 和单独的 LOG_DIR
docker exec sglang-lzx-B103 \
  bash /home/l00889655/sglang/sglang/test/manual/hicache/bench_glm52_layout_ttft_16npu.sh
```

归档目录：`/home/l00889655/sglang/evidence_glm52_layer_first_20261023/`。

- `8npu/`：最终成功日志及 0.75 内存阈值、无 DeepEP、16K/6K/HCCL 等失败日志。
- `16npu/roundtrip_server.log`：16 卡往返服务日志。
- `16npu_gsm8k/gsm8k_eval.log`、`gsm8k_server.log`：准确率和服务日志。
- `scripts/`：运行时脚本及共享评估程序快照。
- `ttft/`：两种顺序的比较结果、JSON 样本和脚本快照。
- `SHA256SUMS`：归档文件校验值。

本文件摘录关键证据；归档日志用于进一步核对。运行原路径的日志可能被后续测试覆盖，复查时以该归档为准。
