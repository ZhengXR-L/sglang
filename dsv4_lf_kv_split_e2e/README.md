# DeepSeek-V4-Flash 的 DSV4 HiCache AIV 双向验证

本目录针对远端 SGLang `cjz_op` 与 sgl-kernel-npu `npu/hicache-sparse-copy`。
当前 `cjz_op` 原有 NPU 参数解析会把 DSV4 的 `layer_first` 改为
`page_first_direct`；已应用本目录的 `npu_dsv4_layout.patch`，令 DSV4
保留 `layer_first_kv_split`。运行前需确保这处补丁仍在工作区。
默认权重为 `/mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp`，该目录的
tokenizer、索引和 70 个 W8A8 权重分片已核对齐全，服务使用 `modelslim` 加载。
原 W8A8 路径 `/home/weights/DeepSeek-V4-Flash-0731-w8a8` 是失效软连接，
目标 `/mnt/paas/weights/linked_to_home/DeepSeek-V4-Flash-0731-w8a8` 已不存在。
`/mnt/paas/weights/DeepSeek-V4-Flash-0731-w8a8` 是另一个仍在下载中的目录，本测试不使用。
曾尝试使用 `/mnt/paas/weights/DeepSeek-V4-Flash-0731/DeepSeek-V4-Flash-0731`
的 FP8/FP4 expert 权重：直接加载报芯片不支持 `customize_dtype`，
启用 `SGLANG_DSV4_FP4_DEQUANT=1` 后又在 FP8 转换时报 NPU 561103，均未进入请求阶段。

## 运行方式

在 `sglang-lzx-B103` 容器中执行。先用 `npu-smi info` 确认八张物理芯片空闲，
输出目录必须不存在。先跑 3 题冒烟，再将第三个参数改为 `100` 做完整样本；
100 题使用一个新目录，脚本自行启动并关闭专用服务。

```bash
docker exec -it sglang-lzx-B103 bash
cd /home/l00889655/sglang/dsv4_lf_kv_split_e2e
npu-smi info
export PORT=32420
export STARTUP_TIMEOUT=3600
export MODEL_PATH=/mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp
export TRAIN_PATH=/home/l00889655/sglang/gsm8k_train.jsonl
export TEST_PATH=/home/l00889655/sglang/gsm8k_test.jsonl
export SGLANG_ROOT=/home/l00889655/sglang/sglang
export KERNEL_ROOT=/home/l00889655/sglang/sgl-kernel-npu
export DP_RANK=0
export EVICTION_TOKENS=65536
python3 -u run_e2e.py /home/l00889655/sglang/dsv4_lf_aiv_smoke_001 8,9,10,11,12,13,14,15 3
```

若要在其他卡上运行，替换八个物理芯片 ID；测试使用 `ASCEND_RT_VISIBLE_DEVICES`，
不会改动 NPU 验签配置。运行前确认端口 32420 空闲。

```bash
cat /home/l00889655/sglang/dsv4_lf_aiv_smoke_001/stage.json
cat /home/l00889655/sglang/dsv4_lf_aiv_smoke_001/results_3/summary.json
cat /home/l00889655/sglang/dsv4_lf_aiv_smoke_001/report.md
```

## 测试原理

`gsm8k_aiv.py plan` 从独立的 GSM8K train/test JSONL 中固定取测试集第 5 题起的
3 或 100 题，用训练集示例构造至少 8192 token 的真实 few-shot 提示。
每题保存完整 input IDs、真值和哈希，A 与 R 使用完全相同的 input IDs。

每题串行执行 `flush_cache → A → B → R`：A 是冷 KV 缓存生成，
`write_through` 触发 D2H；B 使用不同前缀、65536 token 的请求驱逐 L1；
R 重放 A，要求 L1 命中 0、L2 命中至少 2048 token，触发 H2D。
请求固定路由到 `DP_RANK`（默认 0），独占专用服务，避免其他请求污染阶段审计。

私有 `sitecustomize` 在 SGLang worker 中包装当前分支的
`DeepSeekV4PagedHostPool` 与 `DeepSeekV4StateHostPool`。每次真正调用
`transfer_kv_all_layer_kv_split` 或 `transfer_kv_per_layer_kv_split` 时，
要求 `layout=layer_first_kv_split` 且 `impl="aiv"`；算子返回后在当前 NPU stream
记录 Event，Event 完成才写入 `aiv_complete`。逐题要求所选 DP rank 的 SWA、
C4 KV、C4 Indexer K、Indexer scale、C4 state、Indexer state、C128 七个池，
各物理层在 A 阶段 D2H 和 R 阶段 H2D 都完成。缺池、缺层、异常或超时即失败。
再比较 A/R 的数值答案和真值；原始 token ID 一致率单独报告，不作默认硬门槛。
Event 证明算子及所在 stream 完成，不是 KV/state 字节级校验。

## 服务启动参数

`run_server.sh` 加载 Ascend customize、custom_transformer、toolkit 和 ATB 环境。
它导出 `ASCEND_RT_VISIBLE_DEVICES`、`SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric`、
`PYTHONPATH`、`HCCL_BUFFSIZE=2048` 等已用于旧 DSV4 测试的 NPU 环境变量。
对于默认 W8A8 模型，脚本设置 `SGLANG_DSV4_FP4_EXPERTS=False` 并传入
`--quantization modelslim`；
具体完整列表见脚本。`gsm8k_aiv.py serve` 再将审计 hook 目录放在
`PYTHONPATH` 最前，并记录 `audit/launch.json`。最终服务命令为：

```bash
python3 -m sglang.launch_server \
  --model-path /mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp \
  --host 127.0.0.1 --port 32420 --trust-remote-code \
  --tp-size 8 --ep-size 8 --dp-size 8 --enable-dp-attention --enable-dp-lm-head \
  --device npu --attention-backend dsv4 --quantization modelslim --kv-cache-dtype bfloat16 \
  --page-size 128 --swa-full-tokens-ratio 0.1 --max-total-tokens 66000 \
  --max-running-requests 16 --chunked-prefill-size 32768 --mem-fraction-static 0.8 \
  --moe-a2a-backend deepep --deepep-mode auto --watchdog-timeout 9000 --random-seed 0 \
  --context-length 1048576 --cuda-graph-backend-prefill disabled \
  --cuda-graph-backend-decode disabled \
  --enable-hicache --hicache-mem-layout layer_first \
  --hicache-io-backend kernel_ascend --hicache-write-policy write_through \
  --hicache-ratio 2.5
```

`layer_first` 在 NPU DSV4 host pool 中解析为 `layer_first_kv_split`。
`--quantization modelslim` 用于当前 W8A8 MTP 权重。
此脚本先覆盖一个指定 DP rank 的七池双向路径；它不宣称 8 个 DP rank
都被逐题验证，也不评价并发和性能。

## 前八卡实测

2026-11-02 在物理卡 `0,1,2,3,4,5,6,7` 运行 3 题，结果位于
`/home/l00889655/sglang/dsv4_lf_aiv_smoke_front8_w8a8mtp_layout_20261102`。
3/3 题覆盖七池全部物理层的 AIV D2H/H2D，L1 重放命中 0、L2 每题命中
8192 token，3/3 的 A/R 答案一致且与真值一致。详细统计见该目录 `report.md`。

输出目录包含 `server.log`、`eval.log`、`audit/calls-*.jsonl`、
逐题请求与响应、`results_<题数>/summary.json`、`run_status.json` 和 `report.md`。
