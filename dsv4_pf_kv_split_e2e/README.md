# DeepSeek-V4-Flash page_first_kv_split AIV 双向验证

运行所用 SGLang 和 sgl-kernel-npu 源码需支持 NPU DSV4 的
`page_first_kv_split` 布局及其 AIV 双向搬运算子。默认模型
`/mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp` 有 tokenizer 和索引列出的全部
70 个 W8A8 权重分片。服务采用 TP8/EP8/attention-DP8、`modelslim`、BF16 KV、
`--hicache-mem-layout page_first`、`kernel_ascend`、`write_through`、Memfabric。
`page_first` 解析为 `page_first_kv_split`。

## 运行

在 `sglang-lzx-B103` 容器中执行；输出目录应尚不存在，前八张物理 NPU 和端口
32420 应空闲。脚本启动专用服务，结束时关闭该服务。

```bash
cd /home/l00889655/sglang/dsv4_pf_kv_split_e2e
export MODEL_PATH=/mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp
export SGLANG_ROOT=/home/l00889655/sglang/sglang
export KERNEL_ROOT=/home/l00889655/sglang/sgl-kernel-npu
export TRAIN_PATH=/home/l00889655/sglang/gsm8k_train.jsonl
export TEST_PATH=/home/l00889655/sglang/gsm8k_test.jsonl
export PORT=32420
export DP_RANK=0
export EVICTION_TOKENS=65536
export STARTUP_TIMEOUT=3600
python3 -u run_e2e.py /home/l00889655/sglang/dsv4_pf_aiv_50_front8_w8a8mtp_20261102 0,1,2,3,4,5,6,7 50
```

先跑 3 题冒烟时，把最后的 `50` 改为 `3`，并使用另一个新输出目录。
`run_server.sh` 包含完整服务启动命令和 Ascend 环境变量。它调用的最终服务参数为：

```bash
python3 -m sglang.launch_server \
  --model-path /mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp \
  --host 127.0.0.1 --port 32420 --trust-remote-code \
  --tp-size 8 --ep-size 8 --dp-size 8 --enable-dp-attention --enable-dp-lm-head \
  --device npu --attention-backend dsv4 --quantization modelslim \
  --kv-cache-dtype bfloat16 --page-size 128 --swa-full-tokens-ratio 0.1 \
  --max-total-tokens 66000 --max-running-requests 16 \
  --chunked-prefill-size 32768 --mem-fraction-static 0.8 \
  --moe-a2a-backend deepep --deepep-mode auto --watchdog-timeout 9000 \
  --random-seed 0 --context-length 1048576 \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled \
  --enable-hicache --hicache-mem-layout page_first \
  --hicache-io-backend kernel_ascend --hicache-write-policy write_through \
  --hicache-ratio 2.5
```

## 验证方法

每题固定相同 input IDs，执行 `flush_cache → A → B → R`。A 冷生成触发
D2H；B 使用不同的 65536 token 前缀驱逐 L1；R 重放 A，要求 L1 命中 0、
L2 至少命中 2048 token。私有审计 hook 包装 DSV4 七个 host pool 的真实算子：
D2H 为 `transfer_kv_all_layer_kv_split_lf_pf`，H2D 为
`transfer_kv_per_layer_kv_split_pf_lf`。只有显式 `impl="aiv"`、返回后 NPU
stream Event 完成，才计入该池该层覆盖。逐题要求七池全部物理层双向覆盖，
并检查 A/R 答案和真值。Event 证明算子完成，不是 KV/state 逐字节比对。

结果目录包含 `report.md`、`results_50/summary.json`、逐题 verdict、
`audit/calls-*.jsonl`、服务日志和冻结的脚本副本。只审计指定 DP rank，默认 0。
