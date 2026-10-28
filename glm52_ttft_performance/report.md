# GLM-5.2 W4A8C8 kernel_ascend 布局 TTFT 对比

比较 layer_first_kv_split 与 page_first_kv_split 的实际服务路径。

TP8/EP8、NPU 4,5,6,7,8,9,14,15；并发 2，每请求约 1000 输入 token、最多 16 输出 token；KV 容量 3072 token，HiCache ratio 2.5，write_through。

每条测试采用 A 填充、B 驱逐、R 重放；每次服务预热一轮，再测五轮，两条并发请求每次服务产生 10 条正式 TTFT 样本。四次服务顺序为 layer/page/page/layer。

TTFT 从流式 /v1/completions 请求发送到首次非空文本。每轮要求 D2H/H2D 日志标记和预设答案一致；A/R 文本一致率单独报告。这些日志标记验证发生了传输，但不提供每请求精确的 L1/L2 命中 token 数。

## 结果

| 布局 | 样本数 | 平均 TTFT ms | 中位数 ms | p95 ms |
|---|---:|---:|---:|---:|
| layer_first | 20 | 1067.62 | 1068.52 | 1344.91 |
| page_first_kv_split | 20 | 1069.77 | 1067.30 | 1304.22 |

完整四轮比较：True。
layer 相对 page 的平均 TTFT 变化：-0.20%（负数表示更快）。

## 各轮结果

| 轮次 | 布局 | 平均 ms | D2H 标记 | H2D 标记 | A/R 文本一致 | 答案一致 |
|---|---|---:|---:|---:|---:|---:|
| 1_layer_first | layer_first | 1070.57 | 192 | 6240 | 7/10 | 10/10 |
| 2_page_first_kv_split | page_first_kv_split | 1084.14 | 192 | 6240 | 7/10 | 10/10 |
| 3_page_first_kv_split | page_first_kv_split | 1055.40 | 192 | 6240 | 8/10 | 10/10 |
| 4_layer_first | layer_first | 1064.68 | 192 | 6240 | 6/10 | 10/10 |

## 参数与证据

```json
{
  "start_utc": "2026-10-28T03:32:39.617573+00:00",
  "devices": "4,5,6,7,8,9,14,15",
  "model_path": "/mnt/paas/weights/GLM-5.2-w4a8c8",
  "commit": "183512c7f180befd45203d7783f3fd55c02fee47",
  "git_status": " D gsm8k_aiv_long_fewshot_100_20261027/README.md\n D gsm8k_aiv_long_fewshot_100_20261027/aiv_audit_hook.py\n D gsm8k_aiv_long_fewshot_100_20261027/gsm8k_aiv.py\n D gsm8k_aiv_long_fewshot_100_20261027/npu_report_100_20261027.md\n D gsm8k_aiv_long_fewshot_100_20261027/report_npu.py\n D gsm8k_aiv_long_fewshot_100_20261027/run_100.py\n D gsm8k_aiv_long_fewshot_100_20261027/run_server.sh\n D hicache_parameter_cause_20261026/analysis.md\n?? coder_v2_lite_aiv_accuracy_20261027/\n?? coder_v2_lite_ttft_performance_20261028/\n?? dsv4_gsm8k_aiv_long_fewshot_100_20261027/\n?? dsv4_hicache_parameter_cause_20261026/\n?? glm52_aiv_accuracy_20261027/\n?? glm52_ttft_performance/\n",
  "script_sha256": {
    "run_server.sh": "65ddaf8b7c7fb5cb0a49ed346a20fab95e82fb07bc3b97b268cb2cef35392ca6",
    "bench.py": "71825c06681b7d90de4da89b8e6786dbb3cbdb54c5b0ab3b4b5b44e6d73d502e",
    "run_compare.py": "ff707f14b9ee5c7529be6e38b0e02290f610ab177f4164d9888d9168750c16f4"
  },
  "order": [
    "layer_first",
    "page_first_kv_split",
    "page_first_kv_split",
    "layer_first"
  ],
  "blocks": [
    {
      "name": "1_layer_first",
      "layout": "layer_first"
    },
    {
      "name": "2_page_first_kv_split",
      "layout": "page_first_kv_split"
    },
    {
      "name": "3_page_first_kv_split",
      "layout": "page_first_kv_split"
    },
    {
      "name": "4_layer_first",
      "layout": "layer_first"
    }
  ]
}
```

逐轮请求结果、server_info、日志和 NPU 快照保存在源码目录外的运行目录：/home/l00889655/sglang/glm52_ttft_20261028_03。

本测试沿用早期 GLM 的双并发工作负载并按最新 TP8 精度测试配置缩短前缀；因此不能与此前 TP16、约 3000-token 前缀的绝对 TTFT 直接比较。共享服务器负载和有限样本数也限制了外推。
