# DeepSeek-Coder-V2-Lite-Instruct HiCache 布局 TTFT 对比

比较 kernel_ascend 的 layer_first_kv_split 与 page_first_kv_split。

固定 TP1、并发 1；同一张 NPU，按 layer/page/page/layer 顺序。

每场景输入为缓存前缀长度 +128 token；A 重算，B=10000 token 驱逐，R 原输入回迁。A/R 输出16 token，B输出8 token，temperature=0、ignore_eos=true。

每种长度每次服务先预热一轮（i0，不计时统计），再测五轮。TTFT 从请求发送到首个含 output_ids 的事件；同时保存首个非空文本时间。

强制 A 无缓存、R 的 L1=0 且 L2 命中指定长度、A/R 输出 IDs 全等；任一失败中止该次比较，不能剔除后继续报通过。

不加载逐调用 event/JSONL audit hook。使用 info 日志；计时没有逐层日志 I/O。B 请求可能因填满 KV 容量而不生成 token，是否有效由后续 R 的 L1=0、L2 精确命中及 A/R 输出一致共同判定。缓存写入可能与计算重叠，A TTFT 不等于 D2H 总耗时，R TTFT 也不等于纯算子耗时。

## 结果

| L2 前缀 | layer 均值 ms | page 均值 ms | layer p50 | page p50 | layer p95 | page p95 | layer 相对变化 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 107.13 | 105.66 | 106.80 | 104.45 | 112.76 | 122.03 | +1.40% |
| 4096 | 107.54 | 109.28 | 107.38 | 108.46 | 113.48 | 121.28 | -1.59% |
| 8192 | 112.16 | 110.15 | 111.30 | 107.26 | 123.28 | 126.74 | +1.83% |

完整四次服务比较：True。负的相对变化表示 layer 更快。

## 各次服务结果

| 次序 | 布局 | L2 前缀 | 样本数 | R TTFT 均值 ms |
|---:|---|---:|---:|---:|
| 1_layer_first | layer_first | 1024 | 5 | 107.82 |
| 1_layer_first | layer_first | 4096 | 5 | 110.20 |
| 1_layer_first | layer_first | 8192 | 5 | 113.71 |
| 2_page_first | page_first | 1024 | 5 | 106.47 |
| 2_page_first | page_first | 4096 | 5 | 106.67 |
| 2_page_first | page_first | 8192 | 5 | 108.68 |
| 3_page_first | page_first | 1024 | 5 | 104.85 |
| 3_page_first | page_first | 4096 | 5 | 111.89 |
| 3_page_first | page_first | 8192 | 5 | 111.61 |
| 4_layer_first | layer_first | 1024 | 5 | 106.45 |
| 4_layer_first | layer_first | 4096 | 5 | 104.87 |
| 4_layer_first | layer_first | 8192 | 5 | 110.61 |

## 参数与证据

```json
{
  "start_utc": "2026-10-28T02:26:32.543793+00:00",
  "devices": "10",
  "source_plan": "/home/l00889655/sglang/coder_mla_aiv_100_20261027_211414/plan.json",
  "commit": "183512c7f180befd45203d7783f3fd55c02fee47",
  "git_status": " D gsm8k_aiv_long_fewshot_100_20261027/README.md\n D gsm8k_aiv_long_fewshot_100_20261027/aiv_audit_hook.py\n D gsm8k_aiv_long_fewshot_100_20261027/gsm8k_aiv.py\n D gsm8k_aiv_long_fewshot_100_20261027/npu_report_100_20261027.md\n D gsm8k_aiv_long_fewshot_100_20261027/report_npu.py\n D gsm8k_aiv_long_fewshot_100_20261027/run_100.py\n D gsm8k_aiv_long_fewshot_100_20261027/run_server.sh\n D hicache_parameter_cause_20261026/analysis.md\n?? coder_v2_lite_ttft/\n?? deepseek_coder_v2_lite_aiv_accuracy_20261027/\n?? dsv4_gsm8k_aiv_long_fewshot_100_20261027/\n?? dsv4_hicache_parameter_cause_20261026/\n?? glm52_aiv_accuracy_20261027/\n",
  "script_sha256": {
    "run_server.sh": "f0f55833164db756e1dc444becc782c6bdd2107296dfcccb69e82a4d585a557e",
    "bench.py": "15bcb1bf6ceb959b3b16ea5727dccc45b6b0bc43daf380f553eac15242361b35",
    "run_compare.py": "81e7ee54c5a22f5f54b878c32de4bd241c827281b4aa95f3c736a39adf680f3c"
  },
  "order": [
    "layer_first",
    "page_first",
    "page_first",
    "layer_first"
  ],
  "blocks": [
    {
      "name": "1_layer_first",
      "layout": "layer_first"
    },
    {
      "name": "2_page_first",
      "layout": "page_first"
    },
    {
      "name": "3_page_first",
      "layout": "page_first"
    },
    {
      "name": "4_layer_first",
      "layout": "layer_first"
    }
  ]
}
```

每次服务的 server_info.json、environment.json、server.log，以及每样本 A/B/R 完整请求/响应均在同目录保存。

源配置和权重不变，比较两种 host 布局的实际 kernel_ascend 路径。这里只覆盖并发1；共享服务器负载变化、测量样本数限制了对其他工作负载的外推。

本轮计时不逐调用观测算子；当前通路依据服务实际解析布局和源码版本确定。
