# DSV4 HiCache 布局 TTFT 对比

比较实际实现组合：layer_first_kv_split + AIV 与 page_first_direct + 原有传输路径。

固定 TP8/EP8/attention-DP8、routed_dp_rank=0、并发 1；同一组 NPU，按 layer/page/page/layer 顺序。

每场景输入为缓存前缀长度 +128 token；A 重算，B=65536 token 驱逐，R 原输入回迁。A/R 输出16 token，B输出8 token，temperature=0、ignore_eos=true。

每种长度每次服务先预热一轮（i0，不计时统计），再测五轮。TTFT 从请求发送到首个含 output_ids 的事件；同时保存首个非空文本时间。

强制 A 无缓存、R 的 L1=0 且 L2 命中指定长度、A/R 输出 IDs 全等；任一失败中止该次比较，不能剔除后继续报通过。

不加载逐调用 event/JSONL audit hook。使用 warning 日志，双方禁用 AIV trace；计时没有逐层日志 I/O。缓存写入可能与计算重叠，A TTFT 不等于 D2H 总耗时，R TTFT 也不等于纯算子耗时。

## 结果

| L2 前缀 | layer 均值 ms | page 均值 ms | layer p50 | page p50 | layer p95 | page p95 | layer 相对变化 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 447.34 | 561.28 | 450.37 | 553.93 | 454.75 | 616.66 | -20.30% |
| 32768 | 483.47 | 806.07 | 484.01 | 797.52 | 495.36 | 884.12 | -40.02% |
| 61440 | 522.45 | 1116.66 | 522.71 | 1077.32 | 534.24 | 1387.67 | -53.21% |

完整四次服务比较：True。负的相对变化表示 layer 更快。

## 各次服务结果

| 次序 | 布局 | L2 前缀 | 样本数 | R TTFT 均值 ms |
|---:|---|---:|---:|---:|
| 1_layer_first | layer_first | 8192 | 5 | 448.86 |
| 1_layer_first | layer_first | 32768 | 5 | 483.62 |
| 1_layer_first | layer_first | 61440 | 5 | 523.55 |
| 2_page_first | page_first | 8192 | 5 | 551.67 |
| 2_page_first | page_first | 32768 | 5 | 793.66 |
| 2_page_first | page_first | 61440 | 5 | 1158.94 |
| 3_page_first | page_first | 8192 | 5 | 570.88 |
| 3_page_first | page_first | 32768 | 5 | 818.49 |
| 3_page_first | page_first | 61440 | 5 | 1074.39 |
| 4_layer_first | layer_first | 8192 | 5 | 445.81 |
| 4_layer_first | layer_first | 32768 | 5 | 483.32 |
| 4_layer_first | layer_first | 61440 | 5 | 521.36 |

## 参数与证据

```json
{
  "start_utc": "2026-10-27T10:26:13.433501+00:00",
  "devices": "0,1,2,3,4,5,6,7",
  "source_plan": "/home/l00889655/sglang/gsm8k_aiv_npu_20261027_100_01/plan_100.json",
  "commit": "356c078865ae63511685837082e1f1a49d697b62",
  "git_status": "?? test/manual/hicache/bench_dsv4_layout_ttft_8npu.py\n?? test/manual/hicache/bench_dsv4_layout_ttft_8npu.sh\n?? test/manual/hicache/e2e_layer_first_dsv4_gsm8k.sh\n",
  "script_sha256": {
    "run_server.sh": "f32cd187cc64ccf5bddf6bd1ae623f7fe7e97c0284ad854a537d5e76cd323892",
    "bench.py": "9dd680c8227c33026413b882686b863b4505739c9da5916c4b6eea383d51fc72",
    "run_compare.py": "d54665158bd3a57875f7888521c9e57945f21496f6201c541d2c96baa8970fd4"
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

源配置和权重不变，变化同时包括 host 布局及其对应传输实现，因此不能把差异全部归因于内存布局。这里只覆盖并发1；共享服务器负载变化、测量样本数限制了对其他工作负载的外推。

此前同提交的100题功能测试已证明layer路径双向AIV完成；本轮计时不逐调用观测算子，当前通路依据实际解析布局及冻结源码分支确定。

## 测试原理与设计目的

### 要回答的问题与计时口径

本测试比较两套实际实现组合：`layer_first_kv_split` + AIV，以及 `page_first_direct` + 原有传输路径。当请求的长前缀已写入主机侧 HiCache、但不在 NPU 缓存中时，比较两者让首个输出 token 到达的时间。报告中的布局统计使用回迁请求 R 的 TTFT；A、B 的 TTFT 也逐样本记录，但不参与布局均值。

客户端在发出流式 `/generate` 请求前启动单调时钟，到收到第一个包含非空 `output_ids` 的流事件时停止 TTFT 计时；另行记录首个非空文本出现的时间。R 的 TTFT 包含请求处理、调度、主机缓存读取、剩余计算、生成和首个流事件返回，并非纯 KV 传输耗时。

### 单条样本的 A → B → R 流程

每条样本开始前调用 `/flush_cache`，随后把三个请求串行发送到同一服务，均固定 `routed_dp_rank=0`、`temperature=0`、`ignore_eos=true`。A 与 R 使用相同输入，输出上限为 16 token；B 的输出上限为 8 token。

1. **A：冷算并建立缓存。** 输入由目标长度的自然文本前缀加 128 token 尾部构成。脚本要求 A 的 `cached_tokens=0`，以排除已有缓存命中。A 完成后两种布局均等待 0.5 秒，给待完成的 `write_through` 工作相同的缓冲时间。
2. **B：驱逐设备侧前缀。** 发送独立生成的 65,536 token 序列，占用缓存空间，使随后 R 需要从主机侧取回 A 的前缀。B 用于构造缓存状态，不是主要比较指标。
3. **R：重放 A 的原输入。** 脚本要求 R 的设备侧命中数为 0，主机侧命中数恰好等于目标前缀长度（8,192、32,768 或 61,440 token）。这验证测量对象是 L2 主机缓存回迁，而非意外的 L1 设备缓存命中。
4. **正确性校验。** A 与 R 的最终输出 token ID 必须完全一致；空输出、全零输出、错误的 DP rank、缓存层级或命中长度都会使测试失败，不会作为有效样本计入结果。

每种前缀长度在每次服务中先运行一次预热样本（`i0`），随后运行五次正式样本（`i1`–`i5`）。因此每轮日志有 3 × 6 = 18 条记录，其中 15 条参与统计；四轮总计 72 条记录、60 条正式样本。同一布局的两轮合并后，每种长度有 10 条正式样本。前缀长度覆盖约 8K、32K、60K，用于观察回迁规模增加时 TTFT 差距如何变化。

### 四轮 ABBA 顺序的目的

| 轮次 | 布局与传输路径 | 用途 |
|---|---|---|
| 1 | `layer_first_kv_split` + AIV | 第一轮 layer 测量 |
| 2 | `page_first_direct` + 原有传输路径 | 第一轮 page 测量 |
| 3 | `page_first_direct` + 原有传输路径 | 第二轮 page 测量 |
| 4 | `layer_first_kv_split` + AIV | 第二轮 layer 测量 |

每轮都在同一组 8 张 NPU 上重新启动服务，使用相同模型、输入计划与服务参数。按 A–B–B–A 排列，使两种实现的测试时间中心相同：第一、四轮的位置平均为 2.5，第二、三轮的位置平均也为 2.5。若环境性能随时间近似线性漂移，合并两轮可减弱这种漂移对布局差值的影响。独立重启还能检验差异是否依赖某一次服务实例；此设计不能消除任意形态的共享机器负载波动。

脚本在各轮开始前检查 Git HEAD 未变化；每轮检查实际解析的 HiCache 布局、`kernel_ascend` 后端、TP8、attention DP8、`write_through`、流式间隔、模型路径和每 worker 的实际 token 容量。它保存脚本副本与 SHA-256、服务信息、环境、NPU 快照以及逐样本的 A/B/R 请求和响应，便于复核测试条件。

### 结果的解释范围

本次四轮均完成。两次 layer 服务在 61,440 token 下的 R TTFT 均值分别为 523.55 ms 和 521.36 ms；两次 page 服务分别为 1158.94 ms 和 1074.39 ms。合并正式样本后，layer 相对 page 的平均 R TTFT 在 8,192、32,768、61,440 token 下分别降低 20.30%、40.02%、53.21%。

比较同时改变了主机缓存布局及其对应的传输实现，因此这些数字代表两套实际实现组合的端到端差异，不能将全部差距单独归因于内存布局。本测试固定 TP8、attention DP8、同一组 NPU、`routed_dp_rank=0` 和单并发；结果不直接代表多并发或其他工作负载。A 的 TTFT 不等于完整 D2H 写入耗时，R 的 TTFT 也不等于纯回迁算子耗时。
