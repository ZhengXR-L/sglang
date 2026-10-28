# DSV4 HiCache 布局 TTFT 对比复现

`scripts/run_compare.py` 在同一组八张 NPU 上按 `layer_first → page_first → page_first → layer_first` 启动四次独立服务。`scripts/bench.py` 准备固定输入、执行 A（重算）→ B（65536 token 驱逐）→ R（原输入 L2 回迁），并生成报告；`scripts/run_server.sh` 保存服务参数。每种长度先预热一次，再测五次。R 必须 L1=0、L2 命中目标长度、A/R 输出 token 完全相同，才纳入 TTFT 统计。历史数据在 `report.md`。

## 在原环境运行

在 `sglang-lzx-B103` 容器内确认八卡空闲、端口 `32430` 未占用，且源计划 `/home/l00889655/sglang/gsm8k_aiv_npu_20261027_100_01/plan_100.json` 存在：

```bash
HERE=/home/l00889655/sglang/sglang/dsv4_ttft_performance
RUN=/home/l00889655/sglang/dsv4_ttft_$(date +%Y%m%d_%H%M%S)
python3 -u "$HERE/scripts/run_compare.py" "$RUN" 0,1,2,3,4,5,6,7 2>&1 | tee "${RUN}_controller.log"
```

第二个参数是八个不同的空闲物理 NPU 编号；RUN 必须尚不存在。控制器默认等待卡空闲；仅在已独占这些卡时设置 `DSV4_PERF_START_IMMEDIATELY=1`。测试期间不要往专用端口发其他请求。

## 迁移到另一台服务器

准备兼容的 Ascend 驱动/工具链、SGLang DSV4 分支、`sgl_kernel_npu`/Memfabric、模型权重及源 AIV 计划。`bench.py prepare` 读取源计划的 `questions[i].input_ids`、`qid`、`model_path`，至少需要六题，再构造 8192/32768/61440 token 的输入；源计划必须与服务使用同一模型 tokenizer。`run_compare.py` 中的仓库路径、源计划路径和端口 `32430` 是写死值；`run_server.sh` 的默认模型路径、Ascend 环境脚本路径和端口也需核对。迁移时同步修改，并确保 `SGLANG_ROOT` 指向包含 `python/sglang` 的仓库。可设置 `MODEL_PATH` 覆盖服务模型，但必须与计划里的 `model_path` 完全一致。

为保持可比性，使用 TP8/EP8/attention-DP8、modelslim、BF16 KV、`kernel_ascend`、write-through、HiCache ratio 2.5、相同输入计划和轮次顺序。`layer_first` 是 `layer_first_kv_split + AIV`，`page_first` 是 `page_first_direct` 原有路径；这是实际实现组合的比较，不能解释成仅改了内存排列。

## 结果与边界

查看新 RUN 的 `manifest.json`、`run_status.json`、四轮 `results/summary.json`、逐请求响应与生成的 `report.md`。四轮都完成且每轮三种长度各有 1 次预热与 5 次有效样本，才能引用平均值、p50/p95。TTFT 定义为请求发出至首个带 `output_ids` 的流事件。此脚本为性能测试，关闭 AIV trace、未装逐层 event 审计；若要证明各层 D2H/H2D 完成，应另跑精度审计测试。
