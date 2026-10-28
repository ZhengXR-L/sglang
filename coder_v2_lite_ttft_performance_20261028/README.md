# DeepSeek-Coder-V2-Lite-Instruct HiCache TTFT 对比复现

本目录的 `scripts/run_compare.py` 连续启动四次独立服务，按 `layer_first → page_first → page_first → layer_first` 比较同一张 NPU 上的 L2 回迁 TTFT。`scripts/bench.py` 负责生成计划、逐轮测量和汇总；`scripts/run_server.sh` 是服务配置；`scripts/launch.sh` 是原服务器的快捷入口；历史结果见 `report.md`。每个前缀长度先预热一次，再测五次，A 是冷态重算，B 是驱逐请求，R 是原输入的 L2 重放。脚本要求 R 的 L1 为 0、L2 命中指定长度且 A/R 输出 token 全等；TTFT 是请求发送至首个非空 `output_ids` 事件的时间。

## 在原环境运行

在 `sglang-lzx-B103` 容器内，确认单张 NPU 空闲、端口 `32431` 未占用，且源计划 `/home/l00889655/sglang/coder_mla_aiv_100_20261027_211414/plan.json` 存在。然后执行：

```bash
HERE=/home/l00889655/sglang/sglang/coder_v2_lite_ttft_performance_20261028
RUN=/home/l00889655/sglang/coder_v2_lite_ttft_$(date +%Y%m%d_%H%M%S)
python3 -u "$HERE/scripts/run_compare.py" "$RUN" 10 2>&1 | tee "${RUN}_controller.log"
```

第二个参数是单张物理 NPU 编号，示例 `10` 仅适用于该卡空闲时。RUN 必须尚不存在。默认等待卡空闲；只在已独占目标卡时才设置 `DSV4_PERF_START_IMMEDIATELY=1`。快捷脚本 `bash "$HERE/scripts/launch.sh" "$RUN"` 固定使用 NPU 10，并把控制日志和退出码写到 RUN 旁边。

## 换服务器前检查和修改

需要 Ascend 驱动/工具链、可用单卡、兼容 SGLang、`sgl_kernel_npu`/Memfabric、模型权重，以及含至少六题的旧 AIV `plan.json`。`bench.py prepare` 从源计划读取 `questions[i].input_ids`、`qid` 和 `model_path`，按固定规则生成 1024/4096/8192 token 的测量输入；不能只复制本目录脚本而不准备源计划。若重新生成源计划，须保持同一模型 tokenizer 和输入协议，记录新旧计划不可直接比较。

当前 `run_compare.py` 写死仓库路径、源计划路径和端口 `32431`；`run_server.sh` 写死模型路径及 `/usr/local/Ascend/...` 环境脚本；`launch.sh` 写死脚本路径和 NPU 10。迁移时先把这些路径与端口改成新服务器值，确保服务模型与源计划 `model_path` 完全一致，并使 `SGLANG_ROOT` 指向含 `python/sglang` 的仓库根目录。保留 TP1、HiCache ratio 2.0、`kernel_ascend`、write-through、池容量和测量顺序；改配置需另记实验。

## 结果核对

新 RUN 包含 `manifest.json`、`plan.json`、四个 `1_layer_first` 等轮次目录、`run_status.json` 和汇总 `report.md`。每轮 `results/summary.json` 与原始请求/响应可复查样本；`stage.json`、`server.log`、`bench.log` 用于定位失败。确认四轮完成、每轮三种长度各 1 次预热 + 5 次测量且路径/输出校验通过，再引用 TTFT 汇总。此性能测试没有逐层 AIV event 审计，不能用它单独证明 D2H/H2D 每层完成。
