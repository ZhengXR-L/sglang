# GLM-5.2 两种 KV split 布局的 L2 回迁 TTFT 对比

本用例在同一组八张 NPU 上按 `layer_first → page_first → page_first → layer_first`
顺序启动四次独立服务。两种启动参数仅 `--hicache-mem-layout` 不同；
当前源码分别解析为 `layer_first_kv_split` 与 `page_first_kv_split`。
两种布局的 D2H/H2D 都显式选择 AIV，因此报告比较的是两条实际服务路径。
服务沿用 GLM-5.2 W4A8C8 的
TP8/EP8、Ascend attention、Memfabric、`kernel_ascend`、`write_through`、
HiCache ratio 2.5 和 3072 token 的配置。

## 运行

在能访问模型和八张空闲 NPU 的容器内执行。输出目录须尚不存在；
脚本会等待指定 NPU 空闲，自行启动和关闭每轮服务。默认端口 32432。

```bash
cd /home/l00889655/sglang/sglang/glm52_ttft_compare
export SGLANG_ROOT=/home/l00889655/sglang/sglang
export KERNEL_ROOT=/home/l00889655/sglang/sgl-kernel-npu
export MODEL_PATH=/mnt/paas/weights/GLM-5.2-w4a8c8
export PORT=32432
export STARTUP_TIMEOUT=2400
python3 -u run_compare.py /home/l00889655/sglang/glm52_split_ttft_back8_001 8,9,10,11,12,13,14,15
```

运行完检查输出目录的 `run_status.json`、`comparison.json` 和 `report.md`。
`run_status.json` 的 `error` 应为 `null`、`finished_blocks` 应为 4；
`comparison.json` 的 `complete` 应为 `true`。每轮的服务日志、
`server_info.json`、逐题 A/B/R 请求及响应、NPU 快照也都保存在输出目录。
输入计划由本目录 `bench.py prepare` 使用模型 tokenizer 和固定算术文本生成，
四轮共用同一份 `plan.json`，无需其他测试产生的文件。

## 测量与判定

每轮对目标前缀 512、1024、1536 token 分别预热一次、计时五次。
每次先清空缓存，再串行执行 A（冷计算）、B（2816 token 驱逐）、
R（重放 A）。用流式 `/generate` 的首个 `output_ids` 事件计算 R 的 TTFT。
样本只有在 A 无缓存、R 的设备缓存命中为 0、主机缓存命中至少 128 token、
且 A/R 都包含预设的正确数值时才有效；A/R 逐 token 相同率单独记录，
不作为硬门槛。任一轮失败即停止比较。

脚本在 `/server_info` 中检查实际布局、TP8/EP8、IO 后端、
`write_through`、模型路径和服务报告的 KV 容量。记录两个仓库的
Git 提交、工作区状态及脚本哈希。计时期间不加载逐调用审计钩子；
缓存层级信息证明回迁命中，但 TTFT 包含调度、计算和传输时间，
不能当作纯 H2D 算子耗时。与旧 `glm52_ttft_performance` 的双并发负载不同，
本用例采用单并发以逐请求校验缓存来源；两组绝对时间不直接比较。

## 2026-11-03 后八卡实测

完整四轮结果见 `report_20261103.md` 和 `comparison_20261103.json`，
原始请求、响应与服务日志保存在远端
`/home/l00889655/sglang/glm52_split_ttft_back8_20261103_aiv_v4`。
每种长度每种布局有 10 个正式样本。512、1024、1536 token 前缀的
R 请求平均 TTFT 分别为 layer 847.75/826.99/823.02 ms、
page 840.66/831.94/838.14 ms。A/R 逐 token 相同数两种布局各为 16/30；
全部样本的 A/R 数值答案均正确，R 均为纯 L2 命中。
