# GLM-5.2 W4A8C8 HiCache 布局 TTFT 测试

本目录只保存执行脚本和最终报告。逐轮请求结果、服务日志、NPU 快照等放在源码目录外。`report.md` 是 2026-10-28 在当前服务器上完成的一次 TP8 对比结果；换服务器后应重新运行，不能直接沿用其中的数值。

## 测什么

比较 `kernel_ascend` 下的 `layer_first_kv_split` 和 `page_first_kv_split`。控制器在同一组 8 张 NPU 上依次重新启动四次服务，顺序为 layer/page/page/layer。服务固定 TP8/EP8、`write_through`、HiCache ratio 2.5、KV 容量 3072 token。每次服务先预热一轮，再测五轮；每轮并发两个约 1000 输入 token 的请求，每种布局合计 20 条正式 TTFT 样本。

一轮请求按 A（填充缓存）→ B（用不同提示驱逐设备缓存）→ R（重放 A）执行。TTFT 是客户端发送流式 `/v1/completions` 请求到收到首段非空文本的时间。脚本要求 D2H/H2D 日志标记和预设答案一致，另行记录 A/R 文本完全一致数。日志标记可证明发生传输，但**不能**给出每条请求精确的 L1/L2 命中 token 数；报告也不代表纯传输耗时。

## 在当前服务器上运行

在能访问 8 张 Ascend NPU、模型权重及本仓库的 `sglang-lzx-B103` 容器中执行。先用 `npu-smi info` 确认设备编号；以下编号是本次运行采用的物理 NPU ID，按实际空闲情况替换。运行目录必须是**尚不存在**、可写且位于源码目录外的路径。

```bash
PERF=/home/l00889655/sglang/sglang/glm52_ttft_performance
RUN=/home/l00889655/sglang/glm52_ttft_$(date +%Y%m%d_%H%M%S)
DEVICES=4,5,6,7,8,9,14,15
bash "$PERF/scripts/launch.sh" "$RUN" "$DEVICES"
```

`launch.sh` 把控制器输出写入 `${RUN}_controller.log`，退出码写入 `${RUN}_exit_code`。它自身只负责启动和记录退出码，因此以退出码文件、`run_status.json` 和 `comparison.json` 判断结果：

```bash
cat "${RUN}_exit_code"               # 期望 0
cat "$RUN/run_status.json"           # error 应为 null，finished_blocks 应为 4
cat "$RUN/comparison.json"           # complete 应为 true，每种布局 n=20
cat "$RUN/report.md"
```

控制器会在每轮启动前等待选定的 8 张 NPU 全部空闲，最长 12 小时；不会主动结束其他进程。每轮会保存 `server_info.json` 并检查实际 TP 数、模型路径、布局及 `kernel_ascend` 后端。若失败，先看 `$RUN/stage.json`、`${RUN}_controller.log`，再看对应轮次的 `server.log` 和 `bench.log`。失败或不完整的运行目录不能当作完整比较。

确认报告无误后，仅把最终报告复制到本目录；原始证据继续留在源码目录外：

```bash
cp "$RUN/report.md" "$PERF/report.md"
```

## 换到另一台服务器

先在新服务器的容器中准备同一版 SGLang 源码、可用的 Ascend/CANN 与 `torch_npu` 环境、`transformers`、`npu-smi`、`memfabric` 后端，以及 GLM-5.2 W4A8C8 权重。确认容器能看到 8 张完全空闲的物理 NPU，且这些设备的剩余显存足以加载 TP8 模型。端口 32432 也必须空闲。

复制本目录后，检查并修改以下**当前服务器专用**的位置，再运行上面的命令：

| 文件 | 需要核对的值 |
|---|---|
| `scripts/launch.sh` | `SCRIPT` 的源码绝对路径；默认 `BASE` 的源码目录外输出路径。显式传入 `RUN` 仍可覆盖 `BASE`。 |
| `scripts/run_compare.py` | `repo`（Git 工作树）、`model`（权重路径）、`port`、默认 `devices`；设备 ID 的断言目前限定为 0–15。 |
| `scripts/run_server.sh` | `PYTHONPATH` 中的仓库及 CANN 路径、`--model-path`；需与控制器中的 `model` 完全一致。 |

控制器从 `npu-smi info` 的进程表按 **物理 ID = 2 × card + chip** 判断占用，这是当前双芯片卡服务器的格式。如果新机器的设备编号或 `npu-smi` 输出不同，先修改 `scripts/run_compare.py` 中的 `busy_devices()`，避免错误地把繁忙设备当作空闲设备。

为与本报告可比，保持 TP8/EP8、相同模型与权重、两条并发请求、1000/16 token 负载、3072 token KV 容量、HiCache 参数和四轮顺序不变。若必须改变这些设置，在新报告中明确记录；尤其要保证 A 可放入 L1、A+B 会驱逐 L1，且能保留于 L2。`scripts/bench.py` 负责生成提示、计时和传输标记检查；控制器的参数与服务容量需一起调整。
