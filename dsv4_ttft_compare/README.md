# DSV4 两种 split 布局的 L2 回迁 TTFT 对比

本脚本参考 `sglang/dsv4_ttft_performance`，在相同的八张 NPU 上按
`layer_first → page_first → page_first → layer_first`（ABBA）分别启动四次独立服务。
当前源码将 `layer_first`、`page_first` 分别解析为
`layer_first_kv_split`、`page_first_kv_split`；两条路径都调用 AIV 算子。
默认模型为已核对完整的 `/mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp`，
使用 `modelslim`、BF16 KV、TP8/EP8/attention-DP8、`kernel_ascend`、
`write_through`、HiCache ratio 2.5。

## 运行命令

在远端 `sglang-lzx-B103` 容器内执行。输出目录必须尚不存在，端口 32430
应空闲，指定的八张物理卡应空闲；控制器也会检查卡空闲后再启动每轮服务。
不需要 GSM8K 数据集，也不需要其他测试产生的 `plan_50.json`。

```bash
cd /home/l00889655/sglang/sglang/dsv4_ttft_compare
export SGLANG_ROOT=/home/l00889655/sglang/sglang
export KERNEL_ROOT=/home/l00889655/sglang/sgl-kernel-npu
export PYTHONPATH="$KERNEL_ROOT/python/sgl_kernel_npu${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH=/mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp
export PORT=32430
python3 -u run_compare.py /home/l00889655/sglang/dsv4_split_ttft_back8_004 8,9,10,11,12,13,14,15
```

每次运行都要指定一个尚不存在的输出目录；上面的 `_004` 只是示例。
当前 `run_server.sh` 已将 `$SGLANG_ROOT/python` 和
`$KERNEL_ROOT/python/sgl_kernel_npu` 加入 `PYTHONPATH`。命令中的显式
`PYTHONPATH` 设置也便于在启动测试前单独检查算子导入。此前远端旧版脚本
只加入 SGLang 源码路径，导致以下导入错误；使用本目录的新版脚本即可修复。

若第一轮报 `ModuleNotFoundError: No module named 'sgl_kernel_npu.operators'`，
说明 Python 导入了容器中已安装的旧版 `sgl_kernel_npu`，该版本没有
`operators` 子包；当前算子源码目录中有该子包。可用以下命令确认导入位置：

```bash
python3 -c 'import sgl_kernel_npu; print(sgl_kernel_npu.__file__); import sgl_kernel_npu.operators; print(sgl_kernel_npu.operators.__file__)'
```

预期两个路径都位于 `$KERNEL_ROOT/python/sgl_kernel_npu/sgl_kernel_npu/` 下。
此类启动失败时，`run_status.json` 中的 `finished_blocks: 0` 表示尚未开始性能测量；
`report_exit: 0` 仅表示失败报告生成成功。具体错误见输出目录中的
`1_layer_first/server.log`。

`bench.py prepare` 用模型目录中的 tokenizer 和脚本内固定的算术文本生成
`plan.json`，记录随机种子、输入 ID 哈希与计划哈希。四轮服务共用该计划；
不读取其他测试的结果或输入。新的输入生成方式与 2026-11-02 的历史结果不同，
新旧 TTFT 数值不应直接比较。

每轮服务使用相同输入计划、同一个 DP rank 0 和单并发。脚本生成
8192、32768、61440 token 三种目标 L2 前缀长度，每种长度先预热 1 次，
再测 5 次。每次执行 `flush_cache → A（冷计算）→ B（65536 token 驱逐）→
R（重放 A）`；只有 A 无缓存、R 的 L1 命中 0 且 L2 命中目标长度、
A/R 输出 token 完全一致，才计入有效 TTFT。TTFT 是请求发出到流式响应中
首个包含 `output_ids` 的事件的时间。双方关闭逐调用审计，以免计时包含审计 I/O。

运行目录保存四轮的 `server.log`、`server_info.json`、每次 A/B/R 请求和响应、
`samples.jsonl`、`manifest.json`、`comparison.json`、`report.md` 和 `run_status.json`。
只有四轮都完成且每轮三种长度各有 5 个有效计时样本，报告中的对比才视为完整。
布局和对应算子同时变化，因此差异代表两种实现组合的效果。

已归档 2026-11-02 的两轮完整结果：`report_initial_20261102.md` 和
`report_rerun_20261102.md`。这两轮使用旧测试计划；当前脚本改为自行生成
测试计划，因此新运行的绝对 TTFT 数值不宜直接与这些历史结果比较。
