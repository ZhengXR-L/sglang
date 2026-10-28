# 新脚本避开 R 全 0 的启动参数归因

日期：2026-10-26/27（服务器日志为 UTC）。本文补充上一轮仅定位 C4 计算路径、未确定启动参数的缺口。

**已定位到原新配置路线中的有效启动参数组：大池 `max-total-tokens=66000`，配合 `dp-size=8 + enable-dp-attention + enable-dp-lm-head`（SWA=0.1）。固定大池和 SWA=0.1 后，DP1 的冷态旧短请求失败，DP8 参数组的冷态旧短请求通过；DP8 旧小池仍失败。因此这是联合作用，不能归结成“DP8 单独有效”或“66000 单独有效”。另有已验证的替代路线：DP1 下 `66000 + SWA=1.0`。两条路线都覆盖了旧短输入及新 64K 输入的真实 L2 回迁。运行历史还会影响结果，参数规避不等于 C4 缺陷修复。**

## 固定条件

- 服务器 `113.46.15.88`，容器 `sglang-lzx-B103`。
- 独立 main clone `/home/l00889655/sglang/hicache_ablation_20261026/main_source`，HEAD `d6ccb495da418e0d202223d159362fbd04b96655`，无 C4 修复、无数值诊断补丁。
- 原 `sgl_kernel_npu` binary SHA256：`7e14ee6cec76874bf92e1b837935b054eb45da69b01ecb2f19ea626e6d330c1a`。
- 模型 `/mnt/paas/weights/DeepSeek-V4-Flash-0731-w8a8`；TP8、EP8；除明确标出的 DP8 对照外为 DP1；modelslim、BF16 KV；MTP 和 graphs 关闭。
- 本轮使用物理卡 `6,7,8,9,10,11,14,15`。原 4–5 卡占用，因此先在新卡组重现旧失败配置，所有本轮对照使用同一组卡。
- CLI `page_first` 实际为 `page_first_direct`；`kernel_ascend`、memfabric、write_through。
- HiCache ratio 全部固定 2.5，而不是原新脚本的 2.0，以便保持参数归因对照一致；实际 prefill chunk 全部固定 4096，DP1 的 CLI 值为 4096，DP8 的 CLI 值为 32768；global max-running-requests=16；mem-fraction-static=0.8；random-seed=0。WAR 两项均为 1。
- temperature=0、max_new_tokens=8、ignore_eos=false；原生 `/generate`，固定 rank 0。每个小组先 flush，每组重复两次。

除明确的 DP 交叉对照外，HiCache ratio、实际 chunk、接口、输出长度和 C4 代码均不变。“冷态”指服务完成内置启动 warmup/health 后、尚未执行业务长前缀往返的阶段，不是声称完全未执行过任何算子。

## 请求集合与判据

| 小组 | 顺序与输入 |
|---|---|
| pure_l1 | 旧 W 的原始 7841-token 自然语言 A → 完全相同 R，无其他请求 |
| old_evict | 同一 A → 原 E1=7840 token → 原 E2=1057 token → 同一 R |
| long_evict | seed=60000 的 65536-token 随机 A → seed=60001 的不同 65536-token B → 同一 A 的 R |
| short_long_evict | 旧 7841-token 自然语言 A → seed=60001 的 65536-token B → 同一 A 的 R |

所有短 A 的 input IDs 与此前旧脚本一致。随机请求由原 `bench_ids_dsv4.py` 的 `_make_ids` 生成，排除特殊 token。请求、响应、原始输出 IDs 均保存为 JSONL。

短请求匹配用于比较相同请求在不同启动参数下的行为。大池的 old_evict 如果仍命中 L1，只计为 L1 对照，不算 L2 验收。L2 对照必须看到 device=0、host>0，并且 A/R 全部 8 个 token 一致且非全 0。

## 现有结果

| 目录 | max-total-tokens / 实际 full 容量 | SWA ratio | 结果 |
|---|---|---|---|
| baseline_8192_swa1_retry | 8192 / 8192 | 1.0 | pure_l1 两次全 0；old_evict 两次全 0，后者 device=0 / host=6144 |
| cap12288_swa1 | 12288 / 12288 | 1.0 | pure_l1 两次全 0；old_evict 两次全 0，后者 device=4096 / host=2048，属于混合命中 |
| cap66000_swa1 | 66000 / 65920 | 1.0 | 相同 pure_l1、old_evict 各两次 A/R 一致；均为 device=6144 / host=0 |
| cap66000_swa1_exact | 66000 / 65920 | 1.0 | long_evict 两次通过，device=0 / host=61440；short_long_evict 两次通过，device=0 / host=6144 |
| cap66000_swa01 | 66000 / 65920 | 0.1 | pure_l1、old_evict、short_long_evict 各两次全 0；long_evict 两次通过 |
| baseline_8192_return | 8192 / 8192 | 1.0 | 大池测试之后回拨旧参数，pure_l1、old_evict 各两次再次全 0；old_evict 的 device=0 / host=6144 |
| dp8_old8192_swa1 | 8192 / 8192 | 1.0 | 只切换到 DP8 attention + DP lm-head，保持实际 chunk=4096；pure_l1、old_evict 各两次全 0，后者 device=0 / host=6144 |
| dp8_new66000_swa01 | 66000 / 65920 | 0.1 | DP8 参数组：冷态 short_long_evict 两次通过；64K long_warmup 两次通过；后续短请求两次通过；均为 device=0 的 L2 命中 |
| dp8_new66000_cold_restart | 66000 / 65920 | 0.1 | 独立重启相同 DP8 参数组，仅跑旧短请求，两次均通过，device=0 / host=6144 |

`cap66000_swa1` 的后续 65664-token 驱逐探针触发 prefill 缓存分配失败，进程 exit_code=1；此前完成的四个短请求小组仍保留独立结果。65664 不是原新脚本的输入长度，这个探针不用于宣称 64K 通过或失败。后续 `cap66000_swa1_exact` 使用准确的 65536-token 请求，四组均完成且功能通过。

第一轮 `baseline_8192_swa1` 在处理 flush_cache 纯文本响应时，客户端错误地尝试 JSON 解析，未发送业务请求，不纳入结果。已修正后完整重跑为 `baseline_8192_swa1_retry`。

## 已可确认的因果关系

固定 SWA ratio=1.0 和其余参数时，将 `--max-total-tokens` 从 8192 改为 66000，使完全相同的短请求重放从全 0 恢复为 token 一致。准确的 64K 强制 L2 回迁和旧短输入强制 L2 回迁，也都在 SWA=1.0 时通过。因此无需把 SWA ratio 降到 0.1，便能获得这两组 L2 通过结果。

这里测试的是启动参数 max-total-tokens 的整体作用。它不只改变一个内部张量：HiCache ratio 固定时，L2 的派生容量也会变化；SWA ratio 固定时，SWA/状态池容量也随之变化。不能把此结果进一步表述为已经证明单独某个底层池容量或某个 C4 指针地址是唯一原因。

SWA=0.1 不是充分的规避条件。对于完全相同的短 A、64K B、相同缓存命中分项（device=0 / host=6144），固定 max-total-tokens=66000，只把 SWA ratio 从 1.0 改为 0.1，R 从一致输出变成全 0。对于相同 64K 随机 A/B，两种 SWA ratio 都通过。因此需要把参数效果与请求负载结合解释，不能寻找一个对所有输入都成立的单参数结论。

旧参数回拨再次失败，支持该组合对当前场景具有可重复影响；这不等于证明任意运行历史下都不会异常。DP8 旧小池对照也表明，DP8 本身并不足以消除旧请求的故障。

## 相同启动参数下的输入/运行历史反例

`input_alignment_swa01` 全程 DP1、max=66000、SWA=0.1、ratio=2.5、实际 chunk=4096，不修改服务参数。自然语言变体仅在填充叙述中删/加 token，保留原 secret passcode 与最后的问题，所有组的 B 均为同一 65536-token 随机输入，每组先 flush。

| A/R 输入 token 数 | R 的 device / host | A/R 比较 |
|---:|---|---|
| 7841（开头） | 0 / 6144 | R 全 0 |
| 7840 | 0 / 6144 | R 全 0 |
| 7848 | 0 / 6144 | R 全 0 |
| 8192 | 0 / 4096 | 完全一致且非全 0 |
| 8193 | 0 / 8192 | 完全一致且非全 0 |
| 65536 | 0 / 61440 | 完全一致且非全 0 |
| 7841（末尾，与开头 input IDs、B、参数相同） | 0 / 6144 | 完全一致且非全 0 |

7840/7848 已经按 TP8 对齐仍失败，8193 未按 TP8 或 C128 完整原始页组对齐却通过，因此不能把原因简单归结为 token 长度能否被 8/2048 整除。尤其首尾同一 7841 请求发生翻转，证明除了参数和静态输入长度，服务运行历史也在改变触发条件。不能从该顺序实验断言“单个 64K 请求就是充分预热”：中间还执行过 8192/8193 等变体，下面另用最小序列检查。

`cold_long_warm_short` 采用同一启动配置，按“旧短 A/64K B/R 两次 → 64K A/B/R 两次 → 相同旧短 A/64K B/R 两次”执行，每组依旧先 flush，结果如下：

| 阶段 | 输入 A/R | R 的 device / host | 两次结果 |
|---|---:|---|---|
| 先跑旧短请求 | 7841 | 0 / 6144 | 均全 0，均不等于 A |
| 完成长前缀 L2 往返 | 65536 | 0 / 61440 | 均与 A 完全一致且非全 0 |
| 再跑同一旧短请求 | 7841 | 0 / 6144 | 均与 A 完全一致且非全 0 |

关键输出摘录：

```text
short_long_evict, repeat=0/1:
  A=[223,24980,511,271,671,8613,2281,8308]
  R=[0,0,0,0,0,0,0,0], device=0, host=6144
long_warmup, repeat=0/1:
  identical=true, all_zero=false, device=0, host=61440
short_after_warmup, repeat=0/1:
  A=R=[223,24980,511,271,671,8613,2281,8308], device=0, host=6144
```

最终脚本 `verify_parameter_findings.py` 已逐项核对：忽略唯一请求标识 rid 后，前后两组短请求的 A/B/R 完整 payload 完全相同；A 的原始输出也相同；所有轮次 kernel 哈希与 SGLang commit 相同。机器可读结论见 `verified_findings.json`。

这证实长前缀 L2 往返产生了可观察的预热/运行历史效应。不能把它简单说成“只要发一个长 B 就会恢复”，因为冷态失败的短请求组本身已经包含了一个 64K B。实测使后续短请求恢复的是中间完整的 64K A/B/R 往返序列；该实验没有进一步证明究竟初始化了哪一块状态内存、改变了哪个内部缓存或时序。

这也解释了上一轮“大池 + 默认 0.1 的短请求曾通过”与本轮冷态失败的差异：上一轮顺序先执行了 aligned 64K 往返，然后才执行短输入。不能把预热后的结果当作冷态参数对照。

## 补齐原新脚本的 DP8 大池冷态交叉

在 `cold_long_warm_short`（DP1）与 `dp8_new66000_swa01`（DP8）之间，max-total-tokens=66000、SWA=0.1、HiCache ratio=2.5、实际 chunk=4096 均一致，A/B/R 的 input IDs、sampling_params、routed_dp_rank 和执行顺序一致。服务均在启动后的第一组业务测试执行旧短 A → 64K B → 旧短 R，尚未执行过 64K A/R 长往返。

| 固定大池与 SWA=0.1 | 首两次短请求 R | 缓存分项 |
|---|---|---|
| DP1，无 DP attention / DP lm-head | 两次全 0 | device=0 / host=6144 |
| DP8 + DP attention + DP lm-head | 两次与 A 完全一致，非全 0 | device=0 / host=6144 |

DP8 随后的两轮 64K 以及两轮旧短请求也全部通过。独立进程 `dp8_new66000_cold_restart` 仅执行两次冷态旧短请求，同样全部通过。因此该参数组在两次独立服务启动中、执行长 A/R 往返前，共 4 次旧短请求通过；每次均为 device=0 / host=6144。不是仅凭长往返预热后的结果宣称参数有效。

可复现的核心启动参数（其余完整参数以该目录 `launch_command.txt` 为准）：

```bash
--tp-size 8 --ep-size 8 \
--dp-size 8 --enable-dp-attention --enable-dp-lm-head \
--max-total-tokens 66000 --max-running-requests 16 \
--chunked-prefill-size 32768 --swa-full-tokens-ratio 0.1
```

实际 full 容量为 65920；实际每 worker chunk=4096；实际每 worker max-running-requests=2。SWA 是有效值 0.1；原新脚本不显式传该参数时，model override 同样解析为 0.1，其运行日志显示 ratio(explicit) 模式，因此本次不是靠另一个自动 SWA 规划模式取得通过。

这次验证的是参数组整体：切换 DP8 同时将 attention TP 从 8 改为 1，将每 worker 请求上限从 16 改为 2，并启用 DP lm-head 等派生行为。没有分别拆开证明其中某个内部开关是唯一原因。此前“大池 DP1 某些请求也能通过”并不能排除 DP 在 SWA=0.1 冷态下的作用；此前“DP8 旧小池仍失败”也不能排除其与大池的联合作用。

## 最终回答与适用范围

1. **原新配置路线的有效组合是：`--max-total-tokens 66000 --dp-size 8 --enable-dp-attention --enable-dp-lm-head`，保留 SWA=0.1；global max-running-requests=16，CLI chunk=32768，实际 chunk=4096。**本轮 TP8/EP8、HiCache ratio=2.5。固定其余参数时，DP1 冷态失败、DP8 冷态通过，且 DP8 的 64K 负载也通过。这是原新脚本核心参数组的受控对照，不是宣称 HiCache ratio=2.0 的原命令被逐字复跑了所有冷态探针。
2. **替代规避路线是 DP1 下 `--max-total-tokens 66000 --swa-full-tokens-ratio 1.0`。**实际 full/SWA 容量均为 65920，两种输入的 R 都在明确 device=0 的 L2 命中下连续两次与 A 完全一致。这里没有证明 66000 是最小阈值，也没有证明任意其他负载均安全。
3. **仅说 `66000 + SWA=0.1` 不够，必须说明 DP 配置与是否预热。**DP1 下该组合的冷态短请求失败，长往返之后又可通过。参数效果与输入/缓存命中边界、运行历史存在交互；DP8 单独、SWA=0.1 单独、较小 chunk 单独都不能被表述为正式修复。
4. 正式验证应区分冷态与预热态。`flush_cache` 并不能在这里恢复到与进程初启完全相同的行为；短请求探针应在任何长往返之前执行，并保留 main 的失败基线。比较两种布局时必须保持启动配置和请求顺序一致。
5. 这是一组当前样本上的规避条件，不是 C4 缺陷修复。本轮没有修改 C4 业务代码、全局 wheel、共享工作区，也没有验证 layer_first/AIV 或 TTFT。

## 证据与复现入口

服务器根目录 `/home/l00889655/sglang/hicache_parameter_cause_20261026`。每轮有实际完整 `launch_command.txt`、`runtime_environment.json`、`resolved_server_info.json`、`server.log`、`run.log` 和 `results/summary.json`，请求响应位于 results 各小组目录。

入口 `run_parameter_cause.sh`；客户端 `parameter_cause_client.py`；顺序矩阵 `run_parameter_matrix.sh`，自动等待前一轮结束并核对目标 NPU 空闲，发生模型加载显存预算错误时原样重试一次。`matrix.log` 记录每轮起止。客户端 exit_code=0 只表示执行完成，功能结论必须看 summary 的 identical、all_zero 和缓存分项。

补充入口为 `run_alignment_after.sh`、`run_dp8_old_after.sh`、`run_warmup_after.sh`、`run_dp8_cold_restart.sh`。DP8 大池首次冷态轮由 docker exec 直接调用通用入口，完整命令见该轮 launch_command.txt。全部已结束，测试卡已释放。独立 clone 的 git status 干净，原 kernel 哈希未变。
