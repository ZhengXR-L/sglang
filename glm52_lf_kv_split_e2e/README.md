# GLM-5.2 `layer_first_kv_split` 端到端验证

## 执行命令

在 `113.46.15.88` 上进入 NPU 容器，先确认八张空闲的物理 NPU，
再替换下面的设备编号。输出目录必须尚不存在。

```bash
ssh root@113.46.15.88
docker exec -it sglang-lzx-B103 bash
npu-smi info

export PORT=32348
python3 -u /home/l00889655/sglang/glm52_lf_kv_split_e2e/run_e2e.py \
  /home/l00889655/sglang/glm52_lf_kv_split_smoke_001 \
  0,1,2,3,4,5,6,7 3
```

最后三个参数为输出目录、八张物理 NPU、题数。先用 3 题冒烟；
正式运行改成 `100` 题并换新的输出目录。端口占用时修改 `PORT`。
脚本自行启动、等待、关闭专用服务。启动脚本会检查所需的 KV split
算子是否可导入，不限制两个源码仓库的分支名。

```bash
cat /home/l00889655/sglang/glm52_lf_kv_split_smoke_001/results/summary.json
cat /home/l00889655/sglang/glm52_lf_kv_split_smoke_001/report.md
```

输出目录还包含 `server.log`、`eval.log`、逐题请求/响应和算子审计事件。

## 环境变量命令

运行前可显式覆盖以下默认路径和端口。`DATA_PATH` 供客户端读取，
其余路径供启动脚本使用；没有覆盖时脚本使用下面的默认值。

```bash
export PORT=32348
export STARTUP_TIMEOUT=3600
export MODEL_PATH=/mnt/paas/weights/GLM-5.2-w4a8c8
export DATA_PATH=/home/l00889655/sglang/gsm8k_test.jsonl
export SGLANG_ROOT=/home/l00889655/sglang/sglang
export KERNEL_ROOT=/home/l00889655/sglang/sgl-kernel-npu
export ASCEND_TOOLKIT_ROOT=/usr/local/Ascend/ascend-toolkit/latest
export CANN_PYTHON_SITE_PACKAGES=/usr/local/Ascend/cann-9.1.0/python/site-packages
export ATB_ENV_SCRIPT=/usr/local/Ascend/nnal/atb/set_env.sh
```

`run_e2e.py` 用 `subprocess.Popen(env=...)` 传入以下值，相当于在
启动 `run_server.sh` 前执行这些命令；**不需要手动重复执行**：

```bash
export AUDIT_DIR="<输出目录>/audit"
export NPU_DEVICES="<第二个命令行参数：八张物理 NPU>"
export PORT="${PORT:-32348}"
export MODEL_PATH="${MODEL_PATH:-/mnt/paas/weights/GLM-5.2-w4a8c8}"
export SGLANG_ROOT="${SGLANG_ROOT:-/home/l00889655/sglang/sglang}"
```

`run_server.sh` 实际执行以下 `export`：

```bash
export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
export HCCL_BUFFSIZE=512
export PYTHONPATH="$SGLANG_ROOT/python:$KERNEL_ROOT/python/sgl_kernel_npu:$ASCEND_TOOLKIT_ROOT/python/site-packages:$CANN_PYTHON_SITE_PACKAGES:${PYTHONPATH:-}"
```

`gsm8k_aiv.py serve` 随后构造服务进程环境，等价于再执行：

```bash
export GLM52_MLA_AIV_AUDIT_DIR="$AUDIT_DIR"
export SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
export PYTHONPATH="$AUDIT_DIR/boot:$SGLANG_ROOT/python:$PYTHONPATH"
```

最终 `PYTHONPATH` 前缀为审计钩子、SGLang 源码（两次）、
sgl-kernel-npu 源码、toolkit Python、CANN Python，然后接原有值。
脚本还会 `source` customize vendor、custom_transformer vendor、
Ascend toolkit、ATB 的 `set_env` 脚本；这些外部脚本额外导出的
运行库和算子变量取决于容器中的安装版本。启动时原有的其他环境
变量也会被继承。

## 服务参数和判定原理

服务使用 GLM-5.2-w4a8c8、TP8/EP8、`--device npu`、
`--attention-backend ascend`、`--enable-hicache`、
`--hicache-mem-layout layer_first`（解析为 `layer_first_kv_split`）、
`--hicache-io-backend kernel_ascend`、`--hicache-write-policy write_through`、
`--hicache-ratio 2.5`、`--max-total-tokens 3072`。完整启动参数见
`run_server.sh` 和运行结果中的 `audit/launch.json`。

审计启动器最终 `exec` 的 **SGLang 服务命令**如下。这里展开了脚本的
默认模型路径和端口；如果运行前设置了 `MODEL_PATH` 或 `PORT`，
对应参数会使用覆盖后的值：

```bash
python3 -m sglang.launch_server \
  --model-path /mnt/paas/weights/GLM-5.2-w4a8c8 \
  --host 127.0.0.1 --port 32348 \
  --tp-size 8 --ep 8 \
  --moe-a2a-backend deepep --deepep-mode normal \
  --deepep-dispatcher-output-dtype bf16 \
  --enable-hicache --hicache-mem-layout layer_first \
  --hicache-io-backend kernel_ascend \
  --hicache-write-policy write_through \
  --hicache-ratio 2.5 \
  --mem-fraction-static 0.86 \
  --max-total-tokens 3072 \
  --device npu --attention-backend ascend \
  --disable-cuda-graph --log-level info
```

端到端脚本会先准备上一节的环境变量和审计钩子，然后以这些参数
启动服务。单独执行上面的命令不会自动配置审计钩子。

每题执行 `flush_cache → A → B → R`：A 冷缓存写穿触发
`transfer_kv_all_layer_kv_split` D2H；B 用不同长前缀驱逐 L1；
R 重放与 A 相同的 input IDs，要求 L1 命中 0、L2 命中至少 128 token，
触发 `transfer_kv_per_layer_kv_split` H2D。审计钩子在算子返回后等待
NPU stream event 完成，并要求 D2H、H2D 都显式选择 `impl="aiv"`。
八个 TP rank 的 K、V、Indexer K 均须覆盖
78 层；同时比较 A/R 最终答案，记录输出 token 是否一致。100 题运行
还会独立复核原始证据。

GLM-5.2 经 `MLATokenToKVPoolHost` 调用这两个算子；此测试覆盖
GLM 的 MLA 路径，不覆盖 DSV4 专用 host pool。NPU event 完成不
等同于 KV 逐字节比对。
