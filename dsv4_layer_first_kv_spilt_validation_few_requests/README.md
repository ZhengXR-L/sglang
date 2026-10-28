# DSV4 layer_first 稀疏拷贝少量请求验证复现

本目录保存 `layer_first_ascendc_sparse_copy_validation_20261026.md` 报告和少量请求测试脚本。入口 `run_layer_first_verify.sh` 调用 `test_hicache/test_cached_token.sh`，后者启动专用服务并调用 `test_l2.sh`；实际请求由 `bench_ids_dsv4.py` 发送。原报告在目标提交 `356c078865ae63511685837082e1f1a49d697b62` 上，对 `page_first` 与 `layer_first` 两次服务作比较；两轮完整命令、结果和原始证据位于报告中标明的 `/home/l00889655/sglang/sgl_notebook_l2_2n_20261026/`。本目录当前的入口只运行 **layer_first 一轮**，不会自动生成报告里的 page_first 对照轮。

## 运行前准备

需要 Linux、Ascend 八卡/驱动与工具链、目标 SGLang 分支与兼容的 `sgl_kernel_npu`/Memfabric、DSV4 模型权重。确认八张卡空闲，测试端口可用，`curl`、`python3`、模型 tokenizer 可用。原入口把 `SGL_SRC_HOME`、`MODEL_PATH`、`NPU_DEVICES=4,5,6,7,8,9,10,11`、TP/DP/EP、容量等值直接 `export` 为固定值，还要求 Git 当前分支名等于 `layer_first_ascendc_sparse_copy`。迁移时先编辑入口里的这些值；仅在环境变量里覆盖卡号、仓库或模型路径会被入口重新赋值。`run_l2_server.sh` 还含 `/home/cx`、`/mnt/weight` 默认路径，但入口会覆盖主要值；Ascend 环境脚本和 CPU/sysctl 设置同样要按新机器核对。

## 执行一轮

先将入口中仓库、权重和八卡列表改为实际值，并使用新证据目录：

```bash
HERE=/home/l00889655/sglang/sglang/dsv4_layer_first_kv_spilt_validation_few_requests
LOG_DIR=/home/l00889655/sglang/layer_first_verify_$(date +%Y%m%d_%H%M%S) \
  bash "$HERE/run_layer_first_verify.sh"
```

`LOG_DIR` 必须尚不存在。脚本默认端口 `32367`，请求长度 65536 token、8 条 A 和 8 条不同前缀 B，再重放 A；按 DP rank 路由，检查 L2 命中。运行日志写到 `$LOG_DIR/run.log`，服务命令和环境、Git 版本、内核哈希、server log、结果及 `exit_code` 同处该目录。脚本会尝试只停止自己启动的服务；仍应确认测试结束后没有残留进程。

## 复现 page_first 对照和判读

要复现原报告的双轮对照，分别用全新的 LOG_DIR、相同模型/代码/卡组/请求参数运行 `HICACHE_LAYOUT=page_first` 和 `HICACHE_LAYOUT=layer_first`；当前入口允许 `HICACHE_LAYOUT` 覆盖，但两轮默认使用同一端口，必须串行执行。原报告使用独立端口 32370/32371，可按需要设置 `PORT`。比较实际 `launch_command.txt`、`runtime_environment.json`、`sglang_commit` 和输出结果，确认仅布局不同。由于入口会把 `SGLANG_DSV4_AIV_TRACE` 固定为 1，trace 可用于核对 layer_first AIV 调用及 page_first 无 AIV；trace 对每个进程的 pool/方向只记首条，不能把日志行数当实际调用次数或异步完成事件数。

验收重点是 R 的 `cached_tokens_details` 显示 device=0、host>0，原报告是 host=61440；A/R 输出 8 个 token 完全一致、非全 0。原报告还检查了 8 rank × 7 池 × D2H/H2D 的 trace 覆盖。改模型、代码、内核或卡组后须重新检查真实容量和路径，不能直接沿用原报告的通过结论。
