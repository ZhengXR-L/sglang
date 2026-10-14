# HiCache storage prefetch retry 验证方法

以下命令在配置好 SGLang、Qwen3-0.6B 模型和 NPU 的 Linux 主机上执行。仓库根目录在现有脚本中设为 `/home/l00889655/sglang/sglang`；如果实际路径或模型路径不同，先修改 `scripts/server.sh` 和 `scripts/client.py` 中的路径。

仓库中的服务启动文件是 `scripts/server.sh`，客户端文件是 `scripts/client.py`。`client.py` 虽然后缀为 `.py`，内容是 Bash，不能用 `python3 client.py` 运行。

## 单元测试

从仓库根目录运行本次关注的测试：

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pytest -v \
  test/registered/unit/mem_cache/test_storage_prefetch_lifecycle.py::TestStagedPrefetchLifecycle::test_retry_budget_bounds_reissues_and_paces_capacity_misses
```

日志探针本身也有 CPU 单元测试，可一并运行：

```bash
python3 scripts/test_hicache_retry_probe.py
```

两条命令均应以退出码 `0` 结束。

## NPU 端到端测试

对 `8`、`1`、`0` **分别完整执行**下面的步骤。每轮只运行一个服务实例，上一轮结束后先停止服务，再启动下一轮。

1. 将 `scripts/server.sh` 中的 `INTERVAL=8` 改为本轮数值。确认服务参数 `--hicache-storage-prefetch-retry-poll-interval "$INTERVAL"` 使用该变量；`--port` 为 `7777`。服务日志会写入 `/tmp/hicache-retry-i<数值>.log`。
2. 在终端 A，从仓库根目录启动服务：

   ```bash
   bash scripts/server.sh
   ```

   等待服务就绪，可在另一终端运行 `curl -f http://127.0.0.1:7777/health` 检查。
3. 同步修改 `scripts/client.py` 中的 `--interval` 和 `--log` **文件名**。日志文件名中的 `i` 后面的数值必须与 `server.sh` 的 `INTERVAL` 一致：

   | `server.sh` | `client.py` 的参数 |
   | --- | --- |
   | `INTERVAL=8` | `--interval 8 --log /tmp/hicache-retry-i8.log` |
   | `INTERVAL=1` | `--interval 1 --log /tmp/hicache-retry-i1.log` |
   | `INTERVAL=0` | `--interval 0 --log /tmp/hicache-retry-i0.log` |

   每轮修改后，在终端 B 运行：

   ```bash
   cd scripts
   bash client.py
   ```

   如果从 Windows 复制的 `client.py` 带有 CRLF 换行，先转换成 LF，例如 `sed -i 's/\r$//' client.py`。也可以不改客户端文件，在仓库根目录直接运行等价的探针命令：

   ```bash
   interval=8  # 改为本轮的 8、1 或 0
   python3 scripts/verify_hicache_retry_interval_npu.py \
     --interval "$interval" --port 7777 --log "/tmp/hicache-retry-i${interval}.log"
   ```
4. 记录终端 B 的输出和退出码。正常通过时退出码为 `0`：

   | `INTERVAL` | 预期输出 |
   | --- | --- |
   | `8` | 出现 `verified ... delta=8`，并输出 `PASS: interval=8 matches actual retry step delta` |
   | `1` | 出现 `verified ... delta=1`，并输出 `PASS: interval=1 matches actual retry step delta` |
   | `0` | 输出 `PASS: interval=0 disables miss polling`；不应出现轮询 `fired` 事件 |

   退出码 `1` 表示验证失败。退出码 `2` 和 `INCONCLUSIVE` 表示没有收集到足够的排队 miss 证据，不能算通过；检查 debug 日志、L3 配置和请求是否执行完毕。
5. 在终端 A 按 `Ctrl+C` 停止服务，确认端口 `7777` 已释放，再进行下一轮。

三轮都通过且两项单元测试通过，才算完成本次验证。
