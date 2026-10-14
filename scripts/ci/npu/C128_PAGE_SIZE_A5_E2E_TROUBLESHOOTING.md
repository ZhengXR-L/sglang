# A5 环境运行 C128 page size 端到端用例：问题与处理

本文记录在 Ascend A5 上运行 `test_npu_c128_page_size.py` 时遇到的问题。用例使用 DeepSeek-V4-Flash W8A8、NPU DSV4 attention 和 `--c128-page-size 16`，以 100 道 GSM8K 题检查跨 C128 物理页推理及答案正确率。

## 运行前提

- 模型目录包含与权重匹配、能由 `AutoTokenizer.from_pretrained` 加载的 tokenizer 文件。
- 使用 8 张卡时，`--tp-size 8` 与 `--attn-dp-size 8` 配套设置。若实际改用 4 张卡，则两者均设为 4；还需确认模型在 4 卡上能装载。
- 运行器 `scripts/ci/npu/run_c128_page_size_tests.sh` 会加载 CANN 9.1.0 的 customize 和 custom_transformer 环境。
- GSM8K 官方 `test.jsonl` 在容器中可读。本工作区的用例可通过 `SGLANG_GSM8K_DATA_PATH` 指定文件；未包含该修改的旧版用例会检查缓存文件 `/tmp/test.jsonl`。

当前仓库的用例默认配置面向 8 卡 A3，**不会自动加入 A5 所需的 `--attn-dp-size 8`**。在 A5 上运行前，应在 `DEEPSEEK_V4_FLASH_W8A8_OTHER_ARGS` 的 `--tp-size 8` 后加入：

```python
"--attn-dp-size",
"8",
```

## 1. 模型 tokenizer 无法加载

**现象：** 服务启动时 `AutoTokenizer.from_pretrained` 报 `Couldn't instantiate the backend tokenizer`，随后 `torch_npu` 的异常打印钩子还可能出现 `RecursionError`。

**原因：** 模型路径指向的目录缺少可用的 tokenizer 文件，或文件与该权重不匹配。该环境已经安装 `sentencepiece` 和 `tiktoken`，因此不能仅凭错误提示认定是缺少这两个 Python 包。此时服务尚未进入 NPU C128 推理。

**处理：** 检查目录中的 `tokenizer.json`、`tokenizer_config.json` 等文件，并从**同一模型对应的完整权重目录**补齐。先单独验证：

```bash
python - <<'PY'
from transformers import AutoTokenizer

model = "/home/weights/DeepSeek-V4-Flash-0731-w8a8"
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
print(type(tokenizer).__name__)
PY
```

## 2. A5 DeepEP 输出 FP8 激活，与 INT8 MoE 权重不匹配

**现象：** 启动阶段捕获 decode graph 时，`npu_grouped_matmul` 报 `aclnnGroupedMatmulWeightNz failed`，底层日志给出 `x=DT_FLOAT8_E4M3FN, weight=DT_INT8`。

**原因：** W8A8 MoE 权重为 INT8，但 A5 上 DeepEP 分发给该算子的激活成为 FP8。这个 dtype 组合无法用于当前 grouped matmul。`Capture cuda graph failed` 是上层汇总错误；禁用 graph capture 不能修正输入 dtype。

**处理：** 本次验证采用 BF16 DeepEP 分发，让 W8A8 MoE 内核在计算前自行将激活量化为 INT8：

```bash
export SGLANG_DEEPEP_BF16_DISPATCH=1
```

该环境变量已弃用。长期配置应在服务参数中使用 `--deepep-dispatcher-output-dtype bf16`，并重新跑完整精度测试。

## 3. A5 attention metadata 的每 rank 头数不满足算子要求

**现象：** 推理开始后，Python 报 `call aclnnKvQuantSparseAttnSharedkvMetadata failed`。CANN plog 的首个具体错误是：

```text
errno[161002] num_heads_q should be 64 or 128, but got 8
```

后续 `errno[561000]`、`errno[561102]` 是参数检查失败后的连带错误。

**原因：** 模型共有 64 个注意力头。仅设置 `--tp-size 8` 时，attention TP 宽度为 8，每 rank 得到 `64 / 8 = 8` 个头，不满足 A5 量化 KV metadata 算子要求。失败发生在 C1A metadata 的首次调用，尚未进入 C128 metadata。

**处理：** 8 卡时加 `--attn-dp-size 8`。此时 attention TP 宽度为 `8 / 8 = 1`，每 rank 保留 64 个注意力头；MoE 仍在 8 卡上执行。4 卡时相应使用 `--tp-size 4 --attn-dp-size 4`。不要通过改模型 `num_attention_heads` 规避算子检查。

## 4. 下载 GSM8K 时证书校验失败

**现象：** 服务已启动，但 `GSM8KEval` 下载官方 `test.jsonl` 时，`requests` 报 `SSLCertVerificationError: self-signed certificate in certificate chain`。

**原因：** 容器经代理访问 `raw.githubusercontent.com` 时，Python `requests` 不信任返回证书的签发链。这是测试数据下载问题，不是模型或 C128 算子错误。

**处理：** 本次验证将官方 GSM8K `test.jsonl` 下载到容器本地。可在包含 `SGLANG_GSM8K_DATA_PATH` 支持的用例中指定：

```bash
export SGLANG_GSM8K_DATA_PATH=/实际路径/test.jsonl
```

旧版用例的 `download_and_cache_file` 会直接复用 `/tmp/test.jsonl`，因此也可以把已校验的官方文件放在那里。若仍希望通过代理在线下载，应为 `requests` 配置受信任的 CA bundle，并用 Python `requests.get()` 验证；不建议关闭证书校验。

## 5. `post-warmup freeze_gc failed` 警告

**现象：** 服务向本机 `/freeze_gc` 发请求时，短暂出现 `ConnectionRefusedError`。

**判断：** 该异常在服务代码中被捕获并记录为警告。本次运行随后完成了 100 次 `/v1/chat/completions` 请求，pytest 最终通过，因此它不是本次用例失败原因。如果服务后续不能提供请求，再单独排查 HTTP 服务的启动和监听状态。

## 本次 A5 验证结果

在修正 MoE 分发和 attention 并行配置、提供本地 GSM8K 数据后，日志显示：

```text
GSM8K: score=0.970, requests=100, min_prompt_tokens=3637, prefix_tokens=3606, padding_examples=0
1 passed, 15 warnings in 315.07s (0:05:15)
```

用例要求 `score >= 0.93`，并检查每个服务端上报的 prompt token 数均大于 `128 * (16 + 1) = 2176`。本次最短 prompt 为 3637 tokens，满足跨页条件。此结果说明**这组 A5 配置下的端到端用例通过**；`post-warmup freeze_gc failed` 没有改变 pytest 结果。

## 复现命令

确保 A5 容器中的用例已经加入 `--attn-dp-size 8`，模型 tokenizer 可加载，GSM8K 数据文件存在，然后在仓库根目录执行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_DSV4_W8A8_MODEL_PATH=/home/weights/DeepSeek-V4-Flash-0731-w8a8
export SGLANG_DEEPEP_BF16_DISPATCH=1
export SGLANG_GSM8K_DATA_PATH=/实际路径/test.jsonl

set -o pipefail
bash scripts/ci/npu/run_c128_page_size_tests.sh e2e 2>&1 | tee /tmp/c128-a5.log
```

如使用旧版用例，先将官方数据文件放到 `/tmp/test.jsonl`，再运行相同命令；旧版不会读取 `SGLANG_GSM8K_DATA_PATH`。
