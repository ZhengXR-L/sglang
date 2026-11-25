import os
import subprocess
import sys
from pathlib import Path


out, rc, result = sys.argv[1:]
root = os.environ.get("SGLANG_ROOT")
if root:
    revision = subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip() or "未知"
else:
    revision = "未知"

fields = (
    ("SGLang 路径", "SGLANG_ROOT"),
    ("公共模型", "SOURCE_MODEL"),
    ("私有五层模型", "PRIVATE_MODEL"),
    ("NPU 卡号", "NPU_DEVICES"),
    ("端口", "PORT"),
    ("量化", "QUANTIZATION"),
    ("共享前缀 token 数", "PREFIX_TOKENS"),
    ("计划请求数", "REQUEST_COUNT"),
    ("Host 内存后端", "SGLANG_HICACHE_HOST_MEM_BACKEND"),
    ("Mamba I/O 模式", "SGLANG_NPU_HICACHE_MAMBA_IO"),
    ("HiCache 布局", "HICACHE_MEM_LAYOUT"),
)
lines = [
    "# Kimi K3 HiCache L2 复现",
    "",
    f"{result}（退出码 {rc}）",
    "",
    f"- SGLang 提交 `{revision}`",
    *(f"- {label} `{os.environ.get(key) or '未设置'}`" for label, key in fields),
    "",
    "请求进度和响应元数据见 [eval.log](eval.log)，服务栈见 [server.log](server.log)。",
]
Path(out, "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
