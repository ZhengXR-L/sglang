"""单机四卡、五层 Kimi K3 的 HiCache L2 执行路径复现入口。

运行 python3 dts_test.py <output_dir>。
模型、卡号、代码路径和请求配置统一读取同目录 env.sh。
"""

import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 dts_test.py <output_dir>")

    here = Path(__file__).resolve().parent
    runner = here / "run_test.sh"
    child = subprocess.Popen(
        ["bash", str(runner), sys.argv[1]],
        env=os.environ.copy(),
        start_new_session=True,
    )

    def forward_signal(signum, frame):
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)
    raise SystemExit(child.wait())


if __name__ == "__main__":
    main()
