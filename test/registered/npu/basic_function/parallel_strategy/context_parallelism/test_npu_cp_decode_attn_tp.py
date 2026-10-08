"""GSM8K accuracy regression for NPU CP decode attention TP with W8A8 weights.

The test serves one DeepSeek-V4-Flash model with decode attention TP enabled.
It checks the server configuration and evaluates GSM8K instead of comparing
individual tokens or logprobs against a second server.
"""

import json
import os
import unittest
from pathlib import Path

import requests

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=3600, suite="full-8-npu-a3")
register_npu_ci(est_time=7200, suite="nightly-8-npu-a3", nightly=True)

PARALLEL_SIZE = 8
MODEL_NAME = "DeepSeek-V4-Flash-0731-w8a8"
MODEL_PATH_ENV = "SGLANG_TEST_CP_DECODE_MODEL_PATH"

TEST_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    # EP=8 low-latency combine with dispatch capacity 128 requires >=557 MB.
    "HCCL_BUFFSIZE": "1024",
    "DEEPEP_HCCL_BUFFSIZE": "1024",
    "SGLANG_ZBAL_LOCAL_MEM_SIZE": "0",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "0",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
}

SERVER_ARGS = [
    "--tp-size",
    str(PARALLEL_SIZE),
    "--attn-cp-size",
    str(PARALLEL_SIZE),
    "--dp-size",
    "1",
    "--ep-size",
    str(PARALLEL_SIZE),
    "--enable-prefill-cp",
    "--cp-strategy",
    "interleave",
    "--enable-cp-decode-attn-tp",
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--quantization",
    "modelslim",
    "--kv-cache-dtype",
    "bfloat16",
    "--page-size",
    "128",
    "--mem-fraction-static",
    # Eight-card W8A8 weights use ~43.84 GiB/rank; 0.7 leaves no KV budget.
    "0.8",
    "--chunked-prefill-size",
    "4096",
    "--prefill-max-requests",
    "8",
    "--max-running-requests",
    "8",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--watchdog-timeout",
    "600",
    "--random-seed",
    "0",
    "--disable-radix-cache",
    "--cuda-graph-backend-prefill",
    "disabled",
    "--cuda-graph-backend-decode",
    "full",
    "--cuda-graph-bs-decode",
    "1",
    "2",
    "4",
    "8",
]


def resolve_model_path():
    override = os.environ.get(MODEL_PATH_ENV)
    candidates = (
        [Path(override)]
        if override
        else [
            Path("/home/weights") / MODEL_NAME,
            Path("/mnt/pass/weights") / MODEL_NAME,
            Path("/root/.cache/modelscope/hub/models/Eco-Tech") / MODEL_NAME,
        ]
    )
    for path in candidates:
        if (path / "config.json").is_file() and (
            path / "quant_model_description.json"
        ).is_file():
            return str(path)
    raise FileNotFoundError(
        f"Set {MODEL_PATH_ENV} to a local ModelSlim W8A8 model directory; "
        f"searched: {candidates}. This test does not download model weights."
    )


class TestNPUCpDecodeAttnTP(GSM8KAscendMixin, CustomTestCase):
    """Evaluate the enabled CP decode attention TP path on eight NPUs."""

    timeout_for_server_launch = 1800  # Seconds, not milliseconds.
    other_args = SERVER_ARGS
    env = {**os.environ, **TEST_ENVS}
    accuracy = 0.93
    # The evaluator uses five test-set examples as shots, leaving 1314 scored.
    num_questions = 1314

    @classmethod
    def setUpClass(cls):
        cls.model = resolve_model_path()
        with open(Path(cls.model) / "config.json") as f:
            config = json.load(f)
        if config.get("architectures", [None])[0] != "DeepseekV4ForCausalLM":
            raise ValueError("This regression requires DeepseekV4ForCausalLM")
        for field in ("num_attention_heads", "o_groups"):
            value = config[field]
            if value < PARALLEL_SIZE or value % PARALLEL_SIZE:
                raise ValueError(
                    f"{field}={value} must be positive and divisible by CP={PARALLEL_SIZE}"
                )
        with open(Path(cls.model) / "quant_model_description.json") as f:
            quant_config = json.load(f)
        for suffix in ("attn.wq_b.weight", "attn.wo_b.weight"):
            entries = [v for k, v in quant_config.items() if k.endswith(suffix)]
            if not entries or any(v != "W8A8_DYNAMIC" for v in entries):
                raise ValueError(f"Expected W8A8_DYNAMIC entries for {suffix}")

        super().setUpClass()
        try:
            response = requests.get(f"{cls.base_url}/server_info", timeout=30)
            response.raise_for_status()
            info = response.json()
            expected = {
                "enable_cp_decode_attn_tp": True,
                "enable_prefill_cp": True,
                "attn_cp_size": PARALLEL_SIZE,
                "tp_size": PARALLEL_SIZE,
                "dp_size": 1,
                "cp_strategy": "interleave",
            }
            for key, value in expected.items():
                if info.get(key) != value:
                    raise AssertionError(
                        f"server_info.{key}={info.get(key)!r}, expected {value!r}"
                    )
            # CudaGraphConfig.to_dict omits the default decode backend (full).
            decode_graph = info["cuda_graph_config"]["decode"]
            if decode_graph.get("backend", "full") != "full":
                raise AssertionError(f"Unexpected decode graph backend: {decode_graph}")
        except Exception:
            cls.tearDownClass()
            raise


if __name__ == "__main__":
    unittest.main()
