"""GSM8K accuracy regression for Qwen3.8-Flash-Next with the QSA indexer in
BF16 on Ascend NPUs.

Enables ``--qsa-indexer-dtype bfloat16`` (the only QSA indexer dtype allowed
on NPU; fp8_e4m3 is CUDA-SM90/SM100-gated) and asserts the GSM8K accuracy
stays above the configured threshold.

Model path, accuracy threshold, question count and few-shot count are all
overridable via environment variables. ``GSM8K_ACCURACY`` is mandatory and
must be set slightly below the measured accuracy of this exact configuration
(run once without the assertion threshold to record it, then set it).

[Test Category] Model
[Test Target] Qwen3.8-Flash-Next --qsa-indexer-dtype bfloat16
"""

import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=1800, suite="full-8-npu-a3", nightly=True)

# Mandatory: measured accuracy of this configuration minus a small margin.
ACCURACY = float(os.environ["GSM8K_ACCURACY"])
MODEL = os.environ.get("GSM8K_MODEL_PATH", "/mnt/paas/weights/Qwen3.8-Flash-Next")
NUM_QUESTIONS = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))
NUM_SHOTS = int(os.environ.get("GSM8K_NUM_SHOTS", "5"))

TEST_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
}

BASE_ARGS = [
    "--trust-remote-code",
    "--device",
    "npu",
    "--tp-size",
    "8",
    "--attention-backend",
    "ascend",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--mem-fraction-static",
    "0.8",
    # Hybrid GDN on NPU: full-sequence prefill, no chunking.
    "--chunked-prefill-size",
    "-1",
    "--context-length",
    "8192",
    "--max-running-requests",
    "32",
    # QSA needs page-aligned full-KV slots (page_size 64 is set by the
    # Qwen4-Exp override) without the mamba extra-buffer strategy.
    "--disable-radix-cache",
    "--disable-cuda-graph",
]

# Feature under test, kept separate so dropping it re-runs the same
# configuration as the baseline.
FEATURE_ARGS = ["--qsa-indexer-dtype", "bfloat16"]


class TestQwen38FlashNextGSM8KQsaBf16(GSM8KAscendMixin, CustomTestCase):
    """QSA indexer cache stored in BF16; accuracy must not drop."""

    model = MODEL
    other_args = BASE_ARGS + FEATURE_ARGS
    env = {**os.environ, **TEST_ENVS}
    accuracy = ACCURACY
    num_questions = NUM_QUESTIONS
    gsm8k_num_shots = NUM_SHOTS
    timeout_for_server_launch = 3600


if __name__ == "__main__":
    unittest.main()
