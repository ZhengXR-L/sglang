"""End-to-end BF16 QSA indexer check on eight Ascend NPUs.

Run a long greedy request through the full Qwen3.8-Flash-Next checkpoint,
covering sparse prefill and repeated decode reads of the compressed index cache.
"""

import math
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=1800, suite="full-8-npu-a3", nightly=True)

MODEL = "/mnt/paas/weights/Qwen3.8-Flash-Next"
# The model's indexer_budget is 2048. Exceed it to exercise sparse top-k.
PROMPT_REPEAT = int(os.getenv("SGLANG_QSA_TEST_PROMPT_REPEAT", "320"))
PROMPT = "The QSA indexer checks compressed sparse attention. " * PROMPT_REPEAT
MAX_NEW_TOKENS = 16
TEST_URL = os.getenv("SGLANG_QSA_TEST_URL", DEFAULT_URL_FOR_TEST)
INDEXER_DTYPE = os.getenv("SGLANG_QSA_TEST_INDEXER_DTYPE", "bfloat16")


class TestNPUQSAIndexerBFloat16(CustomTestCase):
    def test_qsa_indexer_bfloat16_prefill_and_decode(self):
        process = popen_launch_server(
            MODEL,
            TEST_URL,
            timeout=1200,
            other_args=[
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
                "--qsa-indexer-dtype",
                INDEXER_DTYPE,
                "--context-length",
                "4096",
                "--max-total-tokens",
                "4096",
                "--max-running-requests",
                "4",
                "--disable-radix-cache",
                "--disable-cuda-graph",
                "--skip-server-warmup",
            ],
        )
        try:
            response = requests.post(
                f"{TEST_URL}/generate",
                json={
                    "text": PROMPT,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                    "logprob_start_len": 0,
                },
                timeout=600,
            )
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            meta = result["meta_info"]
            self.assertGreater(meta["prompt_tokens"], 2048)
            self.assertLess(meta["prompt_tokens"] + MAX_NEW_TOKENS, 4096)
            self.assertEqual(meta["completion_tokens"], MAX_NEW_TOKENS)
            self.assertEqual(len(meta["output_token_logprobs"]), MAX_NEW_TOKENS)
            prompt_ids = [item[1] for item in meta["input_token_logprobs"]]
            decode_logprobs = meta["output_token_logprobs"]
            output_ids = [item[1] for item in decode_logprobs]
            self.assertEqual(len(prompt_ids), meta["prompt_tokens"])
            score_response = requests.post(
                f"{TEST_URL}/generate",
                json={
                    "input_ids": prompt_ids + output_ids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 0},
                    "return_logprob": True,
                    "logprob_start_len": 0,
                },
                timeout=600,
            )
            self.assertEqual(score_response.status_code, 200, score_response.text)
            score_logprobs = score_response.json()["meta_info"]["input_token_logprobs"][
                meta["prompt_tokens"] :
            ]
            self.assertEqual(len(score_logprobs), MAX_NEW_TOKENS)
            self.assertEqual([item[1] for item in score_logprobs], output_ids)
            logprob_diffs = []
            for decode_lp, score_lp in zip(decode_logprobs, score_logprobs):
                self.assertTrue(math.isfinite(decode_lp[0]))
                self.assertTrue(math.isfinite(score_lp[0]))
                logprob_diffs.append(abs(decode_lp[0] - score_lp[0]))
            print(
                f"QSA dtype={INDEXER_DTYPE}, prompt_tokens={meta['prompt_tokens']}, "
                f"output_ids={output_ids}, logprob_diffs={logprob_diffs}",
                flush=True,
            )
            # The first generated token is scored by a different prefill
            # boundary on the replay request. The remaining decode tokens
            # should agree more closely across the two execution paths.
            self.assertLess(logprob_diffs[0], 1.0)
            self.assertLess(max(logprob_diffs[1:]), 0.255)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
