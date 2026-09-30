"""NPU CP decode attention TP regression: baseline, eager, then graph.

This is a correctness test, not a throughput benchmark. The W8A8 implementation
must support runtime slicing; implementation errors must fail, not be skipped.
Run on eight reserved A3 NPUs. SGLANG_TEST_CP_DECODE_MODEL_PATH overrides local
weight discovery. PR jobs run all differential probes; other jobs also run GSM8K.
"""

import json
import math
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests

from sglang.srt.utils.network import get_open_port
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

# Three sequential server launches; only eight devices are used at a time.
register_npu_ci(est_time=3600, suite="full-8-npu-a3")
register_npu_ci(est_time=7200, suite="nightly-8-npu-a3", nightly=True)

PARALLEL_SIZE = 8
MODEL_NAME = "DeepSeek-V4-Flash-0731-w8a8"
MODEL_PATH_ENV = "SGLANG_TEST_CP_DECODE_MODEL_PATH"

# Keep only settings used by this execution path. In particular, do not force
# the performance case's 61-GB ZBAL allocator, fixed bootstrap port or MTP flags.
TEST_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_BUFFSIZE": "200",
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


def server_args(enable_decode_tp, graph_backend):
    # TP=CP and DP=1 leave attention linears replicated before runtime slicing.
    # Eight CP ranks divide both 64 heads and 8 output groups in V4-Flash.
    args = [
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
        "0.7",
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
        graph_backend,
    ]
    if enable_decode_tp:
        args.append("--enable-cp-decode-attn-tp")
    if graph_backend == "full":
        args.extend(["--cuda-graph-bs-decode", "1", "2", "4", "8"])
    return args


class TestNPUCpDecodeAttnTP(CustomTestCase):
    launch_timeout = 1800  # Seconds, not milliseconds.
    request_timeout = 600
    output_tokens = 32
    # Regression tolerance for W8A8 reduction-order differences, not a claimed
    # measured bound. Greedy token IDs must still match exactly.
    logprob_atol = 0.15

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
        # Do not let a FLOAT-only model make the W8A8 regression appear to pass.
        for suffix in ("attn.wq_b.weight", "attn.wo_b.weight"):
            entries = [v for k, v in quant_config.items() if k.endswith(suffix)]
            if not entries or any(v != "W8A8_DYNAMIC" for v in entries):
                raise ValueError(f"Expected W8A8_DYNAMIC entries for {suffix}")

    def _probe(self, base_url):
        outputs = []
        # Every completed request is followed by a fresh prefill, exercising
        # weight restoration. 3 also exercises padding to a graph bucket.
        for round_id, batch_size in enumerate((1, 2, 4, 8, 3, 1)):
            prompts = [
                (
                    f"Record {round_id}-{i}. "
                    + "The store has red, green, and blue apples. " * (16 + i)
                    + "\nQuestion: There are 12 apples and 5 are sold. "
                    "How many remain? Explain briefly.\nAnswer:"
                )
                for i in range(batch_size)
            ]
            response = requests.post(
                f"{base_url}/generate",
                json={
                    "text": prompts,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": self.output_tokens,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                },
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            results = response.json()
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), batch_size)
            for result in results:
                meta = result["meta_info"]
                self.assertGreaterEqual(meta["prompt_tokens"], PARALLEL_SIZE)
                self.assertEqual(meta["completion_tokens"], self.output_tokens)
                logprobs = meta["output_token_logprobs"]
                self.assertEqual(len(logprobs), self.output_tokens)
                self.assertTrue(all(math.isfinite(item[0]) for item in logprobs))
                outputs.append(logprobs)
        return outputs

    def _compare(self, reference, actual, variant):
        self.assertEqual(len(reference), len(actual), variant)
        for request_id, (expected, observed) in enumerate(zip(reference, actual)):
            self.assertEqual(len(expected), len(observed), variant)
            self.assertEqual(
                [item[1] for item in expected],
                [item[1] for item in observed],
                f"{variant}: request {request_id} greedy token IDs differ",
            )
            for token_id, (left, right) in enumerate(zip(expected, observed)):
                self.assertAlmostEqual(
                    left[0],
                    right[0],
                    delta=self.logprob_atol,
                    msg=f"{variant}: request {request_id}, token {token_id} logprob",
                )

    def test_decode_tp_regression(self):
        reference = None
        variants = (
            ("baseline-eager", False, "disabled"),
            ("decode-tp-eager", True, "disabled"),
            ("decode-tp-graph", True, "full"),
        )
        for name, enabled, graph_backend in variants:
            base_url = f"http://127.0.0.1:{get_open_port()}"
            print(f"CP decode attention TP variant: {name}", flush=True)
            process = popen_launch_server(
                self.model,
                base_url,
                timeout=self.launch_timeout,
                other_args=server_args(enabled, graph_backend),
                env={**os.environ, **TEST_ENVS},
                device="npu",
            )
            try:
                response = requests.get(f"{base_url}/server_info", timeout=30)
                response.raise_for_status()
                info = response.json()
                self.assertEqual(info["enable_cp_decode_attn_tp"], enabled)
                self.assertTrue(info["enable_prefill_cp"])
                self.assertEqual(info["attn_cp_size"], PARALLEL_SIZE)
                self.assertEqual(info["tp_size"], PARALLEL_SIZE)
                self.assertEqual(info["dp_size"], 1)
                self.assertEqual(info["cp_strategy"], "interleave")
                # CudaGraphConfig.to_dict omits default values (decode=full).
                decode_graph = info["cuda_graph_config"]["decode"]
                self.assertEqual(decode_graph.get("backend", "full"), graph_backend)
                outputs = self._probe(base_url)
                if reference is None:
                    reference = outputs
                else:
                    self._compare(reference, outputs, name)

                # PRs still run ALL three differential variants above; only the
                # expensive dataset evaluation is omitted. No Paris-only bypass.
                if name == "decode-tp-graph" and os.environ.get(
                    "GITHUB_EVENT_NAME"
                ) != "pull_request":
                    metrics = run_eval(
                        SimpleNamespace(
                            max_tokens=512,
                            base_url=base_url,
                            model=self.model,
                            eval_name="gsm8k",
                            api="completion",
                            num_examples=1319,
                            num_threads=8,
                            num_shots=5,
                            temperature=0,
                        )
                    )
                    print(f"CP decode attention TP GSM8K: {metrics}", flush=True)
                    self.assertGreaterEqual(metrics["score"], 0.93)
            finally:
                # Only stop the server owned by this variant, including failures.
                terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
