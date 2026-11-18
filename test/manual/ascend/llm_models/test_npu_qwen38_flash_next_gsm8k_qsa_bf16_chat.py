"""EvalScope chat-path GSM8K verification for Qwen3.8-Flash-Next with the QSA
indexer in BF16 on Ascend NPUs.

Unlike the registered few-shot completions test
(`test/registered/npu/llm_models/test_npu_qwen38_flash_next_gsm8k_qsa_bf16.py`),
this manual test drives the model through the OpenAI **chat** API with
thinking mode enabled, which is the recommended interaction path for this
instruct multimodal model.

Measured results (2026-11-18, tp8 NPU, sglang qsa_1117 + sgl-kernel-npu main
3f4d922):
- 100 questions, indexer dtype auto (resolves to bfloat16 on NPU): 98%
- 5 questions, explicit ``--qsa-indexer-dtype bfloat16``: 100%

Model path, question count and accuracy threshold are overridable via
environment variables. Run with::

    GSM8K_MODEL_PATH=/mnt/paas/weights/Qwen3.8-Flash-Next \
    GSM8K_DATASET_PATH=/path/to/gsm8k \
    python test/manual/ascend/llm_models/test_npu_qwen38_flash_next_gsm8k_qsa_bf16_chat.py

[Test Category] Model
[Test Target] Qwen3.8-Flash-Next --qsa-indexer-dtype bfloat16 (chat path)
"""

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_accuracy_utils import assert_metrics
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)


@unittest.skipUnless(
    os.environ.get("QWEN38_FLASH_NEXT_MODEL_PATH"),
    "Set QWEN38_FLASH_NEXT_MODEL_PATH to the Qwen3.8-Flash-Next checkpoint",
)
class TestQwen38FlashNextGSM8KQsaBf16Chat(CustomTestCase):
    """QSA indexer cache stored in BF16, verified through the chat API."""

    model = os.environ.get("QWEN38_FLASH_NEXT_MODEL_PATH")
    accuracy = float(os.environ.get("GSM8K_ACCURACY", "0.90"))
    num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "100"))
    served_model_name = "qwen3.8-flash-next"
    base_url = "http://127.0.0.1:6678"
    env = {
        **os.environ,
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
    }
    other_args = [
        "--served-model-name",
        served_model_name,
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
        "--chunked-prefill-size",
        "4096",
        "--max-prefill-tokens",
        "52000",
        "--disable-radix-cache",
        "--max-running-requests",
        "8",
        "--mem-fraction-static",
        "0.8",
        "--enable-multimodal",
        "--mm-attention-backend",
        "ascend_attn",
        "--mm-enable-dp-encoder",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        # Feature under test.
        "--qsa-indexer-dtype",
        "bfloat16",
        "--trust-remote-code",
    ]
    generation_config = {
        "temperature": 0.9,
        "max_tokens": 250000,
        "seed": 3407,
        "n": 1,
        "timeout": 3600,
        "stream": True,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=cls.other_args,
            env=cls.env,
            device="npu",
        )
        cls.addClassCleanup(terminate_and_kill_process_tree, cls.process)

    def test_gsm8k_chat(self):
        output_root = Path(
            os.environ.get(
                "GSM8K_OUTPUT_DIR", "./outputs/gsm8k_qwen38_flashnext_qsa_bf16_chat"
            )
        )
        output_root.mkdir(parents=True, exist_ok=True)
        # Never reuse answers cached by an earlier evaluation.
        work_dir = Path(tempfile.mkdtemp(prefix="qsa_bf16_chat_", dir=output_root))
        command = [
            "evalscope",
            "eval",
            "--model",
            self.model,
            "--api-url",
            f"{self.base_url}/v1",
            "--api-key",
            "EMPTY",
            "--eval-type",
            "openai_api",
            "--limit",
            str(self.num_questions),
            "--generation-config",
            json.dumps(self.generation_config),
            "--datasets",
            "gsm8k",
            "--dataset-hub",
            "Local",
            "--dataset-args",
            json.dumps(
                {
                    "gsm8k": {
                        "local_path": os.environ.get(
                            "GSM8K_DATASET_PATH", "./gsm8k"
                        ),
                        "shuffle": True,
                        "shuffle_choices": True,
                    }
                }
            ),
            "--eval-batch-size",
            "32",
            "--ignore-errors",
            "--work-dir",
            str(work_dir),
        ]
        report_path = None
        with (
            (work_dir / "evalscope.log").open("w", encoding="utf-8") as log,
            subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            ) as process,
        ):
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    match = re.search(r"Dump report to:\s*(\S+)", line)
                    if match:
                        report_path = Path(match.group(1))
                self.assertEqual(process.wait(), 0, f"See {work_dir}/evalscope.log")
            finally:
                if process.poll() is None:
                    kill_process_tree(process.pid)
        self.assertIsNotNone(report_path, f"No EvalScope report: {work_dir}")
        with report_path.open(encoding="utf-8") as report_file:
            score = float(json.load(report_file)["score"])
        print(f"GSM8K chat: score={score}, report={report_path}", flush=True)
        assert_metrics(self, {"accuracy": score})


if __name__ == "__main__":
    unittest.main()
