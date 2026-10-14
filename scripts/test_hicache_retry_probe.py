#!/usr/bin/env python3
"""CPU-only checks of the real retry implementation and the NPU log probe.

Run: python3 scripts/test_hicache_retry_probe.py
These checks do not establish that an NPU server can start or perform inference.
"""

import contextlib
import importlib.util
import io
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


retry = load("retry_under_test", ROOT / "python/sglang/srt/mem_cache/storage_prefetch.py")
probe = load("retry_probe", ROOT / "scripts/verify_hicache_retry_interval_npu.py")


class RetryProbeTest(unittest.TestCase):
    def check_log(self, log, interval):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(log, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return probe.check(path, interval)

    def test_real_retry_schedule_and_log_checker(self):
        for interval in (0, 1, 8):
            with self.subTest(interval=interval):
                stream = io.StringIO()
                handler = logging.StreamHandler(stream)
                previous_level = retry.logger.level
                retry.logger.addHandler(handler)
                retry.logger.setLevel(logging.DEBUG)
                try:
                    retries = retry.StoragePrefetchRetries()
                    head = SimpleNamespace(rid="head", storage_prefetch_retry_attempts=0)
                    queued = SimpleNamespace(rid="queued", storage_prefetch_retry_attempts=0)
                    retries.poll_miss(queued.rid)
                    for step in range(1, interval + 3):
                        ready = retries.pop_ready([head, queued], interval, 3)
                        expected = [(queued, None)] if interval > 0 and step == interval + 1 else []
                        self.assertEqual(ready, expected)
                finally:
                    retry.logger.removeHandler(handler)
                    retry.logger.setLevel(previous_level)
                self.assertEqual(self.check_log(stream.getvalue(), interval), 0)
                if interval:
                    self.assertIn(
                        f"scheduled_step=1 actual_interval={interval} "
                        "interval_match=True unit=scheduling_passes",
                        stream.getvalue(),
                    )
                    self.assertEqual(self.check_log(stream.getvalue(), interval + 1), 1)
                else:
                    self.assertIn("reason=disabled", stream.getvalue())
                    self.assertNotIn("event=fired", stream.getvalue())

    def test_no_evidence_is_inconclusive(self):
        self.assertEqual(self.check_log("HTTP 200 OK\n", 8), 2)
        self.assertEqual(self.check_log("HTTP 200 OK\n", 0), 2)

    def test_early_retry_fails(self):
        self.assertEqual(self.check_log(
            "HiCache storage retry event=scheduled pid=1 req=x step=10 interval=8 due_step=18\n"
            "HiCache storage retry event=fired pid=1 req=x step=17 interval=8 mode=poll due_step=18\n",
            8,
        ), 1)

    def test_head_budget_and_immediate(self):
        head = SimpleNamespace(rid="head", storage_prefetch_retry_attempts=0)
        queued = SimpleNamespace(rid="queued", storage_prefetch_retry_attempts=0)
        retries = retry.StoragePrefetchRetries()
        retries.poll_miss(head.rid)
        self.assertEqual(retries.pop_ready([head], 1, 3), [])
        self.assertEqual(retries.pop_ready([head], 1, 3), [])
        retries.refetch(queued.rid, 256)
        self.assertEqual(retries.pop_ready([head, queued], 0, 3), [(queued, 256)])
        retries.refetch(queued.rid, 256)
        self.assertEqual(retries.pop_ready([head, queued], 8, 0), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
