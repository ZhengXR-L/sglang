"""Scheduling-pass coverage for HiCache L3 prefetch retry polling."""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=10, suite="base-b-test-1-npu-a3")


class TestNPUHiCacheStoragePrefetchRetry(unittest.TestCase):
    """Verify --hicache-storage-prefetch-retry-poll-interval without a server."""

    def test_miss_retry_follows_poll_interval(self):
        for interval in (0, 1, 8):
            with self.subTest(interval=interval):
                retries = StoragePrefetchRetries()
                head = SimpleNamespace(rid="head", storage_prefetch_retry_attempts=0)
                queued = SimpleNamespace(rid="queued", storage_prefetch_retry_attempts=0)
                retries.poll_miss(queued.rid)

                # A speculative miss on the queue head is cancelled, so keep
                # the request under test behind another waiting request.
                for scheduling_pass in range(1, interval + 2):
                    ready = retries.pop_ready([head, queued], interval, 3)
                    if interval > 0 and scheduling_pass == interval + 1:
                        self.assertEqual(ready, [(queued, None)])
                    else:
                        self.assertEqual(ready, [])

                # 0 disables miss polling, rather than leaving a retry pending.
                if interval == 0:
                    self.assertEqual(retries.pop_ready([head, queued], interval, 3), [])

    def test_immediate_refetch_ignores_poll_interval(self):
        for interval in (0, 8):
            with self.subTest(interval=interval):
                retries = StoragePrefetchRetries()
                head = SimpleNamespace(rid="head", storage_prefetch_retry_attempts=0)
                queued = SimpleNamespace(rid="queued", storage_prefetch_retry_attempts=0)
                retries.refetch(queued.rid, storage_hit_end=256)

                self.assertEqual(
                    retries.pop_ready([head, queued], interval, 3),
                    [(queued, 256)],
                )


if __name__ == "__main__":
    unittest.main()
