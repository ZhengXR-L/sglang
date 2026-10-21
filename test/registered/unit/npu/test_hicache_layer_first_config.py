"""Unit tests for NPU HiCache layer_first configuration (Step 10).

Coverage:
- layer_first layout is not overwritten by ``set_default_server_args``.
- page-first default projection is preserved for non-layer_first inputs.
- layer_first + any non-empty L3 storage backend is rejected at config time
  on the NPU (kernel_ascend) path only; GPU rewrites are unchanged.
- layer_first + ``SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric`` is rejected at
  startup.
"""

import os
import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.hicache_hook import resolve_storage_layout_compatibility
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.hardware_backend.npu.utils import set_default_server_args
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

NPU_UTILS_MODULE = "sglang.srt.hardware_backend.npu.utils"


def _make_args(**overrides) -> ServerArgs:
    return ServerArgs(model_path="dummy", **overrides)


def _run_set_default(use_mla: bool, **overrides) -> ServerArgs:
    args = _make_args(**overrides)
    with (
        patch(
            f"{NPU_UTILS_MODULE}.get_npu_memory_capacity",
            return_value=128 * 1024,
        ),
        patch(
            f"{NPU_UTILS_MODULE}.use_mla_backend",
            return_value=use_mla,
        ),
    ):
        set_default_server_args(args)
    return args


class TestHiCacheLayerFirstConfig(unittest.TestCase):
    def test_layer_first_not_overwritten_for_mla(self):
        args = _run_set_default(
            use_mla=True,
            enable_hierarchical_cache=True,
            hicache_mem_layout="layer_first",
        )
        self.assertEqual(resolution_result(args, "hicache_mem_layout"), "layer_first")
        self.assertEqual(
            resolution_result(args, "hicache_io_backend"), "kernel_ascend"
        )

    def test_layer_first_not_overwritten_for_mha(self):
        args = _run_set_default(
            use_mla=False,
            enable_hierarchical_cache=True,
            hicache_mem_layout="layer_first",
        )
        self.assertEqual(resolution_result(args, "hicache_mem_layout"), "layer_first")

    def test_page_first_default_still_projected(self):
        mla = _run_set_default(use_mla=True, enable_hierarchical_cache=True)
        self.assertEqual(
            resolution_result(mla, "hicache_mem_layout"), "page_first_kv_split"
        )
        mha = _run_set_default(use_mla=False, enable_hierarchical_cache=True)
        self.assertEqual(
            resolution_result(mha, "hicache_mem_layout"), "page_first_direct"
        )

    def test_layer_first_rejects_any_storage_backend_on_npu(self):
        for backend in ("mooncake", "npu_memcache", "file", "nixl", "hf3fs"):
            with self.subTest(backend=backend):
                args = _make_args(
                    hicache_mem_layout="layer_first",
                    hicache_storage_backend=backend,
                    hicache_io_backend="kernel_ascend",
                )
                with self.assertRaisesRegex(
                    ValueError, "layer_first layout does not support"
                ):
                    resolve_storage_layout_compatibility(args)

    def test_layer_first_pure_l1l2_passes(self):
        args = _make_args(
            hicache_mem_layout="layer_first",
            hicache_storage_backend=None,
            hicache_io_backend="kernel_ascend",
        )
        resolve_storage_layout_compatibility(args)
        self.assertEqual(resolution_result(args, "hicache_mem_layout"), "layer_first")

    def test_gpu_direct_rewrite_unchanged(self):
        args = _make_args(
            hicache_mem_layout="layer_first",
            hicache_storage_backend="mooncake",
            hicache_io_backend="direct",
        )
        resolve_storage_layout_compatibility(args)
        self.assertEqual(
            resolution_result(args, "hicache_mem_layout"), "page_first_direct"
        )

    def test_gpu_kernel_rewrite_unchanged(self):
        args = _make_args(
            hicache_mem_layout="layer_first",
            hicache_storage_backend="mooncake",
            hicache_io_backend="kernel",
        )
        resolve_storage_layout_compatibility(args)
        self.assertEqual(resolution_result(args, "hicache_mem_layout"), "page_first")

    def test_layer_first_rejects_memfabric(self):
        with patch.dict(
            os.environ, {"SGLANG_HICACHE_HOST_MEM_BACKEND": "memfabric"}
        ):
            with self.assertRaisesRegex(ValueError, "does not support"):
                _run_set_default(
                    use_mla=True,
                    enable_hierarchical_cache=True,
                    hicache_mem_layout="layer_first",
                )

    def test_layer_first_without_memfabric_passes(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_HICACHE_HOST_MEM_BACKEND", None)
            args = _run_set_default(
                use_mla=True,
                enable_hierarchical_cache=True,
                hicache_mem_layout="layer_first",
            )
        self.assertEqual(resolution_result(args, "hicache_mem_layout"), "layer_first")


if __name__ == "__main__":
    unittest.main()