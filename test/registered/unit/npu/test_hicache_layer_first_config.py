"""Unit tests for NPU HiCache layer_first config enforcement (Step 9).

Coverage:
- ``hicache_hook.resolve_storage_layout_compatibility`` rejects
  ``layer_first + non-empty L3 + kernel_ascend`` at config time (and leaves
  the pure L1/L2 and non-kernel_ascend GPU paths untouched).
- ``npu.utils.set_default_server_args`` no longer projects an explicit
  ``--hicache-mem-layout layer_first`` back to the page_first defaults.
- The generic MHA/MLA host pool ``init_kv_buffer`` requires memfabric for
  ``layer_first`` on NPU, rather than silently falling back to ordinary
  pinned host memory.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.arg_groups.hicache_hook import resolve_storage_layout_compatibility
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.hardware_backend.npu import utils as npu_utils
from sglang.srt.mem_cache.device_pool_info import (
    KVBuffers,
    KVBufferLayout,
    MHABufferInfo,
    MLABufferInfo,
    MLASplitBuffers,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MLA_MODULE = "sglang.srt.mem_cache.pool_host.mla"
MHA_MODULE = "sglang.srt.mem_cache.pool_host.mha"

DTYPE = torch.float16


def _build_layer_first_mla_host():
    page_size = 16
    layer_num = 2
    kv_lora = 8
    qk_rope = 4
    k = torch.zeros((layer_num, 3, page_size, 1, kv_lora), dtype=DTYPE)
    v = torch.zeros((layer_num, 3, page_size, 1, qk_rope), dtype=DTYPE)

    def native_buffers():
        return MLASplitBuffers(
            k=k,
            v=v,
            index_k=None,
            index_scales=None,
            index_layer_indices=(),
            k_width=kv_lora,
            v_width=qk_rope,
            index_width=6,
        )

    info = MLABufferInfo(
        buffers=native_buffers(),
        data_ptrs=None,
        layer_indices=(0, 1),
        device_layer_num=layer_num,
        page_size=page_size,
        kv_lora_rank=kv_lora,
        qk_rope_head_dim=qk_rope,
        kv_cache_dim=kv_lora + qk_rope,
        dtype=DTYPE,
        device=k.device,
        packed_fp8=False,
        scales=None,
        native_buffers=native_buffers,
    )
    return MLATokenToKVPoolHost(
        info,
        2,
        0,
        page_size,
        "layer_first_kv_split",
        pin_memory=False,
        device="cpu",
        draft_buffer_infos=(),
        device_capacity_tokens=32,
        start_layer=0,
        end_layer=layer_num,
        target_layer_num=layer_num,
    )


def _build_layer_first_mha_host():
    page_size = 4
    layer_num = 2
    head_num = 2
    head_dim = 4
    device_capacity = 8
    ratio = 1
    size = (device_capacity * ratio // page_size + 1) * page_size
    page_num = size // page_size
    per_layer_k = tuple(
        torch.zeros((size, head_num, head_dim), dtype=DTYPE) for _ in range(layer_num)
    )
    per_layer_v = tuple(
        torch.zeros((size, head_num, head_dim), dtype=DTYPE) for _ in range(layer_num)
    )
    buffers = KVBuffers(
        k=per_layer_k,
        v=per_layer_v,
        k_data_ptrs=None,
        v_data_ptrs=None,
        layout=KVBufferLayout.NHD,
    )

    def transfer_buffers():
        tk = torch.zeros((layer_num, page_num, page_size, head_num, head_dim), dtype=DTYPE)
        tv = torch.zeros((layer_num, page_num, page_size, head_num, head_dim), dtype=DTYPE)
        return tk, tv

    info = MHABufferInfo(
        buffers=buffers,
        page_size=page_size,
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=head_dim,
        dtype=DTYPE,
        device=per_layer_k[0].device,
        scales=None,
        transfer_buffers=transfer_buffers,
    )
    return MHATokenToKVPoolHost(
        info,
        ratio,
        0,
        page_size,
        "layer_first_kv_split",
        pin_memory=False,
        device="cpu",
        draft_buffer_infos=(),
        device_capacity_tokens=device_capacity,
        start_layer=0,
        end_layer=layer_num,
    )


class TestLayerFirstStorageCompatibility(unittest.TestCase):
    def test_layer_first_rejects_l3_with_kernel_ascend(self):
        for backend in ("file", "mooncake", "npu_memcache"):
            with self.subTest(backend=backend):
                args = ServerArgs(
                    model_path="dummy",
                    hicache_mem_layout="layer_first",
                    hicache_storage_backend=backend,
                    hicache_io_backend="kernel_ascend",
                )
                with self.assertRaisesRegex(ValueError, "storage backend"):
                    resolve_storage_layout_compatibility(args)

    def test_layer_first_allows_pure_l1l2_kernel_ascend(self):
        args = ServerArgs(
            model_path="dummy",
            hicache_mem_layout="layer_first",
            hicache_storage_backend=None,
            hicache_io_backend="kernel_ascend",
        )
        resolve_storage_layout_compatibility(args)

    def test_gpu_layer_first_l3_is_not_rejected(self):
        args = ServerArgs(
            model_path="dummy",
            hicache_mem_layout="layer_first",
            hicache_storage_backend="file",
            hicache_io_backend="kernel",
        )
        resolve_storage_layout_compatibility(args)


class TestLayerFirstMemfabricEnforcement(unittest.TestCase):
    def test_mla_layer_first_requires_memfabric(self):
        with (
            mock.patch(f"{MLA_MODULE}._is_npu", True),
            mock.patch(f"{MLA_MODULE}.memfabric_host_memory_enabled", return_value=False),
        ):
            with self.assertRaisesRegex(
                ValueError, "SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric"
            ):
                _build_layer_first_mla_host()

    def test_mha_layer_first_requires_memfabric(self):
        with (
            mock.patch(f"{MHA_MODULE}._is_npu", True),
            mock.patch(f"{MHA_MODULE}.memfabric_host_memory_enabled", return_value=False),
        ):
            with self.assertRaisesRegex(
                ValueError, "SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric"
            ):
                _build_layer_first_mha_host()


class TestLayerFirstLayoutProjection(unittest.TestCase):
    def test_layer_first_layout_not_overridden_by_npu_default(self):
        args = ServerArgs(
            model_path="dummy",
            enable_hierarchical_cache=True,
            hicache_mem_layout="layer_first",
        )
        with (
            mock.patch.object(
                npu_utils, "get_npu_memory_capacity", return_value=128 * 1024
            ),
            mock.patch.object(npu_utils, "use_mla_backend", return_value=False),
        ):
            npu_utils.set_default_server_args(args)
        self.assertEqual(
            resolution_result(args, "hicache_mem_layout"), "layer_first_kv_split"
        )

    def test_page_first_default_still_projected(self):
        args = ServerArgs(
            model_path="dummy",
            enable_hierarchical_cache=True,
        )
        with (
            mock.patch.object(
                npu_utils, "get_npu_memory_capacity", return_value=128 * 1024
            ),
            mock.patch.object(npu_utils, "use_mla_backend", return_value=False),
        ):
            npu_utils.set_default_server_args(args)
        self.assertEqual(
            resolution_result(args, "hicache_mem_layout"), "page_first_direct"
        )


if __name__ == "__main__":
    unittest.main()