"""Unit tests for NPU HiCache MLA layer_first special paths (indexer + FP8 DSA).

Coverage:
- Step 7: index_k / index_k_scale as independent components, ``_indexer_slot_range_for_layer``
  slot mapping, absolute-slot positioning and None skipping.
- Step 8: FP8 DSA empty V component (``v_width == 0``) allocation + component skip.
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.device_pool_info import MLABufferInfo, MLASplitBuffers
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MLA_POOL_HOST_MODULE = "sglang.srt.mem_cache.pool_host.mla"

PAGE_SIZE = 16
KV_LORA_RANK = 8
QK_ROPE_HEAD_DIM = 4
KV_CACHE_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
PACKED_K_WIDTH = 16  # FP8 DSA packed (latent + scale + rope) row width
LAYER_NUM = 3
DEVICE_PAGES = 3
INDEX_WIDTH = 6
INDEX_LAYERS = (1, 2)  # subset of layers that carry an indexer
DTYPE = torch.float16


def _make_split_info(
    *,
    indexer: bool,
    index_layer_indices: tuple = (),
    packed_fp8: bool = False,
    k_width: int = KV_LORA_RANK,
    v_width: int = QK_ROPE_HEAD_DIM,
    dtype: torch.dtype = DTYPE,
) -> MLABufferInfo:
    """Build an MLASplitBuffers-backed MLA info with CPU device tensors."""
    device_k = torch.zeros(
        (LAYER_NUM, DEVICE_PAGES, PAGE_SIZE, 1, k_width), dtype=dtype
    )
    device_v = torch.zeros(
        (LAYER_NUM, DEVICE_PAGES, PAGE_SIZE, 1, v_width), dtype=dtype
    )
    index_k = None
    index_scales = None
    if indexer:
        index_k = torch.zeros(
            (len(index_layer_indices), DEVICE_PAGES, PAGE_SIZE, 1, INDEX_WIDTH),
            dtype=dtype,
        )
        index_scales = torch.zeros(
            (len(index_layer_indices), DEVICE_PAGES, PAGE_SIZE, 1, 1),
            dtype=torch.float32,
        )

    def native_buffers() -> MLASplitBuffers:
        return MLASplitBuffers(
            k=device_k,
            v=device_v,
            index_k=index_k,
            index_scales=index_scales,
            index_layer_indices=index_layer_indices,
            k_width=k_width,
            v_width=v_width,
            index_width=INDEX_WIDTH,
        )

    return MLABufferInfo(
        buffers=native_buffers(),
        data_ptrs=None,
        layer_indices=tuple(range(LAYER_NUM)),
        device_layer_num=LAYER_NUM,
        page_size=PAGE_SIZE,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        kv_cache_dim=KV_CACHE_DIM,
        dtype=dtype,
        device=device_k.device,
        packed_fp8=packed_fp8,
        scales=None,
        native_buffers=native_buffers,
    )


def _fake_alloc_memfabric(dims, dtype, device, pin_memory, allocator):
    return torch.empty(dims, dtype=dtype)


@contextmanager
def _memfabric_pool_patch():
    # Step 9 forces memfabric for layer_first on NPU (is_npu() is True in this
    # container). Patch the memfabric allocation boundary out so pool building
    # exercises the real shape/capacity logic on ordinary CPU tensors.
    with (
        mock.patch(
            f"{MLA_POOL_HOST_MODULE}.memfabric_host_memory_enabled",
            return_value=True,
        ),
        mock.patch(
            f"{MLA_POOL_HOST_MODULE}.ensure_memfabric_capacity",
            return_value=None,
        ),
        mock.patch(
            f"{MLA_POOL_HOST_MODULE}.alloc_with_memfabric",
            side_effect=_fake_alloc_memfabric,
        ),
    ):
        yield


def _build_host(
    info: MLABufferInfo, *, override_kv_cache_dim: int | None = None
) -> MLATokenToKVPoolHost:
    with _memfabric_pool_patch():
        return MLATokenToKVPoolHost(
            info,
            2,
            0,
            PAGE_SIZE,
            "layer_first_kv_split",
            pin_memory=False,
            device="cpu",
            draft_buffer_infos=(),
            device_capacity_tokens=32,
            start_layer=0,
            end_layer=LAYER_NUM,
            target_layer_num=LAYER_NUM,
            override_kv_cache_dim=override_kv_cache_dim,
        )


def _fill_fp8(tensor: torch.Tensor, base: float) -> None:
    """Fill each layer with a distinct constant (exact in FP8) so mis-routing
    surfaces; ``copy_`` roundtrips bit-exactly for any fill pattern."""
    for layer in range(tensor.shape[0]):
        src = torch.full(tensor[layer].shape, float(base + layer), dtype=torch.float32)
        tensor[layer].copy_(src.to(tensor.dtype))


def _make_transfer(captured: list, host):
    """Mock ``_transfer_ascendc_sparse_copy`` with a layer-first CPU page copy.

    Indices are page-level here (a simplification of the real token-level
    indices); empty components (``numel() == 0``, e.g. FP8 DSA V) are skipped.
    ``buffers`` carries the device split tensors; the host mirrors come from
    ``host``.
    """

    def transfer(
        buffers,
        host_indices,
        device_indices,
        direction,
        layer_start=0,
        layer_num=-1,
        index_k_layer_start=0,
        index_k_layer_num=-1,
    ):
        device_k = buffers.k
        device_v = buffers.v
        host_k = host.k_buffer
        host_v = host.v_buffer
        captured.append(
            {
                "device_k": device_k,
                "device_v": device_v,
                "host_k": host_k,
                "host_v": host_v,
                "device_indices": device_indices,
                "host_indices": host_indices,
                "direction": direction,
                "layer_start": layer_start,
                "layer_num": layer_num,
            }
        )
        layers = device_k.shape[0] if layer_num < 0 else layer_num
        for device_buffer, host_buffer in ((device_k, host_k), (device_v, host_v)):
            if device_buffer.numel() == 0 or host_buffer.numel() == 0:
                continue
            for layer in range(layers):
                src_layer = layer_start + layer
                if direction == "D2H":
                    for di, hi in zip(device_indices.tolist(), host_indices.tolist()):
                        host_buffer[src_layer, hi].copy_(device_buffer[src_layer, di])
                else:
                    for di, hi in zip(device_indices.tolist(), host_indices.tolist()):
                        device_buffer[src_layer, di].copy_(host_buffer[src_layer, hi])

    return transfer


@contextmanager
def _patch_ascend_transfer(captured: list, host):
    with (
        mock.patch(
            f"{MLA_POOL_HOST_MODULE}.TransferDirection",
            SimpleNamespace(H2D="H2D", D2H="D2H"),
            create=True,
        ),
        mock.patch.object(
            MLATokenToKVPoolHost,
            "_transfer_ascendc_sparse_copy",
            side_effect=_make_transfer(captured, host),
        ),
    ):
        yield


class TestNpuMlaLayerFirstIndexer(unittest.TestCase):
    def test_indexer_slot_range_mapping(self):
        host = _build_host(
            _make_split_info(indexer=True, index_layer_indices=INDEX_LAYERS)
        )
        self.addCleanup(host.destroy)
        native = host._native_buffers

        # Hit: layer 2 is the second indexer slot.
        self.assertEqual(host._indexer_slot_range_for_layer(native, 2), (1, 1))
        # Hit: layer 1 is the first indexer slot.
        self.assertEqual(host._indexer_slot_range_for_layer(native, 1), (0, 1))
        # Miss: layer 0 has no indexer.
        self.assertEqual(host._indexer_slot_range_for_layer(native, 0), (0, 0))

    def test_indexer_slot_range_no_indexer(self):
        host = _build_host(_make_split_info(indexer=False))
        self.addCleanup(host.destroy)
        native = host._native_buffers
        self.assertEqual(host._indexer_slot_range_for_layer(native, 1), (0, -1))

    def test_build_component_meta_includes_indexer(self):
        host = _build_host(
            _make_split_info(indexer=True, index_layer_indices=INDEX_LAYERS)
        )
        self.addCleanup(host.destroy)
        native = host._native_buffers
        n = native.k.shape[0]
        num_indexer = len(INDEX_LAYERS)
        itemsize = DTYPE.itemsize

        comps = host._build_ascendc_component_meta(native, 0, n, 0, -1, 1)
        # K + V + index_k + index_k_scale.
        self.assertEqual(len(comps), 4)
        index_k_comp = comps[2]
        # host index_k is 5-D layer-first [num_indexer, page, page_size, 1, w].
        self.assertEqual(index_k_comp[4], host.index_k_buffer.stride(1) * itemsize)
        self.assertEqual(index_k_comp[5], host.index_k_buffer.stride(0) * itemsize)
        self.assertEqual(index_k_comp[6], PAGE_SIZE * INDEX_WIDTH * itemsize)
        self.assertEqual(index_k_comp[7], 0)
        self.assertEqual(index_k_comp[8], num_indexer)

    def test_build_component_meta_absolute_slot_positioning(self):
        host = _build_host(
            _make_split_info(indexer=True, index_layer_indices=INDEX_LAYERS)
        )
        self.addCleanup(host.destroy)
        native = host._native_buffers
        n = native.k.shape[0]

        # ik_start=1 selects the second indexer slot without double-offsetting
        # the (already base-aligned) index_k buffer.
        comps = host._build_ascendc_component_meta(native, 0, n, 1, 1, 1)
        self.assertEqual(len(comps), 4)
        self.assertEqual(comps[2][7], 1)
        self.assertEqual(comps[2][8], 2)
        self.assertEqual(comps[3][7], 1)
        self.assertEqual(comps[3][8], 2)

    def test_build_component_meta_skips_missing_indexer(self):
        # No index_k: K + V only.
        host = _build_host(_make_split_info(indexer=False))
        self.addCleanup(host.destroy)
        native = host._native_buffers
        n = native.k.shape[0]
        comps = host._build_ascendc_component_meta(native, 0, n, 0, -1, 1)
        self.assertEqual(len(comps), 2)

    def test_build_component_meta_skips_missing_scale(self):
        # index_k present but no index_scales: K + V + index_k only.
        info = _make_split_info(indexer=True, index_layer_indices=INDEX_LAYERS)
        # Drop the scales by re-wrapping native_buffers without them.
        native_buffers = info.native_buffers
        original = native_buffers()
        info_no_scale = MLABufferInfo(
            buffers=MLASplitBuffers(
                k=original.k,
                v=original.v,
                index_k=original.index_k,
                index_scales=None,
                index_layer_indices=original.index_layer_indices,
                k_width=original.k_width,
                v_width=original.v_width,
                index_width=original.index_width,
            ),
            data_ptrs=None,
            layer_indices=info.layer_indices,
            device_layer_num=info.device_layer_num,
            page_size=info.page_size,
            kv_lora_rank=info.kv_lora_rank,
            qk_rope_head_dim=info.qk_rope_head_dim,
            kv_cache_dim=info.kv_cache_dim,
            dtype=info.dtype,
            device=info.device,
            packed_fp8=info.packed_fp8,
            scales=None,
            native_buffers=info.native_buffers,
        )
        host = _build_host(info_no_scale)
        self.addCleanup(host.destroy)
        native = host._native_buffers
        n = native.k.shape[0]
        comps = host._build_ascendc_component_meta(native, 0, n, 0, -1, 1)
        self.assertEqual(len(comps), 3)


class TestNpuMlaLayerFirstFp8Dsa(unittest.TestCase):
    def test_fp8_dsa_empty_v_buffer(self):
        host = _build_host(
            _make_split_info(
                indexer=False, packed_fp8=True, k_width=PACKED_K_WIDTH, v_width=0
            )
        )
        self.addCleanup(host.destroy)
        page_num = host.page_num

        # V is packed into K for FP8 DSA: the host v mirror is a 5-D
        # zero-element tensor (not None), contributing no capacity.
        self.assertEqual(host.v_buffer.numel(), 0)
        self.assertEqual(
            host.v_buffer.shape, (LAYER_NUM, page_num, PAGE_SIZE, 1, 0)
        )
        # K uses the packed row width.
        self.assertEqual(host.k_buffer.shape[-1], PACKED_K_WIDTH)

    def test_fp8_dsa_skips_empty_v_component(self):
        host = _build_host(
            _make_split_info(
                indexer=False, packed_fp8=True, k_width=PACKED_K_WIDTH, v_width=0
            )
        )
        self.addCleanup(host.destroy)
        native = host._native_buffers
        n = native.k.shape[0]
        itemsize = DTYPE.itemsize

        comps = host._build_ascendc_component_meta(native, 0, n, 0, -1, 1)
        # Empty V component is skipped; only packed K remains.
        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0][6], PAGE_SIZE * PACKED_K_WIDTH * itemsize)

    def test_fp8_dsa_capacity_matches_packed_width(self):
        fp8 = torch.float8_e4m3fn
        host = _build_host(
            _make_split_info(
                indexer=False,
                packed_fp8=True,
                k_width=PACKED_K_WIDTH,
                v_width=0,
                dtype=fp8,
            ),
            override_kv_cache_dim=PACKED_K_WIDTH,
        )
        self.addCleanup(host.destroy)
        itemsize = fp8.itemsize

        self.assertEqual(host.k_buffer.dtype, fp8)
        self.assertEqual(host.k_buffer.shape[-1], PACKED_K_WIDTH)
        self.assertEqual(host.v_buffer.numel(), 0)
        # size_per_token uses the packed width (not kv_cache_dim=12) and
        # excludes the packed-away V component.
        self.assertEqual(
            host.get_size_per_token(), PACKED_K_WIDTH * itemsize * LAYER_NUM
        )
        # Capacity closure over host bytes.
        self.assertEqual(
            host.k_buffer.numel() * itemsize,
            host.size * host.get_size_per_token(),
        )

    def test_fp8_dsa_backup_then_load_roundtrip(self):
        fp8 = torch.float8_e4m3fn
        info = _make_split_info(
            indexer=False,
            packed_fp8=True,
            k_width=PACKED_K_WIDTH,
            v_width=0,
            dtype=fp8,
        )
        host = _build_host(info, override_kv_cache_dim=PACKED_K_WIDTH)
        self.addCleanup(host.destroy)
        native = host._native_buffers
        device_k = native.k
        _fill_fp8(device_k, base=1.0)
        snapshot = device_k.clone()

        device_indices = torch.arange(DEVICE_PAGES, dtype=torch.int64)
        host_indices = torch.arange(DEVICE_PAGES, dtype=torch.int64)

        captured = []
        with _patch_ascend_transfer(captured, host):
            host.backup_from_device_all_layer(
                object(), host_indices, device_indices, "kernel_ascend"
            )
            device_k.copy_(torch.zeros_like(device_k))
            for layer_id in range(LAYER_NUM):
                host.load_to_device_per_layer(
                    object(),
                    host_indices,
                    device_indices,
                    layer_id,
                    "kernel_ascend",
                    is_draft=False,
                )

        # 1 D2H (full segment) + LAYER_NUM H2D (per layer).
        self.assertEqual(len(captured), 1 + LAYER_NUM)
        self.assertTrue(torch.equal(device_k.float(), snapshot.float()))
        # V stayed empty (packed into K) and was never transferred.
        self.assertEqual(native.v.numel(), 0)


if __name__ == "__main__":
    unittest.main()