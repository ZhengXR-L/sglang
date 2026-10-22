"""Unit tests for NPU HiCache MLA layer_first (AscendC sparse-copy path).

Coverage:
- Step 2: ``init_kv_buffer`` split allocation + ``_size_per_token`` closure.
- Step 3: ``_ascendc_comp_meta`` / ``_build_ascendc_component_meta``
  ``host_layout_mode=1`` host stride swap + ``backup_from_device_all_layer``
  D2H (target once + each draft once).
- Step 4: ``load_to_device_per_layer`` H2D single-layer semantics.
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
LAYER_NUM = 2
DEVICE_PAGES = 3
INDEX_WIDTH = 6
DTYPE = torch.float16


def _indices(start: int, end: int) -> torch.Tensor:
    return torch.arange(start, end, dtype=torch.int64)


def _fill_layerwise(tensor: torch.Tensor, base: float) -> None:
    """Fill each layer with ``arange + base + layer`` so routing bugs surface."""
    for layer in range(tensor.shape[0]):
        tensor[layer].copy_(
            torch.arange(tensor[layer].numel(), dtype=DTYPE).reshape(
                tensor[layer].shape
            )
            + base
            + layer
        )


def _make_mla_info(indexer: bool) -> MLABufferInfo:
    """Build an MLASplitBuffers-backed MLA info with CPU device tensors."""
    device_k = torch.zeros(
        (LAYER_NUM, DEVICE_PAGES, PAGE_SIZE, 1, KV_LORA_RANK), dtype=DTYPE
    )
    device_v = torch.zeros(
        (LAYER_NUM, DEVICE_PAGES, PAGE_SIZE, 1, QK_ROPE_HEAD_DIM), dtype=DTYPE
    )
    index_k = None
    index_scales = None
    index_layer_indices = ()
    if indexer:
        index_layer_indices = tuple(range(LAYER_NUM))
        index_k = torch.zeros(
            (LAYER_NUM, DEVICE_PAGES, PAGE_SIZE, 1, INDEX_WIDTH), dtype=DTYPE
        )
        index_scales = torch.zeros(
            (LAYER_NUM, DEVICE_PAGES, PAGE_SIZE, 1, 1), dtype=torch.float32
        )

    def native_buffers() -> MLASplitBuffers:
        return MLASplitBuffers(
            k=device_k,
            v=device_v,
            index_k=index_k,
            index_scales=index_scales,
            index_layer_indices=index_layer_indices,
            k_width=KV_LORA_RANK,
            v_width=QK_ROPE_HEAD_DIM,
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
        dtype=DTYPE,
        device=device_k.device,
        packed_fp8=False,
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


def _build_host(layout: str, info: MLABufferInfo) -> MLATokenToKVPoolHost:
    with _memfabric_pool_patch():
        return MLATokenToKVPoolHost(
            info,
            2,
            0,
            PAGE_SIZE,
            layout,
            pin_memory=False,
            device="cpu",
            draft_buffer_infos=(),
            device_capacity_tokens=32,
            start_layer=0,
            end_layer=LAYER_NUM,
            target_layer_num=LAYER_NUM,
        )


def _make_split(layer_num: int):
    """Return (k, v, native_buffers) for a split MLA info with ``layer_num`` layers."""
    k = torch.zeros(
        (layer_num, DEVICE_PAGES, PAGE_SIZE, 1, KV_LORA_RANK), dtype=DTYPE
    )
    v = torch.zeros(
        (layer_num, DEVICE_PAGES, PAGE_SIZE, 1, QK_ROPE_HEAD_DIM), dtype=DTYPE
    )

    def native_buffers():
        return MLASplitBuffers(
            k=k,
            v=v,
            index_k=None,
            index_scales=None,
            index_layer_indices=(),
            k_width=KV_LORA_RANK,
            v_width=QK_ROPE_HEAD_DIM,
            index_width=INDEX_WIDTH,
        )

    return k, v, native_buffers


def _make_info(k, v, native_buffers, *, layer_indices, device_layer_num) -> MLABufferInfo:
    return MLABufferInfo(
        buffers=native_buffers(),
        data_ptrs=None,
        layer_indices=layer_indices,
        device_layer_num=device_layer_num,
        page_size=PAGE_SIZE,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        kv_cache_dim=KV_CACHE_DIM,
        dtype=DTYPE,
        device=k.device,
        packed_fp8=False,
        scales=None,
        native_buffers=native_buffers,
    )


def _build_draft_host():
    """Build a layer_first host over 2 target + 1 draft MLA split layers."""
    target_k, target_v, target_native = _make_split(2)
    draft_k, draft_v, draft_native = _make_split(1)
    target_info = _make_info(
        target_k, target_v, target_native, layer_indices=(0, 1), device_layer_num=2
    )
    draft_info = _make_info(
        draft_k, draft_v, draft_native, layer_indices=(0,), device_layer_num=1
    )
    with _memfabric_pool_patch():
        host = MLATokenToKVPoolHost(
            target_info,
            2,
            0,
            PAGE_SIZE,
            "layer_first",
            pin_memory=False,
            device="cpu",
            draft_buffer_infos=(draft_info,),
            device_capacity_tokens=32,
            start_layer=0,
            end_layer=2,
            target_layer_num=2,
        )
    return host, [(target_k, target_v), (draft_k, draft_v)]


def _make_transfer(captured: list, host):
    """Mock ``_transfer_ascendc_sparse_copy`` with a CPU page copy over the
    ``buffers`` signature; the host mirrors come from ``host``.
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


class TestNpuMlaLayerFirstAllocation(unittest.TestCase):
    def test_layer_first_split_allocates_separate_kv(self):
        host = _build_host("layer_first", _make_mla_info(indexer=False))
        self.addCleanup(host.destroy)

        page_num = host.page_num
        self.assertEqual(
            host.k_buffer.shape,
            (LAYER_NUM, page_num, PAGE_SIZE, 1, KV_LORA_RANK),
        )
        self.assertEqual(
            host.v_buffer.shape,
            (LAYER_NUM, page_num, PAGE_SIZE, 1, QK_ROPE_HEAD_DIM),
        )
        self.assertIsNone(host.index_k_buffer)
        self.assertIsNone(host.index_k_scale_buffer)
        self.assertEqual(len(host.data_refs), LAYER_NUM)
        self.assertEqual(
            host.data_refs[0].shape, (page_num, PAGE_SIZE, 1, KV_LORA_RANK)
        )

    def test_layer_first_capacity_matches_page_first(self):
        lf = _build_host("layer_first", _make_mla_info(indexer=False))
        pf = _build_host("page_first_kv_split", _make_mla_info(indexer=False))
        self.addCleanup(lf.destroy)
        self.addCleanup(pf.destroy)

        self.assertEqual(lf.size, pf.size)
        self.assertEqual(lf.page_num, pf.page_num)
        self.assertEqual(lf.get_size_per_token(), pf.get_size_per_token())
        self.assertEqual(lf.k_buffer.numel(), pf.k_buffer.numel())
        self.assertEqual(lf.v_buffer.numel(), pf.v_buffer.numel())

    def test_layer_first_indexer_buffers_and_capacity(self):
        host = _build_host("layer_first", _make_mla_info(indexer=True))
        self.addCleanup(host.destroy)

        page_num = host.page_num
        self.assertEqual(
            host.index_k_buffer.shape,
            (LAYER_NUM, page_num, PAGE_SIZE, 1, INDEX_WIDTH),
        )
        self.assertEqual(
            host.index_k_scale_buffer.shape,
            (LAYER_NUM, page_num, PAGE_SIZE, 1, 1),
        )
        expected = (
            KV_CACHE_DIM * DTYPE.itemsize * LAYER_NUM
            + INDEX_WIDTH * DTYPE.itemsize * LAYER_NUM
            + 4 * LAYER_NUM
        )
        self.assertEqual(host.get_size_per_token(), expected)


class TestNpuMlaLayerFirstComponentMeta(unittest.TestCase):
    def test_comp_meta_swaps_host_stride_for_layer_first(self):
        dev_k = torch.zeros(
            (LAYER_NUM, DEVICE_PAGES, PAGE_SIZE, 1, KV_LORA_RANK), dtype=DTYPE
        )
        host_k = torch.zeros(
            (LAYER_NUM, 4, PAGE_SIZE, 1, KV_LORA_RANK), dtype=DTYPE
        )
        itemsize = DTYPE.itemsize
        comp = MLATokenToKVPoolHost._ascendc_comp_meta(dev_k, host_k, 0, 2, 1)
        # Tuple: (dev_ptr, host_ptr, dev_s0, dev_s1, host_page_stride,
        #         host_layer_stride, width, lo, hi)
        self.assertEqual(comp[4], host_k.stride(1) * itemsize)
        self.assertEqual(comp[5], host_k.stride(0) * itemsize)
        self.assertEqual(comp[6], PAGE_SIZE * KV_LORA_RANK * itemsize)

    def test_build_component_meta_layer_first_kv(self):
        host = _build_host("layer_first", _make_mla_info(indexer=False))
        self.addCleanup(host.destroy)
        native = host._native_buffers
        n = native.k.shape[0]
        comps = host._build_ascendc_component_meta(native, 0, n, 0, -1, 1)
        self.assertEqual(len(comps), 2)
        k_comp = comps[0]
        itemsize = DTYPE.itemsize
        # host_page_stride = stride(1), host_layer_stride = stride(0)
        self.assertEqual(k_comp[4], host.k_buffer.stride(1) * itemsize)
        self.assertEqual(k_comp[5], host.k_buffer.stride(0) * itemsize)
        self.assertEqual(k_comp[6], PAGE_SIZE * KV_LORA_RANK * itemsize)
        self.assertEqual(k_comp[7], 0)
        self.assertEqual(k_comp[8], n)


class TestNpuMlaLayerFirstTransfer(unittest.TestCase):
    def test_layer_first_d2h_backs_up_target_only(self):
        host, splits = _build_draft_host()
        self.addCleanup(host.destroy)
        target_k, target_v = splits[0]
        draft_k, draft_v = splits[1]
        _fill_layerwise(target_k, base=0)
        _fill_layerwise(target_v, base=10)
        _fill_layerwise(draft_k, base=20)
        _fill_layerwise(draft_v, base=30)

        device_indices = _indices(0, DEVICE_PAGES)
        host_indices = _indices(0, DEVICE_PAGES)

        captured = []
        with _patch_ascend_transfer(captured, host):
            host.backup_from_device_all_layer(
                object(), host_indices, device_indices, "kernel_ascend"
            )

        # D2H covers the target pool only (draft is ignored, like
        # page_first_kv_split).
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["device_k"].shape[0], 2)
        self.assertEqual(captured[0]["layer_num"], -1)
        self.assertEqual(captured[0]["direction"], "D2H")

        for layer_id in range(2):
            for page in range(DEVICE_PAGES):
                self.assertTrue(
                    torch.equal(host.k_buffer[layer_id][page], target_k[layer_id][page]),
                    f"host K layer {layer_id} page {page} mismatch",
                )
                self.assertTrue(
                    torch.equal(
                        host.v_buffer[layer_id][page], target_v[layer_id][page]
                    ),
                    f"host V layer {layer_id} page {page} mismatch",
                )
        # Draft host layer (index 2) is never backed up and stays zero.
        self.assertTrue(torch.equal(host.k_buffer[2], torch.zeros_like(host.k_buffer[2])))
        self.assertTrue(torch.equal(host.v_buffer[2], torch.zeros_like(host.v_buffer[2])))

    def test_layer_first_backup_then_load_roundtrip(self):
        host, splits = _build_draft_host()
        self.addCleanup(host.destroy)
        target_k, target_v = splits[0]
        _fill_layerwise(target_k, base=0)
        _fill_layerwise(target_v, base=10)

        snapshot = (target_k.clone(), target_v.clone())

        device_indices = _indices(0, DEVICE_PAGES)
        host_indices = _indices(0, DEVICE_PAGES)

        captured = []
        with _patch_ascend_transfer(captured, host):
            host.backup_from_device_all_layer(
                object(), host_indices, device_indices, "kernel_ascend"
            )
            target_k.zero_()
            target_v.zero_()
            for layer_id in range(host.target_layer_num):
                host.load_to_device_per_layer(
                    object(),
                    host_indices,
                    device_indices,
                    layer_id,
                    "kernel_ascend",
                    is_draft=False,
                )

        # 1 D2H (target only) + target_layer_num H2D calls; draft is ignored.
        self.assertEqual(len(captured), 1 + host.target_layer_num)
        self.assertTrue(torch.equal(target_k, snapshot[0]))
        self.assertTrue(torch.equal(target_v, snapshot[1]))

    def test_layer_first_h2d_loads_only_requested_layer(self):
        host, splits = _build_draft_host()
        self.addCleanup(host.destroy)
        target_k, target_v = splits[0]
        draft_k, draft_v = splits[1]
        _fill_layerwise(target_k, base=0)
        _fill_layerwise(target_v, base=10)
        _fill_layerwise(draft_k, base=20)
        _fill_layerwise(draft_v, base=30)

        snapshot = [
            (target_k.clone(), target_v.clone()),
            (draft_k.clone(), draft_v.clone()),
        ]

        device_indices = _indices(0, DEVICE_PAGES)
        host_indices = _indices(0, DEVICE_PAGES)
        with _patch_ascend_transfer([], host):
            host.backup_from_device_all_layer(
                object(), host_indices, device_indices, "kernel_ascend"
            )

        for k, v in ((target_k, target_v), (draft_k, draft_v)):
            k.zero_()
            v.zero_()

        captured = []
        with _patch_ascend_transfer(captured, host):
            host.load_to_device_per_layer(
                object(),
                host_indices,
                device_indices,
                1,
                "kernel_ascend",
                is_draft=False,
            )

        self.assertEqual(len(captured), 1)
        # Only target layer 1 is restored.
        self.assertTrue(torch.equal(target_k[1], snapshot[0][0][1]))
        self.assertTrue(torch.equal(target_v[1], snapshot[0][1][1]))
        # All other layers stay zero.
        self.assertTrue(torch.equal(target_k[0], torch.zeros_like(target_k[0])))
        self.assertTrue(torch.equal(target_v[0], torch.zeros_like(target_v[0])))
        self.assertTrue(torch.equal(draft_k[0], torch.zeros_like(draft_k[0])))
        self.assertTrue(torch.equal(draft_v[0], torch.zeros_like(draft_v[0])))


if __name__ == "__main__":
    unittest.main()