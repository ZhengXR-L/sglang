"""Unit tests for NPU HiCache MLA layer_first.

Coverage:
- Step 3: ``init_kv_buffer`` split allocation + ``_size_per_token`` closure.
- Step 4: ``backup_from_device_all_layer`` D2H (target + draft, one layer/call).
- Step 5: ``load_to_device_per_layer`` H2D (single layer semantics).
- Step 6: layer-sharded ``_backup_from_device_per_layer`` kernel_ascend branch.
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


def _build_host(layout: str, info: MLABufferInfo) -> MLATokenToKVPoolHost:
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


def _build_sharded_host():
    """Build a layer-sharded host owning layer 0 of a 2-layer device pool."""
    k, v, native = _make_split(2)
    info = _make_info(k, v, native, layer_indices=(0,), device_layer_num=2)
    host = MLATokenToKVPoolHost(
        info,
        2,
        0,
        PAGE_SIZE,
        "layer_first",
        pin_memory=False,
        device="cpu",
        draft_buffer_infos=(),
        device_capacity_tokens=32,
        start_layer=0,
        end_layer=1,
        target_layer_num=1,
        layer_shard_enabled=True,
    )
    return host, k, v


def _make_transfer(captured: list):
    """Mock ``transfer_kv_dim_exchange`` with a per-index page copy (D2H/H2D)."""

    def transfer(
        *,
        device_k,
        device_v,
        host_k,
        host_v,
        device_indices,
        host_indices,
        page_size,
        direction,
        **kwargs,
    ):
        captured.append(
            {
                "device_k": device_k,
                "device_v": device_v,
                "host_k": host_k,
                "host_v": host_v,
                "device_indices": device_indices,
                "host_indices": host_indices,
                "direction": direction,
            }
        )
        for device_buffer, host_buffer in ((device_k, host_k), (device_v, host_v)):
            if direction == "D2H":
                for di, hi in zip(device_indices.tolist(), host_indices.tolist()):
                    host_buffer[hi, 0].copy_(device_buffer[0, di])
            else:
                for di, hi in zip(device_indices.tolist(), host_indices.tolist()):
                    device_buffer[0, di].copy_(host_buffer[hi, 0])

    return transfer


@contextmanager
def _patch_ascend_transfer(captured: list):
    with (
        mock.patch(
            f"{MLA_POOL_HOST_MODULE}.TransferDirection",
            SimpleNamespace(H2D="H2D", D2H="D2H"),
            create=True,
        ),
        mock.patch(
            f"{MLA_POOL_HOST_MODULE}.transfer_kv_dim_exchange",
            side_effect=_make_transfer(captured),
            create=True,
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


class TestNpuMlaLayerFirstTransfer(unittest.TestCase):
    def test_layer_first_d2h_backs_up_target_and_draft(self):
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
        with _patch_ascend_transfer(captured):
            host.backup_from_device_all_layer(
                object(), host_indices, device_indices, "kernel_ascend"
            )

        # 2 target + 1 draft -> one call per layer, N=1.
        self.assertEqual(len(captured), 3)
        for call in captured:
            self.assertEqual(call["device_k"].shape[0], 1)
            self.assertEqual(call["host_k"].shape[1], 1)
            self.assertEqual(call["device_v"].shape[0], 1)
            self.assertEqual(call["host_v"].shape[1], 1)
            self.assertEqual(call["host_k"].shape[2], PAGE_SIZE)
            self.assertEqual(call["direction"], "D2H")

        # Host rows 0/1 map to target layers 0/1; host row 2 maps to the draft.
        expected = [
            (target_k[0], target_v[0]),
            (target_k[1], target_v[1]),
            (draft_k[0], draft_v[0]),
        ]
        for layer_id, (k_src, v_src) in enumerate(expected):
            for page in range(DEVICE_PAGES):
                self.assertTrue(
                    torch.equal(host.k_buffer[layer_id][page], k_src[page]),
                    f"host K layer {layer_id} page {page} mismatch",
                )
                self.assertTrue(
                    torch.equal(host.v_buffer[layer_id][page], v_src[page]),
                    f"host V layer {layer_id} page {page} mismatch",
                )

    def test_layer_first_backup_then_load_roundtrip(self):
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

        captured = []
        with _patch_ascend_transfer(captured):
            host.backup_from_device_all_layer(
                object(), host_indices, device_indices, "kernel_ascend"
            )
            for k, v in ((target_k, target_v), (draft_k, draft_v)):
                k.zero_()
                v.zero_()
            for layer_id in range(host.layer_num):
                host.load_to_device_per_layer(
                    object(),
                    host_indices,
                    device_indices,
                    layer_id,
                    "kernel_ascend",
                    is_draft=layer_id >= host.target_layer_num,
                )

        # 3 D2H + 3 H2D calls.
        self.assertEqual(len(captured), 6)
        for (k_snap, v_snap), (k, v) in zip(
            snapshot, ((target_k, target_v), (draft_k, draft_v))
        ):
            self.assertTrue(torch.equal(k, k_snap))
            self.assertTrue(torch.equal(v, v_snap))

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
        with _patch_ascend_transfer([]):
            host.backup_from_device_all_layer(
                object(), host_indices, device_indices, "kernel_ascend"
            )

        for k, v in ((target_k, target_v), (draft_k, draft_v)):
            k.zero_()
            v.zero_()

        captured = []
        with _patch_ascend_transfer(captured):
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

    def test_layer_sharded_backup_uses_kernel_ascend_branch(self):
        host, k, v = _build_sharded_host()
        self.addCleanup(host.destroy)
        _fill_layerwise(k, base=0)
        _fill_layerwise(v, base=10)

        device_indices = _indices(0, DEVICE_PAGES)
        host_indices = _indices(0, DEVICE_PAGES)

        captured = []
        with _patch_ascend_transfer(captured):
            host.backup_from_device_all_layer(
                object(), host_indices, device_indices, "kernel_ascend"
            )

        # Only the owned layer 0 is backed up.
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["direction"], "D2H")
        self.assertTrue(torch.equal(host.k_buffer[0][:DEVICE_PAGES], k[0]))
        self.assertTrue(torch.equal(host.v_buffer[0][:DEVICE_PAGES], v[0]))


if __name__ == "__main__":
    unittest.main()