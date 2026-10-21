"""Unit tests for NPU HiCache MLA layer_first indexer transfer (Step 7).

Verifies ``_transfer_one_layer`` routes ``index_k`` / ``index_k_scale`` for the
indexer subset of layers, maps device layer -> indexer slot correctly, and
passes the host scale as a 5-D tensor.
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
# FP8 DSA packs latent + rope + scale into a single wider k row; V is folded in
# so the device v_buffer carries width 0.
DSA_K_WIDTH = 12
DTYPE = torch.float16


def _indices(start: int, end: int) -> torch.Tensor:
    return torch.arange(start, end, dtype=torch.int64)


def _fill_layerwise(tensor: torch.Tensor, base: float, dtype=DTYPE) -> None:
    for layer in range(tensor.shape[0]):
        tensor[layer].copy_(
            torch.arange(tensor[layer].numel(), dtype=dtype).reshape(
                tensor[layer].shape
            )
            + base
            + layer
        )


def _build_indexer_host():
    """Build a layer_first host where only layer 1 owns an indexer cache."""
    k = torch.zeros(
        (LAYER_NUM, DEVICE_PAGES, PAGE_SIZE, 1, KV_LORA_RANK), dtype=DTYPE
    )
    v = torch.zeros(
        (LAYER_NUM, DEVICE_PAGES, PAGE_SIZE, 1, QK_ROPE_HEAD_DIM), dtype=DTYPE
    )
    # One indexer slot, owned by device layer 1 (index_layer_indices=(1,)).
    index_k = torch.zeros(
        (1, DEVICE_PAGES, PAGE_SIZE, 1, INDEX_WIDTH), dtype=DTYPE
    )
    index_scales = torch.zeros(
        (1, DEVICE_PAGES, PAGE_SIZE, 1), dtype=torch.float32
    )

    def native_buffers():
        return MLASplitBuffers(
            k=k,
            v=v,
            index_k=index_k,
            index_scales=index_scales,
            index_layer_indices=(1,),
            k_width=KV_LORA_RANK,
            v_width=QK_ROPE_HEAD_DIM,
            index_width=INDEX_WIDTH,
        )

    info = MLABufferInfo(
        buffers=native_buffers(),
        data_ptrs=None,
        layer_indices=(0, 1),
        device_layer_num=LAYER_NUM,
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
        end_layer=LAYER_NUM,
        target_layer_num=LAYER_NUM,
    )
    return host, (k, v), (index_k, index_scales)


def _build_dsa_host():
    """Build a layer_first host for an FP8 DSA pool: v_width == 0.

    The device v_buffer is a 5-D zero-element tensor (numel == 0), folding the
    V component into the packed k_buffer. No indexer keeps the fixture focused
    on the empty-V path.
    """
    k = torch.zeros((LAYER_NUM, DEVICE_PAGES, PAGE_SIZE, 1, DSA_K_WIDTH), dtype=DTYPE)
    v = torch.zeros((LAYER_NUM, DEVICE_PAGES, PAGE_SIZE, 1, 0), dtype=DTYPE)

    def native_buffers():
        return MLASplitBuffers(
            k=k,
            v=v,
            index_k=None,
            index_scales=None,
            index_layer_indices=(),
            k_width=DSA_K_WIDTH,
            v_width=0,
            index_width=0,
        )

    info = MLABufferInfo(
        buffers=native_buffers(),
        data_ptrs=None,
        layer_indices=(0, 1),
        device_layer_num=LAYER_NUM,
        page_size=PAGE_SIZE,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        kv_cache_dim=DSA_K_WIDTH,
        dtype=DTYPE,
        device=k.device,
        packed_fp8=True,
        scales=None,
        native_buffers=native_buffers,
    )

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
        end_layer=LAYER_NUM,
        target_layer_num=LAYER_NUM,
        override_kv_cache_dim=DSA_K_WIDTH,
    )
    return host, (k, v)


def _make_transfer(captured: list):
    """Mock ``transfer_kv_dim_exchange`` copying K/V plus optional indexers."""

    def transfer(
        *,
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k=None,
        host_index_k=None,
        device_index_k_scale=None,
        host_index_k_scale=None,
        device_indices,
        host_indices,
        page_size,
        direction,
        **kwargs,
    ):
        captured.append(
            {
                "device_k": device_k,
                "host_k": host_k,
                "device_v": device_v,
                "host_v": host_v,
                "device_index_k": device_index_k,
                "host_index_k": host_index_k,
                "device_index_k_scale": device_index_k_scale,
                "host_index_k_scale": host_index_k_scale,
                "device_indices": device_indices,
                "host_indices": host_indices,
                "direction": direction,
            }
        )

        def copy_pages(device_buffer, host_buffer):
            if direction == "D2H":
                for di, hi in zip(device_indices.tolist(), host_indices.tolist()):
                    host_buffer[hi, 0].copy_(device_buffer[0, di])
            else:
                for di, hi in zip(device_indices.tolist(), host_indices.tolist()):
                    device_buffer[0, di].copy_(host_buffer[hi, 0])

        copy_pages(device_k, host_k)
        copy_pages(device_v, host_v)
        if device_index_k is not None and host_index_k is not None:
            copy_pages(device_index_k, host_index_k)
        if device_index_k_scale is not None and host_index_k_scale is not None:
            # Mirror the wrapper's trailing-singleton pad for the 4-D device
            # scale operand before the per-page copy.
            if device_index_k_scale.dim() == 4:
                device_index_k_scale = device_index_k_scale.unsqueeze(-1)
            copy_pages(device_index_k_scale, host_index_k_scale)

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


class TestNpuMlaLayerFirstIndexerTransfer(unittest.TestCase):
    def test_indexer_backup_routes_index_k_and_scale(self):
        host, (k, v), (index_k, index_scales) = _build_indexer_host()
        self.addCleanup(host.destroy)
        _fill_layerwise(k, base=0)
        _fill_layerwise(v, base=10)
        _fill_layerwise(index_k, base=100)
        _fill_layerwise(index_scales, base=200, dtype=torch.float32)

        device_indices = _indices(0, DEVICE_PAGES)
        host_indices = _indices(0, DEVICE_PAGES)

        captured = []
        with _patch_ascend_transfer(captured):
            host.backup_from_device_all_layer(
                object(), host_indices, device_indices, "kernel_ascend"
            )

        self.assertEqual(len(captured), 2)
        # Layer 0 owns no indexer slot -> index params left as None.
        self.assertIsNone(captured[0]["device_index_k"])
        self.assertIsNone(captured[0]["host_index_k"])
        self.assertIsNone(captured[0]["device_index_k_scale"])
        self.assertIsNone(captured[0]["host_index_k_scale"])
        # Layer 1 -> slot 0 (not slot 1); scale operands 4-D device / 5-D host.
        self.assertIsNotNone(captured[1]["device_index_k"])
        self.assertIsNotNone(captured[1]["host_index_k"])
        self.assertEqual(captured[1]["device_index_k_scale"].dim(), 4)
        self.assertEqual(captured[1]["host_index_k_scale"].dim(), 5)

        # slot mapping + data: index_k/scale of layer 1 land in host slot 0.
        for page in range(DEVICE_PAGES):
            self.assertTrue(
                torch.equal(host.index_k_buffer[0][page], index_k[0][page])
            )
            self.assertTrue(
                torch.equal(
                    host.index_k_scale_buffer[0][page],
                    index_scales[0][page].unsqueeze(-1),
                )
            )

    def test_indexer_backup_then_load_roundtrip(self):
        host, (k, v), (index_k, index_scales) = _build_indexer_host()
        self.addCleanup(host.destroy)
        _fill_layerwise(k, base=0)
        _fill_layerwise(v, base=10)
        _fill_layerwise(index_k, base=100)
        _fill_layerwise(index_scales, base=200, dtype=torch.float32)

        index_k_snapshot = index_k.clone()
        index_scales_snapshot = index_scales.clone()

        device_indices = _indices(0, DEVICE_PAGES)
        host_indices = _indices(0, DEVICE_PAGES)

        captured = []
        with _patch_ascend_transfer(captured):
            host.backup_from_device_all_layer(
                object(), host_indices, device_indices, "kernel_ascend"
            )
            index_k.zero_()
            index_scales.zero_()
            for layer_id in range(LAYER_NUM):
                host.load_to_device_per_layer(
                    object(),
                    host_indices,
                    device_indices,
                    layer_id,
                    "kernel_ascend",
                    is_draft=False,
                )

        self.assertTrue(torch.equal(index_k, index_k_snapshot))
        self.assertTrue(torch.equal(index_scales, index_scales_snapshot))


class TestNpuMlaLayerFirstDsaEmptyV(unittest.TestCase):
    def test_dsa_empty_v_buffer_shape_and_capacity(self):
        host, (k, v) = _build_dsa_host()
        self.addCleanup(host.destroy)
        # v_buffer is a 5-D zero-element tensor (not None): the operator's V
        # operand stays 5-D while numel()==0 skips the copy branch.
        self.assertIsNotNone(host.v_buffer)
        self.assertEqual(host.v_buffer.dim(), 5)
        self.assertEqual(host.v_buffer.shape[-1], 0)
        self.assertEqual(host.v_buffer.numel(), 0)
        self.assertEqual(
            tuple(host.k_buffer.shape),
            (LAYER_NUM, host.page_num, PAGE_SIZE, 1, DSA_K_WIDTH),
        )
        # Capacity closure: packed k width only; no separate V component.
        self.assertEqual(
            host.get_size_per_token(),
            DSA_K_WIDTH * DTYPE.itemsize * LAYER_NUM,
        )

    def test_dsa_backup_then_load_empty_v(self):
        host, (k, v) = _build_dsa_host()
        self.addCleanup(host.destroy)
        _fill_layerwise(k, base=0)

        snapshot = k.clone()

        device_indices = _indices(0, DEVICE_PAGES)
        host_indices = _indices(0, DEVICE_PAGES)

        captured = []
        with _patch_ascend_transfer(captured):
            host.backup_from_device_all_layer(
                object(), host_indices, device_indices, "kernel_ascend"
            )

        # Every per-layer call passes 5-D zero-element V operands.
        self.assertEqual(len(captured), LAYER_NUM)
        for call in captured:
            self.assertEqual(call["device_v"].dim(), 5)
            self.assertEqual(call["host_v"].dim(), 5)
            self.assertEqual(call["device_v"].numel(), 0)
            self.assertEqual(call["host_v"].numel(), 0)

        k.zero_()
        load_captured = []
        with _patch_ascend_transfer(load_captured):
            for layer_id in range(LAYER_NUM):
                host.load_to_device_per_layer(
                    object(),
                    host_indices,
                    device_indices,
                    layer_id,
                    "kernel_ascend",
                    is_draft=False,
                )

        self.assertTrue(torch.equal(k, snapshot))


if __name__ == "__main__":
    unittest.main()