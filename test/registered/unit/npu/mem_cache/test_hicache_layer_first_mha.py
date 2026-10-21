"""Unit tests for NPU HiCache MHA layer_first D2H/H2D transfer."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MHA_POOL_HOST_MODULE = "sglang.srt.mem_cache.pool_host.mha"

PAGE_SIZE = 16
HEAD_NUM = 2
HEAD_DIM = 4


def _indices(start: int, end: int) -> torch.Tensor:
    return torch.arange(start, end, dtype=torch.int64)


def _fill_layerwise(tensor: torch.Tensor, base: float) -> None:
    """Fill each layer with `arange + base + layer` so routing bugs are caught.

    The per-layer offset makes every (layer, position) value distinct; the
    arange structure preserves per-page/per-element fidelity under the mock
    page copy so page routing mistakes also fail the comparison.
    """
    for layer in range(tensor.shape[0]):
        tensor[layer].copy_(
            torch.arange(tensor[layer].numel(), dtype=torch.float16).reshape(
                tensor[layer].shape
            )
            + base
            + layer
        )


def _build_layer_first_host():
    """Build a layer_first host pool over 2 target + 1 draft MHA layers."""
    with mock.patch.dict("os.environ", {"ASCEND_USE_FIA": "True"}):
        pools = tuple(
            NPUMHATokenToKVPool(
                size=32,
                page_size=PAGE_SIZE,
                dtype=torch.float16,
                head_num=HEAD_NUM,
                head_dim=HEAD_DIM,
                layer_num=layers,
                device="cpu",
                enable_memory_saver=False,
                enable_alt_stream=False,
            )
            for layers in (2, 1)
        )
    infos = tuple(pool.get_device_pool_infos()[0].buffer_info for pool in pools)
    for pool in pools:
        pool.get_hicache_transfer_buffers = mock.Mock(
            side_effect=AssertionError("runtime owner query")
        )
    host = MHATokenToKVPoolHost(
        infos[0],
        2,
        0,
        PAGE_SIZE,
        "layer_first",
        pin_memory=False,
        device_capacity_tokens=32,
        start_layer=0,
        end_layer=2,
        draft_buffer_infos=infos[1:],
    )
    return host, infos


def _make_transfer(captured: list):
    """Mock `transfer_kv_dim_exchange` with a per-index page copy (D2H/H2D)."""

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
            f"{MHA_POOL_HOST_MODULE}.TransferDirection",
            SimpleNamespace(H2D="H2D", D2H="D2H"),
            create=True,
        ),
        mock.patch(
            f"{MHA_POOL_HOST_MODULE}.transfer_kv_dim_exchange",
            side_effect=_make_transfer(captured),
            create=True,
        ),
    ):
        yield


class TestNpuMhaLayerFirstTransfer(unittest.TestCase):
    def test_layer_first_d2h_transfers_one_layer_per_call(self):
        host, infos = _build_layer_first_host()
        self.addCleanup(host.destroy)

        target_k, target_v = infos[0].transfer_buffers()
        draft_k, draft_v = infos[1].transfer_buffers()
        _fill_layerwise(target_k, base=0)
        _fill_layerwise(target_v, base=10)
        _fill_layerwise(draft_k, base=20)
        _fill_layerwise(draft_v, base=30)

        # 3 device pages; transfer them to host pages 0..2.
        device_indices = _indices(0, 3)
        host_indices = _indices(0, 3)

        captured = []
        with _patch_ascend_transfer(captured):
            host.backup_from_device_all_layer(
                object(), host_indices, device_indices, "kernel_ascend"
            )

        # Target layers 0..1 plus one draft layer -> one call per layer.
        self.assertEqual(len(captured), 3)

        for call in captured:
            # The exchange operator requires a single layer per call (N=1):
            # device layer is dim 0, host layer is dim 1.
            self.assertEqual(call["device_k"].shape[0], 1)
            self.assertEqual(call["host_k"].shape[1], 1)
            self.assertEqual(call["device_v"].shape[0], 1)
            self.assertEqual(call["host_v"].shape[1], 1)
            self.assertEqual(call["host_k"].shape[0], host.page_num)
            self.assertEqual(call["host_k"].shape[2], PAGE_SIZE)
            self.assertEqual(call["direction"], "D2H")

        # Host layers 0/1 map to target layers 0/1; host layer 2 maps to the
        # single draft layer. The layer_first host buffer has no page dimension,
        # so each layer is paged as [page_num, page_size, head, dim].
        expected = [
            (target_k[0], target_v[0]),
            (target_k[1], target_v[1]),
            (draft_k[0], draft_v[0]),
        ]
        host_k = host.k_buffer
        host_v = host.v_buffer
        for layer_id, (k_src, v_src) in enumerate(expected):
            k_view = host_k[layer_id].view(
                host.page_num, PAGE_SIZE, HEAD_NUM, HEAD_DIM
            )
            v_view = host_v[layer_id].view(
                host.page_num, PAGE_SIZE, HEAD_NUM, HEAD_DIM
            )
            for page in range(3):
                self.assertTrue(
                    torch.equal(k_view[page], k_src[page]),
                    f"host K layer {layer_id} page {page} mismatch",
                )
                self.assertTrue(
                    torch.equal(v_view[page], v_src[page]),
                    f"host V layer {layer_id} page {page} mismatch",
                )

    def test_layer_first_backup_then_load_roundtrip(self):
        host, infos = _build_layer_first_host()
        self.addCleanup(host.destroy)

        target_k, target_v = infos[0].transfer_buffers()
        draft_k, draft_v = infos[1].transfer_buffers()
        _fill_layerwise(target_k, base=0)
        _fill_layerwise(target_v, base=10)
        _fill_layerwise(draft_k, base=20)
        _fill_layerwise(draft_v, base=30)

        snapshot = [
            (target_k.clone(), target_v.clone()),
            (draft_k.clone(), draft_v.clone()),
        ]

        device_indices = _indices(0, 3)
        host_indices = _indices(0, 3)

        captured = []
        with _patch_ascend_transfer(captured):
            # D2H: device -> host.
            host.backup_from_device_all_layer(
                object(), host_indices, device_indices, "kernel_ascend"
            )
            # Wipe the device buffers before loading them back.
            for k, v in ((target_k, target_v), (draft_k, draft_v)):
                k.zero_()
                v.zero_()
            # H2D: host -> device, one layer per call (mirrors the transfer
            # engine calling load_to_device_per_layer once per layer).
            for layer_id in range(host.layer_num):
                host.load_to_device_per_layer(
                    object(),
                    host_indices,
                    device_indices,
                    layer_id,
                    "kernel_ascend",
                    is_draft=layer_id >= host.target_layer_num,
                )

        # 3 D2H calls + 3 H2D calls.
        self.assertEqual(len(captured), 6)
        self.assertEqual([c["direction"] for c in captured], ["D2H"] * 3 + ["H2D"] * 3)

        # After the roundtrip each device layer matches its original fill.
        for (k_src, v_src), (k_dst, v_dst) in zip(
            snapshot, ((target_k, target_v), (draft_k, draft_v))
        ):
            self.assertTrue(torch.equal(k_dst, k_src))
            self.assertTrue(torch.equal(v_dst, v_src))

    def test_layer_first_h2d_loads_only_requested_layer(self):
        host, infos = _build_layer_first_host()
        self.addCleanup(host.destroy)

        target_k, target_v = infos[0].transfer_buffers()
        draft_k, draft_v = infos[1].transfer_buffers()

        # Seed host layers with distinct data and zero the device buffers.
        _fill_layerwise(host.k_buffer, base=0)
        _fill_layerwise(host.v_buffer, base=500)
        for k, v in ((target_k, target_v), (draft_k, draft_v)):
            k.zero_()
            v.zero_()

        device_indices = _indices(0, 3)
        host_indices = _indices(0, 3)

        captured = []
        with _patch_ascend_transfer(captured):
            # Load only global layer 1 (target layer 1).
            host.load_to_device_per_layer(
                object(), host_indices, device_indices, 1, "kernel_ascend"
            )

        # Exactly one exchange call, targeting layer 1 only.
        self.assertEqual(len(captured), 1)
        call = captured[0]
        self.assertEqual(call["direction"], "H2D")
        self.assertEqual(call["device_k"].shape[0], 1)
        self.assertEqual(call["host_k"].shape[1], 1)

        expected_k = host.k_buffer[1].view(
            host.page_num, PAGE_SIZE, HEAD_NUM, HEAD_DIM
        )[:3]
        expected_v = host.v_buffer[1].view(
            host.page_num, PAGE_SIZE, HEAD_NUM, HEAD_DIM
        )[:3]
        self.assertTrue(torch.equal(target_k[1], expected_k))
        self.assertTrue(torch.equal(target_v[1], expected_v))

        # The other layers remain untouched.
        self.assertTrue(torch.equal(target_k[0], torch.zeros_like(target_k[0])))
        self.assertTrue(torch.equal(target_v[0], torch.zeros_like(target_v[0])))
        self.assertTrue(torch.equal(draft_k[0], torch.zeros_like(draft_k[0])))
        self.assertTrue(torch.equal(draft_v[0], torch.zeros_like(draft_v[0])))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))