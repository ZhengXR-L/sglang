"""Unit tests for NPU HiCache MHA layer_first (AscendC sparse-copy path).

Coverage:
- Step 5: ``init_kv_buffer`` layer_first host buffer shapes + ``_ascendc_comp_meta``
  layer-first stride swap / width derivation.
- Step 6: ``backup_from_device_all_layer`` D2H by buffer_info segment and
  ``load_to_device_per_layer`` H2D single-layer relocation.
"""

import enum
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.device_pool_info import (
    KVBuffers,
    KVBufferLayout,
    MHABufferInfo,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MHA_POOL_HOST_MODULE = "sglang.srt.mem_cache.pool_host.mha"

PAGE_SIZE = 4
HEAD_NUM = 2
HEAD_DIM = 4
V_HEAD_DIM = 4  # symmetric MHA only (head_dim == v_head_dim)
DTYPE = torch.float16
DEVICE_CAPACITY_TOKENS = 8
HOST_TO_DEVICE_RATIO = 1
SIZE = (DEVICE_CAPACITY_TOKENS * HOST_TO_DEVICE_RATIO // PAGE_SIZE + 1) * PAGE_SIZE
PAGE_NUM = SIZE // PAGE_SIZE
TARGET_LAYER_NUM = 2
DRAFT_LAYER_NUM = 1

_Direction = enum.IntEnum("_Direction", {"H2D": 0, "D2H": 1})


def _indices(num_pages: int) -> torch.Tensor:
    return torch.arange(num_pages, dtype=torch.int64)


def _fill_layerwise(tensor: torch.Tensor, base: float) -> None:
    """Fill each layer so routing bugs surface on copy."""
    for layer in range(tensor.shape[0]):
        tensor[layer].copy_(
            torch.arange(tensor[layer].numel(), dtype=DTYPE).reshape(
                tensor[layer].shape
            )
            + base
            + layer
        )


def _make_mha_info(layer_num: int, transfer_buffers) -> MHABufferInfo:
    """Build an MHABufferInfo backed by ``transfer_buffers`` on CPU tensors."""
    per_layer_k = tuple(
        torch.zeros((SIZE, HEAD_NUM, HEAD_DIM), dtype=DTYPE) for _ in range(layer_num)
    )
    per_layer_v = tuple(
        torch.zeros((SIZE, HEAD_NUM, V_HEAD_DIM), dtype=DTYPE)
        for _ in range(layer_num)
    )
    buffers = KVBuffers(
        k=per_layer_k,
        v=per_layer_v,
        k_data_ptrs=None,
        v_data_ptrs=None,
        layout=KVBufferLayout.NHD,
    )
    return MHABufferInfo(
        buffers=buffers,
        page_size=PAGE_SIZE,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        dtype=DTYPE,
        device=per_layer_k[0].device,
        scales=None,
        transfer_buffers=transfer_buffers,
    )


def _fake_alloc_memfabric(dims, dtype, device, pin_memory, allocator):
    return torch.empty(dims, dtype=dtype)


@contextmanager
def _memfabric_pool_patch():
    # Step 9 forces memfabric for layer_first on NPU (is_npu() is True in this
    # container). Patch the memfabric allocation boundary out so pool building
    # exercises the real shape logic on ordinary CPU tensors.
    with (
        mock.patch(
            f"{MHA_POOL_HOST_MODULE}.memfabric_host_memory_enabled",
            return_value=True,
        ),
        mock.patch(
            f"{MHA_POOL_HOST_MODULE}.ensure_memfabric_capacity",
            return_value=None,
        ),
        mock.patch(
            f"{MHA_POOL_HOST_MODULE}.alloc_with_memfabric",
            side_effect=_fake_alloc_memfabric,
        ),
    ):
        yield


def _build_draft_host():
    """Build a layer_first MHA host over 2 target + 1 draft layers.

    ``transfer_buffers`` mirrors the production NPU 5-D layout
    ``[layer, page, page_size, head, dim]`` (see memory_pool_npu.py), while the
    host mirror stays 4-D ``[layer, size, head, dim]``.
    """
    target_k = torch.zeros(
        (TARGET_LAYER_NUM, PAGE_NUM, PAGE_SIZE, HEAD_NUM, HEAD_DIM), dtype=DTYPE
    )
    target_v = torch.zeros(
        (TARGET_LAYER_NUM, PAGE_NUM, PAGE_SIZE, HEAD_NUM, V_HEAD_DIM), dtype=DTYPE
    )
    draft_k = torch.zeros(
        (DRAFT_LAYER_NUM, PAGE_NUM, PAGE_SIZE, HEAD_NUM, HEAD_DIM), dtype=DTYPE
    )
    draft_v = torch.zeros(
        (DRAFT_LAYER_NUM, PAGE_NUM, PAGE_SIZE, HEAD_NUM, V_HEAD_DIM), dtype=DTYPE
    )

    def target_transfer_buffers():
        return target_k, target_v

    def draft_transfer_buffers():
        return draft_k, draft_v

    target_info = _make_mha_info(TARGET_LAYER_NUM, target_transfer_buffers)
    draft_info = _make_mha_info(DRAFT_LAYER_NUM, draft_transfer_buffers)
    with _memfabric_pool_patch():
        host = MHATokenToKVPoolHost(
            target_info,
            HOST_TO_DEVICE_RATIO,
            0,
            PAGE_SIZE,
            "layer_first",
            pin_memory=False,
            device="cpu",
            draft_buffer_infos=(draft_info,),
            device_capacity_tokens=DEVICE_CAPACITY_TOKENS,
            start_layer=0,
            end_layer=TARGET_LAYER_NUM,
        )
    return host, [(target_k, target_v), (draft_k, draft_v)]


def _make_transfer(captured: list):
    """Mock ``_transfer_ascendc_sparse_copy`` with a layer-first CPU page copy."""

    def transfer(
        device_k,
        device_v,
        host_k,
        host_v,
        host_indices,
        device_indices,
        direction,
        layer_start=0,
        layer_num=-1,
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
                "layer_start": layer_start,
                "layer_num": layer_num,
            }
        )
        for device_buffer, host_buffer in ((device_k, host_k), (device_v, host_v)):
            for layer in range(device_buffer.shape[0]):
                for di, hi in zip(device_indices.tolist(), host_indices.tolist()):
                    # device_buffer is 5-D [layer, page, page_size, head, dim]:
                    # index the page axis directly (di is a page index here).
                    d_slice = device_buffer[layer, di]
                    h_slice = host_buffer[
                        layer, hi * PAGE_SIZE : (hi + 1) * PAGE_SIZE
                    ]
                    if direction == "D2H":
                        h_slice.copy_(d_slice)
                    else:
                        d_slice.copy_(h_slice)

    return transfer


@contextmanager
def _patch_ascend_transfer(captured: list):
    with (
        mock.patch(
            f"{MHA_POOL_HOST_MODULE}.TransferDirection",
            SimpleNamespace(H2D="H2D", D2H="D2H"),
            create=True,
        ),
        mock.patch.object(
            MHATokenToKVPoolHost,
            "_transfer_ascendc_sparse_copy",
            side_effect=_make_transfer(captured),
        ),
    ):
        yield


class TestNpuMhaLayerFirstBuffer(unittest.TestCase):
    def test_layer_first_host_buffer_shapes(self):
        host, _ = _build_draft_host()
        self.addCleanup(host.destroy)

        self.assertEqual(host.layer_num, TARGET_LAYER_NUM + DRAFT_LAYER_NUM)
        self.assertEqual(
            host.k_buffer.shape,
            (TARGET_LAYER_NUM + DRAFT_LAYER_NUM, SIZE, HEAD_NUM, HEAD_DIM),
        )
        self.assertEqual(
            host.v_buffer.shape,
            (TARGET_LAYER_NUM + DRAFT_LAYER_NUM, SIZE, HEAD_NUM, V_HEAD_DIM),
        )
        self.assertEqual(host.k_buffer[0].shape, (SIZE, HEAD_NUM, HEAD_DIM))

    def test_comp_meta_swaps_host_stride_for_layer_first(self):
        n = 2
        pages = 3
        dev_k = torch.zeros((n, pages, PAGE_SIZE, HEAD_NUM, HEAD_DIM), dtype=DTYPE)
        host_k = torch.zeros((n, 4, PAGE_SIZE, HEAD_NUM, HEAD_DIM), dtype=DTYPE)
        itemsize = DTYPE.itemsize
        comp = MHATokenToKVPoolHost._ascendc_comp_meta(dev_k, host_k, 0, 2, 1)
        # Tuple: (dev_ptr, host_ptr, dev_s0, dev_s1, host_page_stride,
        #         host_layer_stride, width, lo, hi)
        self.assertEqual(comp[4], host_k.stride(1) * itemsize)
        self.assertEqual(comp[5], host_k.stride(0) * itemsize)
        self.assertEqual(comp[6], PAGE_SIZE * HEAD_NUM * HEAD_DIM * itemsize)


class TestNpuMhaLayerFirstTransfer(unittest.TestCase):
    def test_transfer_ascendc_shape_uses_5d_device(self):
        """Shape handling runs for real: 5-D device + 4-D host.

        Regression for treating ``device_k.shape[1]`` (the page count) as a
        token count and folding it into a bogus page axis. ``transfer_buffers``
        already yields ``[layer, page, page_size, head, dim]``; only the host
        mirror's implicit page axis is unfolded. Only the NPU boundary (stream /
        index upload / kernel launch) is mocked, not the shape derivation.
        """
        host, splits = _build_draft_host()
        self.addCleanup(host.destroy)
        target_k, target_v = splits[0]  # 5-D [layer, page, page_size, head, dim]

        # Production passes token-level indices (numel % page_size == 0).
        token_indices = torch.arange(PAGE_NUM * PAGE_SIZE, dtype=torch.int64)
        captured = []

        def fake_launch(vals, device):
            captured.append(vals)

        with (
            mock.patch(
                f"{MHA_POOL_HOST_MODULE}.to_device_no_sync",
                side_effect=lambda t, d: t,
            ),
            mock.patch("torch.npu.current_stream", return_value=None, create=True),
            mock.patch.object(
                torch.Tensor, "record_stream", lambda self, stream: None
            ),
            mock.patch(
                f"{MHA_POOL_HOST_MODULE}.TransferDirection",
                _Direction,
                create=True,
            ),
            mock.patch.object(
                MHATokenToKVPoolHost,
                "_launch_ascendc_kv_exchange",
                side_effect=fake_launch,
            ),
        ):
            host._transfer_ascendc_sparse_copy(
                target_k,
                target_v,
                host.k_buffer[:TARGET_LAYER_NUM],
                host.v_buffer[:TARGET_LAYER_NUM],
                token_indices,
                token_indices,
                _Direction.D2H,
                layer_start=0,
                layer_num=TARGET_LAYER_NUM,
            )

        self.assertEqual(len(captured), 1)
        vals = captured[0]
        itemsize = DTYPE.itemsize
        # header: 8 ints, then 9 ints per component.
        self.assertEqual(vals[0], 2)  # K + V components
        self.assertEqual(vals[1], PAGE_NUM)  # num_pages
        self.assertEqual(vals[2], PAGE_SIZE)
        # K component width (index 6 of its 9-int block).
        self.assertEqual(vals[8 + 6], PAGE_SIZE * HEAD_NUM * HEAD_DIM * itemsize)
        self.assertEqual(
            vals[8 + 9 + 6], PAGE_SIZE * HEAD_NUM * V_HEAD_DIM * itemsize
        )
        # K component lo/hi (indices 7/8 of the block).
        self.assertEqual(vals[8 + 7], 0)
        self.assertEqual(vals[8 + 8], TARGET_LAYER_NUM)

    def test_layer_first_d2h_backs_up_target_and_draft(self):
        host, splits = _build_draft_host()
        self.addCleanup(host.destroy)
        target_k, target_v = splits[0]
        draft_k, draft_v = splits[1]
        _fill_layerwise(target_k, base=0)
        _fill_layerwise(target_v, base=10)
        _fill_layerwise(draft_k, base=20)
        _fill_layerwise(draft_v, base=30)

        device_indices = _indices(PAGE_NUM)
        host_indices = _indices(PAGE_NUM)

        captured = []
        with _patch_ascend_transfer(captured):
            host.backup_from_device_all_layer(
                object(), host_indices, device_indices, "kernel_ascend"
            )

        # target once (n=2) + draft once (n=1).
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0]["device_k"].shape[0], TARGET_LAYER_NUM)
        self.assertEqual(captured[0]["host_k"].shape[0], TARGET_LAYER_NUM)
        self.assertEqual(captured[0]["layer_num"], TARGET_LAYER_NUM)
        self.assertEqual(captured[1]["device_k"].shape[0], DRAFT_LAYER_NUM)
        self.assertEqual(captured[1]["host_k"].shape[0], DRAFT_LAYER_NUM)
        self.assertEqual(captured[1]["layer_num"], DRAFT_LAYER_NUM)
        for call in captured:
            self.assertEqual(call["direction"], "D2H")

        expected = [
            (target_k[0], target_v[0]),
            (target_k[1], target_v[1]),
            (draft_k[0], draft_v[0]),
        ]
        for layer_id, (k_src, v_src) in enumerate(expected):
            # Host mirror is 4-D [size, head, dim]; the 5-D device source
            # flattens to the same page-interleaved token layout.
            self.assertTrue(
                torch.equal(
                    host.k_buffer[layer_id], k_src.reshape(SIZE, HEAD_NUM, HEAD_DIM)
                )
            )
            self.assertTrue(
                torch.equal(
                    host.v_buffer[layer_id], v_src.reshape(SIZE, HEAD_NUM, V_HEAD_DIM)
                )
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

        device_indices = _indices(PAGE_NUM)
        host_indices = _indices(PAGE_NUM)

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

        # 2 D2H + 3 H2D calls.
        self.assertEqual(len(captured), 5)
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

        device_indices = _indices(PAGE_NUM)
        host_indices = _indices(PAGE_NUM)
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
        self.assertTrue(torch.equal(target_k[0], torch.zeros_like(target_k[0])))
        self.assertTrue(torch.equal(target_v[0], torch.zeros_like(target_v[0])))
        self.assertTrue(torch.equal(draft_k[0], torch.zeros_like(draft_k[0])))
        self.assertTrue(torch.equal(draft_v[0], torch.zeros_like(draft_v[0])))

    def test_layer_first_h2d_draft_relocates_to_host_layer(self):
        host, splits = _build_draft_host()
        self.addCleanup(host.destroy)
        target_k, target_v = splits[0]
        draft_k, draft_v = splits[1]
        _fill_layerwise(target_k, base=0)
        _fill_layerwise(target_v, base=10)
        _fill_layerwise(draft_k, base=20)
        _fill_layerwise(draft_v, base=30)

        draft_k_snapshot = draft_k.clone()
        draft_v_snapshot = draft_v.clone()

        device_indices = _indices(PAGE_NUM)
        host_indices = _indices(PAGE_NUM)
        with _patch_ascend_transfer([]):
            host.backup_from_device_all_layer(
                object(), host_indices, device_indices, "kernel_ascend"
            )

        draft_k.zero_()
        draft_v.zero_()

        captured = []
        draft_layer_id = host.target_layer_num  # global layer id of the draft
        with _patch_ascend_transfer(captured):
            host.load_to_device_per_layer(
                object(),
                host_indices,
                device_indices,
                draft_layer_id,
                "kernel_ascend",
                is_draft=True,
            )

        self.assertEqual(len(captured), 1)
        self.assertTrue(torch.equal(draft_k[0], draft_k_snapshot[0]))
        self.assertTrue(torch.equal(draft_v[0], draft_v_snapshot[0]))


if __name__ == "__main__":
    unittest.main()