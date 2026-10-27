"""Exercise the actual MLA sparse dispatch without importing the NPU runtime."""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]


class Buffer:
    def __init__(self, shape, dtype="bf16"):
        self.shape, self.dtype = shape, dtype

    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def __getitem__(self, key):
        return ("host-slice", self, key[1].start, key[1].stop)


class TestNPUSparseDispatch(unittest.TestCase):
    def setUp(self):
        path = ROOT / "python/sglang/srt/mem_cache/pool_host/mla.py"
        tree = ast.parse(path.read_text())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MLATokenToKVPoolHost"
        )
        names = {
            "load_to_device_per_layer",
            "backup_from_device_all_layer",
            "_npu_transfer_pool_slices",
        }
        methods = ast.Module(
            body=[
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name in names
            ],
            type_ignores=[],
        )
        self.sparse, self.aiv = Mock(), Mock()
        self.ascendc_enabled = Mock(return_value=False)
        self.localize = Mock(side_effect=lambda indices, *_: indices)
        self.namespace = {
            "_is_npu": True,
            "ascendc_io_enabled": self.ascendc_enabled,
            "TransferDirection": SimpleNamespace(H2D=1, D2H=2),
            "maybe_dcp_kernel_indices": self.localize,
            "transfer_kv_sparse_copy": self.sparse,
        }
        exec(compile(methods, str(path), "exec"), self.namespace)
        host_class = type("Host", (), {name: self.namespace[name] for name in names})
        self.host = host_class()
        h = self.host
        h._is_dummy = False
        h.dcp_size, h.dcp_rank = 1, 0
        h._target_layer_indices = (0, 1, 2)
        h._host_layer_by_controller = {0: 0, 1: 1, 2: 2}
        h._device_layer_num = h.layer_num = 3
        h._layer_shard_enabled = False
        h.layout, h.page_size = "page_first_kv_split", 128
        h.k_buffer, h.v_buffer = Buffer((8, 3, 128, 1, 512)), Buffer((8, 3, 128, 1, 64))
        native = SimpleNamespace(
            k=Buffer((3, 8, 128, 1, 512)),
            v=Buffer((3, 8, 128, 1, 64)),
            index_k=None,
            index_scales=None,
        )
        h._native_infos = [SimpleNamespace(native_buffers=lambda: native)]
        h.index_k_buffer = h.index_k_scale_buffer = None
        h._transfer_ascendc_sparse_copy = self.aiv
        h._indexer_slot_range_for_layer = lambda *_: (0, 0)

    def test_h2d_directly_calls_sparse_once_for_all_layers(self):
        self.host.load_to_device_per_layer(
            None, "host-indices", "device-indices", 0, "kernel_ascend"
        )
        self.host.load_to_device_per_layer(
            None, "host-indices", "device-indices", 1, "kernel_ascend"
        )
        self.sparse.assert_called_once()
        self.assertEqual(self.sparse.call_args.kwargs["direction"], 1)
        self.assertEqual(self.sparse.call_args.kwargs["host_indices"], "host-indices")

    def test_d2h_directly_calls_sparse(self):
        self.host.backup_from_device_all_layer(None, "host", "device", "kernel_ascend")
        self.sparse.assert_called_once()
        self.assertEqual(self.sparse.call_args.kwargs["direction"], 2)

    def test_existing_ascendc_h2d_keeps_per_layer_dispatch(self):
        self.ascendc_enabled.return_value = True
        self.host.load_to_device_per_layer(None, "host", "device", 0, "kernel_ascend")
        self.host.load_to_device_per_layer(None, "host", "device", 1, "kernel_ascend")
        self.assertEqual(self.aiv.call_count, 2)
        self.assertEqual(self.aiv.call_args.kwargs["layer_start"], 1)
        self.assertEqual(self.aiv.call_args.kwargs["layer_num"], 1)
        self.sparse.assert_not_called()

    def test_existing_ascendc_d2h_keeps_all_layer_dispatch(self):
        self.ascendc_enabled.return_value = True
        self.host.backup_from_device_all_layer(None, "host", "device", "kernel_ascend")
        self.aiv.assert_called_once()
        self.assertEqual(self.aiv.call_args.args[3], 2)
        self.sparse.assert_not_called()

    def test_draft_keeps_separate_host_layer_slice(self):
        h = self.host
        draft = SimpleNamespace(
            k=Buffer((1, 8, 128, 1, 512)),
            v=Buffer((1, 8, 128, 1, 64)),
            index_k=None,
            index_scales=None,
        )
        h._native_infos.append(SimpleNamespace(native_buffers=lambda: draft))
        h._npu_draft_pool_compatible = lambda _: True
        h.backup_from_device_all_layer(None, "host", "device", "kernel_ascend")
        self.assertEqual(self.sparse.call_count, 2)
        self.assertEqual(self.sparse.call_args.kwargs["host_k"][2:], (3, 4))
        self.assertIsNone(self.sparse.call_args.kwargs["host_index_k"])

    def test_draft_passes_native_indexer_with_absent_host_buffers(self):
        h = self.host
        target = h._native_infos[0].native_buffers()
        target.index_k = Buffer((2, 8, 128, 1, 128))
        target.index_scales = Buffer((2, 8, 128, 1))
        h.index_k_buffer = Buffer((8, 2, 128, 1, 128))
        h.index_k_scale_buffer = Buffer((8, 2, 128, 1, 1))
        draft = SimpleNamespace(
            k=Buffer((1, 8, 128, 1, 512)),
            v=Buffer((1, 8, 128, 1, 64)),
            index_k=Buffer((1, 8, 128, 1, 128)),
            index_scales=Buffer((1, 8, 128, 1)),
        )
        h._native_infos.append(SimpleNamespace(native_buffers=lambda: draft))
        h._npu_draft_pool_compatible = lambda _: True
        h._warn_draft_indexer_skipped = Mock()
        for direction in ("load", "backup"):
            self.sparse.reset_mock()
            if direction == "load":
                h.load_to_device_per_layer(None, "host", "device", 0, "kernel_ascend")
            else:
                h.backup_from_device_all_layer(None, "host", "device", "kernel_ascend")
            self.assertEqual(self.sparse.call_count, 2)
            target_args = self.sparse.call_args_list[0].kwargs
            self.assertIs(target_args["device_index_k"], target.index_k)
            self.assertIs(target_args["host_index_k"], h.index_k_buffer)
            self.assertIs(target_args["device_index_k_scale"], target.index_scales)
            self.assertIs(target_args["host_index_k_scale"], h.index_k_scale_buffer)
            draft_args = self.sparse.call_args.kwargs
            self.assertIs(draft_args["device_index_k"], draft.index_k)
            self.assertIs(draft_args["device_index_k_scale"], draft.index_scales)
            self.assertIsNone(draft_args["host_index_k"])
            self.assertIsNone(draft_args["host_index_k_scale"])
        h._warn_draft_indexer_skipped.assert_called()

    def test_dcp_localization_is_preserved(self):
        self.host.dcp_size, self.host.dcp_rank = 2, 1
        self.localize.side_effect = lambda indices, *_: f"local-{indices}"
        self.host.backup_from_device_all_layer(None, "host", "device", "kernel_ascend")
        self.localize.assert_any_call("host", 2, 1)
        self.localize.assert_any_call("device", 2, 1)
        self.assertEqual(self.sparse.call_args.kwargs["host_indices"], "local-host")
        self.assertEqual(self.sparse.call_args.kwargs["device_indices"], "local-device")

    def controller_move(self):
        path = ROOT / "python/sglang/srt/managers/cache_controller.py"
        tree = ast.parse(path.read_text())
        controller = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "HiCacheController"
        )
        method = next(
            n
            for n in controller.body
            if isinstance(n, ast.FunctionDef) and n.name == "move_indices"
        )
        namespace = dict(self.namespace, torch=SimpleNamespace(Tensor=object))
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        return namespace["move_indices"]

    def test_mla_controller_keeps_indices_on_npu_without_switch(self):
        move = self.controller_move()
        ctl = SimpleNamespace(
            io_backend="kernel_ascend",
            device="npu",
            mem_pool_host=SimpleNamespace(layout="page_first_kv_split"),
        )
        di = SimpleNamespace(
            cpu=Mock(side_effect=AssertionError("unexpected CPU readback"))
        )
        hi = SimpleNamespace(device="npu")
        upload = Mock(return_value=hi)
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.mem_cache.pool_host.npu_memfabric": SimpleNamespace(
                    ascendc_io_enabled=lambda: False, to_device_no_sync=upload
                )
            },
        ):
            self.assertEqual(move(ctl, hi, di), (hi, di))
            cpu_hi = SimpleNamespace(device="cpu")
            self.assertEqual(move(ctl, cpu_hi, di), (hi, di))
            upload.assert_called_once_with(cpu_hi, "npu")

    def test_other_layouts_keep_existing_cpu_index_policy(self):
        move = self.controller_move()
        ctl = SimpleNamespace(
            io_backend="kernel_ascend",
            device="npu",
            mem_pool_host=SimpleNamespace(layout="page_first"),
        )
        di = SimpleNamespace(cpu=Mock(return_value="cpu-indices"))
        hi = SimpleNamespace(device="cpu")
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.mem_cache.pool_host.npu_memfabric": SimpleNamespace(
                    ascendc_io_enabled=lambda: False, to_device_no_sync=Mock()
                )
            },
        ):
            self.assertEqual(move(ctl, hi, di), (hi, "cpu-indices"))
            di.cpu.assert_called_once()


if __name__ == "__main__":
    unittest.main()
