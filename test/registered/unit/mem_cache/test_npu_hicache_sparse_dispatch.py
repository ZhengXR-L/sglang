"""Exercise real MLA PF/LF routing and layer ownership without an NPU runtime."""

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


class TestNPUSparseDispatch(unittest.TestCase):
    def setUp(self):
        path = ROOT / "python/sglang/srt/mem_cache/pool_host/mla.py"
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MLATokenToKVPoolHost"
        )
        names = {
            "load_to_device_per_layer",
            "backup_from_device_all_layer",
            "_init_npu_transfer_bindings",
            "_npu_draft_pool_compatible",
            "_warn_draft_indexer_skipped",
            "init_kv_buffer",
        }
        methods = ast.Module(
            body=[
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name in names
            ],
            type_ignores=[],
        )
        self.ops = {
            name: Mock(name=name)
            for name in (
                "transfer_kv_per_layer_kv_split_pf_lf",
                "transfer_kv_all_layer_kv_split_lf_pf",
                "transfer_kv_per_layer_kv_split",
                "transfer_kv_all_layer_kv_split",
            )
        }
        self.localize = Mock(side_effect=lambda indices, *_: indices)
        self.namespace = dict(
            self.ops,
            _is_npu=True,
            maybe_dcp_kernel_indices=self.localize,
            logger=Mock(),
            memfabric_host_memory_enabled=lambda: False,
        )
        exec(compile(methods, str(path), "exec"), self.namespace)
        host_class = type("Host", (), {name: self.namespace[name] for name in names})
        self.host = h = host_class()
        h._is_dummy = False
        h.dcp_size, h.dcp_rank = 1, 0
        h._target_layer_indices = (0, 1, 2)
        h._host_layer_by_controller = {0: 0, 1: 1, 2: 2, 3: 3}
        h._device_layer_num = h.layer_num = 3
        h._layer_shard_enabled = False
        h.layout, h.page_size = "page_first_kv_split", 128
        h.k_buffer = Buffer((8, 3, 128, 1, 512))
        h.v_buffer = Buffer((8, 3, 128, 1, 64))
        self.native = SimpleNamespace(
            k=Buffer((3, 8, 128, 1, 512)),
            v=Buffer((3, 8, 128, 1, 64)),
            index_k=Buffer((2, 8, 128, 1, 128)),
            index_scales=Buffer((2, 8, 128, 1)),
            index_layer_indices=(0, 2),
            index_width=128,
            k_width=512,
        )
        h._native_infos = [SimpleNamespace(native_buffers=lambda: self.native)]
        h._native_buffers = self.native
        h.index_k_buffer = Buffer((8, 2, 128, 1, 128))
        h.index_k_scale_buffer = Buffer((8, 2, 128, 1, 1))
        h._init_npu_transfer_bindings()

    def test_h2d_transfers_each_layer_and_only_its_indexer_slot(self):
        for layout, name in (
            ("page_first_kv_split", "transfer_kv_per_layer_kv_split_pf_lf"),
            ("layer_first_kv_split", "transfer_kv_per_layer_kv_split"),
        ):
            with self.subTest(layout=layout):
                self.host.layout = layout
                op = self.ops[name]
                for layer in range(3):
                    self.host.load_to_device_per_layer(
                        None, "host", "device", layer, "kernel_ascend"
                    )
                self.assertEqual(op.call_count, 3)
                for layer, call in enumerate(op.call_args_list):
                    kw = call.kwargs
                    self.assertEqual(kw["device_layer_id"], layer)
                    self.assertEqual(kw["host_layer_id"], layer)
                    self.assertEqual(
                        kw["device_index_layer_id"], {0: 0, 2: 1}.get(layer)
                    )
                    self.assertEqual(kw["host_index_layer_id"], {0: 0, 2: 1}.get(layer))
                    self.assertIs(kw["host_k"], self.host.k_buffer)
                    self.assertEqual(kw["impl"], "auto")

    def add_draft(self, width=512):
        draft = SimpleNamespace(
            k=Buffer((1, 8, 128, 1, width)),
            v=Buffer((1, 8, 128, 1, 64)),
            index_k=Buffer((1, 8, 128, 1, 128)),
            index_scales=None,
            index_layer_indices=(0,),
        )
        self.host._native_infos.append(SimpleNamespace(native_buffers=lambda: draft))
        layer_id = self.host._device_layer_num + len(self.host._native_infos) - 2
        self.host._host_layer_by_controller[layer_id] = layer_id
        self.host._init_npu_transfer_bindings()
        return draft

    def test_draft_h2d_selects_only_draft_and_independent_host_layer(self):
        draft = self.add_draft()
        self.host.load_to_device_per_layer(
            None, "host", "device", 3, "kernel_ascend", is_draft=True
        )
        op = self.ops["transfer_kv_per_layer_kv_split_pf_lf"]
        op.assert_called_once()
        kw = op.call_args.kwargs
        self.assertIs(kw["device_k"], draft.k)
        self.assertEqual((kw["device_layer_id"], kw["host_layer_id"]), (0, 3))
        self.assertIsNone(kw["host_index_k"])
        self.assertIsNone(kw["device_index_layer_id"])

    def test_d2h_passes_pool_offsets_and_target_only_indexer(self):
        draft = self.add_draft()
        for layout, name in (
            ("page_first_kv_split", "transfer_kv_all_layer_kv_split_lf_pf"),
            ("layer_first_kv_split", "transfer_kv_all_layer_kv_split"),
        ):
            with self.subTest(layout=layout):
                self.host.layout = layout
                self.host.backup_from_device_all_layer(
                    None, "host", "device", "kernel_ascend"
                )
                calls = self.ops[name].call_args_list
                self.assertEqual(len(calls), 2)
                self.assertEqual([c.kwargs["host_layer_start"] for c in calls], [0, 3])
                self.assertIs(calls[0].kwargs["host_index_k"], self.host.index_k_buffer)
                self.assertIsNone(calls[1].kwargs["host_index_k"])
                self.assertIs(calls[1].kwargs["device_k"], draft.k)

    def test_skipped_draft_preserves_next_pool_host_offset(self):
        self.add_draft(width=999)
        self.add_draft()
        pools = self.host._npu_pool_bindings
        self.assertEqual([offset for _, offset, _ in pools], [0, 4])

    def test_incompatible_draft_skips_both_directions_and_preserves_next_binding(self):
        self.add_draft(width=999)
        draft = self.add_draft()
        self.host.load_to_device_per_layer(
            None, "host", "device", 3, "kernel_ascend", is_draft=True
        )
        op = self.ops["transfer_kv_per_layer_kv_split_pf_lf"]
        op.assert_not_called()
        self.host.load_to_device_per_layer(
            None, "host", "device", 4, "kernel_ascend", is_draft=True
        )
        kw = op.call_args.kwargs
        self.assertIs(kw["device_k"], draft.k)
        self.assertEqual((kw["device_layer_id"], kw["host_layer_id"]), (0, 4))
        self.host.backup_from_device_all_layer(None, "host", "device", "kernel_ascend")
        calls = self.ops["transfer_kv_all_layer_kv_split_lf_pf"].call_args_list
        self.assertEqual([c.kwargs["host_layer_start"] for c in calls], [0, 4])

    def test_transfers_reuse_bindings_without_resolving_pools_or_checking_drafts(self):
        self.add_draft()
        for info in self.host._native_infos:
            info.native_buffers = Mock(side_effect=AssertionError("resolved at runtime"))
        self.host._npu_draft_pool_compatible = Mock(
            side_effect=AssertionError("checked at runtime")
        )
        self.host.load_to_device_per_layer(None, "host", "device", 2, "kernel_ascend")
        self.host.load_to_device_per_layer(
            None, "host", "device", 3, "kernel_ascend", is_draft=True
        )
        self.host.backup_from_device_all_layer(None, "host", "device", "kernel_ascend")
        self.assertEqual(self.ops["transfer_kv_per_layer_kv_split_pf_lf"].call_count, 2)
        self.assertEqual(self.ops["transfer_kv_all_layer_kv_split_lf_pf"].call_count, 2)

    def test_host_layer_mapping_is_not_assumed_equal_to_device_layer(self):
        self.host._host_layer_by_controller[2] = 1
        self.host._init_npu_transfer_bindings()
        self.host.load_to_device_per_layer(None, "host", "device", 2, "kernel_ascend")
        kw = self.ops["transfer_kv_per_layer_kv_split_pf_lf"].call_args.kwargs
        self.assertEqual((kw["device_layer_id"], kw["host_layer_id"]), (2, 1))

    def test_dcp_localization_is_preserved(self):
        self.host.dcp_size, self.host.dcp_rank = 2, 1
        self.localize.side_effect = lambda indices, *_: f"local-{indices}"
        self.host.backup_from_device_all_layer(None, "host", "device", "kernel_ascend")
        kw = self.ops["transfer_kv_all_layer_kv_split_lf_pf"].call_args.kwargs
        self.assertEqual(
            (kw["host_indices"], kw["device_indices"]), ("local-host", "local-device")
        )

    def test_pf_and_lf_allocation_mirror_native_component_shapes(self):
        h = self.host
        h.page_num, h.layer_num, h.page_size = 8, 3, 128
        h.dtype = SimpleNamespace(itemsize=2)
        h._transfer_device = SimpleNamespace(type="npu")
        h.device, h.pin_memory, h.allocator = "cpu", True, None
        h.kv_lora_rank, h.qk_rope_head_dim = 512, 64
        h.dsa_kv_cache_store_fp8 = False
        self.namespace["ALLOC_MEMORY_FUNCS"] = {"npu": lambda shape, **_: Buffer(shape)}
        self.namespace["torch"] = SimpleNamespace(float32="fp32")
        for layout, kshape, ishape in (
            ("page_first_kv_split", (8, 3, 128, 1, 512), (8, 2, 128, 1, 128)),
            ("layer_first_kv_split", (3, 8, 128, 1, 512), (2, 8, 128, 1, 128)),
        ):
            with self.subTest(layout=layout):
                h.layout = layout
                self.assertIs(h.init_kv_buffer(), h.k_buffer)
                self.assertEqual(h.k_buffer.shape, kshape)
                self.assertEqual(h.index_k_buffer.shape, ishape)
                self.assertEqual(h.index_k_scale_buffer.shape, (*ishape[:-1], 1))

    def test_controller_keeps_pf_and_lf_indices_on_npu(self):
        path = ROOT / "python/sglang/srt/managers/cache_controller.py"
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "HiCacheController"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "move_indices"
        )
        namespace = {"torch": SimpleNamespace(Tensor=object)}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            namespace,
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
            for layout in ("page_first_kv_split", "layer_first_kv_split"):
                ctl = SimpleNamespace(
                    io_backend="kernel_ascend",
                    device="npu",
                    mem_pool_host=SimpleNamespace(layout=layout),
                )
                self.assertEqual(namespace["move_indices"](ctl, hi, di), (hi, di))


if __name__ == "__main__":
    unittest.main()
