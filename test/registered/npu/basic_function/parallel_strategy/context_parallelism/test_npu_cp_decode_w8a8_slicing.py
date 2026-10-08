"""Single-NPU checks for CP slices of post-load ModelSlim dynamic INT8 weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu

from sglang.srt.layers.cp.cp_decode_attn_tp import CpDecodeAttnTpContext
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.modelslim.schemes.modelslim_w8a8_int8 import (
    ModelSlimW8A8Int8,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="nightly-1-npu-a3", nightly=True)


class TestNPUCpDecodeW8A8Slicing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable"
        cls.addClassCleanup(
            setattr, torch.npu.config, "allow_internal_format", previous
        )
        torch.npu.config.allow_internal_format = True

    def make_layer(self, row, dynamic=True):
        # Avoid creating distributed groups: only the actual quant kernel and
        # CP context are exercised here, reduction is checked by summing shards.
        cls = RowParallelLinear if row else ColumnParallelLinear
        layer = cls.__new__(cls)
        torch.nn.Module.__init__(layer)
        layer.bias = None
        layer.input_size_per_partition = 256
        layer.output_size_per_partition = 256
        layer.use_decode_attn_tp = False
        layer.scheme = ModelSlimW8A8Int8(
            {"test.weight": "W8A8_DYNAMIC" if dynamic else "W8A8"}, "test"
        )
        layer.weight = torch.nn.Parameter(
            torch.randint(-8, 8, (256, 256), device="npu", dtype=torch.int8),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            (torch.arange(256, device="npu", dtype=torch.float32) + 1).view(256, 1)
            / 25600,
            requires_grad=False,
        )
        layer.weight_offset = torch.nn.Parameter(
            torch.zeros((256, 1), device="npu"), requires_grad=False
        )
        if dynamic:
            layer.scheme.process_weights_after_loading(layer)
        return layer

    def context(self, rank):
        with patch(
            "sglang.srt.layers.cp.cp_decode_attn_tp.get_parallel",
            return_value=SimpleNamespace(
                enable_cp_decode_attn_tp=True,
                attn_tp_size=1,
                attn_cp_size=2,
                attn_cp_rank=rank,
            ),
        ):
            ctx = CpDecodeAttnTpContext()
        ctx.set_decode_attn_tp = lambda _: setattr(ctx, "use_decode_attn_tp", True)
        return ctx

    def test_quantized_matmul_and_restore(self):
        torch.manual_seed(0)
        with patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=SimpleNamespace(tp_size=2, attn_cp_size=2),
        ):
            for row in (False, True):
                with self.subTest(row_parallel=row):
                    layer = self.make_layer(row)
                    self.assertEqual(torch_npu.get_npu_format(layer.weight), 29)
                    original = layer.weight.data.clone()
                    weight_ptr = layer.weight.data_ptr()
                    scale_ptr = layer.weight_scale.data_ptr()
                    x = torch.randn(3, 256, device="npu", dtype=torch.bfloat16)
                    # Fix activation quantization to isolate weight slicing:
                    # row-local dynamic quantization has a different scale.
                    xq, xs = torch.ops.npu.npu_dynamic_quant(x)
                    expected = layer.scheme.apply_weights(layer, (xq, xs))
                    parts = []
                    for rank in range(2):
                        ctx = self.context(rank)
                        for repeat in range(2):
                            with ctx.maybe_use_decode_attn_tp(None, [layer]):
                                self.assertEqual(
                                    torch_npu.get_npu_format(layer.weight), 29
                                )
                                self.assertEqual(
                                    tuple(layer.weight.shape),
                                    (128, 256) if row else (256, 128),
                                )
                                self.assertEqual(
                                    layer.weight_scale.numel(), 256 if row else 128
                                )
                                if row:
                                    self.assertEqual(
                                        layer.weight_scale.data_ptr(), scale_ptr
                                    )
                                local_x = (
                                    xq[:, rank * 128 : (rank + 1) * 128].contiguous()
                                    if row
                                    else xq
                                )
                                result = layer.scheme.apply_weights(
                                    layer, (local_x, xs)
                                )
                                if repeat == 0:
                                    local_ptr = layer.weight.data_ptr()
                                    parts.append(result.float())
                                else:
                                    self.assertEqual(layer.weight.data_ptr(), local_ptr)
                            self.assertEqual(layer.weight.data_ptr(), weight_ptr)
                            self.assertEqual(layer.weight_scale.data_ptr(), scale_ptr)
                            self.assertFalse(layer.use_decode_attn_tp)
                            torch.testing.assert_close(layer.weight.data, original)
                    actual = sum(parts) if row else torch.cat(parts, dim=-1)
                    torch.testing.assert_close(
                        actual, expected.float(), rtol=0.015, atol=0.015
                    )

    def test_exception_restores_and_static_rejected(self):
        with patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=SimpleNamespace(tp_size=2, attn_cp_size=2),
        ):
            layer = self.make_layer(False)
            ptr = layer.weight.data_ptr()
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with self.context(0).maybe_use_decode_attn_tp(None, [layer]):
                    raise RuntimeError("injected")
            self.assertEqual(layer.weight.data_ptr(), ptr)
            self.assertEqual(layer.output_size_per_partition, 256)
            with self.assertRaisesRegex(ValueError, "dynamic ModelSlim"):
                with self.context(0).maybe_use_decode_attn_tp(
                    None, [self.make_layer(True, False)]
                ):
                    pass


if __name__ == "__main__":
    unittest.main()
