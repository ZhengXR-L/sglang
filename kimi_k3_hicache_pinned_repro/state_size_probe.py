import argparse
import json
import math
from pathlib import Path

import torch
import torch_npu

from sglang.srt.mem_cache.pool_host.npu_host_memory import (
    alloc_mmap,
    get_host_memory_backend,
)
from sgl_kernel_npu.operators.data_move import transfer_mamba_state
from sgl_kernel_npu.operators.data_move._types import TransferDirection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tp", required=True, type=int)
    parser.add_argument("--slots", required=True, type=int)
    parser.add_argument("--kind", choices=("ssm", "conv"), required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    text = config.get("text_config", config)
    attention = text["linear_attn_config"]
    layers = len(attention["kda_layers"])
    heads = attention["num_heads"] // args.tp
    head_dim = attention["head_dim"]
    if attention["num_heads"] % args.tp or args.slots < 2:
        raise ValueError("Require divisible heads and at least two host slots")
    if args.kind == "ssm":
        state_shape = (heads, head_dim, head_dim)
        dtype = torch.float32
    else:
        state_shape = (attention["short_conv_kernel_size"] - 1, 3 * heads * head_dim)
        dtype = torch.bfloat16
    torch.npu.set_device(0)
    host_shape = (args.slots, layers, 1, *state_shape)
    host = alloc_mmap(host_shape, dtype)
    record = {
        "config": args.config,
        "tp": args.tp,
        "kind": args.kind,
        "host_shape": host_shape,
        "bytes": math.prod(host_shape) * dtype.itemsize,
        "ptr": hex(host.data_ptr()),
        "before_pinned": host.is_pinned(),
    }
    print(json.dumps(record), flush=True)
    backend = get_host_memory_backend()
    if type(backend).__name__ != "AclHostMemoryBackend":
        raise RuntimeError("This probe requires the ACL backend")
    backend._register_block(host)
    record["after_pinned"] = host.is_pinned()
    print(json.dumps(record), flush=True)

    device = torch.empty((layers, 2, *state_shape), dtype=dtype, device="npu")
    pattern = torch.arange(1, layers + 1, dtype=dtype, device="npu")
    device.copy_(pattern.reshape(layers, *([1] * (device.ndim - 1))))
    device_indices = torch.tensor([0, 1], dtype=torch.int64)
    host_indices = torch.tensor([0, args.slots - 1], dtype=torch.int64)
    transfer_mamba_state(
        device, host, device_indices, host_indices, TransferDirection.D2H, impl="aclrt"
    )
    torch.npu.synchronize()
    torch.testing.assert_close(host[0, :, 0], device[:, 0].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(host[-1, :, 0], device[:, 1].cpu(), rtol=0, atol=0)
    record["d2h_ok"] = True

    restored = torch.zeros_like(device)
    transfer_mamba_state(
        restored, host, device_indices, host_indices, TransferDirection.H2D, impl="aclrt"
    )
    torch.npu.synchronize()
    if not torch.equal(restored, device):
        raise RuntimeError("H2D data mismatch")
    record["h2d_ok"] = True
    print("PROBE_OK " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
