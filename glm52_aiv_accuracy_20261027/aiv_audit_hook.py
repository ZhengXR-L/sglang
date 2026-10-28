"""Per-request completion audit for GLM-5.2 MLA/Indexer AscendC exchange."""
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

TARGET = "sglang.srt.mem_cache.pool_host.mla"
ENV = "GLM52_MLA_AIV_AUDIT_DIR"
_lock = threading.Lock()
_pending = queue.Queue()
_started = False


def emit(row):
    path = Path(os.environ[ENV]) / f"calls-{os.getpid()}.jsonl"
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.time(), "pid": os.getpid(), **row}) + "\n")


def marker():
    try:
        return json.loads((Path(os.environ[ENV]) / "phase.json").read_text())
    except FileNotFoundError:
        return {}


def drain():
    import torch
    while True:
        event, device, row = _pending.get()
        try:
            torch.npu.set_device(device)
            deadline = time.monotonic() + 600
            while not event.query():
                if time.monotonic() >= deadline:
                    raise TimeoutError("AIV event did not complete")
                time.sleep(0.005)
            emit({**row, "event": "aiv_complete", "return_code": 0})
        except Exception as exc:
            emit({**row, "event": "audit_error", "error": repr(exc)})


def patch(module):
    global _started
    import torch
    from memfabric_hybrid import offload
    from sgl_kernel_npu.kvcacheio import TransferDirection
    cls = module.MLATokenToKVPoolHost
    real_launch = cls._launch_ascendc_kv_exchange
    if getattr(real_launch, "_coder_audit", False):
        return
    real_copy = offload.kv_exchange_copy

    def audited_launch(self, vals, device):
        # ABI: 8-int header; each component has 9 ints ending in [lo, hi).
        n = int(vals[0]); pages = int(vals[1]); page_size = int(vals[2])
        if n not in (2, 3) or len(vals) != 8 + 9 * n:
            raise RuntimeError(f"Expected GLM MLA K/V/Indexer ABI, got components={n}, length={len(vals)}")
        if int(vals[6]) != 1 or self.layout != "layer_first_kv_split":
            raise RuntimeError("Expected layer_first_kv_split metadata")
        direction_name = TransferDirection(int(vals[3])).name
        components = [{"name": name, "lo": int(vals[8 + 9*i + 7]),
                       "hi": int(vals[8 + 9*i + 8]),
                       "width_bytes": int(vals[8 + 9*i + 6])}
                      for i, name in enumerate(("k", "v", "index_k")[:n])]
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        row = {**marker(), "direction": direction_name, "pages": pages,
               "page_size": page_size, "layout": self.layout,
               "components": components, "layers": int(self.target_layer_num),
               "indexer_layers": list(self._native_buffers.index_layer_indices),
               "rank": rank, "device": str(device),
               "bytes": pages * sum(c["width_bytes"] * (c["hi"] - c["lo"]) for c in components)}

        def audited_copy(*args, **kwargs):
            rc = real_copy(*args, **kwargs)
            if rc != 0:
                emit({**row, "event": "audit_error", "return_code": int(rc)})
            else:
                try:
                    event = torch.npu.Event(enable_timing=False)
                    event.record(torch.npu.current_stream())
                    _pending.put((event, device, row))
                except Exception as exc:
                    emit({**row, "event": "audit_error", "error": repr(exc)})
                    raise
            return rc

        # The launch invokes the operator synchronously on the Python side.
        offload.kv_exchange_copy = audited_copy
        try:
            return real_launch(self, vals, device)
        finally:
            offload.kv_exchange_copy = real_copy

    audited_launch._coder_audit = True
    cls._launch_ascendc_kv_exchange = audited_launch
    if not _started:
        threading.Thread(target=drain, daemon=True, name="coder-mla-aiv-audit").start()
        _started = True
    emit({"event": "hook_ready", "module": module.__file__, "audit_version": 1})


class Loader(importlib.abc.Loader):
    def __init__(self, wrapped): self.wrapped = wrapped
    def create_module(self, spec):
        fn = getattr(self.wrapped, "create_module", None)
        return fn(spec) if fn else None
    def exec_module(self, module):
        self.wrapped.exec_module(module)
        patch(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET: return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None: raise ImportError(TARGET)
        spec.loader = Loader(spec.loader)
        return spec


def install():
    if os.environ.get(ENV):
        if TARGET in sys.modules: patch(sys.modules[TARGET])
        else: sys.meta_path.insert(0, Finder())
