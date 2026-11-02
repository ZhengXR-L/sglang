"""Audit completed GLM-5.2 layer-first KV-split operator transfers per TP rank."""

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
    with _lock, path.open("a", encoding="utf-8") as out:
        out.write(json.dumps({"time": time.time(), "pid": os.getpid(), **row}) + "\n")


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
                    raise TimeoutError("NPU transfer event did not complete")
                time.sleep(0.005)
            emit({**row, "event": "aiv_complete", "return_code": 0})
        except Exception as exc:
            emit({**row, "event": "audit_error", "error": repr(exc)})


def component(name, tensor, lo, hi):
    if tensor is None or tensor.numel() == 0:
        return None
    if tensor.ndim != 5:
        raise ValueError(f"{name} must be 5D; got {tuple(tensor.shape)}")
    if not 0 <= lo < hi <= tensor.shape[0]:
        raise ValueError(f"{name} layer range [{lo}, {hi}) exceeds {tensor.shape[0]}")
    return {"name": name, "lo": lo, "hi": hi,
            "width_bytes": tensor.shape[3] * tensor.shape[4] * tensor.element_size()}


def patch(module):
    global _started
    import torch

    names = ("transfer_kv_all_layer_kv_split", "transfer_kv_per_layer_kv_split")
    if all(getattr(getattr(module, name), "_coder_audit", False) for name in names):
        return

    def wrap(name, direction):
        original = getattr(module, name)

        def audited(*args, **kw):
            if args:
                raise ValueError(f"{name}: expected keyword call from MLA host pool")
            if kw.get("impl") != "aiv":
                raise ValueError(
                    f"Expected explicit AIV for {direction}; got {kw.get('impl')!r}"
                )
            page_size = int(kw["page_size"])
            device_indices, host_indices = kw["device_indices"], kw["host_indices"]
            if device_indices.dtype != torch.int64 or host_indices.dtype != torch.int64:
                raise ValueError("Sparse-copy indices must be int64")
            if device_indices.numel() != host_indices.numel() or device_indices.numel() % page_size:
                raise ValueError("Sparse-copy indices must describe complete paired pages")
            pages = device_indices.numel() // page_size
            if direction == "D2H":
                lo = int(kw.get("host_layer_start", 0))
                hi = lo + kw["device_k"].shape[0]
                index_lo = 0
                index_hi = kw["device_index_k"].shape[0] if kw.get("device_index_k") is not None else 0
            else:
                lo = int(kw["host_layer_id"])
                hi = lo + 1
                index_lo = kw.get("host_index_layer_id")
                index_hi = index_lo + 1 if index_lo is not None else 0
            parts = [component("k", kw["host_k"], lo, hi),
                     component("v", kw["host_v"], lo, hi)]
            if index_hi:
                parts.append(component("index_k", kw.get("host_index_k"), index_lo, index_hi))
            parts = [part for part in parts if part is not None]
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            row = {**marker(), "direction": direction, "operator": name,
                   "implementation": kw["impl"],
                   "pages": pages, "page_size": page_size,
                   "layout": "layer_first_kv_split", "components": parts,
                   "layers": kw["host_k"].shape[0],
                   "indexer_layers": list(range(kw["host_index_k"].shape[0]))
                   if kw.get("host_index_k") is not None else [],
                   "rank": rank, "device": str(kw["device_k"].device),
                   "bytes": pages * page_size * sum(
                       p["width_bytes"] * (p["hi"] - p["lo"]) for p in parts)}
            try:
                result = original(**kw)
                event = torch.npu.Event(enable_timing=False)
                event.record(torch.npu.current_stream())
                _pending.put((event, kw["device_k"].device, row))
                return result
            except Exception as exc:
                emit({**row, "event": "audit_error", "error": repr(exc)})
                raise

        audited._coder_audit = True
        return audited

    for name, direction in zip(names, ("D2H", "H2D")):
        setattr(module, name, wrap(name, direction))
    if not _started:
        threading.Thread(target=drain, daemon=True, name="coder-kv-split-audit").start()
        _started = True
    emit({"event": "hook_ready", "module": module.__file__, "audit_version": 2})


class Loader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self.wrapped, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        patch(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError(TARGET)
        spec.loader = Loader(spec.loader)
        return spec


def install():
    if os.environ.get(ENV):
        if TARGET in sys.modules:
            patch(sys.modules[TARGET])
        else:
            sys.meta_path.insert(0, Finder())
