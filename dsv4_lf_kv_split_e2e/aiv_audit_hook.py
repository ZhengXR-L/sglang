"""Audit the current DSV4 host pools' real KV-split AIV calls.

Loaded only by the dedicated test server through a private sitecustomize.
The wrappers do not synchronize an NPU stream; they record an event after each
operator returns and report completion once that event has finished.
"""

import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time


TARGET = "sglang.srt.mem_cache.memory_pool_host"
ENV = "DSV4_GSM_AIV_AUDIT_DIR"
_tls = threading.local()
_write_lock = threading.Lock()
_pending = queue.Queue()
_drain_started = False


def emit(row):
    path = Path(os.environ[ENV]) / f"calls-{os.getpid()}.jsonl"
    with _write_lock, path.open("a", encoding="utf-8") as out:
        out.write(json.dumps({"time": time.time(), "pid": os.getpid(), **row}) + "\n")


def marker():
    try:
        return json.loads((Path(os.environ[ENV]) / "phase.json").read_text())
    except FileNotFoundError:
        return {}


def pool_value(name):
    value = getattr(name, "value", str(name))
    return value.removeprefix("PoolName.").lower()


def drain():
    import torch

    while True:
        event, device, row = _pending.get()
        try:
            torch.npu.set_device(device)
            deadline = time.monotonic() + 600
            while not event.query():
                if time.monotonic() >= deadline:
                    raise TimeoutError("DSV4 AIV completion event timed out")
                time.sleep(0.005)
            emit({**row, "event": "aiv_complete", "return_code": 0})
        except Exception as exc:
            emit({**row, "event": "audit_error", "error": repr(exc)})


def patch(module):
    global _drain_started
    import torch

    if getattr(module, "_dsv4_aiv_audited", False):
        return

    def wrap_host_method(cls, method_name):
        original = getattr(cls, method_name)

        def audited(self, *args, **kw):
            previous = getattr(_tls, "context", None)
            context = {
                "pool": pool_value(self.pool_name),
                "layers": int(self.layer_num),
                "layout": self.layout,
                "layer": None,
            }
            if method_name == "load_to_device_per_layer":
                context["layer"] = int(kw.get("layer_id", args[3] if len(args) > 3 else -1))
            _tls.context = context
            try:
                return original(self, *args, **kw)
            finally:
                _tls.context = previous

        setattr(cls, method_name, audited)

    def wrap_host_view(cls, method_name):
        original = getattr(cls, method_name)

        def audited(self, *args, **kw):
            context = getattr(_tls, "context", None)
            if context is not None:
                context["layer"] = int(args[0] if args else kw["l"])
            return original(self, *args, **kw)

        setattr(cls, method_name, audited)

    for cls, view_name in (
        (module.DeepSeekV4PagedHostPool, "_host_page_view"),
        (module.DeepSeekV4StateHostPool, "_state_host_page_view"),
    ):
        wrap_host_method(cls, "backup_from_device_all_layer")
        wrap_host_method(cls, "load_to_device_per_layer")
        wrap_host_view(cls, view_name)

    def wrap_operator(name, direction):
        original = getattr(module, name)

        def audited(*args, **kw):
            if args:
                raise ValueError(f"{name}: expected keyword-only DSV4 call")
            context = getattr(_tls, "context", None)
            if context is None or context["layer"] is None:
                raise RuntimeError(f"{name}: missing DSV4 pool/layer context")
            if context["layout"] != "layer_first_kv_split" or kw.get("impl") != "aiv":
                raise ValueError(f"{name}: expected layer_first_kv_split with explicit AIV")

            device_indices = kw["device_indices"]
            host_indices = kw["host_indices"]
            page_size = int(kw["page_size"])
            if page_size <= 0 or device_indices.numel() != host_indices.numel():
                raise ValueError(f"{name}: invalid paired transfer indices")
            if device_indices.numel() == 0 or device_indices.numel() % page_size:
                raise ValueError(f"{name}: transfer must contain complete pages")

            device_k = kw["device_k"]
            host_k = kw["host_k"]
            if device_k.ndim != 5 or host_k.ndim != 5 or device_k.shape[0] != 1 or host_k.shape[0] != 1:
                raise ValueError(f"{name}: expected one 5D fused-KV layer view")
            if kw["device_v"].numel() or kw["host_v"].numel():
                raise ValueError(f"{name}: DSV4 fused KV must use empty V tensors")

            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            row = {
                **marker(), **context,
                "direction": direction,
                "operator": name,
                "implementation": kw["impl"],
                "pages": int(device_indices.numel() // page_size),
                "page_size": page_size,
                "bytes": int(device_indices.numel() * device_k.shape[3] * device_k.shape[4] * device_k.element_size()),
                "rank": rank,
                "device": str(device_k.device),
            }
            if not 0 <= row["layer"] < row["layers"]:
                raise ValueError(f"{name}: layer {row['layer']} outside pool")
            try:
                result = original(**kw)
                event = torch.npu.Event(enable_timing=False)
                event.record(torch.npu.current_stream())
                _pending.put((event, device_k.device, row))
                return result
            except Exception as exc:
                emit({**row, "event": "audit_error", "error": repr(exc)})
                raise

        setattr(module, name, audited)

    wrap_operator("transfer_kv_all_layer_kv_split", "D2H")
    wrap_operator("transfer_kv_per_layer_kv_split", "H2D")
    if not _drain_started:
        threading.Thread(target=drain, daemon=True, name="dsv4-aiv-audit").start()
        _drain_started = True
    module._dsv4_aiv_audited = True
    emit({"event": "hook_ready", "module": module.__file__, "audit_version": 3})


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
