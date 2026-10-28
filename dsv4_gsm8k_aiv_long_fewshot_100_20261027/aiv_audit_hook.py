"""Test-only, per-call audit of the real Memfabric AIV operator.

Loaded through a private sitecustomize in every SGLang worker. No source patch
and no additional stream synchronization; completion is observed using events.
"""
import importlib.abc
import importlib.machinery
import json
import inspect
import os
from pathlib import Path
import queue
import sys
import threading
import time

TARGET = "sglang.srt.mem_cache.memory_pool_host"
_tls = threading.local()
_patch_lock = threading.Lock()
_write_lock = threading.Lock()
_pending = queue.Queue()
_thread_started = False


def emit(record):
    root = Path(os.environ["DSV4_GSM_AIV_AUDIT_DIR"])
    record = {"time": time.time(), "pid": os.getpid(), **record}
    with _write_lock:
        with (root / f"calls-{os.getpid()}.jsonl").open("a") as out:
            out.write(json.dumps(record) + "\n")


def marker():
    try:
        return json.loads((Path(os.environ["DSV4_GSM_AIV_AUDIT_DIR"]) / "phase.json").read_text())
    except FileNotFoundError:
        return {}


def drain():
    import torch
    while True:
        event, device, record = _pending.get()
        try:
            torch.npu.set_device(device)
            deadline = time.monotonic() + 600
            while not event.query():
                if time.monotonic() > deadline:
                    raise TimeoutError("AIV completion event timed out")
                time.sleep(0.005)
            emit({**record, "event": "aiv_complete", "return_code": 0})
        except Exception as exc:
            emit({**record, "event": "audit_error", "error": repr(exc)})


def patch_module(module):
    global _thread_started
    original = getattr(module, "_dsv4_ascendc_page_copy", None)
    if original is None:
        raise RuntimeError("AIV audit requires the DSV4 _dsv4_ascendc_page_copy helper")
    parameters = inspect.signature(original).parameters
    required = {"pool_name", "layer_id", "layer_num", "layout", "direction"}
    if not required <= set(parameters) and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        raise RuntimeError("DSV4 helper ABI lacks pool/layer metadata; use the documented target branch")
    if getattr(original, "_gsm_audit", False):
        return

    def audited_helper(**kw):
        global _thread_started
        import torch
        from memfabric_hybrid import offload
        with _patch_lock:
            if not getattr(offload.kv_exchange_copy, "_gsm_audit", False):
                real_copy = offload.kv_exchange_copy

                def audited_copy(*args, **kwargs):
                    context = getattr(_tls, "context", None)
                    rc = real_copy(*args, **kwargs)
                    if context is not None:
                        record = {**marker(), **context}
                        if rc != 0:
                            emit({**record, "event": "audit_error", "return_code": int(rc)})
                        else:
                            try:
                                event = torch.npu.Event(enable_timing=False)
                                event.record(torch.npu.current_stream())
                                _pending.put((event, context["device"], record))
                            except Exception as exc:
                                emit({**record, "event": "audit_error", "error": repr(exc)})
                                raise
                    return rc

                audited_copy._gsm_audit = True
                offload.kv_exchange_copy = audited_copy
            if not _thread_started:
                threading.Thread(target=drain, daemon=True, name="gsm-aiv-audit").start()
                _thread_started = True
        pool = kw["pool_name"]
        direction = kw["direction"]
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else None
        prior = getattr(_tls, "context", None)
        _tls.context = {
            "pool": getattr(pool, "value", str(pool)),
            "direction": getattr(direction, "name", str(direction)),
            "layer": int(kw["layer_id"]), "layers": int(kw["layer_num"]),
            "pages": int(kw["device_rows"].numel()),
            "bytes": int(kw["device_rows"].numel()) * int(kw["item_bytes"]),
            "layout": kw["layout"], "rank": rank,
            "device": str(kw["device_buffer"].device),
        }
        try:
            return original(**kw)
        finally:
            _tls.context = prior

    audited_helper._gsm_audit = True
    module._dsv4_ascendc_page_copy = audited_helper
    emit({"event": "hook_ready", "module": module.__file__, "audit_version": 1})


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        fn = getattr(self.original, "create_module", None)
        return fn(spec) if fn else None

    def exec_module(self, module):
        self.original.exec_module(module)
        patch_module(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot find {TARGET}")
        spec.loader = Loader(spec.loader)
        return spec


def install():
    if os.environ.get("DSV4_GSM_AIV_AUDIT_DIR"):
        if TARGET in sys.modules:
            patch_module(sys.modules[TARGET])
        else:
            sys.meta_path.insert(0, Finder())
