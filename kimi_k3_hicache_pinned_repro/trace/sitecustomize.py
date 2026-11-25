import importlib.abc
import importlib.machinery
import os
import sys
import traceback


class TraceMamba(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sglang.srt.mem_cache.pool_host.mamba":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        original = spec.loader.exec_module

        def execute(module):
            original(module)
            transfer = module.transfer_mamba_state
            print(f"MAMBA_SOURCE {module.__file__}", flush=True)
            if transfer is None:
                print("MAMBA_OP_UNAVAILABLE", flush=True)
                return

            def traced(**kwargs):
                host = kwargs["host_buf"]
                print(
                    f"MAMBA_OP_BEGIN pid={os.getpid()} "
                    f"backend={os.environ.get('SGLANG_HICACHE_HOST_MEM_BACKEND')} "
                    f"pinned={host.is_pinned()} operator={transfer.__module__}",
                    flush=True,
                )
                traceback.print_stack(limit=6)
                result = transfer(**kwargs)
                import torch
                torch.npu.synchronize()
                print(f"MAMBA_OP_OK pid={os.getpid()}", flush=True)
                return result

            module.transfer_mamba_state = traced

        spec.loader.exec_module = execute
        return spec


sys.meta_path.insert(0, TraceMamba())
