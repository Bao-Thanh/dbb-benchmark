"""Compatibility entrypoint for workers.dbb_batch_worker; canonical implementation is in Code/workers/."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys
if __name__ == "__main__":
    _runpy.run_module("workers.dbb_batch_worker", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("workers.dbb_batch_worker")
