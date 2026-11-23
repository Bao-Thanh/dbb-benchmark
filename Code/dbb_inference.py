"""Backward-compatible import for core.dbb_inference; canonical code is in Code/core/."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys
if __name__ == "__main__":
    _runpy.run_module("core.dbb_inference", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("core.dbb_inference")
