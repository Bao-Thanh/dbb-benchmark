"""Compatibility entrypoint for scripts.dbb_pipeline; implementation: Code/scripts/."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys
if __name__ == "__main__":
    _runpy.run_module("scripts.dbb_pipeline", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("scripts.dbb_pipeline")
