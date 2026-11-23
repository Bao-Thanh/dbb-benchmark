"""Compatibility entrypoint for scripts.compare_models; implementation: Code/scripts/."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys
if __name__ == "__main__":
    _runpy.run_module("scripts.compare_models", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("scripts.compare_models")
