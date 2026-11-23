"""Compatibility entrypoint for apps.experiment2; canonical implementation is in Code/apps/."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys
if __name__ == "__main__":
    _runpy.run_module("apps.experiment2", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("apps.experiment2")
