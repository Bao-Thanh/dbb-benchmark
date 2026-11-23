"""Backward-compatible import for reporting.dbb_charts; canonical code is in Code/reporting/."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys
if __name__ == "__main__":
    _runpy.run_module("reporting.dbb_charts", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("reporting.dbb_charts")
