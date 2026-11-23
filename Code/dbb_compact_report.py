"""Compatibility entrypoint for reporting.dbb_compact_report; implementation: Code/reporting/."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys
if __name__ == "__main__":
    _runpy.run_module("reporting.dbb_compact_report", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("reporting.dbb_compact_report")
