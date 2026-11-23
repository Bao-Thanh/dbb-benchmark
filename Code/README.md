# Code — organized source (no duplicate root-level Python files)

All **25 active Python implementations** are stored once, inside their respective functional folders. The previous duplicate `Code/*.py` compatibility wrappers have been **deleted**.

```text
Code/
├── __init__.py
├── apps/                   # Streamlit pages, app entrypoint, demos
│   ├── dbb_web.py
│   ├── experiment1.py
│   ├── experiment2.py
│   ├── experiment3.py
│   ├── experiment3_dbb_bbq.py
│   ├── dbb_chat_demo.py
│   ├── dbb_demo.py
│   ├── dbb_ui.py
│   └── dbb_simple_report_ui.py
├── core/                   # Dataset, inference, metrics, runner, state, storage
│   ├── dbb_data.py
│   ├── dbb_inference.py
│   ├── dbb_metrics.py
│   ├── dbb_runner.py
│   ├── dbb_storage.py
│   ├── dbb_status.py
│   ├── dbb_counterfactual.py
│   └── dbb_demo_prompts.py
├── workers/                # Background processing
│   ├── dbb_batch_worker.py
│   └── dbb_compact_worker.py
├── reporting/              # Charts and report export
│   ├── dbb_charts.py
│   └── dbb_compact_report.py
├── scripts/                # Data preparation and command-line utilities
│   ├── compare_models.py
│   ├── dbb_pipeline.py
│   ├── build_dbb_pair_index.py
│   └── download_benchmarks.py
├── notebooks/              # Four contributed notebooks (EDA/experiments/legacy)
├── legacy/archive_versions/ # Historical snapshots (not active)
└── README.md
```

## Run from the repository root

```powershell
python -m streamlit run Code/apps/dbb_web.py --server.port 8500
python -m streamlit run Code/apps/experiment1.py --server.port 8501
python -m streamlit run Code/apps/experiment2.py --server.port 8502
python -m streamlit run Code/apps/experiment3.py --server.port 8503
python Code/scripts/dbb_pipeline.py --help
python Code/scripts/build_dbb_pair_index.py --help
python -m pytest -q tests
python -m compileall -q Code
```

Imports between moved Python modules use qualified names, e.g. `from Code.core.dbb_data import load_dbb`, so no `Code/dbb_data.py` duplicate is needed. Scripts and Streamlit entrypoints bootstrap the repository root on `sys.path` when executed by file path. Background worker paths are updated.

**Breaking change:** Old paths like `Code/dbb_web.py` and `Code/dbb_pipeline.py` have been removed at the user's request. Update shortcuts, task scheduler entries, scripts, and shell commands accordingly.

Tests were updated to locate the new entrypoints and alias legacy import names during test execution. **Full tests and a real Ollama/Streamlit smoke test have not been run** by this GitHub connector refactor; please run them before relying on the app.
