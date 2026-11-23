# DBB source code organization

The **canonical Python implementations** are grouped by responsibility. The thin `Code/*.py` files are backwards-compatible wrappers so existing Streamlit commands, imports, tests, and worker subprocesses can continue to work without requiring users to change commands.

```text
Code/
├── apps/                   # Streamlit UI, demos and experiment entrypoints
│   ├── dbb_web.py
│   ├── experiment1.py
│   ├── experiment2.py
│   ├── experiment3.py
│   ├── experiment3_dbb_bbq.py
│   ├── dbb_chat_demo.py
│   ├── dbb_demo.py
│   ├── dbb_ui.py
│   └── dbb_simple_report_ui.py
├── core/                   # Data, inference, scoring, state, storage, counterfactuals
│   ├── dbb_data.py
│   ├── dbb_inference.py
│   ├── dbb_metrics.py
│   ├── dbb_runner.py
│   ├── dbb_storage.py
│   ├── dbb_status.py
│   ├── dbb_counterfactual.py
│   └── dbb_demo_prompts.py
├── workers/                # Background batch processing
│   ├── dbb_batch_worker.py
│   └── dbb_compact_worker.py
├── reporting/              # Plotting and result export
│   ├── dbb_charts.py
│   └── dbb_compact_report.py
├── scripts/                # CLI and dataset preparation
│   ├── compare_models.py
│   ├── dbb_pipeline.py
│   ├── build_dbb_pair_index.py
│   └── download_benchmarks.py
├── notebooks/              # Four team-contributed notebooks + guide
│   ├── eda/
│   ├── experiments/
│   └── legacy/
├── legacy/archive_versions/ # 12 historical Python versions
├── dbb_web.py              # Compatibility entrypoint (wrapper)
├── experiment1.py          # Compatibility entrypoint (wrapper)
├── ...                     # Other compatibility wrappers
└── README.md
```

The top-level `tests/` folder contains automated tests. The dataset remains in `Dataset/`.

## Usage

Existing commands remain supported:

```powershell
python -m streamlit run Code/dbb_web.py --server.port 8500
python -m streamlit run Code/experiment1.py --server.port 8501
python -m streamlit run Code/experiment2.py --server.port 8502
python -m streamlit run Code/experiment3.py --server.port 8503
python Code/dbb_pipeline.py --help
python -m pytest -q tests
python -m compileall -q Code
```

For new imports inside `Code/`, use the actual packages (e.g. `from core.dbb_metrics import interpret`) once all callers have migrated. Existing imports such as `from dbb_metrics import interpret` continue to work through compatibility wrappers.

## Compatibility and review

- Active implementations are in the grouped folders, not duplicated in the wrappers.
- `Path(__file__)` references inside moved modules were adjusted for the extra directory level.
- Streamlit entrypoints and background worker command paths remain at their original public locations.
- No dataset or experiment outputs were modified; notebooks remain distinct from the Phase 2 logit-based experiment.
- **Runtime testing has not been executed in this GitHub-only refactor.** Run the tests and Streamlit/Ollama smoke tests locally before presenting or releasing.
- The old `Code/legacy/archive_versions/` files are historical snapshots, not production entrypoints.
