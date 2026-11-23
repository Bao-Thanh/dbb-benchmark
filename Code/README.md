# Code directory — DBB research and applications

All project source notebooks and Python scripts live under `Code/`.

## Layout

| Path | Purpose |
| --- | --- |
| `Code/dbb_web.py` | Main Streamlit application entrypoint; unchanged |
| `Code/experiment1.py`, `experiment2.py`, `experiment3.py` | Standalone Streamlit experiment entrypoints; unchanged |
| `Code/dbb_*.py`, `compare_models.py` | Shared application, inference, data, scoring, reports and demos; kept at this level to preserve existing imports |
| `Code/build_dbb_pair_index.py`, `download_benchmarks.py`, `download_models.ps1` | Existing data/model setup scripts; original invocation paths preserved |
| [`notebooks/eda/`](notebooks/eda/) | Exploratory data analysis notebooks |
| [`notebooks/experiments/`](notebooks/experiments/) | Additional team-contributed generation-based LLM experiments |
| [`notebooks/legacy/`](notebooks/legacy/) | Older exploratory notebook, preserved for provenance |
| [`legacy/archive_versions/`](legacy/archive_versions/) | Historical versions of the Streamlit app and workers; not production entrypoints |

See [notebook guide](notebooks/README.md) for execution requirements and scientific cautions.

## Why active Python modules stay at the top of Code/

The active application currently uses imports such as `from experiment1 import render` and `from dbb_data import ...`. Moving these modules into new Python packages without updating every import, subprocess entrypoint, test, and Streamlit invocation would break the running app. This refactor therefore **organizes the four new notebooks and all archived code** while preserving all active file paths and public commands. A future package migration should be a separate tested change, not an unverified file move.

## Existing commands (unchanged)

```bash
python -m streamlit run Code/dbb_web.py --server.port 8500
python -m streamlit run Code/experiment1.py --server.port 8501
python -m pytest -q tests
python -m compileall -q Code
```

The notebook paths changed from `notebooks/...` to `Code/notebooks/...`. The top-level `Dataset/` directory remains unchanged.
