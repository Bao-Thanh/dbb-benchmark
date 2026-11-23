# Code architecture — Gradio Research Studio

A local-first research dashboard for the **Description-based Bias Benchmark (DBB)**. The website now uses **Gradio**, not Streamlit. It retains the existing audited data, model inference, experiment runner, report generator and append-only storage.

**Reference:** Pan et al., *What's Not Said Still Hurts*, Findings EMNLP 2025: https://aclanthology.org/2025.findings-emnlp.76/

## Quick start (Windows PowerShell)

Run these commands from the repository root:

```powershell
git pull origin main
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama list
python -m Code.apps.gradio_web
```

Open **http://127.0.0.1:7860**. Ollama must be running at **http://127.0.0.1:11434** and at least one model must already be installed. No cloud API keys are required. The default Gradio bind address is localhost only; set `DBB_HOST` or `DBB_PORT` to change it.

## Four research workspaces

1. **DBB Benchmark** — choose local models, category and an even number of questions; start an audited subprocess; refresh progress, request safe stop, view actual DBB S≥20 category charts and exploratory diagnostics, and download PNG/CSV/ZIP.
2. **Four benchmarks** — compare DBB, BBQ, CrowS-Pairs and StereoSet using their separate exploratory measurements. The scores are *not* interchangeable.
3. **DBB vs BBQ** — run the two different benchmarks on the same model(s). These are not matched questions and the scores are not directly comparable.
4. **AI Bias Lab** — local Ollama chat with a conservative lexical indicator display and JSONL audit; indicators are **not** evidence of discrimination.

Every experiment uses real Ollama outputs. If the dataset is missing or the model is offline, the UI shows an error; it does not fabricate a result.

## Data and scientific caveats

Place original audited files under `Dataset/`:
- `Bias-Dataset.csv` — required for DBB
- `BBQ-Dataset.csv`, `CrowS-Pairs.csv`, `StereoSet.csv` — required for their respective comparison tabs

DBB requires the audited SHA256. DBB S is computed from repeated counterfactual questions; each question needs at least 10 valid responses. The two additional plotted diagnostics are exploratory, and **True Social Bias is N/A** rather than a made-up score. BBQ accuracy is not BBQ bias score; CrowS-Pairs and StereoSet forced-choice proxies are not the original likelihood-based metrics.

Results are written under `Results/explainable/`; exported Gradio files under `Results/gradio_exports/`. The interface never overwrites old experiment runs. Large datasets, models and results should remain out of Git.

## Code layout

- `Code/apps/gradio_web.py` — single Gradio website entrypoint
- `Code/core/` — data loading, Ollama inference, metrics, storage, counterfactuals
- `Code/workers/` — experiment subprocesses
- `Code/reporting/` — charts and report archives
- `Code/scripts/` — CLI and dataset setup
- `Code/notebooks/` — four contributed notebooks
- `Code/legacy/archive_versions/` — archived historical scripts

The old Streamlit UI source files were removed from the active `Code/apps/` directory. Existing worker/reporting and test logic is preserved.

## Tests

```powershell
python -m compileall -q Code
python -m pytest -q tests
```

The GitHub-only migration has not been exercised against a real Windows Ollama GPU environment. Verify model connectivity, UI actions, background workers and ZIP exports locally before the presentation.
