"""Regression: Gradio runs must use the worker's accepted protocol identifier."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_gradio_workflow_matches_worker():
    app = (ROOT / "Code/apps/gradio_web.py").read_text(encoding="utf-8")
    worker = (ROOT / "Code/workers/dbb_compact_worker.py").read_text(encoding="utf-8")
    ast.parse(app)
    ast.parse(worker)
    assert "'workflow':'compact_multi_v1'" in app
    assert "cfg.get('workflow') != 'compact_multi_v1'" in worker

def test_progress_shows_worker_error():
    app = (ROOT / "Code/apps/gradio_web.py").read_text(encoding="utf-8")
    assert "state.get('error')" in app
    assert "worker.log" in app
