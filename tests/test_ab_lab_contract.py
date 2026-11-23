"""Static regression checks for the Gradio A/B lab integration."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_lab_syntax_and_integration():
    lab=(ROOT/'Code/apps/ab_lab.py').read_text(encoding='utf-8')
    web=(ROOT/'Code/apps/gradio_web.py').read_text(encoding='utf-8')
    ast.parse(lab)
    ast.parse(web)
    assert 'from Code.apps.ab_lab import run_ab' in web
    assert 'go4.click(run_ab' in web
    assert 'show_progress=\'hidden\'' in web

def test_ab_audit_and_original_option_mapping():
    lab=(ROOT/'Code/apps/ab_lab.py').read_text(encoding='utf-8')
    assert 'parse_choice(' in lab
    assert 'selected_original' in lab
    assert 'display_order' in lab
    assert 'manifest.json' in lab
    assert 'trials.csv' in lab
    assert 'INVALID' in lab
