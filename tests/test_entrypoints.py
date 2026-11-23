import ast
from pathlib import Path


def test_gradio_entrypoint_exists_and_syntax():
    source=(Path(__file__).resolve().parents[1]/'Code'/'apps'/'gradio_web.py').read_text(encoding='utf-8')
    ast.parse(source)
    assert 'def build():' in source
    assert 'gr.Blocks' in source
    assert 'streamlit' not in source.lower()
