import ast
from pathlib import Path


def test_all_four_entrypoints_exist_and_syntax():
    root=Path(__file__).resolve().parents[1]/'Code'
    for name in ['dbb_web.py','experiment1.py','experiment2.py','experiment3.py']:
        source=(root/name).read_text(encoding='utf-8')
        assert ast.parse(source)
    text=(root/'dbb_web.py').read_text(encoding='utf-8')
    for n in ['experiment1','experiment2','experiment3']:
        assert f'from {n} import render' in text
