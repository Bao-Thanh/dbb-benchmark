import ast
from pathlib import Path


def test_all_four_entrypoints_exist_and_syntax():
    root=Path(__file__).resolve().parents[1]/'Code'
    for name in ['apps/dbb_web.py','apps/experiment1.py','apps/experiment2.py','apps/experiment3.py']:
        source=(root/name).read_text(encoding='utf-8')
        assert ast.parse(source)
    text=(root/'apps'/'dbb_web.py').read_text(encoding='utf-8')
    for n in ['experiment1','experiment2','experiment3']:
        assert f'from Code.apps.{n} import render' in text
