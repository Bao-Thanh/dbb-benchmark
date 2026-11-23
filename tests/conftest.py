import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Code'))

# Legacy test imports below resolve to the canonical Code packages.
import importlib
for _name, _group in {
    'dbb_data':'core','dbb_inference':'core','dbb_metrics':'core','dbb_runner':'core',
    'dbb_storage':'core','dbb_status':'core','dbb_counterfactual':'core',
    'dbb_charts':'reporting','dbb_compact_report':'reporting',
    'dbb_compact_worker':'workers',
}.items():
    sys.modules[_name] = importlib.import_module(f'Code.{_group}.{_name}')
