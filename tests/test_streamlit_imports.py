"""Import smoke without an installed Streamlit distribution or a running Ollama."""
import importlib
import sys
import types


def test_import_all_pages_and_main_router(monkeypatch):
    st=types.ModuleType('streamlit')
    st.cache_data=lambda **kwargs: lambda fn:fn
    for method in ('set_page_config','title','caption','markdown','divider'):
        setattr(st,method,lambda *args,**kwargs:None)
    st.radio=lambda *args,**kwargs:'1 · DBB cặp đối chứng'
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules,'streamlit',st)
        one=importlib.import_module('experiment1')
        two=importlib.import_module('experiment2')
        three=importlib.import_module('experiment3')
        called=[]
        patch.setattr(one,'render',lambda:called.append(1))
        patch.setattr(two,'render',lambda:called.append(2))
        patch.setattr(three,'render',lambda:called.append(3))
        sys.modules.pop('dbb_web',None)
        importlib.import_module('dbb_web')
        assert called==[1]
        sys.modules.pop('dbb_web',None)
