"""DBB Explainable AI Lab: three experiments, one Streamlit server.

Standalone experiment files can be launched separately if desired.
Main website: python -m streamlit run Code/dbb_web.py --server.port 8500
"""
import streamlit as st
import sys as _dbb_sys
from pathlib import Path as _dbb_Path
_dbb_root = str(_dbb_Path(__file__).resolve().parents[2])
if _dbb_root not in _dbb_sys.path:
    _dbb_sys.path.insert(0, _dbb_root)

from Code.apps.experiment1 import render as render_experiment1
from Code.apps.experiment2 import render as render_experiment2
from Code.apps.experiment3 import render as render_experiment3
from Code.apps.dbb_chat_demo import render as render_chat_demo

st.set_page_config(page_title='DBB · Explainable AI Research Lab',page_icon='🧪',layout='wide')
st.title('🧪 DBB · Explainable AI Research Lab')
st.caption('Đánh giá thiên kiến xã hội của LLM | Streamlit + Ollama chạy cục bộ | 3 thí nghiệm độc lập + demo chat AI')
st.markdown('**Dữ liệu → Prompt → Phản hồi thô → Parser → Quy tắc chấm → Kết quả → Giới hạn kết luận.**')
selected=st.radio('Trang',['1 · DBB cặp đối chứng','2 · Bốn benchmark','3 · DBB vs BBQ','4 · Chat AI & thiên kiến'],horizontal=True)
if selected.startswith('1'): render_experiment1()
elif selected.startswith('2'):render_experiment2()
elif selected.startswith('3'):render_experiment3()
else:render_chat_demo()
st.divider()
st.caption('Nguồn: Pan et al., EMNLP Findings 2025 · https://aclanthology.org/2025.findings-emnlp.76/  |  '\
           'Đây là giải thích đầu vào/đầu ra/quy tắc đánh giá, không phải truy xuất suy nghĩ nội bộ của LLM.')
