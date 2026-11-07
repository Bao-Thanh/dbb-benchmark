"""DBB Explainable AI Lab: three experiments, one Streamlit server.

Standalone experiment files can be launched separately if desired.
Main website: python -m streamlit run Code/dbb_web.py --server.port 8500
"""
import streamlit as st
from experiment1 import render as render_experiment1
from experiment2 import render as render_experiment2
from experiment3 import render as render_experiment3

st.set_page_config(page_title='DBB · Explainable AI Research Lab',page_icon='🧪',layout='wide')
st.title('🧪 DBB · Explainable AI Research Lab')
st.caption('Đánh giá thiên kiến xã hội của LLM | Streamlit + Ollama chạy cục bộ | 3 thí nghiệm độc lập')
st.markdown('**Dữ liệu → Prompt → Phản hồi thô → Parser → Quy tắc chấm → Kết quả → Giới hạn kết luận.**')
selected=st.radio('Trang',['1 · DBB cặp đối chứng','2 · Bốn benchmark','3 · DBB vs BBQ'],horizontal=True)
if selected.startswith('1'): render_experiment1()
elif selected.startswith('2'):render_experiment2()
else:render_experiment3()
st.divider()
st.caption('Nguồn: Pan et al., EMNLP Findings 2025 · https://aclanthology.org/2025.findings-emnlp.76/  |  '\
           'Đây là giải thích đầu vào/đầu ra/quy tắc đánh giá, không phải truy xuất suy nghĩ nội bộ của LLM.')
