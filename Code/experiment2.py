"""Experiment 2: compact multi-benchmark reports; no run management UI."""
import streamlit as st
from dbb_simple_report_ui import render_compact

def render():
    render_compact('experiment2', ['DBB', 'BBQ', 'CrowS-Pairs', 'StereoSet'],
                   'Thực nghiệm 2 · So sánh bốn benchmark',
                   'Chọn model, số mẫu, benchmark → chạy → xem biểu đồ và tải báo cáo. '
                   'Mỗi benchmark có định nghĩa chỉ số khác nhau.')

if __name__ == '__main__':
    st.set_page_config(page_title='DBB · Thực nghiệm 2', layout='wide')
    render()
