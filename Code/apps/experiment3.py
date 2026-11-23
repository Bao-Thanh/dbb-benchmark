"""Experiment 3: compact DBB/BBQ descriptive reports; no run management UI."""
import streamlit as st
from dbb_simple_report_ui import render_compact

def render():
    render_compact('experiment3', ['DBB', 'BBQ'],
                   'Thực nghiệm 3 · DBB và BBQ trên cùng model',
                   'Chọn model và số mẫu → chạy → xem biểu đồ, tải báo cáo. '
                   'DBB và BBQ không có cặp câu hỏi tương đương; '
                   'kết quả chỉ mô tả hai tập câu hỏi khác nhau.')

if __name__ == '__main__':
    st.set_page_config(page_title='DBB · Thực nghiệm 3', layout='wide')
    render()
