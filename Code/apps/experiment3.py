"""Experiment 3: compact DBB/BBQ descriptive reports; no run management UI."""
import streamlit as st
import sys as _dbb_sys
from pathlib import Path as _dbb_Path
_dbb_root = str(_dbb_Path(__file__).resolve().parents[2])
if _dbb_root not in _dbb_sys.path:
    _dbb_sys.path.insert(0, _dbb_root)

from Code.apps.dbb_simple_report_ui import render_compact

def render():
    render_compact('experiment3', ['DBB', 'BBQ'],
                   'Thực nghiệm 3 · DBB và BBQ trên cùng model',
                   'Chọn model và số mẫu → chạy → xem biểu đồ, tải báo cáo. '
                   'DBB và BBQ không có cặp câu hỏi tương đương; '
                   'kết quả chỉ mô tả hai tập câu hỏi khác nhau.')

if __name__ == '__main__':
    st.set_page_config(page_title='DBB · Thực nghiệm 3', layout='wide')
    render()
