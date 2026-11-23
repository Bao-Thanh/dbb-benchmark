"""Experiment 2: compact multi-benchmark reports; no run management UI."""
import streamlit as st
import sys as _dbb_sys
from pathlib import Path as _dbb_Path
_dbb_root = str(_dbb_Path(__file__).resolve().parents[2])
if _dbb_root not in _dbb_sys.path:
    _dbb_sys.path.insert(0, _dbb_root)

from Code.apps.dbb_simple_report_ui import render_compact

def render():
    st.caption('Giao diện báo cáo mới · Không cần tạo run hoặc chọn run cũ')
    render_compact('experiment2', ['DBB', 'BBQ', 'CrowS-Pairs', 'StereoSet'],
                   'Thực nghiệm 2 · So sánh bốn benchmark',
                   'Chọn model, số mẫu, benchmark → chạy → xem biểu đồ và tải báo cáo. '
                   'Mỗi benchmark có định nghĩa chỉ số khác nhau.')

if __name__ == '__main__':
    st.set_page_config(page_title='DBB · Thực nghiệm 2', layout='wide')
    render()
