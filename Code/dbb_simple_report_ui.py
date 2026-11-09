"""Compact, report-first UI shared by Experiments 2 and 3.

Runs are stored internally for audit; the UI has no create/resume/run picker.
Each click evaluates fresh samples, displays a descriptive report and offers ZIP/CSV.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import streamlit as st

from dbb_data import (DATASET, audited_pairs, choose_pairs, file_sha256,
                      load_dbb, load_bbq, load_crows, load_stereoset, sample_rows)
from dbb_inference import Settings, models_available
from dbb_metrics import benchmark_summary, latest_attempts
from dbb_runner import dbb_tasks, bbq_tasks, crows_tasks, stereoset_tasks, evaluate
from dbb_storage import create_run, export_csv, load_run

DATASETS = {
    'DBB': ('Bias-Dataset.csv', load_dbb, dbb_tasks),
    'BBQ': ('BBQ-Dataset.csv', load_bbq, bbq_tasks),
    'CrowS-Pairs': ('CrowS-Pairs.csv', load_crows, crows_tasks),
    'StereoSet': ('StereoSet.csv', load_stereoset, stereoset_tasks),
}
EXPLANATIONS = {
    'DBB': 'Tỷ lệ chọn ans1 chỉ là mô tả một lượt; KHÔNG phải DBB S.',
    'BBQ': 'Accuracy trên câu hợp lệ; KHÔNG phải BBQ bias score.',
    'CrowS-Pairs': 'Tỷ lệ chọn câu stereotype là proxy; KHÔNG phải điểm log-likelihood gốc.',
    'StereoSet': 'Tỷ lệ chọn stereotype là proxy; KHÔNG phải ICAT.',
}


@st.cache_data(show_spinner=False)
def dataset(name: str, mtime: int) -> pd.DataFrame:
    return DATASETS[name][1]()


def build_tasks(name: str, count: int, seed: int, cfg: dict) -> list[dict]:
    path = DATASET / DATASETS[name][0]
    df = dataset(name, path.stat().st_mtime_ns)
    if name == 'DBB':
        pairs = choose_pairs(audited_pairs(df), count, 'Tất cả', seed)
        return dbb_tasks(df, pairs, cfg)
    sampled = sample_rows(df, count, seed)
    return DATASETS[name][2](sampled, cfg)


def _summary(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(latest_attempts(records))
    if df.empty:
        return pd.DataFrame()
    output = []
    for model, group in df.groupby('model', sort=True):
        part = benchmark_summary(group.to_dict('records'))
        if not part.empty:
            part.insert(0, 'Model', model)
            output.append(part)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def _report(kind: str, run_ids: list[str], benchmarks: list[str]) -> None:
    attempts = []
    for rid in run_ids:
        _, records = load_run(kind, rid)
        attempts.extend(records)
    summary = _summary(attempts)
    if summary.empty:
        st.warning('Chưa có phản hồi để lập báo cáo.')
        return
    st.subheader('Báo cáo kết quả')
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.caption('Các chỉ số có định nghĩa khác nhau; không xếp hạng thiên kiến giữa benchmark.')
    for name in benchmarks:
        st.caption('**' + name + ':** ' + EXPLANATIONS[name])
    # Compare all selected models on the SAME benchmark and the SAME sampled rows.
    # Metric and answer-format compliance have different denominators and must
    # never be presented as interchangeable or as a universal bias ranking.
    st.subheader('So sánh các model trên cùng benchmark')
    st.caption('Mỗi biểu đồ gom tất cả model. N/A = không có phản hồi hợp lệ, '
               'không được hiểu là 0%. Mọi model dùng cùng mẫu (seed 42).')
    for name in benchmarks:
        part = summary.loc[summary['Benchmark'].eq(name)].copy()
        if part.empty:
            continue
        st.markdown('**' + name + '** — ' + EXPLANATIONS[name])
        metric_label = str(part['Chỉ số (không so trực tiếp)'].iloc[0])
        part['Giá trị (%)'] = pd.to_numeric(part['Giá trị (%)'], errors='coerce')
        valid_metric = part.dropna(subset=['Giá trị (%)'])
        if valid_metric.empty:
            st.info('N/A — chưa có phản hồi hợp lệ để tính ' + metric_label + '.')
        else:
            st.caption(metric_label + ' · % trên phản hồi hợp lệ')
            st.bar_chart(valid_metric.set_index('Model')[['Giá trị (%)']],
                         y_label='Phần trăm (%)', use_container_width=True)
        st.dataframe(part[['Model', 'Đã thử', 'Hợp lệ', 'Không hợp lệ',
                           'Tỷ lệ hợp lệ (%)', 'Giá trị (%)']],
                     hide_index=True, use_container_width=True)

    st.subheader('Tỷ lệ phản hồi hợp lệ — tất cả model và benchmark')
    validity = summary.pivot(index='Model', columns='Benchmark',
                             values='Tỷ lệ hợp lệ (%)')
    ordered = [name for name in benchmarks if name in validity.columns]
    if ordered:
        st.bar_chart(validity[ordered], y_label='Phần trăm hợp lệ (%)',
                     use_container_width=True)
    st.caption('Biểu đồ hợp lệ đo khả năng trả lời đúng định dạng, KHÔNG đo thiên kiến. '
               'Để đánh giá DBB S theo paper, dùng Thực nghiệm 1 với đủ lượt hợp lệ.')

    raw = export_csv(attempts)
    aggregate = summary.to_csv(index=False).encode('utf-8-sig')
    explanation = ('Các chỉ số chỉ mang tính mô tả/thăm dò, không thay thế metric gốc.\n'
                   + '\n'.join(f'{n}: {EXPLANATIONS[n]}' for n in benchmarks)).encode('utf-8')
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('chi_tiet.csv', raw)
        z.writestr('tong_hop.csv', aggregate)
        z.writestr('phuong_phap.txt', explanation)
    left, right = st.columns(2)
    with left:
        st.download_button('⬇ CSV tổng hợp', aggregate,
                           file_name=f'{kind}_tong_hop.csv', mime='text/csv',
                           key=kind+'_summary_download', use_container_width=True)
        st.download_button('⬇ CSV chi tiết', raw,
                           file_name=f'{kind}_chi_tiet.csv', mime='text/csv',
                           key=kind+'_detail_download', use_container_width=True)
    with right:
        st.download_button('⬇ Tải toàn bộ báo cáo (ZIP)', archive.getvalue(),
                           file_name=f'{kind}_report.zip', mime='application/zip',
                           key=kind+'_zip_download', type='primary',
                           use_container_width=True)


def render_compact(kind: str, benchmarks: list[str], title: str, subtitle: str) -> None:
    st.header(title)
    st.caption(subtitle)
    available = models_available()
    if not available:
        st.warning('Chưa kết nối được Ollama hoặc chưa cài model.')
    models = st.multiselect('Chọn model Ollama', available,
                            default=available[:1], key=kind+'_models')
    max_count = 20 if kind == 'experiment2' else 10
    counts = [2, 4, 10, 20] if kind == 'experiment2' else [2, 4, 10]
    count = st.selectbox('Số mẫu mỗi benchmark (DBB tính theo cặp)',
                         counts, key=kind+'_count')
    chosen = (st.multiselect('Chọn benchmark', benchmarks, default=benchmarks,
                             key=kind+'_benchmarks')
              if kind == 'experiment2' else benchmarks)
    total = count * sum(2 if name == 'DBB' else 1 for name in chosen) * len(models)
    st.info(f'{len(models)} model · {len(chosen)} benchmark · khoảng {total} câu hỏi, '
            'chưa tính retry INVALID.')
    st.caption('Seed 42 · temperature 0 · 1 lượt/câu, retry tối đa 1 lần nếu INVALID. '
               'Kết quả lưu tự động, không cần tạo hay chọn run.')
    if st.button('▶ Chạy và tạo báo cáo', type='primary', use_container_width=True,
                 disabled=not models or not chosen, key=kind+'_start'):
        run_ids = []
        bar = st.progress(0)
        total_steps = len(models) * len(chosen)
        try:
            for model_index, model in enumerate(models):
                cfg = {'model': model, 'settings': Settings(temperature=0.0, seed=42,
                         num_predict=96, timeout=120).metadata(),
                       'repeats': 1, 'workflow': 'compact_report_v1',
                       'metric_scope': 'exploratory_only',
                       'benchmarks': chosen,
                       'dataset_hashes': {n: file_sha256(DATASET / DATASETS[n][0])
                                          for n in chosen}}
                for bench_index, name in enumerate(chosen):
                    tasks = build_tasks(name, count, 42, cfg)
                    # Internal audit storage only; no run management in the UI.
                    rid = create_run(kind, cfg)
                    run_ids.append(rid)
                    step = model_index * len(chosen) + bench_index
                    st.caption(f'Đang đánh giá: {model} · {name}')
                    evaluate(kind, rid, tasks,
                             progress=lambda i, n, step=step: bar.progress(
                                 min(1.0, (step + i / max(n, 1)) / total_steps)))
            st.session_state[kind+'_completed_ids'] = run_ids
            st.session_state[kind+'_completed_benchmarks'] = chosen
            st.success('Đã hoàn tất. Báo cáo được tạo từ các phản hồi thực tế.')
        except Exception as exc:
            st.session_state[kind+'_completed_ids'] = run_ids
            st.session_state[kind+'_completed_benchmarks'] = chosen
            st.error('Đánh giá chưa hoàn tất: ' + str(exc))
            st.warning('Các lượt đã ghi vẫn được giữ lại; báo cáo bên dưới có thể chưa đủ dữ liệu.')
    ids = st.session_state.get(kind+'_completed_ids', [])
    if ids:
        try:
            _report(kind, ids, st.session_state.get(kind+'_completed_benchmarks', benchmarks))
        except Exception as exc:
            st.error('Không thể lập báo cáo: ' + str(exc))
