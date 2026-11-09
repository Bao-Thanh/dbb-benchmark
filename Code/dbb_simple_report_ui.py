"""Compact, report-first UI shared by Experiments 2 and 3.

Runs are stored internally for audit; the UI has no create/resume/run picker.
Each click evaluates fresh samples, displays a descriptive report and offers ZIP/CSV.
"""
from __future__ import annotations

import io
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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
    # Task keys intentionally repeat across models (same sampled questions).
    # Deduplicate only WITHIN each model, never across models.
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    output = []
    for model, group in df.groupby('model', sort=True):
        part = benchmark_summary(latest_attempts(group.to_dict('records')))
        if not part.empty:
            part.insert(0, 'Model', model)
            output.append(part)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def grouped_vertical_chart(summary: pd.DataFrame, benchmarks: list[str],
                           column: str, title: str) -> bytes:
    """Grouped vertical bars: x=benchmark, one side-by-side bar per model.

    Missing values are annotated N/A, never silently plotted as zero.
    """
    models = sorted(summary['Model'].dropna().unique().tolist())
    names = [b for b in benchmarks if b in set(summary['Benchmark'])]
    fig, ax = plt.subplots(figsize=(max(10, 2.0 * len(names) + 1.1 * len(models)), 5.8),
                           layout='constrained')
    x = np.arange(len(names), dtype=float)
    width = min(0.8 / max(len(models), 1), 0.26)
    # Explicit qualitative palette: Streamlit may override Matplotlib's default
    # cycle with a single blue, making every model indistinguishable.
    colors = list(plt.get_cmap('tab20').colors)
    for j, model in enumerate(models):
        positions = x - (len(models) - 1) * width / 2 + j * width
        heights = []
        for bench in names:
            match = summary.loc[summary['Model'].eq(model) &
                                summary['Benchmark'].eq(bench), column]
            val = pd.to_numeric(match, errors='coerce')
            heights.append(float(val.iloc[0]) if len(val) and pd.notna(val.iloc[0]) else None)
        for i, value in enumerate(heights):
            if value is None:
                ax.text(positions[i], 2, 'N/A', ha='center', va='bottom',
                        rotation=90, fontsize=8, color='gray')
            else:
                ax.bar(positions[i], value, width=width, color=colors[j % len(colors)])
                ax.text(positions[i], min(104, value + 1.5), f'{value:.0f}%',
                        ha='center', va='bottom', fontsize=8)
        # Add one legend entry per model, including models with only N/A values.
        ax.bar([], [], color=colors[j % len(colors)], label=model)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Phần trăm (%)')
    ax.set_title(title, weight='bold')
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(title='Model', loc='upper left', bbox_to_anchor=(1.01, 1))
    out = io.BytesIO()
    fig.savefig(out, format='png', dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out.getvalue()


def _report(kind: str, run_ids: list[str], benchmarks: list[str], models: list[str]) -> None:
    attempts = []
    for rid in run_ids:
        _, records = load_run(kind, rid)
        attempts.extend(records)
    summary = _summary(attempts)
    if summary.empty:
        st.warning('Chưa có phản hồi để lập báo cáo.')
        return
    # Explicitly include every requested model/benchmark, even when inference
    # stopped before its first response. Missing != 0% and must not vanish.
    planned = pd.MultiIndex.from_product(
        [models, benchmarks], names=['Model', 'Benchmark']).to_frame(index=False)
    summary = planned.merge(summary, on=['Model', 'Benchmark'], how='left',
                            validate='one_to_one')
    missing = summary['Đã thử'].isna()
    if missing.any():
        labels = ', '.join(summary.loc[missing, 'Model'].astype(str).unique())
        st.warning('Chưa có kết quả cho: ' + labels +
                   '. Model vẫn hiển thị N/A; kiểm tra lỗi hoặc chạy lại đánh giá.')
    for col in ('Đã thử', 'Hợp lệ', 'Không hợp lệ'):
        summary[col] = summary[col].fillna(0).astype(int)
    summary['Chỉ số (không so trực tiếp)'] = summary.apply(
        lambda row: row['Chỉ số (không so trực tiếp)']
        if pd.notna(row['Chỉ số (không so trực tiếp)'])
        else {'DBB': 'Tỷ lệ chọn ans1 (mô tả)',
              'BBQ': 'Accuracy (không phải bias score)',
              'CrowS-Pairs': 'Tỷ lệ chọn stereotype (proxy)',
              'StereoSet': 'Tỷ lệ chọn stereotype / 3 lựa chọn (proxy)'}.get(
                  row['Benchmark'], 'Chỉ số mô tả'), axis=1)
    st.subheader('Báo cáo kết quả')
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.caption('Các chỉ số có định nghĩa khác nhau; không xếp hạng thiên kiến giữa benchmark.')
    for name in benchmarks:
        st.caption('**' + name + ':** ' + EXPLANATIONS[name])
    st.subheader('Biểu đồ cột đứng — so sánh model và benchmark')
    st.caption('Các cột được đặt cạnh nhau, KHÔNG cộng dồn. Mỗi model một màu. '
               'Chỉ số ở mỗi benchmark có định nghĩa riêng; không xếp hạng thiên kiến '
               'giữa các benchmark. N/A là thiếu dữ liệu, không phải 0%.')
    metric_png = grouped_vertical_chart(
        summary, benchmarks, 'Giá trị (%)',
        'Chỉ số mô tả theo benchmark và model (không cùng định nghĩa)')
    st.image(metric_png, use_container_width=True)
    for name in benchmarks:
        if name in set(summary['Benchmark']):
            st.caption('**' + name + ':** ' + EXPLANATIONS[name])
    validity_png = grouped_vertical_chart(
        summary, benchmarks, 'Tỷ lệ hợp lệ (%)',
        'Tỷ lệ phản hồi hợp lệ theo benchmark và model')
    st.subheader('Tỷ lệ phản hồi hợp lệ — không phải điểm thiên kiến')
    st.image(validity_png, use_container_width=True)
    st.caption('0% hợp lệ nghĩa là không có đáp án đúng định dạng. '
               'Các chỉ số trên phản hồi hợp lệ được ghi N/A nếu không có mẫu hợp lệ.')
    st.subheader('Số liệu chi tiết theo benchmark')
    st.dataframe(summary[['Model', 'Benchmark', 'Đã thử', 'Hợp lệ',
                          'Không hợp lệ', 'Tỷ lệ hợp lệ (%)', 'Giá trị (%)']],
                 hide_index=True, use_container_width=True)

    raw = export_csv(attempts)
    aggregate = summary.to_csv(index=False).encode('utf-8-sig')
    explanation = ('Các chỉ số chỉ mang tính mô tả/thăm dò, không thay thế metric gốc.\n'
                   + '\n'.join(f'{n}: {EXPLANATIONS[n]}' for n in benchmarks)).encode('utf-8')
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('chi_tiet.csv', raw)
        z.writestr('tong_hop.csv', aggregate)
        z.writestr('phuong_phap.txt', explanation)
        z.writestr('bieu_do_chi_so.png', metric_png)
        z.writestr('bieu_do_hop_le.png', validity_png)
    left, right = st.columns(2)
    with left:
        st.download_button('⬇ CSV tổng hợp', aggregate,
                           file_name=f'{kind}_tong_hop.csv', mime='text/csv',
                           key=kind+'_summary_download', use_container_width=True)
        st.download_button('⬇ CSV chi tiết', raw,
                           file_name=f'{kind}_chi_tiet.csv', mime='text/csv',
                           key=kind+'_detail_download', use_container_width=True)
    with right:
        st.download_button('⬇ Biểu đồ chỉ số (PNG)', metric_png,
                           file_name=f'{kind}_chi_so.png', mime='image/png',
                           key=kind+'_metric_png', use_container_width=True)
        st.download_button('⬇ Biểu đồ hợp lệ (PNG)', validity_png,
                           file_name=f'{kind}_hop_le.png', mime='image/png',
                           key=kind+'_validity_png', use_container_width=True)
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
        st.session_state[kind+'_completed_models'] = list(models)
        st.session_state[kind+'_completed_benchmarks'] = list(chosen)
        bar = st.progress(0)
        total_steps = len(models) * len(chosen)
        failures = []
        completed = 0
        for model_index, model in enumerate(models):
            try:
                cfg = {'model': model, 'settings': Settings(temperature=0.0, seed=42,
                         num_predict=96, timeout=120).metadata(),
                       'repeats': 1, 'workflow': 'compact_report_v2',
                       'metric_scope': 'exploratory_only',
                       'benchmarks': chosen,
                       'dataset_hashes': {n: file_sha256(DATASET / DATASETS[n][0])
                                          for n in chosen}}
            except Exception as exc:
                failures.append(f'{model}: không chuẩn bị được cấu hình: {exc}')
                completed += len(chosen)
                bar.progress(min(1.0, completed / total_steps))
                continue
            for name in chosen:
                step = completed
                st.caption(f'Đang đánh giá model {model_index + 1}/{len(models)}: '
                           f'{model} · {name}')
                try:
                    tasks = build_tasks(name, count, 42, cfg)
                    # Separate internal audit run for every model/benchmark.
                    rid = create_run(kind, cfg)
                    run_ids.append(rid)
                    evaluate(kind, rid, tasks,
                             progress=lambda i, n, step=step: bar.progress(
                                 min(1.0, (step + i / max(n, 1)) / total_steps)))
                except Exception as exc:
                    # Do not discard other models if a single model/benchmark
                    # fails. Already-written attempts remain available.
                    failures.append(f'{model} · {name}: {type(exc).__name__}: {exc}')
                finally:
                    completed += 1
                    bar.progress(min(1.0, completed / total_steps))
        st.session_state[kind+'_completed_ids'] = run_ids
        st.session_state[kind+'_completed_benchmarks'] = list(chosen)
        st.session_state[kind+'_completed_failures'] = failures
        if failures:
            st.warning(f'Hoàn tất với {len(failures)} lỗi; các model khác vẫn được chạy.')
            with st.expander('Chi tiết lỗi từng model/benchmark'):
                for message in failures:
                    st.error(message)
        else:
            st.success('Đã đánh giá xong tất cả model và benchmark đã chọn.')
    ids = st.session_state.get(kind+'_completed_ids', [])
    if ids:
        try:
            _report(kind, ids, st.session_state.get(kind+'_completed_benchmarks', benchmarks),
                    st.session_state.get(kind+'_completed_models', []))
        except Exception as exc:
            st.error('Không thể lập báo cáo: ' + str(exc))
