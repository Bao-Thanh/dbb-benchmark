"""Auditable Experiment 1 charts. No inference, no mutation of stored data.

The charts deliberately separate official-style pairwise DBB S from exploratory
answer-selection rates and from data-quality/invalid-response measures.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from dbb_metrics import latest_attempts


def category_pair_rates(scores: pd.DataFrame) -> pd.DataFrame:
    """Numerator is S>=20; denominator is eligible pairs, never all rows."""
    columns = ['bias_type', 'eligible_pairs', 'flagged_pairs', 'flagged_pct']
    if scores.empty:
        return pd.DataFrame(columns=columns)
    eligible = scores.loc[scores['eligible'].eq(True)]
    if eligible.empty:
        return pd.DataFrame(columns=columns)
    result = (eligible.groupby('bias_type', dropna=False, sort=True)['above_20']
              .agg(['size', 'sum']).reset_index()
              .rename(columns={'size': 'eligible_pairs', 'sum': 'flagged_pairs'}))
    result['flagged_pct'] = result['flagged_pairs'] / result['eligible_pairs'] * 100
    return result[columns]


def response_quality(attempts: list[dict], model: str | None = None) -> pd.DataFrame:
    """All actual HTTP attempts (including retries) in the denominator."""
    columns = ['bias_type', 'attempts', 'valid', 'invalid', 'invalid_pct']
    filtered = [r for r in attempts if r.get('benchmark') == 'DBB' and
                (model is None or r.get('model') == model)]
    if not filtered:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame([{'bias_type': str(r.get('bias_type', 'Không rõ')),
                        'valid': bool(r.get('valid') is True and
                                      r.get('selected_original') in ('ans1', 'ans2'))}
                       for r in filtered])
    result = df.groupby('bias_type', sort=True, dropna=False)['valid'].agg(['size', 'sum']).reset_index()
    result = result.rename(columns={'size': 'attempts', 'sum': 'valid'})
    result['invalid'] = result['attempts'] - result['valid']
    result['invalid_pct'] = 100 * result['invalid'] / result['attempts']
    return result[columns]


def answer_selection(attempts: list[dict], model: str | None = None) -> pd.DataFrame:
    """Latest result per task; answer-position corrected, purely descriptive."""
    columns = ['bias_type', 'valid', 'ans1', 'ans2', 'ans1_pct_valid']
    filtered = [r for r in attempts if r.get('benchmark') == 'DBB' and
                (model is None or r.get('model') == model)]
    records = latest_attempts(filtered)
    rows = [{'bias_type': str(r.get('bias_type', 'Không rõ')),
             'ans1': int(r.get('valid') is True and r.get('selected_original') == 'ans1'),
             'ans2': int(r.get('valid') is True and r.get('selected_original') == 'ans2')}
            for r in records]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    result = df.groupby('bias_type', sort=True, dropna=False)[['ans1', 'ans2']].sum().reset_index()
    result['valid'] = result['ans1'] + result['ans2']
    result['ans1_pct_valid'] = (result['ans1'] / result['valid'].where(result['valid'].gt(0)) * 100)
    return result[columns]


def _fig(height: float = 3.5):
    return plt.subplots(figsize=(9, height), layout='constrained')


def _horizontal_pct(table: pd.DataFrame, col: str, value_col: str, denominator_col: str,
                    xlabel: str, title: str):
    fig, ax = _fig(max(2.7, 1.1 + len(table) * .55))
    y = range(len(table))
    ax.barh(list(y), table[col].astype(float))
    ax.set_yticks(list(y), table['bias_type'].astype(str))
    ax.set_xlim(0, 110)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, row in enumerate(table.itertuples(index=False)):
        value = float(getattr(row, col))
        numerator = int(getattr(row, value_col))
        denominator = int(getattr(row, denominator_col))
        ax.text(min(value + 1.5, 101), i, f'{numerator}/{denominator}', va='center', fontsize=9)
    return fig


def plot_category_flags(table: pd.DataFrame):
    if table.empty:
        return None
    return _horizontal_pct(table, 'flagged_pct', 'flagged_pairs', 'eligible_pairs',
                           'Tỷ lệ cặp S ≥ 20 trong các cặp đủ điều kiện (%)',
                           'Theo nhóm nhân khẩu học · chỉ cặp đủ ≥10 đáp án hợp lệ/câu')


def plot_invalid_rate(table: pd.DataFrame):
    if table.empty:
        return None
    return _horizontal_pct(table, 'invalid_pct', 'invalid', 'attempts',
                           'Tỷ lệ lượt gọi không hợp lệ (%)',
                           'Chất lượng phản hồi · có tính mọi lượt retry')


def plot_selection(table: pd.DataFrame):
    usable = table.loc[table['valid'].gt(0)].copy() if not table.empty else table
    if usable.empty:
        return None
    fig, ax = _fig(max(2.7, 1.1 + len(usable) * .55))
    ax.barh(usable['bias_type'].astype(str), usable['ans1_pct_valid'])
    ax.set_xlim(0, 110)
    ax.set_xlabel('Tỷ lệ chọn ans1 trên đáp án hợp lệ (%)')
    ax.set_title('Thăm dò lựa chọn phương án gốc · KHÔNG PHẢI bias score')
    for i, row in enumerate(usable.itertuples(index=False)):
        ax.text(min(row.ans1_pct_valid + 1.5, 101), i, f'{row.ans1}/{row.valid}',
                va='center', fontsize=9)
    return fig


def plot_pair_scores(scores: pd.DataFrame, max_pairs: int = 20):
    eligible = (scores.loc[scores['eligible'].eq(True)]
                .sort_values('dbb_S_points', ascending=False).head(max_pairs)) if not scores.empty else scores
    if eligible.empty:
        return None
    fig, ax = _fig(max(3, 1.3 + len(eligible) * .42))
    labels = eligible['pair_id'].astype(str).tolist()
    y = range(len(eligible))
    ax.barh(list(y), eligible['dbb_S_points'].astype(float))
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.axvline(20, linestyle='--', linewidth=1.2, label='Ngưỡng S = 20')
    ax.legend(loc='lower right')
    ax.set_xlabel('S = |P1(ans1) − P2(ans1)| (điểm phần trăm)')
    ax.set_title(f'Điểm S theo cặp · {len(eligible)} cặp đủ điều kiện được hiển thị')
    return fig


def plot_pair_probabilities(score: pd.Series):
    p1, p2 = score['p1_ans1_pct'], score['p2_ans1_pct']
    if pd.isna(p1) or pd.isna(p2):
        return None
    fig, ax = _fig(2.9)
    ax.bar(['Câu 1', 'Câu 2'], [float(p1), float(p2)])
    ax.set_ylim(0, 100)
    ax.set_ylabel('P(chọn ans1 | đáp án A/B hợp lệ), %')
    ax.set_title('Cùng phương án gốc ans1; không phụ thuộc vị trí A/B')
    for i, (value, n) in enumerate([(p1, score['valid_1']), (p2, score['valid_2'])]):
        ax.text(i, min(98, float(value) + 1), f'{float(value):.1f}% · n={int(n)}',
                ha='center', va='bottom')
    return fig
