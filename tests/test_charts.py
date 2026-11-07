"""Experiment 1 chart tests: denominators, retry handling, and rendering."""
import matplotlib.pyplot as plt
import pandas as pd
from dbb_charts import (
    category_pair_rates, response_quality, answer_selection,
    plot_pair_scores, plot_category_flags, plot_invalid_rate,
    plot_selection, plot_pair_probabilities,
)


def sample():
    scores=pd.DataFrame([
        dict(pair_id='a',bias_type='age',eligible=True,above_20=True,dbb_S_points=60,
             p1_ans1_pct=80,p2_ans1_pct=20,valid_1=10,valid_2=10),
        dict(pair_id='b',bias_type='age',eligible=True,above_20=False,dbb_S_points=10,
             p1_ans1_pct=30,p2_ans1_pct=40,valid_1=10,valid_2=10),
        dict(pair_id='c',bias_type='gender',eligible=False,above_20=False,dbb_S_points=None,
             p1_ans1_pct=100,p2_ans1_pct=None,valid_1=2,valid_2=0),
    ])
    attempts=[
        dict(benchmark='DBB',model='m',task_key='DBB:0:0',bias_type='age',
             source_row=0,valid=False,selected_original=''),
        dict(benchmark='DBB',model='m',task_key='DBB:0:0',bias_type='age',
             source_row=0,valid=True,selected_original='ans1'),
        dict(benchmark='DBB',model='m',task_key='DBB:1:0',bias_type='age',
             source_row=1,valid=True,selected_original='ans2'),
        dict(benchmark='DBB',model='m',task_key='DBB:4:0',bias_type='gender',
             source_row=4,valid=False,selected_original=''),
        dict(benchmark='DBB',model='other',task_key='other:4:0',bias_type='gender',
             source_row=4,valid=True,selected_original='ans1'),
    ]
    return scores,attempts


def test_flag_rate_only_eligible():
    scores,_=sample()
    result=category_pair_rates(scores)
    assert result.bias_type.tolist()==['age']
    assert result.iloc[0].eligible_pairs==2
    assert result.iloc[0].flagged_pairs==1
    assert result.iloc[0].flagged_pct==50


def test_quality_counts_retry_as_real_attempt():
    _,records=sample()
    q=response_quality(records,'m').set_index('bias_type')
    assert q.loc['age','attempts']==3
    assert q.loc['age','invalid']==1
    assert round(q.loc['age','invalid_pct'],2)==33.33
    assert q.loc['gender','invalid_pct']==100


def test_descriptive_choice_latest_and_model_filter():
    _,records=sample()
    selection=answer_selection(records,'m').set_index('bias_type')
    assert selection.loc['age','ans1']==1
    assert selection.loc['age','valid']==2
    assert selection.loc['age','ans1_pct_valid']==50
    assert pd.isna(selection.loc['gender','ans1_pct_valid'])


def test_all_figures_render_without_streamlit_or_ollama():
    scores,records=sample()
    figs=[
        plot_pair_scores(scores),
        plot_category_flags(category_pair_rates(scores)),
        plot_invalid_rate(response_quality(records,'m')),
        plot_selection(answer_selection(records,'m')),
        plot_pair_probabilities(scores.iloc[0]),
    ]
    assert all(fig is not None and fig.axes for fig in figs)
    assert len(figs[0].axes[0].patches)==2
    for fig in figs: plt.close(fig)


def test_ineligible_pairs_have_no_s_chart():
    scores,_=sample()
    subset=scores.loc[~scores.eligible]
    assert category_pair_rates(subset).empty
    assert plot_pair_scores(subset) is None
    assert plot_pair_probabilities(subset.iloc[0]) is None
    assert plot_invalid_rate(response_quality([])) is None
    assert plot_selection(answer_selection([])) is None
