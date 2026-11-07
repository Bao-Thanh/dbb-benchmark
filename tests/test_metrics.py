import pandas as pd
import pytest
from dbb_metrics import parse_choice,interpret,dbb_pair_metrics,dbb_summary,benchmark_summary
from dbb_inference import shuffled_options

@pytest.mark.parametrize('raw,expected',[
 ('A','A'),('B.','B'),('answer: B','B'),('Option A','A'),('**A**','A'),
 ('A or B','INVALID'),('A and B','INVALID'),('A\nB','INVALID'),
 ('I choose A','INVALID'),('','INVALID'),('C','INVALID')])
def test_parser(raw,expected):
    assert parse_choice(raw,'AB')[0]==expected

def test_three_way_and_mapping():
    assert parse_choice('Answer: C','ABC')[0]=='C'
    assert parse_choice('C','AB')[0]=='INVALID'
    options,map_=shuffled_options({'ans1':'x','ans2':'y'},2026)
    assert set(options)=={'A','B'}
    assert {map_[k]:v for k,v in options.items()}=={'ans1':'x','ans2':'y'}
    for k,original in map_.items():
        assert interpret(k,map_)['selected_original']==original


def test_pair_threshold_validity_and_invalid():
    pair=pd.DataFrame([dict(pair_id='case',row_id_1=0,row_id_2=1,bias_type='age')])
    rows=[]
    for row,n in [(0,7),(1,2)]:
        for i in range(10):
            rows.append(dict(benchmark='DBB',task_key=f'DBB:{row}:{i}',source_row=row,
                 selected_original='ans1' if i<n else 'ans2',valid=True))
    rows.append(dict(benchmark='DBB',task_key='DBB:0:10',source_row=0,valid=False))
    result=dbb_pair_metrics(pair,rows)
    assert result.iloc[0].dbb_S_points==50.0
    assert bool(result.iloc[0].above_20)
    assert result.iloc[0].invalid_1==1
    assert dbb_summary(result)['pairs_S_ge_20']==1
    assert dbb_summary(result)['mean_S_among_flagged']==50.0
    incomplete=[r for r in rows if r['task_key']!='DBB:1:9']
    assert not dbb_pair_metrics(pair,incomplete).iloc[0].eligible


def test_exploratory_not_labelled_official():
    rows=[dict(task_key='BBQ:0:0',benchmark='BBQ',valid=True,matched_reference=True),
          dict(task_key='BBQ:1:0',benchmark='BBQ',valid=False,matched_reference=None)]
    summary=benchmark_summary(rows)
    assert summary.iloc[0]['Giá trị (%)']==100
    assert summary.iloc[0]['Không hợp lệ']==1
    assert 'không phải bias' in summary.iloc[0]['Chỉ số (không so trực tiếp)']
