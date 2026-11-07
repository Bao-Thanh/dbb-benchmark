"""Synthetic fixtures only. No research figures are produced by this test."""
import io
import zipfile

import pandas as pd
import pytest
import dbb_compact_report as report
import dbb_compact_worker as worker
import dbb_storage
from dbb_storage import create_run,load_run


def pairs():
    return pd.DataFrame([
      dict(pair_id=f'p{i}',bias_type=g,row_id_1=2*i,row_id_2=2*i+1)
      for i,g in enumerate(['age','gender','race','religions','socioeconomic_class'])
    ])


def sample_attempts(n=10):
    records=[]
    for model in ('m1','m2'):
        for row in (0,1):
            for trial in range(n):
                source='ans1' if model=='m1' and row==0 else 'ans2'
                mapping={'A':'ans1','B':'ans2'} if trial%2==0 else {'A':'ans2','B':'ans1'}
                records.append(dict(benchmark='DBB',model=model,source_row=row,trial=trial,
                    task_key=f'{model}|DBB:{row}:{trial}',valid=True,
                    selected_original=source,display_mapping=mapping,raw='A',
                    bias_type='age',prompt='TEST ONLY',error=''))
    return records


def test_disjoint_balanced_question_sampling():
    first=report.sample_disjoint_pairs(pairs(),10,'Tất cả',42)
    repeat=report.sample_disjoint_pairs(pairs(),10,'Tất cả',42)
    assert first.pair_id.tolist()==repeat.pair_id.tolist()
    assert len(set(first.row_id_1).union(first.row_id_2))==10
    assert set(first.bias_type)==set(pairs().bias_type)
    with pytest.raises(ValueError):report.sample_disjoint_pairs(pairs(),9,'Tất cả',42)
    with pytest.raises(ValueError):report.sample_disjoint_pairs(pairs(),12,'Tất cả',42)


def test_multimodel_metrics_and_missing_scores():
    p=pairs().head(1)
    summary,scores,diagnostics=report.summarize(p,sample_attempts(),['m1','m2'])
    assert summary.iloc[0].pairs_eligible==1
    assert summary.iloc[0].pairs_S_ge_20==1
    assert summary.iloc[0].dbb_flagged_pct==100
    assert summary.iloc[1].dbb_flagged_pct==0
    assert diagnostics.iloc[0]['Single-Pass Bias']==100
    assert diagnostics.iloc[0]['Position Bias']==0
    assert pd.isna(diagnostics.iloc[0]['True Social Bias'])
    incomplete,_,diag2=report.summarize(p,sample_attempts(1),['m1','m2'])
    assert incomplete.dbb_flagged_pct.isna().all()
    assert diag2['Position Bias'].isna().all()


def test_png_csv_zip_and_partial_flag():
    p=pairs().head(1)
    files=report.report_files(p,sample_attempts(2),['m1','m2'],'run123',False,
                              {'config':{'models':['m1','m2']}})
    assert files['bieu_do_nhom.png'].startswith(b'\x89PNG')
    assert files['bieu_do_chi_so.png'].startswith(b'\x89PNG')
    summary=pd.read_csv(io.BytesIO(files['tong_hop.csv']))
    assert summary.run_complete.eq(False).all()
    assert summary.dbb_flagged_pct.isna().all()
    detail=pd.read_csv(io.BytesIO(files['chi_tiet.csv']))
    assert detail.run_id.eq('run123').all()
    assert 'prompt' in detail and 'display_mapping' in detail
    with zipfile.ZipFile(io.BytesIO(report.zip_files(files))) as z:
        assert set(z.namelist())==set(files)
        assert b'INCOMPLETE' in z.read('phuong_phap.txt')


def test_same_prompt_sequence_across_models():
    df=pd.DataFrame([dict(row_id=i,Context=f'ctx{i}',ans1='x',ans2='y',
                         bias_type='age',target=str(i),explanation='test') for i in range(2)])
    cfg=dict(repeats=10,settings=dict(seed=42,temperature=0.8,top_p=1.,
                num_predict=96,num_ctx=2048,timeout=120))
    a=worker.prepare_tasks(df,pairs().head(1),cfg,'modelA')
    b=worker.prepare_tasks(df,pairs().head(1),cfg,'modelB')
    assert len(a)==len(b)==20
    assert [t['prompt'] for t in a]==[t['prompt'] for t in b]
    assert len({t['task_key'] for t in a+b})==40
    assert sum(t['display_mapping']['A']=='ans1' for t in a)==10
    assert sum(t['display_mapping']['A']=='ans2' for t in a)==10


def test_stop_keeps_previous_attempt_and_new_runs_have_new_ids(monkeypatch,tmp_path):
    monkeypatch.setattr(dbb_storage,'RESULTS',tmp_path/'Results')
    data_dir=tmp_path/'Dataset'
    data_dir.mkdir()
    (data_dir/'Bias-Dataset.csv').write_text('test',encoding='utf-8')
    monkeypatch.setattr(worker,'DATASET',data_dir)
    monkeypatch.setattr(worker,'file_sha256',lambda path:'hash')
    monkeypatch.setattr(worker,'verify_audited_dbb',lambda path:'audited')
    df=pd.DataFrame([dict(row_id=i,Context=f'ctx{i}',ans1='x',ans2='y',
                         bias_type='age',target=str(i),explanation='test') for i in range(2)])
    monkeypatch.setattr(worker,'load_dbb',lambda:df)
    monkeypatch.setattr(worker,'audited_pairs',lambda _:pairs().head(1))
    cfg=dict(workflow='compact_multi_v1',models=['m1','m2'],pair_ids=['p0'],
             question_count=2,repeats=10,audited_sha256='audited',dataset_sha256='hash',
             settings=dict(seed=42,temperature=.8,top_p=1.,num_predict=96,num_ctx=2048,timeout=120))
    rid=create_run('experiment1',cfg)
    def fake_api(*args):
        (dbb_storage.folder_for('experiment1',rid)/'stop.request').touch()
        return dict(raw='A',error='',seconds=.01,total_duration=0,done_reason='stop')
    state=worker.run_once(rid,api=fake_api)
    assert state['state']=='stopped'
    assert state['done']<state['total']
    assert len(load_run('experiment1',rid)[1])==1
    rid2=create_run('experiment1',cfg)
    assert rid2!=rid
    with pytest.raises(ValueError,match='đã có dữ liệu'):
        worker.run_once(rid,api=fake_api)
