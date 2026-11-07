import os
import pytest
from pathlib import Path
import dbb_storage
import dbb_runner
from dbb_inference import Settings
from dbb_metrics import latest_attempts


def test_append_only_resume_and_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(dbb_storage,'RESULTS',tmp_path)
    config={'model':'fake:1b','settings':Settings().metadata(),'repeats':1}
    run=dbb_storage.create_run('experiment2',config)
    task=dbb_runner.new_task('BBQ',0,0,'A question','Which is true?',
                            {'ans0':'a','ans1':'b','ans2':'c'},config,reference='ans1')
    calls=[]
    def mocked(model,prompt,settings):
        calls.append(settings.seed)
        return {'raw':'A or B' if len(calls)==1 else next(k for k,v in task['display_mapping'].items() if v=='ans1'),
                'error':'','seconds':0.1,'total_duration':0,'done_reason':'stop'}
    monkeypatch.setattr(dbb_runner,'ask_ollama',mocked)
    records,valid=dbb_runner.evaluate('experiment2',run,[task])
    assert valid==1 and len(records)==2
    assert records[0]['answer']=='INVALID'
    assert records[1]['matched_reference'] is True
    assert dbb_storage.load_run('experiment2',run)[1]==records
    again,valid=dbb_runner.evaluate('experiment2',run,[task])
    assert len(again)==2 and len(calls)==2
    assert len(dbb_storage.export_csv(again))>20
    with pytest.raises(RuntimeError, match='lock'):
        with dbb_storage.run_lock('experiment2',run):
            with dbb_storage.run_lock('experiment2',run):
                pass
    assert not (dbb_storage.folder_for('experiment2',run)/'.lock').exists()


def test_unsafe_paths_and_storage_validation(tmp_path,monkeypatch):
    monkeypatch.setattr(dbb_storage,'RESULTS',tmp_path)
    for unsafe in ['../secret','a/b','a\\b','']:
        with pytest.raises(ValueError):
            dbb_storage.folder_for('experiment1',unsafe)
