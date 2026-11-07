"""Regression tests: Windows sharing violations must not abort inference."""
import json
import os
import dbb_status
import dbb_storage

def setup_run(monkeypatch,tmp_path):
    monkeypatch.setattr(dbb_storage, 'RESULTS',tmp_path/'results')
    folder=tmp_path/'results'/'experiment1'/'testrun'
    folder.mkdir(parents=True)
    return folder

def test_status_normal(monkeypatch,tmp_path):
    setup_run(monkeypatch,tmp_path)
    dbb_status.atomic_status('testrun',state='running',done=3)
    assert dbb_status.read_status('testrun')['done']==3

def test_windows_permission_error_retry(monkeypatch,tmp_path):
    setup_run(monkeypatch,tmp_path)
    original=os.replace
    count=[0]
    def flaky(src,dst):
        count[0]+=1
        if count[0]<3:
            raise PermissionError(13,'Access is denied',str(dst),5)
        return original(src,dst)
    monkeypatch.setattr(dbb_status.os,'replace',flaky)
    monkeypatch.setattr(dbb_status.time,'sleep',lambda _:None)
    dbb_status.atomic_status('testrun',state='running',done=7)
    assert count[0]==3
    assert dbb_status.read_status('testrun')['done']==7

def test_permanent_lock_preserves_snapshot(monkeypatch,tmp_path):
    folder=setup_run(monkeypatch,tmp_path)
    (folder/'status.json').write_text(json.dumps({'state':'running','done':2}),encoding='utf-8')
    def denied(src,dst):raise PermissionError(13,'Access is denied',str(dst),5)
    monkeypatch.setattr(dbb_status.os,'replace',denied)
    monkeypatch.setattr(dbb_status.time,'sleep',lambda _:None)
    dbb_status.atomic_status('testrun',state='completed',done=10)
    assert dbb_status.read_status('testrun')['done']==10
    assert dbb_status.read_status('testrun')['state']=='completed'
    assert json.loads((folder/'status.json').read_text())['done']==2
    assert list(folder.glob('.status-*.tmp'))

def test_corrupt_snapshot_ignored(monkeypatch,tmp_path):
    folder=setup_run(monkeypatch,tmp_path)
    (folder/'status.json').write_text(json.dumps({'state':'running','done':4}),encoding='utf-8')
    (folder/'.status-999.tmp').write_text('{invalid',encoding='utf-8')
    assert dbb_status.read_status('testrun')['done']==4
