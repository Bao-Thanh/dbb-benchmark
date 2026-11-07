import os
import pytest
import pandas as pd
from dbb_data import audited_pairs,choose_pairs,load_dbb,DBB_SHA256,file_sha256

# Opt-in real-data test when user's audited dataset exists in the test working tree.
def test_audited_full_dataset_if_present():
    from dbb_data import DATASET
    path=DATASET/'Bias-Dataset.csv'
    if not path.exists():
        pytest.skip('Dataset not present in local Git checkout')
    assert file_sha256(path)==DBB_SHA256
    df=load_dbb()
    pairs=audited_pairs(df)
    assert len(pairs)==103649
    assert pairs['pair_id'].is_unique
    assert pairs.groupby('bias_type').size().to_dict()=={
        'age':4641,'gender':6188,'race':61880,'socioeconomic_class':3094,'religions':27846}
    subset=choose_pairs(pairs,3,'age',42)
    assert len(subset)==3
    assert set(subset['bias_type'])=={'age'}


def test_pair_index_rejects_mismatch(monkeypatch,tmp_path):
    from dbb_data import DATASET
    import dbb_data
    fake=tmp_path/'Bias-Dataset.csv'
    fake.write_text('hello')
    monkeypatch.setattr(dbb_data,'DATASET',tmp_path)
    # Even a 77350-row dataframe cannot bypass the SHA check.
    with pytest.raises(ValueError,match='SHA256'):
        audited_pairs(pd.DataFrame())
