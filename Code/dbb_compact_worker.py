"""Spawned subprocess for Streamlit Experiment 1; cooperative stop and atomic progress.

Start only with a freshly created run ID; no replay or resume of old runs.
All outputs stay under Results/explainable/experiment1/<run-id>/.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dbb_data import DATASET, audited_pairs, file_sha256, load_dbb, verify_audited_dbb
from dbb_inference import Settings, ask_ollama, build_prompt
from dbb_metrics import interpret
from dbb_runner import dbb_tasks
from dbb_storage import append_attempt, folder_for, load_run, run_lock
from dbb_status import atomic_status


def prepare_tasks(df, pairs, config: dict, model: str) -> list[dict]:
    """Every model sees the same trial prompts; balanced original A/B positions.

    Trial parity determines whether ans1 is shown at A or B. This is an
    experimental modification relative to the paper; comparisons retain the
    same option semantics via the saved display_mapping.
    """
    subcfg = {**config, 'model': model}
    out = dbb_tasks(df, pairs, subcfg)
    by_id = df.set_index('row_id', drop=False)
    for task in out:
        trial = int(task['trial'])
        row = by_id.loc[int(task['source_row'])]
        mapping = ({'A':'ans1','B':'ans2'} if trial % 2 == 0 else
                   {'A':'ans2','B':'ans1'})
        display = {letter:str(row[src]) for letter,src in mapping.items()}
        task['display_mapping'] = mapping
        task['prompt'] = build_compact_dbb_prompt(str(row['Context']),display)
        task['prompt_version'] = 'dbb_ab_only_v2_20261107'
        task['task_key'] = f'{model}|{task["task_key"]}'
        task['question_id'] = int(task['source_row'])
        task['order'] = 'normal' if trial % 2 == 0 else 'swapped'
    return out


def checked_run(run_id: str):
    manifest, old = load_run('experiment1', run_id)
    cfg = manifest['config']
    if cfg.get('workflow') != 'compact_multi_v1':
        raise ValueError('Run không thuộc giao diện đa model mới.')
    if old:
        raise ValueError('Run này đã có dữ liệu; không tự động tiếp tục.')
    path = DATASET / 'Bias-Dataset.csv'
    verified = verify_audited_dbb(path)
    if cfg['audited_sha256'] != verified or cfg['dataset_sha256'] != file_sha256(path):
        raise ValueError('DBB CSV đã thay đổi so với thời điểm tạo run.')
    df = load_dbb()
    pairs_all = audited_pairs(df)
    ids = cfg['pair_ids']
    pairs = pairs_all.loc[pairs_all['pair_id'].isin(ids)].copy()
    if len(pairs) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError('Danh sách cặp lưu trong run không hợp lệ.')
    questions = set(pairs['row_id_1']).union(pairs['row_id_2'])
    if len(questions) != cfg['question_count']:
        raise ValueError('Các cặp chồng lặp row_id hoặc sai số câu hỏi.')
    return cfg, df, pairs


def run_once(run_id: str, api=ask_ollama) -> dict:
    folder = folder_for('experiment1', run_id)
    atomic_status(run_id, state='initializing', done=0, valid=0, invalid=0, model='', error='')
    try:
        cfg, df, pairs = checked_run(run_id)
        models = cfg['models']
        total = int(cfg['question_count']) * len(models) * int(cfg['repeats'])
        done = valid = invalid = 0
        with run_lock('experiment1', run_id):
            atomic_status(run_id, state='running', total=total, done=0, valid=0, invalid=0)
            for model in models:
                tasks = prepare_tasks(df, pairs, cfg, model)
                for task in tasks:
                    if (folder / 'stop.request').exists():
                        return atomic_status(run_id, state='stopped', done=done, valid=valid,
                                             invalid=invalid, total=total,
                                             message='Đã dừng theo yêu cầu; CSV/PNG có thể tải phần dữ liệu hiện có.')
                    atomic_status(run_id, state='running', total=total, done=done,
                                  valid=valid, invalid=invalid, model=model,
                                  question=task['source_row'])
                    # Retries remain recorded separately; max 2 actual HTTP calls per planned trial.
                    for attempt_number in range(1, 3):
                        if (folder / 'stop.request').exists():
                            return atomic_status(run_id, state='stopped', done=done,
                                                 valid=valid, invalid=invalid, total=total)
                        settings = Settings(**{**cfg['settings'],
                                               'seed':int(task['seed']) + attempt_number - 1})
                        response = api(model,task['prompt'],settings)
                        parsed = interpret(response['raw'],task['display_mapping'],response['error'])
                        record = {k:v for k,v in task.items() if k!='settings'}
                        record.update(response)
                        record.update(parsed)
                        record.update(settings.metadata())
                        record['attempt_number'] = attempt_number
                        record['run_complete'] = False  # final run status stored in status.json
                        append_attempt('experiment1',run_id,record)
                        if parsed['valid']:
                            valid += 1
                            break
                        invalid += 1
                    done += 1
                    atomic_status(run_id, state='running', total=total, done=done,
                                  valid=valid, invalid=invalid, model=model,
                                  question=task['source_row'])
            return atomic_status(run_id, state='completed', done=done, total=total,
                                 valid=valid, invalid=invalid, model='',
                                 message='Đã hoàn tất toàn bộ số lượt dự kiến.')
    except Exception as exc:
        atomic_status(run_id, state='failed', error=str(exc)[:900],
                      message='Thất bại — mọi kết quả đã ghi vẫn được giữ lại.')
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    try:
        run_once(args.run_id)
    except Exception:
        traceback.print_exc()
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
