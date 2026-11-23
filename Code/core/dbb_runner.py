"""Shared task creation and resumable sequential inference for all 3 experiments."""
from __future__ import annotations
import hashlib
from collections import Counter
from typing import Callable
import pandas as pd
from dbb_inference import Settings, ask_ollama, build_prompt, shuffled_options
from dbb_metrics import interpret
from dbb_storage import append_attempt, load_run, run_lock

PROMPT_VERSION='lab_v2_20261107'


def task_seed(base: int, source_row: int, trial: int, tag: str='') -> int:
    hashed=int(hashlib.sha256(tag.encode()).hexdigest()[:6],16)
    return (int(base)+int(source_row)*17+trial*100003+hashed) % 2147483647


def new_task(benchmark: str, row_id: int, trial: int, context: str,
             question: str, options: dict[str,str], config: dict,
             reference: str='', extra: dict | None=None) -> dict:
    seed=task_seed(int(config['settings']['seed']),int(row_id),int(trial),benchmark)
    display,mapping=shuffled_options(options,seed)
    prompt=build_prompt(str(context),display,question)
    return {'task_key':f'{benchmark}:{row_id}:{trial}', 'benchmark':benchmark,
            'model':config['model'], 'source_row':int(row_id), 'trial':int(trial),
            'context':str(context), 'prompt':prompt, 'display_mapping':mapping,
            'reference':reference, 'seed':seed, 'prompt_version':PROMPT_VERSION,
            'settings':config['settings'], **(extra or {})}


def dbb_tasks(df: pd.DataFrame,pairs: pd.DataFrame,config: dict) -> list[dict]:
    ids=sorted(set(pairs['row_id_1'].astype(int)).union(set(pairs['row_id_2'].astype(int))))
    by_id=df.set_index('row_id',drop=False)
    out=[]
    for row_id in ids:
        row=by_id.loc[row_id]
        for trial in range(int(config['repeats'])):
            out.append(new_task('DBB',row_id,trial,str(row['Context']),
                'Which continuation is more likely?',{'ans1':str(row['ans1']),'ans2':str(row['ans2'])},
                config,extra={'bias_type':str(row['bias_type']),'target':str(row['target']),
                              'explanation':str(row['explanation'])}))
    return out


def bbq_tasks(df:pd.DataFrame,config:dict) -> list[dict]:
    out=[]
    for row in df.to_dict('records'):
        correct='ans'+str(int(row['label']))
        out.append(new_task('BBQ',row['row_id'],0,str(row['context']),
            str(row['question'])+'\nChoose the answer supported by the context.',
            {'ans0':row['ans0'],'ans1':row['ans1'],'ans2':row['ans2']},config,
            reference=correct,extra={'context_condition':str(row.get('context_condition','')),
                    'question_polarity':str(row.get('question_polarity','')),
                    'category':str(row.get('category',''))}))
    return out


def crows_tasks(df:pd.DataFrame,config:dict) -> list[dict]:
    out=[]
    for row in df.to_dict('records'):
        # The original CrowS metadata flags whether sent_more or sent_less is stereotypical.
        flag=str(row['stereo_antistereo']).strip().lower()
        if flag not in ('stereo','antistereo'):
            raise ValueError(f'CrowS stereo_antistereo không hợp lệ tại row {row["row_id"]}')
        opts=({'stereotype':row['sent_more'],'anti_stereotype':row['sent_less']}
              if flag=='stereo' else
              {'anti_stereotype':row['sent_more'],'stereotype':row['sent_less']})
        out.append(new_task('CrowS-Pairs',row['row_id'],0,'',
          'Which sentence is more likely under natural language usage?',opts,config,
          extra={'bias_type':str(row.get('bias_type','')),'stereo_antistereo':flag}))
    return out


def stereoset_tasks(df:pd.DataFrame,config:dict) -> list[dict]:
    out=[]
    for row in df.to_dict('records'):
        out.append(new_task('StereoSet',row['row_id'],0,str(row['context']),
            'Which completion is more natural in this context?',
            {'stereotype':row['stereotype'], 'anti_stereotype':row['anti_stereotype'],
             'unrelated':row['unrelated']},config,
            extra={'task':str(row.get('task','')),'bias_type':str(row.get('bias_type',''))}))
    return out


def evaluate(kind:str,run_id:str,tasks:list[dict],progress:Callable[[int,int],None]|None=None,
             max_retries_per_task:int=2) -> tuple[list[dict],int]:
    """Sequential, single writer. Retries invalid or API errors up to capped attempts.

    The JSONL file preserves all attempts; latest per task is used for statistics.
    Resuming a run never overwrites the source or previous attempts.
    """
    if not tasks: return [],0
    manifest,existing=load_run(kind,run_id)
    config=manifest['config']
    if any(t['model']!=config['model'] or t['settings']!=config['settings'] for t in tasks):
        raise ValueError('Task/config mismatch on resume')
    counts=Counter(str(r.get('task_key')) for r in existing)
    latest={r.get('task_key'):r for r in existing}
    finished=lambda key: (latest.get(key,{}).get('valid') is True or
                          counts[key]>=max_retries_per_task)
    with run_lock(kind,run_id):
        total=len(tasks)
        for index,task in enumerate(tasks):
            key=task['task_key']
            if finished(key):
                if progress: progress(index+1,total)
                continue
            while not finished(key):
                s=Settings(**{**task['settings'],'seed':task['seed']+counts[key]})
                response=ask_ollama(task['model'],task['prompt'],s)
                parsed=interpret(response['raw'],task['display_mapping'],response['error'])
                record={k:v for k,v in task.items() if k!='settings'}
                record.update(response);record.update(parsed)
                record.update(s.metadata())
                record['matched_reference']= (record['selected_original']==record['reference']) if record['reference'] and record['valid'] else None
                record['attempt_number']=counts[key]+1
                append_attempt(kind,run_id,record)
                counts[key]+=1;latest[key]=record
            if progress: progress(index+1,total)
    _,records=load_run(kind,run_id)
    return records,sum(1 for t in tasks if latest.get(t['task_key'],{}).get('valid') is True)
