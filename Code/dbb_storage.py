"""Append-only experiment storage; no legacy Results files are ever overwritten."""
from __future__ import annotations
import csv
import io
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'Results' / 'explainable'


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_run(kind: str, config: dict) -> str:
    if kind not in ('experiment1','experiment2','experiment3'):
        raise ValueError('Invalid experiment kind')
    folder=RESULTS/kind
    folder.mkdir(parents=True, exist_ok=True)
    run_id=now()[:10].replace('-','')+'-'+uuid4().hex[:12]
    dest=folder/run_id
    dest.mkdir()  # fail if exists; never overwrite
    (dest/'manifest.json').write_text(json.dumps({
        'run_id':run_id,'kind':kind,'created_at_utc':now(),
        'config':config, 'schema_version':1},ensure_ascii=False,indent=2),encoding='utf-8')
    return run_id


def folder_for(kind: str, run_id: str) -> Path:
    if kind not in ('experiment1','experiment2','experiment3'):
        raise ValueError('Invalid experiment kind')
    if not run_id or '/' in run_id or '\\' in run_id or '..' in run_id:
        raise ValueError('Unsafe run ID')
    return RESULTS/kind/run_id


def load_run(kind: str, run_id: str) -> tuple[dict,list[dict]]:
    folder=folder_for(kind,run_id)
    config=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    entries=[]
    records=folder/'attempts.jsonl'
    if records.exists():
        with records.open(encoding='utf-8') as handle:
            for i,line in enumerate(handle,1):
                if not line.strip(): continue
                try: entries.append(json.loads(line))
                except json.JSONDecodeError as err:
                    # An interrupted final write should be visible, never silently discarded.
                    raise ValueError(f'Bản ghi JSONL hỏng dòng {i} trong {records}: {err}') from err
    return config,entries


def list_runs(kind: str) -> list[str]:
    root=RESULTS/kind
    if not root.is_dir(): return []
    return sorted((x.name for x in root.iterdir() if (x/'manifest.json').is_file()),reverse=True)


@contextmanager
def run_lock(kind: str, run_id: str):
    """A single writer per run. If interrupted, check process before removing .lock."""
    f=folder_for(kind,run_id)/'.lock'
    fd=None
    created=False
    try:
        fd=os.open(f,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        created=True
        os.write(fd,f'pid={os.getpid()} time={now()}'.encode())
        os.close(fd);fd=None
        yield
    except FileExistsError as exc:
        raise RuntimeError(f'Run đang được ghi hoặc còn lock: {f}') from exc
    finally:
        if fd is not None: os.close(fd)
        # Only unlink a lock we created, not another process's lock.
        if 'fd' in locals() and created:
            f.unlink(missing_ok=True)


def append_attempt(kind: str, run_id: str, record: dict):
    """Call inside run_lock; preserve every error and retry as separate line."""
    path=folder_for(kind,run_id)/'attempts.jsonl'
    entry={'timestamp_utc':now(),'run_id':run_id,**record}
    with path.open('a',encoding='utf-8',newline='\n') as out:
        out.write(json.dumps(entry,ensure_ascii=False,default=str)+'\n')
        out.flush()
        os.fsync(out.fileno())


def export_csv(records: list[dict]) -> bytes:
    if not records: return b''
    cols=list(dict.fromkeys(k for r in records for k in r.keys()))
    out=io.StringIO()
    w=csv.DictWriter(out,fieldnames=cols,extrasaction='ignore')
    w.writeheader(); w.writerows(records)
    return out.getvalue().encode('utf-8-sig')


def latest_by_task(records: list[dict]) -> dict[str,dict]:
    return {str(r['task_key']):r for r in records if 'task_key' in r}


def attempts_for_task(records: list[dict], key: str) -> int:
    return sum(x.get('task_key')==key for x in records)
