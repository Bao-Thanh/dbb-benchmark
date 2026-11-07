"""Windows-safe, non-critical progress snapshots for Experiment 1.

A status write must never abort an inference run merely because Windows or an
antivirus temporarily locks status.json while Streamlit reads it.

The append-only attempts.jsonl remains the source of truth.  status.json is a
best-effort UI snapshot.  If an atomic replace stays locked after brief retries,
the complete temporary snapshot is kept and can be read by Streamlit.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dbb_storage import folder_for

_DEFAULT = {'state': 'initializing', 'done': 0, 'total': 0,
            'valid': 0, 'invalid': 0}
_RETRY_DELAYS = (0.02, 0.05, 0.1, 0.2, 0.4)


def _snapshot_paths(folder: Path) -> list[Path]:
    candidates = [folder / 'status.json', *folder.glob('.status-*.tmp')]
    available = []
    for p in candidates:
        try:
            available.append((p.stat().st_mtime_ns, p))
        except OSError:
            pass
    return [p for _, p in sorted(available, key=lambda item: item[0], reverse=True)]


def read_status(run_id: str) -> dict:
    """Read the most recent complete snapshot, including rename-failure files."""
    folder = folder_for('experiment1', run_id)
    for path in _snapshot_paths(folder):
        try:
            state = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(state, dict) and 'state' in state:
                return state
        except (OSError, ValueError):
            # Skip files briefly locked or incomplete in a concurrent write.
            continue
    return dict(_DEFAULT)


def _prune_snapshots(folder: Path, keep: int = 3) -> None:
    """Best-effort cleanup; never remove status.json or stop requests."""
    candidates = [p for p in _snapshot_paths(folder) if p.name.startswith('.status-')]
    for path in candidates[keep:]:
        try:
            path.unlink()
        except OSError:
            pass  # A UI reader may have an old snapshot open on Windows.


def atomic_status(run_id: str, **changes) -> dict:
    """Write status with retries, keeping an immutable fallback on WinError 5/32.

    The file is completely written and closed before the atomic replace. A
    temporary Windows sharing/permission error at rename time is not fatal.
    The fallback is already a fully written JSON snapshot; UI reads it even if
    status.json cannot be replaced. No existing run output is overwritten.
    """
    folder = folder_for('experiment1', run_id)
    status_path = folder / 'status.json'
    state = {**read_status(run_id), **changes,
             'updated_at_utc': datetime.now(timezone.utc).isoformat()}
    temp = folder / f'.status-{time.time_ns()}-{uuid4().hex}.tmp'
    temp.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
    for delay in (0, *_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            os.replace(temp, status_path)
            _prune_snapshots(folder)
            return state
        except PermissionError:
            continue
    # os.replace() was repeatedly denied, usually because a Windows reader or
    # security scanner has status.json open. Keep the COMPLETE temp snapshot.
    print(f'[status] status.json is temporarily locked; preserved progress in {temp.name}',
          file=sys.stderr, flush=True)
    _prune_snapshots(folder)
    return state
