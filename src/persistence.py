from __future__ import annotations
import os, pickle, time
from pathlib import Path

CACHE_DIR = Path(os.environ.get('BOATRACE_CACHE_DIR', '.runtime_cache'))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _path(name: str) -> Path:
    safe = ''.join(c for c in str(name) if c.isalnum() or c in ('-','_','.'))
    return CACHE_DIR / f'{safe}.pkl'


def save_pickle(name: str, obj, meta: dict | None = None) -> Path:
    p = _path(name)
    tmp = p.with_suffix('.tmp')
    payload = {'saved_at': time.time(), 'meta': meta or {}, 'value': obj}
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)
    return p


def load_pickle(name: str, max_age_hours: float | None = None, required_version: str | None = None):
    p = _path(name)
    if not p.exists():
        return None
    try:
        with open(p, 'rb') as f:
            payload = pickle.load(f)
        if max_age_hours is not None:
            age = time.time() - float(payload.get('saved_at', 0))
            if age > float(max_age_hours) * 3600:
                return None
        if required_version is not None:
            if str((payload.get('meta') or {}).get('version')) != str(required_version):
                return None
        return payload
    except Exception:
        return None


def remove_pickle(name: str):
    p = _path(name)
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass
