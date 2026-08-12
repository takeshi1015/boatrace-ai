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


def cache_file_exists(name: str) -> bool:
    return _path(name).exists()


def export_learning_snapshot(names=('models', 'learning_assets')) -> bytes | None:
    """Export runtime learning cache as a portable ZIP kept by the user."""
    import io, json, zipfile
    files = []
    for name in names:
        fp = _path(name)
        if fp.exists() and fp.is_file():
            files.append((name, fp))
    if not files:
        return None

    bio = io.BytesIO()
    manifest = {
        'format': 'boatrace-learning-snapshot-v1',
        'created_at': time.time(),
        'files': [name for name, _ in files],
    }
    with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False))
        for name, fp in files:
            z.writestr(f'cache/{name}.pkl', fp.read_bytes())
    return bio.getvalue()


def import_learning_snapshot(data: bytes, allowed_names=('models', 'learning_assets')) -> dict:
    """Restore a user-owned learning snapshot into the runtime cache."""
    import io, json, zipfile
    result = {'restored': [], 'errors': []}
    if not data:
        result['errors'].append('空のファイルです')
        return result
    try:
        with zipfile.ZipFile(io.BytesIO(data), 'r') as z:
            names = set(z.namelist())
            if 'manifest.json' not in names:
                result['errors'].append('学習スナップショットではありません')
                return result
            manifest = json.loads(z.read('manifest.json').decode('utf-8'))
            if manifest.get('format') != 'boatrace-learning-snapshot-v1':
                result['errors'].append('未対応のスナップショット形式です')
                return result

            for name in allowed_names:
                member = f'cache/{name}.pkl'
                if member not in names:
                    continue
                raw = z.read(member)
                if len(raw) > 100 * 1024 * 1024:
                    result['errors'].append(f'{name} が大きすぎます')
                    continue
                # Validate payload before writing it.
                payload = pickle.loads(raw)
                if not isinstance(payload, dict) or 'value' not in payload or 'meta' not in payload:
                    result['errors'].append(f'{name} の内容が不正です')
                    continue
                target = _path(name)
                tmp = target.with_suffix('.restore.tmp')
                tmp.write_bytes(raw)
                tmp.replace(target)
                result['restored'].append(name)
    except Exception as e:
        result['errors'].append(f'復元失敗: {type(e).__name__}')
    return result
