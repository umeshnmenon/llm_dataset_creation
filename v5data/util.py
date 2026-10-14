import hashlib
import json
import os
from pathlib import Path


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(canonical(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def immutable_json(directory, obj):
    path = Path(directory) / (digest(obj) + ".json")
    if path.exists():
        if read_json(path) != obj:
            raise ValueError("Immutable object was changed: " + str(path))
    else:
        atomic_json(path, obj)
    return path.name


def emit(root, message):
    print(message, flush=True)
    with (Path(root) / "run.log").open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, list) else value
