"""Deterministic JSON serialization and atomic file I/O."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` to a canonical JSON string.

    Keys are sorted and separators are fixed so the same logical object always
    produces the same bytes. Used for hashing configs and samples.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def dump_json(obj: Any, *, indent: int = 2) -> str:
    """Human-readable JSON with stable key order for output files."""
    return json.dumps(obj, indent=indent, sort_keys=False, ensure_ascii=False)


def load_json_file(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    """Write JSON to ``path`` atomically (temp file + rename).

    Prevents corrupt/partial files if the process is interrupted mid-write,
    which matters for checkpoints.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(dump_json(obj, indent=indent))
            f.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def try_parse_json(text: str) -> Any | None:
    """Parse JSON, returning ``None`` instead of raising on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
