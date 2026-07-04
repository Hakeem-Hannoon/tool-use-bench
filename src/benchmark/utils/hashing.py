"""Stable hashing for configs and sample files."""

from __future__ import annotations

import hashlib
from typing import Any

from .json_utils import canonical_json


def sha256_of_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_json(obj: Any) -> str:
    """Hash a JSON-serializable object deterministically.

    The object is serialized with sorted keys and fixed separators so that
    semantically identical objects always hash the same.
    """
    return sha256_of_text(canonical_json(obj))
