"""UTC timestamp helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def local_now_iso() -> str:
    """Current local time as ISO-8601 with UTC offset, if determinable."""
    return datetime.now().astimezone().isoformat(timespec="seconds")
