"""Environment variable access with secret hygiene.

Secrets are only ever read from environment variables. Values are never
logged, printed, or written to output files.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

_DOTENV_LOADED = False


def load_env_once() -> None:
    """Load a local ``.env`` file (if present) exactly once, without
    overriding variables that are already exported."""
    global _DOTENV_LOADED
    if not _DOTENV_LOADED:
        load_dotenv(override=False)
        _DOTENV_LOADED = True


def get_env(name: str) -> str | None:
    load_env_once()
    value = os.environ.get(name)
    if value is not None and value.strip() == "":
        return None
    return value


def require_env(name: str) -> str:
    value = get_env(name)
    if value is None:
        raise MissingCredentialError(
            f"Required environment variable {name!r} is not set. "
            f"Set it in your shell or in a .env file (see .env.example)."
        )
    return value


class MissingCredentialError(RuntimeError):
    """Raised before any API call when a provider credential is missing."""
