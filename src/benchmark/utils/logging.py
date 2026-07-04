"""Console logging via rich."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()

_CONFIGURED = False


def get_logger(name: str = "benchmark") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
        )
        _CONFIGURED = True
    return logging.getLogger(name)
