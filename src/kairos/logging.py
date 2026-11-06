"""Logging for KAIROS: every line carries the run ID.

Use :func:`get_logger` everywhere instead of :func:`logging.getLogger`::

    from kairos.logging import get_logger

    log = get_logger(__name__)
    log.info("fitted %d series", n)

The first call configures the ``kairos`` logger from :mod:`kairos.settings`:
a rich console handler (``log_format="rich"``) or one JSON object per line on
stderr (``log_format="json"``). Worker processes inherit the run ID through the
``KAIROS_RUN_ID`` environment variable so that pooled logs can be joined.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import uuid
from datetime import UTC, datetime

from rich.logging import RichHandler

__all__ = ["JsonFormatter", "RunIdFilter", "configure_logging", "get_logger", "get_run_id"]

ROOT = "kairos"
_RUN_ID_ENV = "KAIROS_RUN_ID"
_lock = threading.Lock()
_configured = False


def _new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def get_run_id() -> str:
    """Return this run's ID, creating and exporting it on first use.

    The ID is stored in ``KAIROS_RUN_ID`` so spawned workers share it.
    """
    run_id = os.environ.get(_RUN_ID_ENV)
    if not run_id:
        from kairos.settings import get_settings

        run_id = get_settings().run_id or _new_run_id()
        os.environ[_RUN_ID_ENV] = run_id
    return run_id


class RunIdFilter(logging.Filter):
    """Attach ``record.run_id`` to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Stamp the record and keep it."""
        record.run_id = get_run_id()
        return True


class JsonFormatter(logging.Formatter):
    """Format records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise ``record`` to JSON."""
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", None),
            "pid": record.process,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    level: str | None = None, fmt: str | None = None, *, force: bool = False
) -> logging.Logger:
    """Configure the ``kairos`` logger once (or again with ``force=True``).

    Args:
        level: Log level name; defaults to ``Settings.log_level``.
        fmt: ``"rich"`` or ``"json"``; defaults to ``Settings.log_format``.
        force: Replace handlers installed by an earlier call.

    Returns:
        The configured ``kairos`` root logger.
    """
    global _configured
    logger = logging.getLogger(ROOT)
    with _lock:
        if _configured and not force:
            return logger
        from kairos.settings import get_settings

        settings = get_settings()
        level = (level or settings.log_level).upper()
        fmt = fmt or settings.log_format
        for old in list(logger.handlers):
            logger.removeHandler(old)
        handler: logging.Handler
        if fmt == "json":
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(JsonFormatter())
        else:
            handler = RichHandler(rich_tracebacks=True, show_path=False, log_time_format="%X")
            handler.setFormatter(logging.Formatter("[%(run_id)s] %(name)s: %(message)s"))
        handler.addFilter(RunIdFilter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        _configured = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``kairos`` hierarchy, configuring logging on first use.

    Args:
        name: Dotted name; ``kairos.`` is prefixed when missing. ``None`` gives the root.
    """
    configure_logging()
    if not name or name == ROOT:
        return logging.getLogger(ROOT)
    if not name.startswith(ROOT + "."):
        name = f"{ROOT}.{name}"
    return logging.getLogger(name)
