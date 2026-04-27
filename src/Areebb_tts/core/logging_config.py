"""Centralised logging configuration helper.

Importing this module has no side effects. Call :func:`configure_logging`
once during application startup (e.g. in ``site.app.run``) if you want a
consistent format across the package and its third-party deps.
"""

from __future__ import annotations

import logging
import os


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int | str | None = None, *, force: bool = False) -> None:
    """Set up a single root handler with a consistent format.

    Args:
        level: Logging level, e.g. ``logging.INFO`` or ``"DEBUG"``. If ``None``,
            falls back to the ``AREEB_LOG_LEVEL`` env var or ``INFO``.
        force: If ``True``, replace any existing handlers (useful when other
            libraries already touched the root logger).
    """
    resolved_level = _resolve_level(level)
    logging.basicConfig(
        level=resolved_level,
        format=_DEFAULT_FORMAT,
        datefmt=_DEFAULT_DATEFMT,
        force=force,
    )
    # httpx is chatty at DEBUG and adds noise during streaming; tame it.
    logging.getLogger("httpx").setLevel(max(resolved_level, logging.WARNING))
    logging.getLogger("httpcore").setLevel(max(resolved_level, logging.WARNING))


def _resolve_level(level: int | str | None) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return logging.getLevelName(level.upper())  # type: ignore[return-value]
    env_level = os.getenv("AREEB_LOG_LEVEL", "INFO").upper()
    return logging.getLevelName(env_level)  # type: ignore[return-value]


__all__ = ["configure_logging"]
