"""Pluggable STT adapter interface.

The realtime WebSocket protocol currently expects ``user_text`` JSON frames
(client-side STT or typed input). This module defines the seam where a future
server-side streaming ASR (e.g. faster-whisper + VAD) can be wired in without
changing the WS protocol or pipeline layout.

Enable a server-side adapter by setting ``AREEB_ENABLE_SERVER_STT=1`` and
returning an :class:`STTAdapter` from :func:`get_stt_adapter` in your fork.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from Areebb_tts.core.settings import settings


class STTAdapter(Protocol):
    """Minimal contract for a streaming ASR backend."""

    async def feed(self, pcm: bytes) -> None: ...

    async def finalize(self) -> str: ...

    def stream(self) -> AsyncIterator[str]:  # pragma: no cover - optional API
        ...


class NoServerSTT:
    """Default no-op adapter that explicitly refuses to accept audio frames."""

    name = "noop"

    async def feed(self, pcm: bytes) -> None:
        raise NotImplementedError(
            "Server-side STT is disabled. Send a 'user_text' JSON frame instead, "
            "or set AREEB_ENABLE_SERVER_STT=1 and provide a real adapter."
        )

    async def finalize(self) -> str:
        raise NotImplementedError("Server-side STT is disabled.")


def get_stt_adapter() -> STTAdapter | None:
    """Return the configured STT adapter, or ``None`` to keep the seam closed."""
    if not settings.server_stt_enabled:
        return None
    return NoServerSTT()


__all__ = ["NoServerSTT", "STTAdapter", "get_stt_adapter"]
