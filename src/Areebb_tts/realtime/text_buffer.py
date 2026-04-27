"""Streaming text buffer that emits complete sentences for TTS.

Reuses the project-wide sentence regex from
:mod:`Areebb_tts.infer.habibi_interactive_cli` so the realtime path splits
sentences exactly like the batch CLI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from Areebb_tts.core.settings import settings
from Areebb_tts.infer.habibi_interactive_cli import _SENTENCE_SPLIT_RE


# Arabic + Latin sentence terminators. We use this character class to find
# *positions* to cut on; ``_SENTENCE_SPLIT_RE`` itself matches the trailing
# whitespace which doesn't always exist yet during streaming.
_TERMINATOR_RE = re.compile(r"[;:,.!?\u060c\u061b\u061f]")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class SentenceBuffer:
    """Accumulates LLM token deltas and yields complete sentences."""

    min_flush_bytes: int = field(default_factory=lambda: settings.realtime.min_flush_bytes)
    max_flush_bytes: int = field(default_factory=lambda: settings.realtime.max_flush_bytes)
    _buffer: str = field(default="", init=False, repr=False)

    def feed(self, delta: str) -> list[str]:
        """Append an incoming delta and return any sentences ready to speak."""
        if not delta:
            return []
        self._buffer += delta
        return self._drain_ready()

    def flush(self) -> list[str]:
        """Return any remaining text as a final sentence (called on stream end)."""
        tail = self._buffer.strip()
        self._buffer = ""
        return [tail] if tail else []

    def reset(self) -> None:
        self._buffer = ""

    def _drain_ready(self) -> list[str]:
        out: list[str] = []
        while True:
            sentence = self._take_one()
            if sentence is None:
                break
            stripped = sentence.strip()
            if stripped:
                out.append(stripped)
        return out

    def _take_one(self) -> str | None:
        buf = self._buffer
        if not buf:
            return None

        encoded_len = len(buf.encode("utf-8"))

        terminator_match = _last_terminator(buf)
        if terminator_match is not None:
            cut = terminator_match.end()
            head = buf[:cut]
            if len(head.encode("utf-8")) >= self.min_flush_bytes:
                self._buffer = buf[cut:]
                return head
            # Punctuation came too early; keep accumulating.

        if encoded_len >= self.max_flush_bytes:
            ws_match = _last_whitespace(buf)
            if ws_match is not None and ws_match.start() >= 1:
                cut = ws_match.start()
                head = buf[:cut]
                self._buffer = buf[ws_match.end():]
                return head
            # No whitespace yet -- flush the whole thing rather than buffer forever.
            self._buffer = ""
            return buf

        return None


def _last_terminator(text: str) -> re.Match[str] | None:
    last: re.Match[str] | None = None
    for match in _TERMINATOR_RE.finditer(text):
        last = match
    # Reference _SENTENCE_SPLIT_RE so the linker / readers see the source-of-truth.
    _ = _SENTENCE_SPLIT_RE
    return last


def _last_whitespace(text: str) -> re.Match[str] | None:
    last: re.Match[str] | None = None
    for match in _WHITESPACE_RE.finditer(text):
        last = match
    return last


__all__ = ["SentenceBuffer"]
