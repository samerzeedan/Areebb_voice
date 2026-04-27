"""On-disk audio storage for the legacy HTTP path.

The realtime WebSocket pipeline keeps everything in memory; only the batch
``/api/tts`` and ``/api/chat`` flows save WAVs and serve them via URL.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from Areebb_tts.core.settings import settings


# Filename shape we generate: ``<32 hex chars>.wav``. Constrains the public
# audio route to reject path traversal / odd inputs.
GENERATED_WAV_NAME = re.compile(r"[0-9a-f]{32}\.wav\Z", re.IGNORECASE)

_READ_BUFFER_BYTES = 1024 * 1024  # 1 MiB


def output_dir() -> Path:
    """Return the directory where generated WAVs live (created if needed)."""
    return settings.output_dir


def save_waveform_wav(waveform: np.ndarray, sample_rate: int) -> str:
    """Persist a waveform to a uniquely-named WAV and return its public URL."""
    file_name = f"{uuid.uuid4().hex}.wav"
    sf.write(output_dir() / file_name, waveform, sample_rate)
    return f"/audio/{file_name}"


def save_bytes_wav(content: bytes) -> str:
    """Persist raw WAV bytes (e.g. from a remote TTS proxy) and return its URL."""
    file_name = f"{uuid.uuid4().hex}.wav"
    (output_dir() / file_name).write_bytes(content)
    return f"/audio/{file_name}"


def resolve_safe_path(file_name: str) -> Path | None:
    """Return the on-disk path for a generated audio file, or ``None`` if invalid."""
    if not GENERATED_WAV_NAME.fullmatch(file_name):
        return None
    path = output_dir() / file_name
    return path if path.is_file() else None


def iter_wav_bytes(path: Path) -> Iterator[bytes]:
    """Yield the WAV file in 1 MiB chunks for ``StreamingResponse``."""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_READ_BUFFER_BYTES)
            if not chunk:
                break
            yield chunk


__all__ = [
    "GENERATED_WAV_NAME",
    "iter_wav_bytes",
    "output_dir",
    "resolve_safe_path",
    "save_bytes_wav",
    "save_waveform_wav",
]
