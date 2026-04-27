"""Per-sentence TTS streaming: F5-TTS in a worker thread + PCM16 chunking."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator

import numpy as np

from Areebb_tts.core.settings import settings
from Areebb_tts.core.tts_engine import TTSResult, synthesize_waveform


LOGGER = logging.getLogger(__name__)

_PCM16_FULLSCALE = 32767.0
_BYTES_PER_SAMPLE = 2  # int16


class _SentenceCancelled(Exception):
    """Internal signal: cancellation observed before/while the sample ran."""


def _waveform_to_pcm16_bytes(wave: np.ndarray) -> bytes:
    """Convert a float32 mono waveform in [-1, 1] to little-endian PCM16 bytes."""
    if wave.size == 0:
        return b""
    clipped = np.clip(wave, -1.0, 1.0)
    pcm = (clipped * _PCM16_FULLSCALE).astype(np.int16, copy=False)
    return pcm.tobytes()


def _slice_into_frames(pcm: bytes, frame_samples: int) -> list[bytes]:
    if not pcm:
        return []
    bytes_per_frame = frame_samples * _BYTES_PER_SAMPLE
    return [pcm[i:i + bytes_per_frame] for i in range(0, len(pcm), bytes_per_frame)]


async def synthesize_sentence_frames(
    character_id: str,
    sentence: str,
    model_type: str,
    *,
    cancel_event: threading.Event,
    frame_samples: int | None = None,
    nfe_step: int | None = None,
) -> AsyncIterator[tuple[bytes, int]]:
    """Yield ``(pcm16_frame_bytes, sample_rate)`` tuples for one sentence.

    ``model.sample`` runs in the default executor so the event loop stays free
    to handle WS reads/writes. Cancellation is checked between frames; the
    sentence-level work cannot be interrupted mid-sample (F5-TTS exposes no
    hook), but each sentence is 1-3 s of audio which is the natural
    interrupt granularity for barge-in.
    """
    if cancel_event.is_set():
        return

    resolved_frame_samples = frame_samples or settings.realtime.frame_samples
    resolved_nfe_step = nfe_step if nfe_step is not None else settings.realtime.nfe_step

    loop = asyncio.get_running_loop()
    try:
        result: TTSResult = await loop.run_in_executor(
            None,
            _synthesize_blocking,
            character_id,
            sentence,
            model_type,
            resolved_nfe_step,
            cancel_event,
        )
    except _SentenceCancelled:
        return

    if cancel_event.is_set() or result.waveform.size == 0:
        return

    pcm = _waveform_to_pcm16_bytes(result.waveform)
    sample_rate = result.sample_rate
    for frame in _slice_into_frames(pcm, resolved_frame_samples):
        if cancel_event.is_set():
            return
        yield frame, sample_rate
        # Yield to the loop so WS sends + cancel checks interleave cleanly.
        await asyncio.sleep(0)


def _synthesize_blocking(
    character_id: str,
    sentence: str,
    model_type: str,
    nfe_step: int,
    cancel_event: threading.Event,
) -> TTSResult:
    if cancel_event.is_set():
        raise _SentenceCancelled
    try:
        return synthesize_waveform(
            character_id=character_id,
            text=sentence,
            model_type=model_type,
            nfe_step=nfe_step,
        )
    except Exception:
        if cancel_event.is_set():
            raise _SentenceCancelled from None
        raise


__all__ = ["synthesize_sentence_frames"]
