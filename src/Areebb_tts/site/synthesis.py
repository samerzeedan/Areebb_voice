"""High-level synthesis orchestrator for the HTTP layer.

Decides between local F5-TTS and the remote TTS proxy, and returns a public
audio URL the client can fetch. Pure orchestration; no FastAPI imports.
"""

from __future__ import annotations

from Areebb_tts.core.characters import get_character
from Areebb_tts.core.exceptions import VoiceConfigError
from Areebb_tts.core.settings import settings
from Areebb_tts.core.tts_engine import synthesize_waveform
from Areebb_tts.site.audio_storage import save_waveform_wav
from Areebb_tts.site.remote_tts import call_remote_tts


def synthesize_text(character_id: str, text: str, model_type: str) -> str:
    """Synthesise ``text`` using the chosen character and return its audio URL."""
    character = get_character(character_id)
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        raise VoiceConfigError("text cannot be empty")

    if settings.remote_tts.enabled:
        return call_remote_tts(character_id, character, cleaned_text, model_type)

    result = synthesize_waveform(character_id, cleaned_text, model_type)
    return save_waveform_wav(result.waveform, result.sample_rate)


__all__ = ["synthesize_text"]
