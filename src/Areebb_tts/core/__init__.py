"""Shared, framework-agnostic primitives used across site/ and realtime/.

Public API:

- :data:`settings` — typed, frozen configuration loaded from env / .env once.
- :data:`CHARACTERS`, :data:`DIALECT_LABELS`, :data:`SPECIALIZED_DIALECTS`,
  :data:`MODEL_STEP_BY_DIALECT` — voice catalog data.
- :class:`VoiceConfigError`, :class:`OllamaError`, :class:`RemoteTTSError`
  — domain exceptions.
- :func:`get_character`, :func:`normalize_model_type` — catalog helpers.
- :func:`get_model`, :func:`get_vocoder`, :func:`synthesize_waveform`,
  :class:`TTSResult` — F5-TTS engine surface.
- :func:`configure_logging` — opt-in logging setup.
"""

from Areebb_tts.core.characters import (
    CHARACTERS,
    DIALECT_LABELS,
    MODEL_STEP_BY_DIALECT,
    SPECIALIZED_DIALECTS,
    get_character,
    normalize_model_type,
)
from Areebb_tts.core.exceptions import (
    OllamaError,
    RemoteTTSError,
    VoiceConfigError,
)
from Areebb_tts.core.logging_config import configure_logging
from Areebb_tts.core.settings import settings
from Areebb_tts.core.tts_engine import (
    TTSResult,
    apply_seed,
    get_model,
    get_model_cfg,
    get_vocoder,
    preprocess_character_ref,
    synthesize_waveform,
)


__all__ = [
    "CHARACTERS",
    "DIALECT_LABELS",
    "MODEL_STEP_BY_DIALECT",
    "OllamaError",
    "RemoteTTSError",
    "SPECIALIZED_DIALECTS",
    "TTSResult",
    "VoiceConfigError",
    "apply_seed",
    "configure_logging",
    "get_character",
    "get_model",
    "get_model_cfg",
    "get_vocoder",
    "normalize_model_type",
    "preprocess_character_ref",
    "settings",
    "synthesize_waveform",
]
