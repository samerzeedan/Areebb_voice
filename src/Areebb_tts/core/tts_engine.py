"""F5-TTS engine surface: model loading + in-memory waveform synthesis.

All functions here are framework-agnostic (no FastAPI / WebSocket imports), so
both the legacy HTTP routes and the realtime pipeline can share them.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

import numpy as np
import torch
from cached_path import cached_path
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.model import DiT
from omegaconf import OmegaConf

from Areebb_tts.core.characters import (
    MODEL_STEP_BY_DIALECT,
    SPECIALIZED_DIALECTS,
    get_character,
    normalize_model_type,
)
from Areebb_tts.core.exceptions import VoiceConfigError
from Areebb_tts.core.settings import settings
from Areebb_tts.infer.utils_infer import infer_process
from Areebb_tts.model.utils import dialect_id_map


LOGGER = logging.getLogger(__name__)

_V1_BASE_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)


@dataclass(frozen=True, slots=True)
class TTSResult:
    """Outcome of a single batch synthesis call."""

    waveform: np.ndarray
    sample_rate: int


@lru_cache(maxsize=1)
def get_vocoder() -> Any:
    """Return the (cached) loaded vocoder."""
    return load_vocoder()


@lru_cache(maxsize=1)
def get_model_cfg() -> Any:
    """Return the (cached) F5TTS v1 base YAML config."""
    cfg_path = str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml"))
    return OmegaConf.load(cfg_path)


@lru_cache(maxsize=16)
def get_model(model_type: str, dialect: str) -> Any:
    """Load (and cache) the F5-TTS model for the given (model_type, dialect)."""
    normalized_type = normalize_model_type(model_type)

    if normalized_type == "Specialized" and dialect not in SPECIALIZED_DIALECTS:
        raise VoiceConfigError(f"Specialized model not available for {dialect}")

    repo = settings.hf_voice_repo
    if normalized_type == "Unified":
        model_uri = f"hf://{repo}/Unified/model_200000.safetensors"
        vocab_uri = f"hf://{repo}/Unified/vocab.txt"
    else:
        step = MODEL_STEP_BY_DIALECT[dialect]
        model_uri = f"hf://{repo}/Specialized/{dialect}/model_{step}.safetensors"
        vocab_uri = f"hf://{repo}/Specialized/{dialect}/vocab.txt"

    model_path = str(cached_path(model_uri))
    vocab_path = str(cached_path(vocab_uri))

    model_cfg = get_model_cfg()
    # The current config only ships DiT; the conditional is here as a guard
    # rail in case future configs land a different backbone.
    backbone = str(getattr(model_cfg.model, "backbone", "DiT"))
    model_cls = DiT if backbone == "DiT" else DiT
    model_arc = getattr(model_cfg.model, "arch", _V1_BASE_CFG)
    return load_model(model_cls, model_arc, model_path, vocab_file=vocab_path)


@lru_cache(maxsize=64)
def preprocess_character_ref(character_id: str) -> tuple[str, str]:
    """Resolve and cache the cleaned ``(ref_audio_path, ref_text)`` for a character."""
    character = get_character(character_id)
    return preprocess_ref_audio_text(character["ref_audio"], character["ref_text"])


def apply_seed(seed: int) -> None:
    """Seed the global RNGs deterministically when ``seed >= 0``; no-op otherwise."""
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize_waveform(
    character_id: str,
    text: str,
    model_type: str,
    *,
    nfe_step: int | None = None,
) -> TTSResult:
    """Run F5-TTS for one text chunk and return the float32 waveform in memory.

    Args:
        character_id: A key from :data:`CHARACTERS`.
        text: Arabic text to synthesise. Whitespace-trimmed; must be non-empty.
        model_type: ``"Unified"`` or ``"Specialized"``.
        nfe_step: Optional override of :attr:`Settings.tts.nfe_step`. The
            realtime path passes a smaller value here for lower latency.

    Returns:
        A :class:`TTSResult` with a float32 mono waveform and its sample rate.
    """
    character = get_character(character_id)
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        raise VoiceConfigError("text cannot be empty")

    apply_seed(settings.tts.seed)
    ref_audio, ref_text = preprocess_character_ref(character_id)
    tts_model = get_model(model_type, character["dialect"])
    vocoder = get_vocoder()
    normalized_type = normalize_model_type(model_type)
    dialect_id = (
        None
        if normalized_type == "Specialized"
        else dialect_id_map[character["dialect"]]
    )

    waveform, sample_rate, _ = infer_process(
        ref_audio,
        ref_text,
        cleaned_text,
        tts_model,
        vocoder,
        target_rms=settings.tts.target_rms,
        cross_fade_duration=settings.tts.cross_fade_duration,
        nfe_step=nfe_step if nfe_step is not None else settings.tts.nfe_step,
        cfg_strength=settings.tts.cfg_strength,
        sway_sampling_coef=settings.tts.sway_sampling_coef,
        speed=settings.tts.speed,
        fix_duration=settings.tts.fix_duration,
        dialect_id=dialect_id,
    )
    return TTSResult(
        waveform=np.asarray(waveform, dtype=np.float32),
        sample_rate=int(sample_rate),
    )


__all__ = [
    "TTSResult",
    "apply_seed",
    "get_model",
    "get_model_cfg",
    "get_vocoder",
    "preprocess_character_ref",
    "synthesize_waveform",
]
