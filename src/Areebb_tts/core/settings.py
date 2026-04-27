"""Typed configuration loaded from environment / .env.

Use the module-level :data:`settings` singleton everywhere. Avoid sprinkling
``os.getenv`` calls across the codebase. ``.env`` is loaded *once* here, before
any settings are read, so module-level config is reliable regardless of import
order.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Idempotent: load_dotenv() does not override existing env vars by default.
load_dotenv(_PROJECT_ROOT / ".env")


_TRUE_TOKENS = {"1", "true", "yes", "on"}
_FALSE_TOKENS = {"0", "false", "no", "off", ""}


def _env_str(key: str, default: str = "", *aliases: str) -> str:
    """Return the first non-empty value of ``key`` or any alias, else default."""
    for candidate in (key, *aliases):
        value = os.getenv(candidate)
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"env var {key!r} is not an int: {raw!r}") from exc


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"env var {key!r} is not a float: {raw!r}") from exc


def _env_optional_float(key: str) -> Optional[float]:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"env var {key!r} is not a float: {raw!r}") from exc


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    token = raw.strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return default if token == "" else False
    raise ValueError(f"env var {key!r} is not a bool: {raw!r}")


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    base_url: str
    model: str


@dataclass(frozen=True, slots=True)
class TTSSettings:
    """Defaults for batch synthesis. Match Gradio/infer pipeline by default."""

    target_rms: float
    cross_fade_duration: float
    nfe_step: int
    cfg_strength: float
    sway_sampling_coef: float
    speed: float
    fix_duration: Optional[float]
    seed: int  # -1 means random per request


@dataclass(frozen=True, slots=True)
class RealtimeSettings:
    """Realtime-specific overrides for low-latency streaming."""

    nfe_step: int
    frame_samples: int
    min_flush_bytes: int
    max_flush_bytes: int
    prewarm_character: str
    prewarm_model_type: str
    prewarm_disabled: bool


@dataclass(frozen=True, slots=True)
class RemoteTTSSettings:
    url: str
    method: str
    json_payload: bool
    timeout: int

    @property
    def enabled(self) -> bool:
        return bool(self.url)


@dataclass(frozen=True, slots=True)
class Settings:
    """Top-level application configuration."""

    project_root: Path
    output_dir: Path
    hf_voice_repo: str
    hf_token: str
    server_stt_enabled: bool
    host: str
    port: int
    ollama: OllamaSettings
    tts: TTSSettings
    realtime: RealtimeSettings
    remote_tts: RemoteTTSSettings


def _load() -> Settings:
    output_dir = _PROJECT_ROOT / "generated_audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        project_root=_PROJECT_ROOT,
        output_dir=output_dir,
        hf_voice_repo=_env_str("AREEB_HF_VOICE_REPO", "SWivid/Habibi-TTS", "HF_VOICE_REPO"),
        hf_token=_env_str("HF_TOKEN", ""),
        server_stt_enabled=_env_bool("AREEB_ENABLE_SERVER_STT", False),
        host=_env_str("AREEB_HOST", "0.0.0.0"),
        port=_env_int("AREEB_PORT", 9000),
        ollama=OllamaSettings(
            base_url=_env_str("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            model=_env_str("OLLAMA_MODEL", "Qwen2.5:7b-instruct-q4_K_M"),
        ),
        tts=TTSSettings(
            target_rms=_env_float("TTS_TARGET_RMS", 0.1),
            cross_fade_duration=_env_float("TTS_CROSS_FADE_DURATION", 0.15),
            nfe_step=_env_int("TTS_NFE_STEP", 32),
            cfg_strength=_env_float("TTS_CFG_STRENGTH", 2.0),
            sway_sampling_coef=_env_float("TTS_SWAY_SAMPLING_COEF", -1.0),
            speed=_env_float("TTS_SPEED", 1.0),
            fix_duration=_env_optional_float("TTS_FIX_DURATION"),
            seed=_env_int("TTS_SEED", -1),
        ),
        realtime=RealtimeSettings(
            nfe_step=_env_int("AREEB_RT_NFE_STEP", 24),
            frame_samples=_env_int("AREEB_RT_FRAME_SAMPLES", 4096),
            min_flush_bytes=_env_int("AREEB_RT_MIN_FLUSH_BYTES", 24),
            max_flush_bytes=_env_int("AREEB_RT_MAX_FLUSH_BYTES", 240),
            prewarm_character=_env_str("AREEB_RT_PREWARM_CHARACTER", "msa_narrator"),
            prewarm_model_type=_env_str("AREEB_RT_PREWARM_MODEL_TYPE", "Unified"),
            prewarm_disabled=_env_bool("AREEB_RT_DISABLE_PREWARM", False),
        ),
        remote_tts=RemoteTTSSettings(
            url=_env_str(
                "AREEBB_TTS_URL",
                "",
                "areebb_tts_url",
                "AREEB_REMOTE_TTS_URL",
                "REMOTE_TTS_URL",
            ),
            method=_env_str("REMOTE_TTS_METHOD", "POST").upper(),
            json_payload=_env_bool("REMOTE_TTS_JSON", True),
            timeout=_env_int("REMOTE_TTS_TIMEOUT", 300),
        ),
    )


settings: Settings = _load()


__all__ = [
    "OllamaSettings",
    "RealtimeSettings",
    "RemoteTTSSettings",
    "Settings",
    "TTSSettings",
    "settings",
]
