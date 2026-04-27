"""Optional remote TTS proxy.

When :attr:`Settings.remote_tts.enabled` is true, the legacy HTTP path forwards
synthesis requests to an external service (e.g. a Vast.ai box running the
F5-TTS Gradio API) instead of running the model in-process.
"""

from __future__ import annotations

import base64
import logging
from typing import Any
from urllib.parse import urljoin

import requests

from Areebb_tts.core.exceptions import RemoteTTSError
from Areebb_tts.core.settings import settings
from Areebb_tts.site.audio_storage import save_bytes_wav


LOGGER = logging.getLogger(__name__)


def call_remote_tts(character_id: str, character: dict[str, str], text: str, model_type: str) -> str:
    """Forward a synthesis request to a configured remote TTS endpoint.

    Returns the public audio URL of the saved WAV. Raises
    :class:`RemoteTTSError` on any transport / protocol failure.
    """
    cfg = settings.remote_tts
    if not cfg.enabled:  # pragma: no cover - defensive
        raise RemoteTTSError("Remote TTS is not configured")

    payload = _build_payload(character_id, character, text, model_type)
    headers = _auth_headers()

    try:
        response = _do_request(payload, headers)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RemoteTTSError(f"Remote TTS request failed: {exc}") from exc

    return _persist_response(response)


def _do_request(payload: dict[str, Any], headers: dict[str, str]) -> requests.Response:
    cfg = settings.remote_tts
    if cfg.method == "GET":
        return requests.get(cfg.url, params=payload, headers=headers, timeout=cfg.timeout)
    if cfg.json_payload:
        return requests.post(cfg.url, json=payload, headers=headers, timeout=cfg.timeout)
    return requests.post(cfg.url, data=payload, headers=headers, timeout=cfg.timeout)


def _build_payload(character_id: str, character: dict[str, str], text: str, model_type: str) -> dict[str, Any]:
    tts = settings.tts
    return {
        "text": text,
        "dialect": character["dialect"],
        "model_type": model_type,
        "character_id": character_id,
        "voice": character_id,
        "target_rms": tts.target_rms,
        "cross_fade_duration": tts.cross_fade_duration,
        "nfe_step": tts.nfe_step,
        "cfg_strength": tts.cfg_strength,
        "sway_sampling_coef": tts.sway_sampling_coef,
        "speed": tts.speed,
        "fix_duration": tts.fix_duration,
        "seed": tts.seed,
    }


def _auth_headers() -> dict[str, str]:
    if settings.hf_token:
        return {"Authorization": f"Bearer {settings.hf_token}"}
    return {}


def _persist_response(response: requests.Response) -> str:
    content_type = (response.headers.get("content-type") or "").lower()
    if "application/json" not in content_type:
        return save_bytes_wav(response.content)

    try:
        data = response.json()
    except ValueError as exc:
        raise RemoteTTSError("Remote TTS returned invalid JSON") from exc

    audio_b64 = data.get("audio_base64") or data.get("audio")
    if audio_b64:
        return save_bytes_wav(base64.b64decode(audio_b64))

    audio_url = data.get("audio_url") or data.get("url")
    if audio_url:
        resolved = urljoin(settings.remote_tts.url, str(audio_url))
        try:
            audio_resp = requests.get(resolved, timeout=settings.remote_tts.timeout)
            audio_resp.raise_for_status()
        except requests.RequestException as exc:
            raise RemoteTTSError(f"Remote TTS audio_url fetch failed: {exc}") from exc
        return save_bytes_wav(audio_resp.content)

    raise RemoteTTSError(
        "Remote TTS JSON must include audio_base64, audio, audio_url, or url"
    )


__all__ = ["call_remote_tts"]
