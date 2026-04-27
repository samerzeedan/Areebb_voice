"""FastAPI integration: ``/ws/voice`` WebSocket + model pre-warm lifespan."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect

from Areebb_tts.core.characters import (
    CHARACTERS,
    get_character,
    normalize_model_type,
)
from Areebb_tts.core.exceptions import VoiceConfigError
from Areebb_tts.core.settings import settings
from Areebb_tts.core.tts_engine import get_model, get_vocoder
from Areebb_tts.realtime.pipeline import run_turn
from Areebb_tts.realtime.session import ConnectionSession


LOGGER = logging.getLogger(__name__)

router = APIRouter()

_SAMPLE_RATE_HZ = 24000


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Pre-load vocoder + default model so the first WS turn isn't cold."""
    if not settings.realtime.prewarm_disabled:
        try:
            await asyncio.to_thread(_prewarm)
            LOGGER.info(
                "Realtime pre-warm complete (character=%s, model=%s)",
                settings.realtime.prewarm_character,
                settings.realtime.prewarm_model_type,
            )
        except Exception as exc:  # pragma: no cover - best-effort
            LOGGER.warning("Realtime pre-warm skipped: %s", exc)
    yield


def _prewarm() -> None:
    get_vocoder()
    character_id = settings.realtime.prewarm_character
    if character_id in CHARACTERS:
        character = CHARACTERS[character_id]
        get_model(settings.realtime.prewarm_model_type, character["dialect"])


@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session = ConnectionSession(ws=websocket)
    LOGGER.info("ws connected id=%s", session.id)
    try:
        await session.send_json(
            {"type": "ready", "sample_rate": _SAMPLE_RATE_HZ, "session_id": session.id}
        )
        await _serve_session(session)
    except WebSocketDisconnect:
        LOGGER.info("ws disconnect id=%s", session.id)
    except Exception:  # pragma: no cover - defensive
        LOGGER.exception("ws fatal error id=%s", session.id)
        with contextlib.suppress(Exception):
            await session.send_json({"type": "error", "message": "internal error"})
    finally:
        await session.interrupt()
        with contextlib.suppress(Exception):
            await websocket.close()


async def _serve_session(session: ConnectionSession) -> None:
    while True:
        message = await session.ws.receive()
        msg_type = message.get("type")
        if msg_type == "websocket.disconnect":
            return

        text = message.get("text")
        if text is None:
            # Binary frames are reserved for future server-side STT input;
            # silently ignore them so a misbehaving client doesn't tear down
            # the session.
            continue

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            await session.send_json({"type": "error", "message": "invalid JSON frame"})
            continue

        await _handle_event(session, payload)


async def _handle_event(session: ConnectionSession, payload: dict) -> None:
    event_type = payload.get("type")
    if event_type == "hello":
        await _handle_hello(session, payload)
        return
    if event_type == "stop":
        await session.interrupt()
        with contextlib.suppress(Exception):
            await session.send_json({"type": "interrupted"})
        return
    if event_type == "user_text":
        await _handle_user_text(session, payload)
        return
    await session.send_json({"type": "error", "message": f"unknown event: {event_type}"})


async def _handle_hello(session: ConnectionSession, payload: dict) -> None:
    character_id = (payload.get("character_id") or "").strip()
    model_type = (payload.get("model_type") or "Unified").strip()
    try:
        get_character(character_id)
        normalize_model_type(model_type)
    except VoiceConfigError as exc:
        await session.send_json({"type": "error", "message": str(exc)})
        return

    session.configure(character_id, model_type)
    session.history = _sanitize_history(payload.get("history"))

    await session.send_json(
        {
            "type": "configured",
            "character_id": character_id,
            "model_type": session.model_type,
        }
    )


def _sanitize_history(raw: object) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    cleaned: list[dict[str, str]] = []
    for turn in raw:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()})
    return cleaned


async def _handle_user_text(session: ConnectionSession, payload: dict) -> None:
    user_text = (payload.get("text") or "").strip()
    if not user_text:
        await session.send_json({"type": "error", "message": "empty user text"})
        return
    if not session.character_id:
        await session.send_json(
            {"type": "error", "message": "session not configured; send hello first"}
        )
        return

    # Any new turn implicitly interrupts the previous one (barge-in).
    await session.interrupt()
    session.begin_turn()

    await session.send_json({"type": "user_text_ack", "text": user_text})
    session.current_task = asyncio.create_task(
        run_turn(session, user_text),
        name=f"rt-turn-{session.id}",
    )


__all__ = ["lifespan", "router"]
