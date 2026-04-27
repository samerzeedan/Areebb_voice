"""Orchestrates a single conversational turn end-to-end.

Flow per turn:

1. Stream tokens from Ollama (async generator).
2. Forward each delta to the WS as ``assistant_text_delta`` and feed it into
   the :class:`SentenceBuffer`.
3. Whenever the buffer yields a complete sentence, push it onto a bounded
   async queue.
4. A consumer task pulls sentences off the queue, runs F5-TTS in an executor,
   and streams PCM16 frames to the WS.
5. When the LLM stream ends, drain the buffer and the queue, then emit
   ``tts_end`` and update the session history.

Cancellation:

- ``session.py_cancel`` is checked at every queue/await boundary.
- ``session.thread_cancel`` is observed inside the executor.
- A single ``asyncio.CancelledError`` cleanly tears down all child tasks.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from Areebb_tts.core.characters import get_character
from Areebb_tts.core.exceptions import OllamaError, VoiceConfigError
from Areebb_tts.realtime.llm_ollama import stream_chat
from Areebb_tts.realtime.session import ConnectionSession
from Areebb_tts.realtime.text_buffer import SentenceBuffer
from Areebb_tts.realtime.tts_streamer import synthesize_sentence_frames


LOGGER = logging.getLogger(__name__)


SENTENCE_QUEUE_MAXSIZE = 8
_SAMPLE_RATE_HZ = 24000

# Sentinel placed on the sentence queue to tell the consumer the producer is done.
_END_OF_STREAM: Any = object()

# Errors that may legitimately happen while sending a tail message after the
# client has gone away. We swallow them; cancellation must always propagate.
_NETWORK_TEARDOWN_ERRORS = (RuntimeError, ConnectionError, OSError)


async def run_turn(session: ConnectionSession, user_text: str) -> None:
    """Run one conversational turn. Safe to schedule as ``asyncio.create_task``."""
    if not session.character_id:
        await session.send_json({"type": "error", "message": "session not configured"})
        return

    try:
        character = get_character(session.character_id)
    except VoiceConfigError as exc:
        await session.send_json({"type": "error", "message": str(exc)})
        return

    persona = character["persona"]
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=SENTENCE_QUEUE_MAXSIZE)
    full_text_parts: list[str] = []

    producer = asyncio.create_task(
        _llm_producer(session, persona, user_text, queue, full_text_parts),
        name=f"rt-llm-{session.id}",
    )
    consumer = asyncio.create_task(
        _tts_consumer(session, queue),
        name=f"rt-tts-{session.id}",
    )

    interrupted = await _await_producer(session, producer)
    interrupted |= await _await_consumer(queue, consumer)

    if interrupted or session.py_cancel.is_set():
        await _safe_send_json(session, {"type": "interrupted"})
        return

    assistant_text = "".join(full_text_parts).strip()
    session.append_turn(user_text, assistant_text)
    await _safe_send_json(
        session,
        {"type": "tts_end", "assistant_text": assistant_text},
    )


async def _await_producer(session: ConnectionSession, producer: asyncio.Task) -> bool:
    """Wait for the LLM producer; return ``True`` if interrupted."""
    try:
        await producer
        return False
    except asyncio.CancelledError:
        return True
    except OllamaError as exc:
        await _safe_send_json(session, {"type": "error", "message": str(exc)})
        return False
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("LLM producer failed")
        await _safe_send_json(session, {"type": "error", "message": f"LLM error: {exc}"})
        return False


async def _await_consumer(queue: asyncio.Queue, consumer: asyncio.Task) -> bool:
    """Signal end-of-stream and wait for the consumer to drain."""
    try:
        queue.put_nowait(_END_OF_STREAM)
    except asyncio.QueueFull:
        consumer.cancel()

    try:
        await consumer
        return False
    except asyncio.CancelledError:
        return True
    except Exception:  # pragma: no cover - defensive
        LOGGER.exception("TTS consumer failed")
        return False


async def _llm_producer(
    session: ConnectionSession,
    persona: str,
    user_text: str,
    queue: asyncio.Queue,
    full_text_parts: list[str],
) -> None:
    """Stream LLM deltas, forward them, and push sentences onto the queue."""
    buffer = SentenceBuffer()
    started_tts = False

    async for delta in stream_chat(persona=persona, user_message=user_text, history=session.history):
        if session.py_cancel.is_set():
            raise asyncio.CancelledError
        full_text_parts.append(delta)
        await _safe_send_json(session, {"type": "assistant_text_delta", "text": delta})
        for sentence in buffer.feed(delta):
            started_tts = await _emit_tts_start_once(session, started_tts)
            await _put_sentence(session, queue, sentence)

    for sentence in buffer.flush():
        started_tts = await _emit_tts_start_once(session, started_tts)
        await _put_sentence(session, queue, sentence)


async def _emit_tts_start_once(session: ConnectionSession, started_tts: bool) -> bool:
    if started_tts:
        return True
    await _safe_send_json(session, {"type": "tts_start", "sample_rate": _SAMPLE_RATE_HZ})
    return True


async def _put_sentence(
    session: ConnectionSession,
    queue: asyncio.Queue,
    sentence: str,
) -> None:
    """Backpressure-friendly enqueue that still observes cancellation."""
    while True:
        if session.py_cancel.is_set():
            raise asyncio.CancelledError
        try:
            queue.put_nowait(sentence)
            return
        except asyncio.QueueFull:
            try:
                await asyncio.wait_for(queue.put(sentence), timeout=0.1)
                return
            except asyncio.TimeoutError:
                continue


async def _tts_consumer(session: ConnectionSession, queue: asyncio.Queue) -> None:
    """Pull sentences and stream PCM16 frames over the WS."""
    while True:
        item = await queue.get()
        if item is _END_OF_STREAM:
            return
        if session.py_cancel.is_set():
            return
        sentence: str = item
        try:
            async for frame, _sr in synthesize_sentence_frames(
                character_id=session.character_id or "",
                sentence=sentence,
                model_type=session.model_type,
                cancel_event=session.thread_cancel,
            ):
                if session.py_cancel.is_set():
                    return
                await _safe_send_bytes(session, frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("TTS sentence failed")
            await _safe_send_json(
                session,
                {"type": "error", "message": f"TTS failed for sentence: {exc}"},
            )


async def _safe_send_json(session: ConnectionSession, payload: dict) -> None:
    """Send a JSON message, swallowing best-effort tail-end network errors."""
    with contextlib.suppress(*_NETWORK_TEARDOWN_ERRORS):
        await session.send_json(payload)


async def _safe_send_bytes(session: ConnectionSession, frame: bytes) -> None:
    with contextlib.suppress(*_NETWORK_TEARDOWN_ERRORS):
        await session.send_bytes(frame)


__all__ = ["run_turn"]
