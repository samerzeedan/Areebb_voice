"""Per-WebSocket connection state, including the cancellable current task."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import uuid
from dataclasses import dataclass, field

from fastapi import WebSocket


LOGGER = logging.getLogger("Areebb_tts.realtime.session")


@dataclass
class ConnectionSession:
    """Mutable state attached to a single WebSocket connection.

    The cancellation primitives are *paired*: ``py_cancel`` is used by async
    code paths (sending frames, awaiting the LLM stream) while ``thread_cancel``
    is observable from inside the F5-TTS executor thread. Both are reset
    together by :meth:`begin_turn` so the next turn starts clean.
    """

    ws: WebSocket
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    character_id: str | None = None
    model_type: str = "Unified"
    history: list[dict[str, str]] = field(default_factory=list)
    current_task: asyncio.Task | None = None
    py_cancel: asyncio.Event = field(default_factory=asyncio.Event)
    thread_cancel: threading.Event = field(default_factory=threading.Event)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def configure(self, character_id: str, model_type: str) -> None:
        self.character_id = character_id
        self.model_type = model_type or "Unified"

    def begin_turn(self) -> None:
        """Reset cancellation events for a fresh turn. Call before scheduling."""
        self.py_cancel = asyncio.Event()
        self.thread_cancel = threading.Event()

    async def interrupt(self) -> None:
        """Cancel any in-flight turn cooperatively and wait for it to wind down."""
        self.py_cancel.set()
        self.thread_cancel.set()
        task = self.current_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self.current_task = None

    def append_turn(self, user_text: str, assistant_text: str) -> None:
        if user_text:
            self.history.append({"role": "user", "content": user_text})
        if assistant_text:
            self.history.append({"role": "assistant", "content": assistant_text})

    async def send_json(self, payload: dict) -> None:
        async with self.send_lock:
            await self.ws.send_json(payload)

    async def send_bytes(self, frame: bytes) -> None:
        async with self.send_lock:
            await self.ws.send_bytes(frame)
