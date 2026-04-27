"""Async streaming Ollama chat client.

Speaks Ollama's ``/api/chat`` HTTP NDJSON stream. Yields incremental message
content deltas so the rest of the pipeline can pipe text into the sentence
buffer as it arrives, instead of waiting for the full response.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterable, Mapping

import httpx

from Areebb_tts.core.exceptions import OllamaError
from Areebb_tts.core.settings import settings
from Areebb_tts.site.ollama_client import DEFAULT_TOPIC_GUARD, build_messages


LOGGER = logging.getLogger(__name__)


# Connect timeout small (model is local), but the read window must be large
# enough to cover slow first-token latency on cold loads.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=30.0)


async def stream_chat(
    persona: str,
    user_message: str,
    history: Iterable[Mapping[str, str]] | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    topic_guard: str | None = DEFAULT_TOPIC_GUARD,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
) -> AsyncIterator[str]:
    """Yield incremental ``message.content`` deltas from Ollama.

    Cancellation: the caller is expected to cancel the surrounding task. The
    underlying ``httpx`` stream is closed cooperatively when the async generator
    is GC'd or when an exception propagates out of the ``async with`` block.
    """
    resolved_model = model or settings.ollama.model
    resolved_base = (base_url or settings.ollama.base_url).rstrip("/")
    payload = {
        "model": resolved_model,
        "messages": build_messages(persona, user_message, history, topic_guard=topic_guard),
        "stream": True,
    }

    url = f"{resolved_base}/api/chat"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    snippet = body.decode("utf-8", errors="replace")[:500]
                    raise OllamaError(
                        f"Ollama returned HTTP {response.status_code}: {snippet}"
                    )
                async for raw_line in response.aiter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        # Skip stray heartbeats / malformed lines.
                        continue
                    delta = (chunk.get("message") or {}).get("content")
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        break
    except httpx.HTTPError as exc:
        raise OllamaError(
            f"Ollama is unavailable. Ensure it is running and model "
            f"'{resolved_model}' is pulled: {exc}"
        ) from exc


__all__ = ["DEFAULT_TIMEOUT", "stream_chat"]
