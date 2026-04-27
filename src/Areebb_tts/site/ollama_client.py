"""Synchronous Ollama chat client for the legacy ``/api/chat`` endpoint.

For the realtime streaming path, see :mod:`Areebb_tts.realtime.llm_ollama`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

import requests

from Areebb_tts.core.exceptions import OllamaError
from Areebb_tts.core.settings import settings


LOGGER = logging.getLogger(__name__)


DEFAULT_TIMEOUT = 180

DEFAULT_TOPIC_GUARD = (
    "Keep the conversation focused on AI topics only (AI tools, models, coding, automation, prompts, "
    "and practical AI use-cases). If the user asks about unrelated topics, gently steer back to AI."
)


def build_messages(
    persona: str,
    user_message: str,
    history: Iterable[Mapping[str, str]] | None = None,
    *,
    topic_guard: str | None = DEFAULT_TOPIC_GUARD,
) -> list[dict[str, str]]:
    """Construct an Ollama-compatible message list (system + history + user)."""
    messages: list[dict[str, str]] = [{"role": "system", "content": persona}]
    if topic_guard:
        messages.append({"role": "system", "content": topic_guard})
    if history:
        for turn in history:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message.strip()})
    return messages


def chat_completion(
    persona: str,
    user_message: str,
    history: Iterable[Mapping[str, str]] | None = None,
    *,
    topic_guard: str | None = DEFAULT_TOPIC_GUARD,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Send a non-streaming chat completion to Ollama and return its content."""
    payload = {
        "model": settings.ollama.model,
        "messages": build_messages(persona, user_message, history, topic_guard=topic_guard),
        "stream": False,
    }

    url = f"{settings.ollama.base_url.rstrip('/')}/api/chat"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaError(
            f"Ollama is unavailable. Ensure it is running and model "
            f"'{settings.ollama.model}' is pulled: {exc}"
        ) from exc

    data = response.json()
    content = (data.get("message") or {}).get("content", "").strip()
    if not content:
        raise OllamaError("Ollama returned an empty response")
    return content


__all__ = ["DEFAULT_TOPIC_GUARD", "build_messages", "chat_completion"]
