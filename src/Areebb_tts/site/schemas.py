"""Pydantic request / response schemas for the FastAPI HTTP layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    """Body for ``POST /api/tts``."""

    character_id: str = Field(..., description="Selected character id")
    text: str = Field(..., min_length=1, description="Text to synthesize")
    model_type: str = Field(default="Unified", description="Unified or Specialized")


class ChatTurn(BaseModel):
    """A single conversation turn for the ``/api/chat`` history."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Body for ``POST /api/chat``."""

    character_id: str
    message: str = Field(..., min_length=1)
    model_type: str = Field(default="Unified")
    history: list[ChatTurn] = Field(default_factory=list)


class TTSResponse(BaseModel):
    """Response for ``POST /api/tts``."""

    audio_url: str


class ChatResponse(BaseModel):
    """Response for ``POST /api/chat``."""

    assistant_text: str
    audio_url: str
    history: list[dict[str, Any]]


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ChatTurn",
    "TTSRequest",
    "TTSResponse",
]
