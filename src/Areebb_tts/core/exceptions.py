"""Domain exceptions used across the codebase."""

from __future__ import annotations


class AreebbError(Exception):
    """Base class for all Areebb-tts domain errors."""


class VoiceConfigError(AreebbError, ValueError):
    """Raised when an invalid voice/character/model combination is requested."""


class OllamaError(AreebbError, RuntimeError):
    """Raised when the Ollama backend is unreachable or returns an error."""


class RemoteTTSError(AreebbError, RuntimeError):
    """Raised when the optional remote TTS proxy returns an error."""


__all__ = ["AreebbError", "OllamaError", "RemoteTTSError", "VoiceConfigError"]
