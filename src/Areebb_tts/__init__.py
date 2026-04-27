"""Areebb-TTS: Arabic dialect TTS + realtime conversational pipeline.

Top-level public API is intentionally tiny. Most code should import from one
of the dedicated subpackages instead of this module:

- :mod:`Areebb_tts.core` — settings, characters, exceptions, TTS engine.
- :mod:`Areebb_tts.realtime` — WebSocket streaming pipeline.
- :mod:`Areebb_tts.site` — FastAPI HTTP layer.
- :mod:`Areebb_tts.infer` / :mod:`Areebb_tts.model` — F5-TTS inference glue.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
