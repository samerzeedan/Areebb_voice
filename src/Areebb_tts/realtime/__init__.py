"""Realtime streaming voice assistant: WebSocket pipeline for low-latency dialogue.

Public entry points:

- :data:`Areebb_tts.realtime.ws_app.router` — FastAPI ``APIRouter`` exposing
  ``GET /ws/voice``.
- :data:`Areebb_tts.realtime.ws_app.lifespan` — model pre-warm hook.

The pipeline streams Ollama deltas, splits them into sentences, runs F5-TTS
per sentence in a worker thread, and pushes PCM16 frames over the WebSocket.
"""
