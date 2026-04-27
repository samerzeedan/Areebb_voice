"""FastAPI HTTP layer.

Public modules:

- :mod:`Areebb_tts.site.app` — FastAPI app + routes (entry point).
- :mod:`Areebb_tts.site.schemas` — Pydantic request / response models.
- :mod:`Areebb_tts.site.synthesis` — local-or-remote TTS dispatch.
- :mod:`Areebb_tts.site.audio_storage` — generated WAV file IO.
- :mod:`Areebb_tts.site.ollama_client` — synchronous Ollama chat client.
- :mod:`Areebb_tts.site.remote_tts` — optional remote TTS proxy.
"""
