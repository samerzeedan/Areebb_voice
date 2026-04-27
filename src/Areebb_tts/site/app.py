"""FastAPI HTTP entry point: thin routing layer over :mod:`Areebb_tts.site`.

All business logic lives in dedicated modules (``synthesis``, ``ollama_client``,
``audio_storage``, ``remote_tts``); routes here only validate input, dispatch,
and translate domain exceptions into HTTP errors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from Areebb_tts.core import (
    CHARACTERS,
    DIALECT_LABELS,
    OllamaError,
    SPECIALIZED_DIALECTS,
    VoiceConfigError,
    configure_logging,
    get_character,
    settings,
)
from Areebb_tts.core.exceptions import RemoteTTSError
from Areebb_tts.realtime.ws_app import lifespan as realtime_lifespan
from Areebb_tts.realtime.ws_app import router as realtime_router
from Areebb_tts.site.audio_storage import iter_wav_bytes, resolve_safe_path
from Areebb_tts.site.ollama_client import chat_completion
from Areebb_tts.site.schemas import (
    ChatRequest,
    ChatResponse,
    ChatTurn,
    TTSRequest,
    TTSResponse,
)
from Areebb_tts.site.synthesis import synthesize_text


LOGGER = logging.getLogger(__name__)

SITE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = SITE_DIR / "templates"
STATIC_DIR = SITE_DIR / "static"


app = FastAPI(title="Areeb Site", lifespan=realtime_lifespan)
app.include_router(realtime_router)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _safe_synthesize(character_id: str, text: str, model_type: str) -> str:
    """Translate domain errors from :func:`synthesize_text` to HTTP responses."""
    try:
        return synthesize_text(character_id, text, model_type)
    except VoiceConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RemoteTTSError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "model_name": settings.ollama.model},
    )


@app.get("/realtime", response_class=HTMLResponse)
def realtime_page(request: Request) -> HTMLResponse:
    template_path = TEMPLATES_DIR / "realtime.html"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Realtime page not available")
    return templates.TemplateResponse(
        "realtime.html",
        {"request": request, "model_name": settings.ollama.model},
    )


@app.get("/audio/{file_name}")
def serve_generated_audio(file_name: str) -> StreamingResponse:
    """Serve a generated WAV with no caching headers."""
    path = resolve_safe_path(file_name)
    if path is None:
        raise HTTPException(status_code=404, detail="Audio not found")
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    }
    return StreamingResponse(
        iter_wav_bytes(path),
        media_type="audio/wav",
        headers=headers,
    )


@app.get("/api/characters")
def list_characters() -> dict[str, Any]:
    """Catalog of available characters + dialects in stable display order."""
    items = [
        {
            "id": character_id,
            "name": cfg["name"],
            "dialect": cfg["dialect"],
            "dialect_label": DIALECT_LABELS.get(cfg["dialect"], cfg["dialect"]),
            "supports_specialized": cfg["dialect"] in SPECIALIZED_DIALECTS,
        }
        for character_id, cfg in CHARACTERS.items()
    ]

    dialect_order = list(DIALECT_LABELS.keys())
    dialects = [
        {"code": code, "label": DIALECT_LABELS[code]}
        for code in dialect_order
        if any(c["dialect"] == code for c in items)
    ]
    items.sort(key=lambda c: (dialect_order.index(c["dialect"]), c["name"]))
    return {
        "characters": items,
        "dialects": dialects,
        "ollama_model": settings.ollama.model,
    }


@app.post("/api/tts", response_model=TTSResponse)
def tts_generate(payload: TTSRequest) -> TTSResponse:
    audio_url = _safe_synthesize(payload.character_id, payload.text, payload.model_type)
    return TTSResponse(audio_url=audio_url)


@app.post("/api/chat", response_model=ChatResponse)
def chat_generate(payload: ChatRequest) -> ChatResponse:
    try:
        character = get_character(payload.character_id)
    except VoiceConfigError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    history_dicts = [turn.model_dump() for turn in payload.history]

    try:
        assistant_text = chat_completion(character["persona"], payload.message, history_dicts)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    audio_url = _safe_synthesize(payload.character_id, assistant_text, payload.model_type)

    new_history = payload.history + [
        ChatTurn(role="user", content=payload.message),
        ChatTurn(role="assistant", content=assistant_text),
    ]
    return ChatResponse(
        assistant_text=assistant_text,
        audio_url=audio_url,
        history=[turn.model_dump() for turn in new_history],
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/favicon.ico")
def favicon() -> Response:
    icon_path = STATIC_DIR / "favicon.ico"
    if icon_path.exists():
        return FileResponse(icon_path)
    return Response(status_code=204)


def run() -> None:
    """Console entry point: ``python -m Areebb_tts.site.app``."""
    import uvicorn

    configure_logging()
    uvicorn.run(
        "Areebb_tts.site.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
