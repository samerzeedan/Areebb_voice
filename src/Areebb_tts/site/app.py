import os
import uuid
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import requests
import soundfile as sf
from cached_path import cached_path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from f5_tts.infer.utils_infer import load_model, load_vocoder, preprocess_ref_audio_text
from f5_tts.model import DiT
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from starlette.requests import Request

from Areebb_tts.infer.utils_infer import infer_process
from Areebb_tts.model.utils import dialect_id_map


SITE_DIR = Path(__file__).resolve().parent
ROOT_DIR = SITE_DIR.parents[2]
OUTPUT_DIR = ROOT_DIR / "generated_audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "Qwen2.5:7b-instruct-q4_K_M")

SPECIALIZED_DIALECTS = {"MSA", "SAU", "UAE", "ALG", "IRQ", "EGY", "MAR"}
MODEL_STEP_BY_DIALECT = {
    "MSA": 200000,
    "SAU": 200000,
    "UAE": 100000,
    "ALG": 100000,
    "IRQ": 100000,
    "EGY": 100000,
    "MAR": 100000,
}
V1_BASE_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)


CHARACTERS: dict[str, dict[str, str]] = {
    "msa_narrator": {
        "name": "MSA Narrator",
        "dialect": "MSA",
        "ref_audio": str(files("Areebb_tts").joinpath("assets/MSA.mp3")),
        "ref_text": "كان اللعيب حاضرًا في العديد من الأنشطة والفعاليات المرتبطة بكأس العالم.",
        "persona": "You are a calm Arabic narrator. Keep your answers clear and practical.",
    },
    "najdi_friend": {
        "name": "Najdi Friend",
        "dialect": "SAU",
        "ref_audio": str(files("Areebb_tts").joinpath("assets/Najdi.wav")),
        "ref_text": "تكفى طمني انا اليوم ماني بنايم ولا هو بداخل عيني النوم الين اتطمن عليه.",
        "persona": "You are a warm Saudi friend speaking naturally and briefly.",
    },
    "egyptian_assistant": {
        "name": "Egyptian Assistant",
        "dialect": "EGY",
        "ref_audio": str(files("Areebb_tts").joinpath("assets/EGY.mp3")),
        "ref_text": "ايه الكلام. بقولك ايه. استخدم صوتي في المحادثات.",
        "persona": "You are an Egyptian Arabic assistant that gives direct and helpful answers.",
    },
    "moroccan_coach": {
        "name": "Moroccan Coach",
        "dialect": "MAR",
        "ref_audio": str(files("Areebb_tts").joinpath("assets/MAR.mp3")),
        "ref_text": "إذا بغيتي شي صوت باللهجة المغربية هذا أحسن واحد غادي تلقاه.",
        "persona": "You are a Moroccan coach. Respond with confidence and short actionable advice.",
    },
}


class TTSRequest(BaseModel):
    character_id: str = Field(..., description="Selected character id")
    text: str = Field(..., min_length=1, description="Text to synthesize")
    model_type: str = Field(default="Unified", description="Unified or Specialized")


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    character_id: str
    message: str = Field(..., min_length=1)
    model_type: str = Field(default="Unified")
    history: list[ChatTurn] = Field(default_factory=list)


app = FastAPI(title="Areeb Site")
templates = Jinja2Templates(directory=str(SITE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(SITE_DIR / "static")), name="static")
app.mount("/audio", StaticFiles(directory=str(OUTPUT_DIR)), name="audio")


@lru_cache(maxsize=1)
def get_vocoder() -> Any:
    return load_vocoder()


@lru_cache(maxsize=1)
def get_model_cfg() -> Any:
    cfg_path = str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml"))
    cfg = OmegaConf.load(cfg_path)
    return cfg


@lru_cache(maxsize=16)
def get_model(model_type: str, dialect: str) -> Any:
    normalized_type = model_type.strip().title()
    if normalized_type not in {"Unified", "Specialized"}:
        raise HTTPException(status_code=400, detail="model_type must be Unified or Specialized")

    if normalized_type == "Specialized" and dialect not in SPECIALIZED_DIALECTS:
        raise HTTPException(status_code=400, detail=f"Specialized model not available for {dialect}")

    if normalized_type == "Unified":
        model_path = str(cached_path("hf://SWivid/Areebb_tts/Unified/model_200000.safetensors"))
        vocab_path = str(cached_path("hf://SWivid/Areebb_tts/Unified/vocab.txt"))
    else:
        step = MODEL_STEP_BY_DIALECT[dialect]
        model_path = str(cached_path(f"hf://SWivid/Areebb_tts/Specialized/{dialect}/model_{step}.safetensors"))
        vocab_path = str(cached_path(f"hf://SWivid/Areebb_tts/Specialized/{dialect}/vocab.txt"))

    model_cfg = get_model_cfg()
    model_cls = DiT if str(model_cfg.model.backbone) == "DiT" else DiT
    model_arc = model_cfg.model.arch if hasattr(model_cfg.model, "arch") else V1_BASE_CFG
    return load_model(model_cls, model_arc, model_path, vocab_file=vocab_path)


def synthesize_text(character_id: str, text: str, model_type: str) -> str:
    character = CHARACTERS.get(character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Unknown character_id: {character_id}")

    cleaned_text = text.strip()
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="text cannot be empty")

    ref_audio, ref_text = preprocess_ref_audio_text(character["ref_audio"], character["ref_text"])
    tts_model = get_model(model_type, character["dialect"])
    vocoder = get_vocoder()
    dialect_id = None if model_type.strip().title() == "Specialized" else dialect_id_map[character["dialect"]]

    waveform, sample_rate, _ = infer_process(
        ref_audio,
        ref_text,
        cleaned_text,
        tts_model,
        vocoder,
        dialect_id=dialect_id,
    )

    file_name = f"{uuid.uuid4().hex}.wav"
    output_path = OUTPUT_DIR / file_name
    sf.write(output_path, waveform, sample_rate)
    return f"/audio/{file_name}"


def call_ollama_chat(persona: str, user_message: str, history: list[ChatTurn]) -> str:
    messages: list[dict[str, str]] = [{"role": "system", "content": persona}]
    for turn in history:
        if turn.role in {"user", "assistant"} and turn.content.strip():
            messages.append({"role": turn.role, "content": turn.content.strip()})
    messages.append({"role": "user", "content": user_message.strip()})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }

    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=180)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama is unavailable. Ensure it is running and model is pulled: {exc}",
        ) from exc

    data = response.json()
    content = (data.get("message") or {}).get("content", "").strip()
    if not content:
        raise HTTPException(status_code=500, detail="Ollama returned an empty response")
    return content


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request, "model_name": OLLAMA_MODEL})


@app.get("/api/characters")
def list_characters() -> dict[str, Any]:
    items = []
    for character_id, cfg in CHARACTERS.items():
        items.append(
            {
                "id": character_id,
                "name": cfg["name"],
                "dialect": cfg["dialect"],
            }
        )
    return {"characters": items, "ollama_model": OLLAMA_MODEL}


@app.post("/api/tts")
def tts_generate(payload: TTSRequest) -> dict[str, str]:
    audio_url = synthesize_text(payload.character_id, payload.text, payload.model_type)
    return {"audio_url": audio_url}


@app.post("/api/chat")
def chat_generate(payload: ChatRequest) -> dict[str, Any]:
    character = CHARACTERS.get(payload.character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Unknown character_id: {payload.character_id}")

    assistant_text = call_ollama_chat(character["persona"], payload.message, payload.history)
    audio_url = synthesize_text(payload.character_id, assistant_text, payload.model_type)

    new_history = payload.history + [
        ChatTurn(role="user", content=payload.message),
        ChatTurn(role="assistant", content=assistant_text),
    ]
    return {
        "assistant_text": assistant_text,
        "audio_url": audio_url,
        "history": [turn.model_dump() for turn in new_history],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    icon_path = SITE_DIR / "static" / "favicon.ico"
    if icon_path.exists():
        return FileResponse(icon_path)
    raise HTTPException(status_code=404)


def run() -> None:
    import uvicorn

    host = os.getenv("AREEB_HOST", "0.0.0.0")
    port = int(os.getenv("AREEB_PORT", "7860"))
    uvicorn.run("Areebb_tts.site.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
