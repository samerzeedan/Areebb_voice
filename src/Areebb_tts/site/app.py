import base64
import os
import random
import uuid
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from dotenv import load_dotenv
import requests
import soundfile as sf
import numpy as np
import torch
from cached_path import cached_path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
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
load_dotenv(ROOT_DIR / ".env")
OUTPUT_DIR = ROOT_DIR / "generated_audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "Qwen2.5:7b-instruct-q4_K_M")
HF_VOICE_REPO = os.getenv("AREEB_HF_VOICE_REPO", os.getenv("HF_VOICE_REPO", "SWivid/Habibi-TTS"))

REMOTE_TTS_URL = (
    os.getenv("AREEBB_TTS_URL")
    or os.getenv("areebb_tts_url")
    or os.getenv("AREEB_REMOTE_TTS_URL")
    or os.getenv("REMOTE_TTS_URL")
    or ""
).strip()
REMOTE_TTS_METHOD = os.getenv("REMOTE_TTS_METHOD", "POST").strip().upper()
REMOTE_TTS_JSON = os.getenv("REMOTE_TTS_JSON", "1").strip() not in {"0", "false", "False", "no", "NO"}
REMOTE_TTS_TIMEOUT = int(os.getenv("REMOTE_TTS_TIMEOUT", "300"))

# Match Gradio/infer defaults by default (overridable via .env).
TTS_TARGET_RMS = float(os.getenv("TTS_TARGET_RMS", "0.1"))
TTS_CROSS_FADE_DURATION = float(os.getenv("TTS_CROSS_FADE_DURATION", "0.15"))
TTS_NFE_STEP = int(os.getenv("TTS_NFE_STEP", "32"))
TTS_CFG_STRENGTH = float(os.getenv("TTS_CFG_STRENGTH", "2.0"))
TTS_SWAY_SAMPLING_COEF = float(os.getenv("TTS_SWAY_SAMPLING_COEF", "-1.0"))
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
_tts_fix_duration_env = os.getenv("TTS_FIX_DURATION", "").strip()
TTS_FIX_DURATION = float(_tts_fix_duration_env) if _tts_fix_duration_env else None
# -1 means random each request (like free sampling); any other value makes output deterministic.
TTS_SEED = int(os.getenv("TTS_SEED", "-1"))

SPECIALIZED_DIALECTS = {"MSA", "SAU", "UAE", "ALG", "IRQ", "EGY", "MAR"}
DIALECT_LABELS = {
    "MSA": "MSA (Modern Standard Arabic)",
    "SAU": "SAU (Saudi)",
    "UAE": "UAE (Emirati)",
    "ALG": "ALG (Algerian)",
    "IRQ": "IRQ (Iraqi)",
    "EGY": "EGY (Egyptian)",
    "MAR": "MAR (Moroccan)",
    "OMN": "OMN (Omani)",
    "TUN": "TUN (Tunisian)",
    "LEV": "LEV (Levantine)",
    "SDN": "SDN (Sudanese)",
    "LBY": "LBY (Libyan)",
    "UNK": "UNK (Unknown/Auto)",
}
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
    "hijazi_voice": {
        "name": "Hijazi Voice",
        "dialect": "SAU",
        "ref_audio": str(files("Areebb_tts").joinpath("assets/Hijazi.wav")),
        "ref_text": "ابغاك تحقق معاه بس بشكل ودي لانه سلطان يمر بظروف صعبة شوية.",
        "persona": "You are a Hijazi Arabic speaker. Keep the response empathetic and natural.",
    },
    "gulf_voice": {
        "name": "Gulf Voice",
        "dialect": "SAU",
        "ref_audio": str(files("Areebb_tts").joinpath("assets/Gulf.wav")),
        "ref_text": "وين تو الناس متى تصحى ومتى تفطر وتغير يبيلك ساعة.",
        "persona": "You are a Gulf Arabic speaker with a clear and direct style.",
    },
    "uae_voice": {
        "name": "UAE Voice",
        "dialect": "UAE",
        "ref_audio": str(files("Areebb_tts").joinpath("assets/UAE.wav")),
        "ref_text": "قمنا نشتريها بشكل متكرر أو لما نلقى ستايل يعجبنا.",
        "persona": "You are an Emirati speaker. Reply in concise and friendly Arabic.",
    },
    "algerian_voice": {
        "name": "Algerian Voice",
        "dialect": "ALG",
        "ref_audio": str(files("Areebb_tts").joinpath("assets/ALG.wav")),
        "ref_text": "أنيا هكا باغية ناكل هكا أني ن نشوف فيها الحاجة هذيكا.",
        "persona": "You are an Algerian Arabic speaker. Reply naturally and confidently.",
    },
    "iraqi_voice": {
        "name": "Iraqi Voice",
        "dialect": "IRQ",
        "ref_audio": str(files("Areebb_tts").joinpath("assets/IRQ.wav")),
        "ref_text": "يعني ااا ما نقدر ناخذ وقت أكثر، لأنه شروط كلش يحتاجلها وقت.",
        "persona": "You are an Iraqi Arabic speaker. Keep responses practical and clear.",
    },
    "omani_voice": {
        "name": "Omani Voice",
        "dialect": "OMN",
        "ref_audio": str(files("Areebb_tts").joinpath("assets/MSA.mp3")),
        "ref_text": "هذا صوت مرجعي موحد لاستخدام اللهجة العمانية ضمن النموذج الموحد.",
        "persona": "You are an Omani Arabic assistant. Keep the tone warm and respectful.",
    },
    "tunisian_voice": {
        "name": "Tunisian Voice",
        "dialect": "TUN",
        "ref_audio": str(files("Areebb_tts").joinpath("assets/MSA.mp3")),
        "ref_text": "هذا صوت مرجعي موحد لاستخدام اللهجة التونسية ضمن النموذج الموحد.",
        "persona": "You are a Tunisian Arabic assistant. Keep answers brief and helpful.",
    },
    "levantine_voice": {
        "name": "Levantine Voice",
        "dialect": "LEV",
        "ref_audio": str(files("Areebb_tts").joinpath("assets/MSA.mp3")),
        "ref_text": "هذا صوت مرجعي موحد لاستخدام اللهجة الشامية ضمن النموذج الموحد.",
        "persona": "You are a Levantine Arabic assistant. Use a friendly conversational tone.",
    },
    "sudanese_voice": {
        "name": "Sudanese Voice",
        "dialect": "SDN",
        "ref_audio": str(files("Areebb_tts").joinpath("assets/MSA.mp3")),
        "ref_text": "هذا صوت مرجعي موحد لاستخدام اللهجة السودانية ضمن النموذج الموحد.",
        "persona": "You are a Sudanese Arabic assistant. Respond clearly and respectfully.",
    },
    "libyan_voice": {
        "name": "Libyan Voice",
        "dialect": "LBY",
        "ref_audio": str(files("Areebb_tts").joinpath("assets/MSA.mp3")),
        "ref_text": "هذا صوت مرجعي موحد لاستخدام اللهجة الليبية ضمن النموذج الموحد.",
        "persona": "You are a Libyan Arabic assistant. Keep responses concise and practical.",
    },
    "unknown_voice": {
        "name": "Unknown / Auto",
        "dialect": "UNK",
        "ref_audio": str(files("Areebb_tts").joinpath("assets/MSA.mp3")),
        "ref_text": "صوت مرجعي افتراضي لحالة اللهجة غير المعروفة.",
        "persona": "You are a generic Arabic assistant that adapts to user style.",
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
        model_path = str(cached_path(f"hf://{HF_VOICE_REPO}/Unified/model_200000.safetensors"))
        vocab_path = str(cached_path(f"hf://{HF_VOICE_REPO}/Unified/vocab.txt"))
    else:
        step = MODEL_STEP_BY_DIALECT[dialect]
        model_path = str(cached_path(f"hf://{HF_VOICE_REPO}/Specialized/{dialect}/model_{step}.safetensors"))
        vocab_path = str(cached_path(f"hf://{HF_VOICE_REPO}/Specialized/{dialect}/vocab.txt"))

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

    if REMOTE_TTS_URL:
        return synthesize_via_remote_tts(character_id, character, cleaned_text, model_type)

    if TTS_SEED >= 0:
        random.seed(TTS_SEED)
        np.random.seed(TTS_SEED)
        torch.manual_seed(TTS_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(TTS_SEED)

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
        target_rms=TTS_TARGET_RMS,
        cross_fade_duration=TTS_CROSS_FADE_DURATION,
        nfe_step=TTS_NFE_STEP,
        cfg_strength=TTS_CFG_STRENGTH,
        sway_sampling_coef=TTS_SWAY_SAMPLING_COEF,
        speed=TTS_SPEED,
        fix_duration=TTS_FIX_DURATION,
        dialect_id=dialect_id,
    )

    file_name = f"{uuid.uuid4().hex}.wav"
    output_path = OUTPUT_DIR / file_name
    sf.write(output_path, waveform, sample_rate)
    return f"/audio/{file_name}"


def synthesize_via_remote_tts(character_id: str, character: dict[str, str], text: str, model_type: str) -> str:
    payload = {
        "text": text,
        "dialect": character["dialect"],
        "model_type": model_type,
        "character_id": character_id,
        "voice": character_id,
        "target_rms": TTS_TARGET_RMS,
        "cross_fade_duration": TTS_CROSS_FADE_DURATION,
        "nfe_step": TTS_NFE_STEP,
        "cfg_strength": TTS_CFG_STRENGTH,
        "sway_sampling_coef": TTS_SWAY_SAMPLING_COEF,
        "speed": TTS_SPEED,
        "fix_duration": TTS_FIX_DURATION,
        "seed": TTS_SEED,
    }
    headers = {}
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    try:
        if REMOTE_TTS_METHOD == "GET":
            response = requests.get(REMOTE_TTS_URL, params=payload, headers=headers, timeout=REMOTE_TTS_TIMEOUT)
        elif REMOTE_TTS_JSON:
            response = requests.post(REMOTE_TTS_URL, json=payload, headers=headers, timeout=REMOTE_TTS_TIMEOUT)
        else:
            response = requests.post(REMOTE_TTS_URL, data=payload, headers=headers, timeout=REMOTE_TTS_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Remote TTS request failed: {exc}") from exc

    content_type = (response.headers.get("content-type") or "").lower()
    file_name = f"{uuid.uuid4().hex}.wav"
    output_path = OUTPUT_DIR / file_name

    if "application/json" in content_type:
        try:
            data = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="Remote TTS returned invalid JSON") from exc
        audio_b64 = data.get("audio_base64") or data.get("audio")
        audio_url = data.get("audio_url") or data.get("url")
        if audio_b64:
            output_path.write_bytes(base64.b64decode(audio_b64))
        elif audio_url:
            resolved_audio_url = urljoin(REMOTE_TTS_URL, str(audio_url))
            try:
                audio_resp = requests.get(resolved_audio_url, timeout=REMOTE_TTS_TIMEOUT)
                audio_resp.raise_for_status()
            except requests.RequestException as exc:
                raise HTTPException(status_code=502, detail=f"Remote TTS audio_url fetch failed: {exc}") from exc
            output_path.write_bytes(audio_resp.content)
        else:
            raise HTTPException(
                status_code=502,
                detail="Remote TTS JSON must include audio_base64, audio, audio_url, or url",
            )
    else:
        output_path.write_bytes(response.content)

    return f"/audio/{file_name}"


def call_ollama_chat(persona: str, user_message: str, history: list[ChatTurn]) -> str:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": persona},
        {
            "role": "system",
            "content": (
                "Keep the conversation focused on AI topics only (AI tools, models, coding, automation, prompts, "
                "and practical AI use-cases). If the user asks about unrelated topics, gently steer back to AI."
            ),
        },
    ]
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
                "dialect_label": DIALECT_LABELS.get(cfg["dialect"], cfg["dialect"]),
                "supports_specialized": cfg["dialect"] in SPECIALIZED_DIALECTS,
            }
        )
    dialect_order = list(DIALECT_LABELS.keys())
    dialects = []
    for code in dialect_order:
        count = sum(1 for c in items if c["dialect"] == code)
        if count > 0:
            dialects.append({"code": code, "label": DIALECT_LABELS.get(code, code)})

    items.sort(key=lambda c: (dialect_order.index(c["dialect"]), c["name"]))
    return {"characters": items, "dialects": dialects, "ollama_model": OLLAMA_MODEL}


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
def favicon() -> Response:
    icon_path = SITE_DIR / "static" / "favicon.ico"
    if icon_path.exists():
        return FileResponse(icon_path)
    return Response(status_code=204)


def run() -> None:
    import uvicorn

    host = os.getenv("AREEB_HOST", "0.0.0.0")
    port = int(os.getenv("AREEB_PORT", "9000"))
    uvicorn.run("Areebb_tts.site.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
