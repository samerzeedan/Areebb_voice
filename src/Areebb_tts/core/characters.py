"""Voice catalog: character definitions, dialect mappings, and lookup helpers.

This module is pure data + small helpers; it imports nothing from FastAPI or
the F5-TTS engine, so it is safe to use from any layer.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Final

from Areebb_tts.core.exceptions import VoiceConfigError


def _asset(path: str) -> str:
    return str(files("Areebb_tts").joinpath(path))


SPECIALIZED_DIALECTS: Final[frozenset[str]] = frozenset(
    {"MSA", "SAU", "UAE", "ALG", "IRQ", "EGY", "MAR"}
)

DIALECT_LABELS: Final[dict[str, str]] = {
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

# Per-dialect checkpoint step for the Specialized model.
MODEL_STEP_BY_DIALECT: Final[dict[str, int]] = {
    "MSA": 200000,
    "SAU": 200000,
    "UAE": 100000,
    "ALG": 100000,
    "IRQ": 100000,
    "EGY": 100000,
    "MAR": 100000,
}

VALID_MODEL_TYPES: Final[frozenset[str]] = frozenset({"Unified", "Specialized"})


CHARACTERS: Final[dict[str, dict[str, str]]] = {
    "msa_narrator": {
        "name": "MSA Narrator",
        "dialect": "MSA",
        "ref_audio": _asset("assets/MSA.mp3"),
        "ref_text": "كان اللعيب حاضرًا في العديد من الأنشطة والفعاليات المرتبطة بكأس العالم.",
        "persona": "You are a calm Arabic narrator. Keep your answers clear and practical.",
    },
    "najdi_friend": {
        "name": "Najdi Friend",
        "dialect": "SAU",
        "ref_audio": _asset("assets/Najdi.wav"),
        "ref_text": "تكفى طمني انا اليوم ماني بنايم ولا هو بداخل عيني النوم الين اتطمن عليه.",
        "persona": "You are a warm Saudi friend speaking naturally and briefly.",
    },
    "egyptian_assistant": {
        "name": "Egyptian Assistant",
        "dialect": "EGY",
        # WAV avoids MP3/ffmpeg edge cases in pydub preprocessing on some servers.
        "ref_audio": _asset("assets/EGY.wav"),
        "ref_text": "ايه الكلام. بقولك ايه. استخدم صوتي في المحادثات.",
        "persona": (
            "You are an Egyptian Arabic assistant. Always reply in Egyptian Arabic (عامية مصرية) "
            "using Arabic script, not English, so the voice engine can speak your answer."
        ),
    },
    "moroccan_coach": {
        "name": "Moroccan Coach",
        "dialect": "MAR",
        "ref_audio": _asset("assets/MAR.mp3"),
        "ref_text": "إذا بغيتي شي صوت باللهجة المغربية هذا أحسن واحد غادي تلقاه.",
        "persona": "You are a Moroccan coach. Respond with confidence and short actionable advice.",
    },
    "hijazi_voice": {
        "name": "Hijazi Voice",
        "dialect": "SAU",
        "ref_audio": _asset("assets/Hijazi.wav"),
        "ref_text": "ابغاك تحقق معاه بس بشكل ودي لانه سلطان يمر بظروف صعبة شوية.",
        "persona": "You are a Hijazi Arabic speaker. Keep the response empathetic and natural.",
    },
    "gulf_voice": {
        "name": "Gulf Voice",
        "dialect": "SAU",
        "ref_audio": _asset("assets/Gulf.wav"),
        "ref_text": "وين تو الناس متى تصحى ومتى تفطر وتغير يبيلك ساعة.",
        "persona": "You are a Gulf Arabic speaker with a clear and direct style.",
    },
    "uae_voice": {
        "name": "UAE Voice",
        "dialect": "UAE",
        "ref_audio": _asset("assets/UAE.wav"),
        "ref_text": "قمنا نشتريها بشكل متكرر أو لما نلقى ستايل يعجبنا.",
        "persona": "You are an Emirati speaker. Reply in concise and friendly Arabic.",
    },
    "algerian_voice": {
        "name": "Algerian Voice",
        "dialect": "ALG",
        "ref_audio": _asset("assets/ALG.wav"),
        "ref_text": "أنيا هكا باغية ناكل هكا أني ن نشوف فيها الحاجة هذيكا.",
        "persona": "You are an Algerian Arabic speaker. Reply naturally and confidently.",
    },
    "iraqi_voice": {
        "name": "Iraqi Voice",
        "dialect": "IRQ",
        "ref_audio": _asset("assets/IRQ.wav"),
        "ref_text": "يعني ااا ما نقدر ناخذ وقت أكثر، لأنه شروط كلش يحتاجلها وقت.",
        "persona": "You are an Iraqi Arabic speaker. Keep responses practical and clear.",
    },
    "omani_voice": {
        "name": "Omani Voice",
        "dialect": "OMN",
        "ref_audio": _asset("assets/MSA.mp3"),
        "ref_text": "هذا صوت مرجعي موحد لاستخدام اللهجة العمانية ضمن النموذج الموحد.",
        "persona": "You are an Omani Arabic assistant. Keep the tone warm and respectful.",
    },
    "tunisian_voice": {
        "name": "Tunisian Voice",
        "dialect": "TUN",
        "ref_audio": _asset("assets/MSA.mp3"),
        "ref_text": "هذا صوت مرجعي موحد لاستخدام اللهجة التونسية ضمن النموذج الموحد.",
        "persona": "You are a Tunisian Arabic assistant. Keep answers brief and helpful.",
    },
    "levantine_voice": {
        "name": "Levantine Voice",
        "dialect": "LEV",
        "ref_audio": _asset("assets/MSA.mp3"),
        "ref_text": "هذا صوت مرجعي موحد لاستخدام اللهجة الشامية ضمن النموذج الموحد.",
        "persona": "You are a Levantine Arabic assistant. Use a friendly conversational tone.",
    },
    "sudanese_voice": {
        "name": "Sudanese Voice",
        "dialect": "SDN",
        "ref_audio": _asset("assets/MSA.mp3"),
        "ref_text": "هذا صوت مرجعي موحد لاستخدام اللهجة السودانية ضمن النموذج الموحد.",
        "persona": "You are a Sudanese Arabic assistant. Respond clearly and respectfully.",
    },
    "libyan_voice": {
        "name": "Libyan Voice",
        "dialect": "LBY",
        "ref_audio": _asset("assets/MSA.mp3"),
        "ref_text": "هذا صوت مرجعي موحد لاستخدام اللهجة الليبية ضمن النموذج الموحد.",
        "persona": "You are a Libyan Arabic assistant. Keep responses concise and practical.",
    },
    "unknown_voice": {
        "name": "Unknown / Auto",
        "dialect": "UNK",
        "ref_audio": _asset("assets/MSA.mp3"),
        "ref_text": "صوت مرجعي افتراضي لحالة اللهجة غير المعروفة.",
        "persona": "You are a generic Arabic assistant that adapts to user style.",
    },
}


def get_character(character_id: str) -> dict[str, str]:
    """Return the character config for ``character_id`` or raise."""
    character = CHARACTERS.get(character_id)
    if not character:
        raise VoiceConfigError(f"Unknown character_id: {character_id}")
    return character


def normalize_model_type(model_type: str) -> str:
    """Validate + canonicalise the model type to ``Unified`` or ``Specialized``."""
    normalized = (model_type or "").strip().title()
    if normalized not in VALID_MODEL_TYPES:
        raise VoiceConfigError("model_type must be Unified or Specialized")
    return normalized


__all__ = [
    "CHARACTERS",
    "DIALECT_LABELS",
    "MODEL_STEP_BY_DIALECT",
    "SPECIALIZED_DIALECTS",
    "VALID_MODEL_TYPES",
    "get_character",
    "normalize_model_type",
]
