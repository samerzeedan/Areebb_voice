const state = {
  characters: [],
  dialects: [],
  currentCharacters: [],
  activeCharacterId: null,
  history: [],
  speechRecognizer: null,
  liveTranscript: "",
  speechSupported: false,
  isVoiceListening: false,
  isProcessing: false,
  isConversationActive: false,
  ttsPlayer: new Audio(),
};

const el = {
  dialectSelect: document.getElementById("dialect-select"),
  modelSelect: document.getElementById("model-select"),
  modelHint: document.getElementById("model-hint"),
  ttsInput: document.getElementById("tts-input"),
  ttsGenerate: document.getElementById("tts-generate"),
  ttsAudio: document.getElementById("tts-audio"),
  ttsStatus: document.getElementById("tts-status"),
  chatToggleBtn: document.getElementById("chat-toggle-btn"),
  chatTranscript: document.getElementById("chat-transcript"),
  chatBox: document.getElementById("chat-box"),
  chatStatus: document.getElementById("chat-status"),
  tabTts: document.getElementById("tab-tts"),
  tabChat: document.getElementById("tab-chat"),
  panelTts: document.getElementById("panel-tts"),
  panelChat: document.getElementById("panel-chat"),
};

function activeTab(isTts) {
  el.tabTts.classList.toggle("active", isTts);
  el.tabChat.classList.toggle("active", !isTts);
  el.panelTts.classList.toggle("active", isTts);
  el.panelChat.classList.toggle("active", !isTts);
}

function renderChatBubble(role, content) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = content;
  el.chatBox.appendChild(div);
  el.chatBox.scrollTop = el.chatBox.scrollHeight;
}

function dialectToLocale(dialectCode) {
  const map = {
    MSA: "ar-SA",
    SAU: "ar-SA",
    UAE: "ar-AE",
    ALG: "ar-DZ",
    IRQ: "ar-IQ",
    EGY: "ar-EG",
    MAR: "ar-MA",
    OMN: "ar-OM",
    TUN: "ar-TN",
    LEV: "ar-LB",
    SDN: "ar-SD",
    LBY: "ar-LY",
    UNK: "ar-SA",
  };
  return map[dialectCode] || "ar-SA";
}

function getActiveCharacter() {
  return state.characters.find((c) => c.id === state.activeCharacterId) || null;
}

function refreshModelOptions() {
  const supportsSpecialized = state.currentCharacters.some((c) => c.supports_specialized);
  const previous = el.modelSelect.value || "Unified";

  el.modelSelect.innerHTML = "";
  const unifiedOption = document.createElement("option");
  unifiedOption.value = "Unified";
  unifiedOption.textContent = "Unified";
  el.modelSelect.appendChild(unifiedOption);

  if (supportsSpecialized) {
    const specializedOption = document.createElement("option");
    specializedOption.value = "Specialized";
    specializedOption.textContent = "Specialized";
    el.modelSelect.appendChild(specializedOption);
  }

  if (supportsSpecialized && previous === "Specialized") {
    el.modelSelect.value = "Specialized";
    el.modelHint.textContent = "This dialect supports Unified and Specialized.";
  } else {
    el.modelSelect.value = "Unified";
    el.modelHint.textContent = supportsSpecialized
      ? "This dialect supports Unified and Specialized."
      : "This dialect supports Unified only.";
  }
}

function renderCharacterOptions(dialectCode) {
  state.currentCharacters = state.characters.filter((c) => c.dialect === dialectCode);
  state.activeCharacterId = state.currentCharacters[0]?.id || null;

  refreshModelOptions();
}

async function parseApiResponse(res) {
  const body = await res.text();
  let data = {};
  try {
    data = body ? JSON.parse(body) : {};
  } catch {
    data = { detail: body || "Unexpected server response." };
  }
  return data;
}

async function loadCharacters() {
  const res = await fetch("/api/characters");
  if (!res.ok) throw new Error("Cannot fetch characters");
  const data = await res.json();
  state.characters = data.characters;
  state.dialects = data.dialects || [];

  el.dialectSelect.innerHTML = "";
  state.dialects.forEach((d) => {
    const option = document.createElement("option");
    option.value = d.code;
    option.textContent = d.label;
    el.dialectSelect.appendChild(option);
  });

  const firstDialect = state.dialects[0]?.code;
  if (firstDialect) {
    renderCharacterOptions(firstDialect);
  }
}

async function onGenerateTts() {
  const text = el.ttsInput.value.trim();
  if (!text) {
    el.ttsStatus.textContent = "Please enter text first.";
    return;
  }

  el.ttsGenerate.disabled = true;
  el.ttsStatus.textContent = "Generating audio...";
  try {
    const res = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        character_id: state.activeCharacterId,
        text: text,
        model_type: el.modelSelect.value,
      }),
    });

    const data = await parseApiResponse(res);
    if (!res.ok) throw new Error(data.detail || "Failed to generate");
    el.ttsAudio.src = data.audio_url + `?t=${Date.now()}`;
    el.ttsStatus.textContent = "Audio generated.";
  } catch (err) {
    el.ttsStatus.textContent = err.message;
  } finally {
    el.ttsGenerate.disabled = false;
  }
}

async function sendChatMessage(message) {
  if (!message) {
    el.chatStatus.textContent = "No speech detected. Try recording again.";
    return;
  }

  renderChatBubble("user", message);
  state.isProcessing = true;
  state.isVoiceListening = false;
  el.chatStatus.textContent = "Thinking and generating voice...";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        character_id: state.activeCharacterId,
        message: message,
        model_type: el.modelSelect.value,
        history: state.history,
      }),
    });

    const data = await parseApiResponse(res);
    if (!res.ok) throw new Error(data.detail || "Chat failed");

    state.history = data.history;
    renderChatBubble("assistant", data.assistant_text);
    state.ttsPlayer.src = data.audio_url + `?t=${Date.now()}`;
    await state.ttsPlayer.play().catch(() => {});
    el.chatStatus.textContent = "Reply ready.";
  } catch (err) {
    el.chatStatus.textContent = err.message;
  } finally {
    state.isProcessing = false;
    if (state.speechSupported && state.isConversationActive && el.panelChat.classList.contains("active")) {
      startListening();
    }
  }
}

function resetChatOnCharacterChange() {
  state.history = [];
  el.chatBox.innerHTML = "";
  state.ttsPlayer.pause();
  state.ttsPlayer.removeAttribute("src");
  state.ttsPlayer.load();
  el.chatStatus.textContent = "Chat reset for selected character.";
  el.chatTranscript.textContent = "Your speech text will appear here...";
  state.liveTranscript = "";
}

function onDialectChange() {
  const dialectCode = el.dialectSelect.value;
  renderCharacterOptions(dialectCode);
  resetChatOnCharacterChange();
  if (state.speechSupported && state.isConversationActive && el.panelChat.classList.contains("active")) {
    startListening();
  }
}

function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    el.chatStatus.textContent = "Browser does not support speech recognition. Use Chrome or Edge.";
    el.chatTranscript.textContent = "Speech recognition is not supported in this browser.";
    return;
  }
  state.speechSupported = true;

  const recognizer = new SpeechRecognition();
  recognizer.continuous = true;
  recognizer.interimResults = true;
  recognizer.maxAlternatives = 1;
  state.speechRecognizer = recognizer;

  recognizer.onstart = () => {
    state.isVoiceListening = true;
    el.chatStatus.textContent = "Listening... speak naturally.";
    state.liveTranscript = "";
    if (!state.isProcessing) {
      el.chatTranscript.textContent = "Listening...";
    }
  };

  recognizer.onresult = (event) => {
    let interim = "";
    let finalText = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const text = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalText += text + " ";
      } else {
        interim += text + " ";
      }
    }
    const merged = (finalText || interim).trim();
    if (merged) {
      state.liveTranscript = merged;
      el.chatTranscript.textContent = merged;
    }
    if (finalText.trim() && !state.isProcessing) {
      recognizer.stop();
    }
  };

  recognizer.onerror = (event) => {
    // Keep speech engine errors silent in UI for cleaner chat experience.
    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      el.chatStatus.textContent = "Microphone permission is required for voice chat.";
    } else {
      el.chatStatus.textContent = "";
    }
    state.isVoiceListening = false;
    if (
      event.error !== "not-allowed" &&
      event.error !== "service-not-allowed" &&
      state.isConversationActive &&
      el.panelChat.classList.contains("active")
    ) {
      setTimeout(startListening, 600);
    }
  };

  recognizer.onend = async () => {
    state.isVoiceListening = false;
    const message = state.liveTranscript.trim();
    if (message && !state.isProcessing) {
      await sendChatMessage(message);
      return;
    }
    if (!state.isProcessing && state.isConversationActive && el.panelChat.classList.contains("active")) {
      setTimeout(startListening, 400);
    }
  };
}

function startListening() {
  if (!state.speechRecognizer || state.isProcessing || state.isVoiceListening) return;
  const selected = getActiveCharacter();
  if (!selected || !el.panelChat.classList.contains("active")) {
    return;
  }
  try {
    state.liveTranscript = "";
    state.speechRecognizer.lang = dialectToLocale(selected.dialect);
    state.speechRecognizer.start();
  } catch {
    // Ignore "already started" browser errors.
  }
}

function stopListening() {
  if (!state.speechRecognizer || !state.isVoiceListening) return;
  state.speechRecognizer.stop();
}

function toggleConversation() {
  if (!state.speechSupported) return;
  state.isConversationActive = !state.isConversationActive;

  if (state.isConversationActive) {
    el.chatToggleBtn.textContent = "Stop Conversation";
    el.chatStatus.textContent = "Conversation started. Speak now...";
    if (el.panelChat.classList.contains("active")) {
      startListening();
    }
  } else {
    el.chatToggleBtn.textContent = "Start Conversation";
    el.chatStatus.textContent = "Conversation stopped.";
    stopListening();
  }
}

async function bootstrap() {
  await loadCharacters();
  initSpeechRecognition();
  activeTab(true);
  el.tabTts.addEventListener("click", () => {
    activeTab(true);
    stopListening();
  });
  el.tabChat.addEventListener("click", () => {
    activeTab(false);
    if (state.speechSupported && state.isConversationActive) {
      startListening();
    }
  });
  el.ttsGenerate.addEventListener("click", onGenerateTts);
  el.dialectSelect.addEventListener("change", onDialectChange);
  el.chatToggleBtn.addEventListener("click", toggleConversation);
}

bootstrap().catch((err) => {
  el.ttsStatus.textContent = err.message;
  el.chatStatus.textContent = err.message;
});
