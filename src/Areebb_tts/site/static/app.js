const state = {
  characters: [],
  history: [],
};

const el = {
  characterSelect: document.getElementById("character-select"),
  modelSelect: document.getElementById("model-select"),
  ttsInput: document.getElementById("tts-input"),
  ttsGenerate: document.getElementById("tts-generate"),
  ttsAudio: document.getElementById("tts-audio"),
  ttsStatus: document.getElementById("tts-status"),
  chatInput: document.getElementById("chat-input"),
  chatSend: document.getElementById("chat-send"),
  chatBox: document.getElementById("chat-box"),
  chatAudio: document.getElementById("chat-audio"),
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

async function loadCharacters() {
  const res = await fetch("/api/characters");
  if (!res.ok) throw new Error("Cannot fetch characters");
  const data = await res.json();
  state.characters = data.characters;
  el.characterSelect.innerHTML = "";
  state.characters.forEach((c) => {
    const option = document.createElement("option");
    option.value = c.id;
    option.textContent = `${c.name} (${c.dialect})`;
    el.characterSelect.appendChild(option);
  });
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
        character_id: el.characterSelect.value,
        text: text,
        model_type: el.modelSelect.value,
      }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to generate");
    el.ttsAudio.src = data.audio_url + `?t=${Date.now()}`;
    el.ttsStatus.textContent = "Audio generated.";
  } catch (err) {
    el.ttsStatus.textContent = err.message;
  } finally {
    el.ttsGenerate.disabled = false;
  }
}

async function onSendChat() {
  const message = el.chatInput.value.trim();
  if (!message) {
    el.chatStatus.textContent = "Please enter a message.";
    return;
  }

  renderChatBubble("user", message);
  el.chatInput.value = "";
  el.chatSend.disabled = true;
  el.chatStatus.textContent = "Thinking and generating voice...";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        character_id: el.characterSelect.value,
        message: message,
        model_type: el.modelSelect.value,
        history: state.history,
      }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Chat failed");

    state.history = data.history;
    renderChatBubble("assistant", data.assistant_text);
    el.chatAudio.src = data.audio_url + `?t=${Date.now()}`;
    el.chatStatus.textContent = "Reply ready.";
  } catch (err) {
    el.chatStatus.textContent = err.message;
  } finally {
    el.chatSend.disabled = false;
  }
}

function resetChatOnCharacterChange() {
  state.history = [];
  el.chatBox.innerHTML = "";
  el.chatAudio.removeAttribute("src");
  el.chatStatus.textContent = "Chat reset for selected character.";
}

async function bootstrap() {
  await loadCharacters();
  activeTab(true);
  el.tabTts.addEventListener("click", () => activeTab(true));
  el.tabChat.addEventListener("click", () => activeTab(false));
  el.ttsGenerate.addEventListener("click", onGenerateTts);
  el.chatSend.addEventListener("click", onSendChat);
  el.characterSelect.addEventListener("change", resetChatOnCharacterChange);
}

bootstrap().catch((err) => {
  el.ttsStatus.textContent = err.message;
  el.chatStatus.textContent = err.message;
});
