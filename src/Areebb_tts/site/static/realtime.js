// Realtime client: WebSocket -> AudioWorklet PCM playback + chat UI.

const SERVER_SAMPLE_RATE = 24000;

const els = {
  character: document.getElementById('rt-character'),
  modelType: document.getElementById('rt-model-type'),
  connect: document.getElementById('rt-connect'),
  stop: document.getElementById('rt-stop'),
  clear: document.getElementById('rt-clear'),
  input: document.getElementById('rt-input'),
  send: document.getElementById('rt-send'),
  status: document.getElementById('rt-status'),
  transcript: document.getElementById('rt-transcript'),
};

const state = {
  ws: null,
  audioCtx: null,
  worklet: null,
  resampleRatio: 1,
  configured: false,
  history: [],
  currentAssistant: '',
  assistantLine: null,
};

function setStatus(text) {
  els.status.textContent = text;
}

function appendLine(role, text) {
  const row = document.createElement('div');
  row.className = `rt-row rt-role-${role === 'user' ? 'user' : 'asst'}`;
  const label = role === 'user' ? 'YOU' : 'ASSISTANT';
  row.textContent = `${label}: ${text}`;
  els.transcript.appendChild(row);
  els.transcript.scrollTop = els.transcript.scrollHeight;
  return row;
}

async function loadCharacters() {
  try {
    const res = await fetch('/api/characters');
    const data = await res.json();
    els.character.innerHTML = '';
    for (const ch of data.characters) {
      const opt = document.createElement('option');
      opt.value = ch.id;
      opt.textContent = `${ch.name} — ${ch.dialect_label}`;
      els.character.appendChild(opt);
    }
  } catch (err) {
    setStatus(`failed to load characters: ${err}`);
  }
}

async function ensureAudio() {
  if (state.audioCtx) return;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  state.audioCtx = new Ctx();
  await state.audioCtx.audioWorklet.addModule('/static/realtime-worklet.js');
  state.worklet = new AudioWorkletNode(state.audioCtx, 'pcm-ring-player', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [1],
  });
  state.worklet.connect(state.audioCtx.destination);
  state.resampleRatio = SERVER_SAMPLE_RATE / state.audioCtx.sampleRate;
}

function pcm16ToFloat32(int16) {
  const out = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    out[i] = int16[i] / 32768.0;
  }
  return out;
}

function resampleLinear(input, ratio) {
  if (Math.abs(ratio - 1.0) < 1e-3) return input;
  const outLen = Math.round(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = i * ratio;
    const i0 = Math.floor(srcIdx);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const t = srcIdx - i0;
    out[i] = input[i0] * (1 - t) + input[i1] * t;
  }
  return out;
}

function pushAudio(arrayBuffer) {
  if (!state.worklet) return;
  const int16 = new Int16Array(arrayBuffer);
  let f32 = pcm16ToFloat32(int16);
  if (state.resampleRatio !== 1) {
    f32 = resampleLinear(f32, state.resampleRatio);
  }
  state.worklet.port.postMessage(f32);
}

function flushAudio() {
  if (state.worklet) state.worklet.port.postMessage('flush');
}

function connect() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    return;
  }
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/ws/voice`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;
  setStatus('connecting...');

  ws.onopen = () => {
    setStatus('connected');
    sendHello();
  };
  ws.onclose = () => {
    setStatus('disconnected');
    els.input.disabled = true;
    els.send.disabled = true;
    els.stop.disabled = true;
    state.configured = false;
  };
  ws.onerror = (event) => {
    setStatus('ws error');
    console.error(event);
  };
  ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
      handleJson(JSON.parse(event.data));
    } else {
      pushAudio(event.data);
    }
  };
}

function sendHello() {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  state.ws.send(
    JSON.stringify({
      type: 'hello',
      character_id: els.character.value,
      model_type: els.modelType.value,
      history: state.history,
    }),
  );
}

function handleJson(msg) {
  switch (msg.type) {
    case 'ready':
      setStatus(`ready (sr=${msg.sample_rate})`);
      break;
    case 'configured':
      state.configured = true;
      els.input.disabled = false;
      els.send.disabled = false;
      setStatus(`ready as ${msg.character_id}`);
      break;
    case 'user_text_ack':
      appendLine('user', msg.text);
      state.currentAssistant = '';
      state.assistantLine = appendLine('assistant', '');
      break;
    case 'assistant_text_delta':
      state.currentAssistant += msg.text;
      if (state.assistantLine) {
        state.assistantLine.textContent = `ASSISTANT: ${state.currentAssistant}`;
        els.transcript.scrollTop = els.transcript.scrollHeight;
      }
      break;
    case 'tts_start':
      els.stop.disabled = false;
      break;
    case 'tts_end':
      els.stop.disabled = true;
      if (msg.assistant_text) {
        state.history.push({ role: 'user', content: state.lastUserText || '' });
        state.history.push({ role: 'assistant', content: msg.assistant_text });
      }
      break;
    case 'interrupted':
      flushAudio();
      els.stop.disabled = true;
      setStatus('interrupted');
      break;
    case 'error':
      setStatus(`error: ${msg.message}`);
      console.error('server error', msg);
      break;
    default:
      console.log('unknown msg', msg);
  }
}

async function sendUserText() {
  const text = els.input.value.trim();
  if (!text) return;
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
    setStatus('not connected');
    return;
  }
  if (!state.configured) {
    setStatus('still configuring; try again in a sec');
    return;
  }
  await ensureAudio();
  if (state.audioCtx.state === 'suspended') {
    await state.audioCtx.resume();
  }
  flushAudio();
  state.lastUserText = text;
  state.ws.send(JSON.stringify({ type: 'user_text', text }));
  els.input.value = '';
}

function sendStop() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: 'stop' }));
  }
  flushAudio();
}

function clearTranscript() {
  els.transcript.innerHTML = '';
  state.history = [];
  state.currentAssistant = '';
  state.assistantLine = null;
}

els.connect.addEventListener('click', () => {
  connect();
});
els.stop.addEventListener('click', sendStop);
els.clear.addEventListener('click', clearTranscript);
els.send.addEventListener('click', sendUserText);
els.input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendUserText();
});
els.character.addEventListener('change', () => {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) sendHello();
});
els.modelType.addEventListener('change', () => {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) sendHello();
});

loadCharacters();
