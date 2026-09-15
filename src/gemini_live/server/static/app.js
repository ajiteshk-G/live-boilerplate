const INPUT_RATE = 16000;
const OUTPUT_RATE = 24000;

const el = (id) => document.getElementById(id);
const state = {
  ws: null,
  micCtx: null,
  playCtx: null,
  playerNode: null,
  recorderNode: null,
  stream: null,
  micOn: false,
  // A call is live from "Start call" until the session is torn down.
  callActive: false,
  // True once the user hung up, so onclose knows this was deliberate.
  ended: false,
  turns: [],
  lastUser: null,
  lastModel: null,
};

// ---------------------------------------------------------------- secure ctx

// getUserMedia requires a secure context. http://<host>.c.googlers.com:8080 is
// NOT secure, so the mic would be blocked with no obvious explanation. Say so.
function checkSecureContext() {
  if (window.isSecureContext) return true;
  el('insecure-banner').hidden = false;
  el('insecure-origin').textContent = window.location.origin;
  el('call-btn').disabled = true;
  return false;
}

// ------------------------------------------------------------------ helpers

function fmt(n) {
  return (n ?? 0).toLocaleString();
}

function modalityChips(map) {
  const entries = Object.entries(map || {}).filter(([, v]) => v > 0);
  if (!entries.length) return '';
  return entries
    .map(([k, v]) => `<span class="chip chip-${k.toLowerCase()}">${k} ${fmt(v)}</span>`)
    .join('');
}

function setStatus(text, cls) {
  const node = el('status');
  node.textContent = text;
  node.className = `status ${cls || ''}`;
}

// -------------------------------------------------------------- transcripts

function appendTranscript(role, text) {
  const list = el('transcript');
  const key = role === 'user' ? 'lastUser' : 'lastModel';
  // The API streams transcripts in fragments; append to the open bubble for this role.
  if (state[key] && state[key].dataset.role === role) {
    state[key].querySelector('.bubble-text').textContent += text;
  } else {
    const item = document.createElement('div');
    item.className = `bubble bubble-${role}`;
    item.dataset.role = role;
    item.innerHTML = `<div class="bubble-role">${role === 'user' ? 'You' : 'Gemini'}</div>
                      <div class="bubble-text"></div>`;
    item.querySelector('.bubble-text').textContent = text;
    list.appendChild(item);
    state[key] = item;
  }
  list.scrollTop = list.scrollHeight;
}

function logEvent(kind, text, cls) {
  const list = el('events');
  const row = document.createElement('div');
  row.className = `event ${cls || ''}`;
  row.innerHTML = `<span class="event-kind">${kind}</span><span>${text}</span>`;
  list.prepend(row);
  while (list.childElementCount > 60) list.lastElementChild.remove();
}

// --------------------------------------------------------------- token panel

function renderTurn(payload) {
  const s = payload.scalars;
  const c = payload.cost || {};
  state.turns.push({ prompt: s.prompt, compression: c.compression_event });

  const row = document.createElement('div');
  row.className = 'turn';
  row.innerHTML = `
    <div class="turn-head">
      <strong>Turn ${payload.turn + 1}</strong>
      <span class="turn-total">${fmt(s.total)} tokens</span>
    </div>
    <div class="turn-grid">
      <div><span class="label">Context rent</span><span class="val">${fmt(c.context_rent ?? s.prompt)}</span></div>
      <div><span class="label">Response</span><span class="val">${fmt(s.response)}</span></div>
      <div><span class="label">Thinking</span><span class="val">${fmt(s.thoughts)}</span></div>
      <div><span class="label">Tool use</span><span class="val">${fmt(s.tool_use)}</span></div>
    </div>
    <div class="rent">
      <div class="rent-bar"><div class="rent-fill" style="width:${((c.rent_ratio ?? 0) * 100).toFixed(1)}%"></div></div>
      <span class="rent-label">${((c.rent_ratio ?? 0) * 100).toFixed(0)}% of this turn was re-reading context</span>
    </div>
    <div class="chips">
      <span class="chips-label">prompt</span>${modalityChips(payload.by_modality.prompt)}
    </div>
    <div class="chips">
      <span class="chips-label">response</span>${modalityChips(payload.by_modality.response)}
    </div>
    ${c.compression_event ? '<div class="compression">Context compression fired (prompt tokens dropped)</div>' : ''}
  `;
  el('turns').prepend(row);
  renderGrowth();

  el('tool-overhead').textContent =
    `${fmt(c.tool_declaration_tokens)} / turn · ${fmt(c.projected_tool_cost)} billed so far`;
}

function renderSession(payload) {
  el('session-total').textContent = fmt(payload.total);
  el('session-turns').textContent = payload.turns;
  el('session-prompt').textContent = fmt(payload.prompt);
  el('session-response').textContent = fmt(payload.response);
}

// Sparkline of prompt tokens per turn. The upward slope IS the billing model:
// every turn re-bills the whole context.
function renderGrowth() {
  const svg = el('growth');
  const pts = state.turns;
  if (pts.length < 2) return;
  const w = 260;
  const h = 56;
  const max = Math.max(...pts.map((p) => p.prompt), 1);
  const step = w / (pts.length - 1);
  const path = pts
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${(i * step).toFixed(1)} ${(h - (p.prompt / max) * h).toFixed(1)}`)
    .join(' ');
  const marks = pts
    .map((p, i) =>
      p.compression
        ? `<circle cx="${(i * step).toFixed(1)}" cy="${(h - (p.prompt / max) * h).toFixed(1)}" r="3.5" class="mark"/>`
        : ''
    )
    .join('');
  svg.innerHTML = `<path d="${path}" class="spark"/>${marks}`;
  el('growth-max').textContent = fmt(max);
}

// --------------------------------------------------------------- citations

function renderCitations(payload) {
  const box = el('citations');
  if (!payload.citations.length) return;
  box.innerHTML =
    `<div class="cite-head">Sources${payload.violations ? ` · ${payload.violations} outside policy` : ''}</div>` +
    payload.citations
      .map(
        (c) => `<div class="cite cite-${c.verdict.toLowerCase()}">
            <span class="verdict">${c.verdict}</span>
            <a href="${c.uri || '#'}" target="_blank" rel="noopener">${c.domain || c.title || c.uri || 'unknown'}</a>
          </div>`
      )
      .join('');
  if (payload.violations) {
    logEvent('policy', `${payload.violations} citation(s) outside the allow-list`, 'warn');
  }
}

// ------------------------------------------------------------------- socket

function connect() {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${window.location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;
  state.ended = false;

  // Resolves once the socket is usable, so startCall() can await it.
  const ready = new Promise((resolve, reject) => {
    ws.onopen = () => {
      setStatus('Connected', 'ok');
      resolve();
    };
    ws.onerror = () => {
      setStatus('Connection error', 'bad');
      reject(new Error('websocket error'));
    };
  });

  ws.onclose = () => {
    // A hang-up closes the socket too; do not report it as a failure.
    setStatus(state.ended ? 'Call ended' : 'Disconnected', state.ended ? '' : 'bad');
    stopMic();
    state.callActive = false;
    setCallButton(false);
  };

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      playPcm(event.data);
      return;
    }
    const msg = JSON.parse(event.data);
    switch (msg.type) {
      case 'connected':
        setStatus(`Live · ${msg.model}`, 'ok');
        logEvent('session', `${msg.tools.length} tool(s) available`);
        break;
      case 'transcript':
        appendTranscript(msg.role, msg.text);
        break;
      case 'text':
        appendTranscript('model', msg.text);
        break;
      case 'interrupted':
        if (state.playerNode) state.playerNode.port.postMessage('flush');
        logEvent('barge-in', 'playback cleared');
        state.lastModel = null;
        break;
      case 'tool_call':
        logEvent('tool', `calling ${msg.names.join(', ')}`);
        break;
      case 'tool_result':
        logEvent('tool', msg.errors.length ? `errors: ${msg.errors.join(', ')}` : 'ok', msg.errors.length ? 'warn' : '');
        break;
      case 'usage_turn':
        renderTurn(msg);
        state.lastUser = null;
        state.lastModel = null;
        break;
      case 'usage_session':
        renderSession(msg);
        break;
      case 'usage_alert':
        logEvent('cost', msg.message, 'warn');
        break;
      case 'citations':
        renderCitations(msg);
        break;
      case 'reconnecting':
        setStatus(`Reconnecting in ${msg.in_seconds}s`, 'warn');
        break;
      case 'session_ended':
        state.ended = true;
        setStatus('Call ended', '');
        logEvent('session', `call ended (${msg.reason || 'no reason given'})`);
        break;
      case 'error':
        logEvent('error', msg.message, 'bad');
        break;
    }
  };

  return ready;
}

// -------------------------------------------------------------------- audio

function playPcm(buffer) {
  if (!state.playerNode) {
    initPlayback().then(() => {
      if (state.playCtx && state.playCtx.state === 'suspended') {
        state.playCtx.resume();
      }
      if (state.playerNode) playPcm(buffer);
    });
    return;
  }
  const pcm = new Int16Array(buffer);
  const floats = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) floats[i] = pcm[i] / 32768;
  state.playerNode.port.postMessage(floats.buffer, [floats.buffer]);
}

async function initPlayback() {
  if (state.playCtx) return;
  state.playCtx = new AudioContext({ sampleRate: OUTPUT_RATE });
  await state.playCtx.audioWorklet.addModule('/static/player-worklet.js');
  state.playerNode = new AudioWorkletNode(state.playCtx, 'player-processor');
  state.playerNode.connect(state.playCtx.destination);
}

async function startMic() {
  await initPlayback();
  await state.playCtx.resume();

  state.stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  state.micCtx = new AudioContext({ sampleRate: INPUT_RATE });
  await state.micCtx.audioWorklet.addModule('/static/recorder-worklet.js');

  const source = state.micCtx.createMediaStreamSource(state.stream);
  state.recorderNode = new AudioWorkletNode(state.micCtx, 'recorder-processor');
  state.recorderNode.port.onmessage = (event) => {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(event.data);
  };
  source.connect(state.recorderNode);

  state.micOn = true;
  setStatus('Listening', 'ok');
}

function stopMic() {
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  if (state.micCtx) state.micCtx.close();
  state.stream = null;
  state.micCtx = null;
  state.recorderNode = null;
  if (state.micOn && state.ws && state.ws.readyState === WebSocket.OPEN) {
    // Flush any audio the server has cached for this utterance.
    state.ws.send(JSON.stringify({ type: 'mic', on: false }));
  }
  state.micOn = false;
}

// ---------------------------------------------------------------------- call

function setCallButton(live) {
  const btn = el('call-btn');
  btn.classList.toggle('live', live);
  btn.textContent = live ? 'Stop call' : 'Start call';
}

// The Live session starts with the socket, so the socket is opened on demand:
// merely loading the page should not start a billed session.
async function ensureConnected() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) return;
  await connect();
}

async function startCall() {
  await ensureConnected();
  await startMic();
  state.callActive = true;
  setCallButton(true);
}

// Stop ends the CALL, not just the microphone: the server tears the Live
// session down, so the model stops generating and billing stops with it.
function endCall() {
  state.ended = true;
  state.callActive = false;
  stopMic();
  if (state.playerNode) state.playerNode.port.postMessage('flush');
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: 'end_call' }));
    // The server closes the socket once it has flushed the final usage; this
    // is the backstop if it does not.
    setTimeout(() => {
      if (state.ws && state.ws.readyState <= WebSocket.OPEN) state.ws.close();
    }, 1500);
  } else if (state.ws) {
    state.ws.close();
  }
  setCallButton(false);
  setStatus('Call ended', '');
}

// --------------------------------------------------------------------- init

el('call-btn').addEventListener('click', async () => {
  try {
    if (state.callActive) endCall();
    else await startCall();
  } catch (err) {
    logEvent('error', `call: ${err.message}`, 'bad');
    state.callActive = false;
    setCallButton(false);
  }
});

// Closing the tab should hang up too, rather than leaving a session billing
// until the server notices the dead socket.
window.addEventListener('beforeunload', () => {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: 'end_call' }));
  }
});

el('text-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = el('text-input');
  const text = input.value.trim();
  if (!text) return;
  try {
    // Typing is a valid way to start (or restart) a session without the mic.
    await ensureConnected();
    await initPlayback();
    if (state.playCtx) await state.playCtx.resume();
  } catch (err) {
    logEvent('error', `send: ${err.message}`, 'bad');
    return;
  }
  state.ws.send(JSON.stringify({ type: 'text', text }));
  state.lastUser = null;
  state.lastModel = null;
  appendTranscript('user', text);
  state.lastUser = null;
  input.value = '';
});

async function loadMeta() {
  try {
    const res = await fetch('/api/config');
    const data = await res.json();
    el('model-name').textContent = data.config.model.name;
    el('tool-count').textContent =
      `${data.tools.selected.length} of ${data.tools.catalog_size}`;
    el('tool-tokens').textContent = `${fmt(data.tools.tokens_per_turn)} tokens/turn`;
    el('tool-list').innerHTML = data.tools.selected
      .map(
        (t) => `<li><span>${t.name}</span><span class="tool-meta">${fmt(t.tokens)}t${t.pinned ? ' · pinned' : ''}</span></li>`
      )
      .join('');
    const domains = data.domains.allow;
    el('domain-list').textContent = domains.length ? domains.join(', ') : 'unrestricted';
    Object.entries(data.tools.mcp_failures || {}).forEach(([name, err]) =>
      logEvent('mcp', `${name} unavailable: ${err}`, 'warn')
    );
  } catch (err) {
    logEvent('error', `config: ${err.message}`, 'bad');
  }
}

// No eager connect: the Live session (and its billing) starts when the user
// presses Start call or sends a message.
checkSecureContext();
setCallButton(false);
setStatus('Idle · press Start call', '');
loadMeta();
