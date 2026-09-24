// Defaults matching the Live API wire format. /api/config is authoritative and
// overwrites these before any AudioContext is created; see loadMeta().
const audio = { inputRate: 16000, outputRate: 24000 };
// Resolves once /api/config has been applied. Assigned during init below.
let metaReady = null;

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
  latencies: [],
  setupLatencyMs: null,
  toolRtts: [],
  lastUser: null,
  lastModel: null,
};

// ---------------------------------------------------------------- secure ctx

// getUserMedia requires a secure context. Serving over plain http:// from a
// remote host is NOT secure, so the mic would be blocked with no obvious
// explanation. Say so, and spell out the tunnel for THIS host and port.
function checkSecureContext() {
  if (window.isSecureContext) return true;
  const loc = window.location;
  const port = loc.port || (loc.protocol === 'https:' ? '443' : '80');
  el('insecure-banner').hidden = false;
  el('insecure-origin').textContent = loc.origin;
  el('secure-url').textContent = `http://localhost:${port}`;
  el('ssh-hint').textContent = `ssh -L ${port}:localhost:${port} ${loc.hostname}`;
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
    const label = role === 'user' ? 'You' : (state.agentName || 'Gemini');
    item.innerHTML = `<div class="bubble-role">${label}</div>
                      <div class="bubble-text"></div>`;
    item.querySelector('.bubble-text').textContent = text;
    list.appendChild(item);
    state[key] = item;
  }
  list.scrollTop = list.scrollHeight;
}

function prettyPayload(raw) {
  if (raw == null) return '';
  let val = raw;
  if (typeof val === 'object' && val !== null && typeof val.result === 'string') {
    try {
      val = JSON.parse(val.result);
    } catch {
      val = val.result;
    }
  }
  return typeof val === 'string' ? val : JSON.stringify(val, null, 2);
}

function renderToolCall(msg) {
  const list = el('transcript');
  // Break active model bubble so post-tool speech starts in a fresh bubble below the card.
  state.lastModel = null;
  const calls = msg.calls && msg.calls.length
    ? msg.calls
    : (msg.names || []).map((name, i) => ({ id: `${name}-${i}`, name, args: {} }));

  calls.forEach((call, idx) => {
    const cardId = call.id || `${call.name}-${Date.now()}-${idx}`;
    const card = document.createElement('div');
    card.className = 'tool-card';
    card.dataset.toolId = cardId;
    card.dataset.toolName = call.name;
    card.innerHTML = `
      <div class="tool-card-head">
        <span class="tool-title">🔧 Tool Call · <code></code></span>
        <span class="tool-badge running">Running…</span>
      </div>
      <div class="tool-section">
        <div class="tool-label">Arguments</div>
        <pre class="tool-json tool-args"></pre>
      </div>
      <div class="tool-section tool-result-box" hidden>
        <div class="tool-label">Result</div>
        <pre class="tool-json tool-res"></pre>
      </div>
    `;
    card.querySelector('code').textContent = call.name;
    card.querySelector('.tool-args').textContent = prettyPayload(call.args);
    list.appendChild(card);
    logEvent('tool', `calling ${call.name}(${JSON.stringify(call.args || {})})`);
  });
  list.scrollTop = list.scrollHeight;
}

function renderToolResult(msg) {
  const list = el('transcript');
  const results = msg.results || [];
  const rtt = msg.round_trip_ms != null ? Math.round(msg.round_trip_ms) : null;
  if (rtt != null) {
    state.toolRtts.push(rtt);
    renderLatencyDashboard();
  }
  results.forEach((res) => {
    // Match by id first, otherwise find the last running card for this tool name
    let card = res.id
      ? list.querySelector(`.tool-card[data-tool-id="${CSS.escape(res.id)}"]`)
      : null;
    if (!card) {
      const candidates = list.querySelectorAll(
        `.tool-card[data-tool-name="${CSS.escape(res.name)}"]`
      );
      card = candidates[candidates.length - 1];
    }
    const isErr =
      res.response && typeof res.response === 'object' && 'error' in res.response;
    if (card) {
      const badge = card.querySelector('.tool-badge');
      const rttSuffix = rtt != null ? ` · ${rtt} ms` : '';
      badge.textContent = isErr ? `⚠ Error${rttSuffix}` : `✓ Completed${rttSuffix}`;
      badge.className = `tool-badge ${isErr ? 'err' : 'ok'}`;
      const resBox = card.querySelector('.tool-result-box');
      resBox.hidden = false;
      card.querySelector('.tool-res').textContent = prettyPayload(res.response);
    }
    logEvent(
      'tool',
      isErr
        ? `error in ${res.name}`
        : `${res.name} completed${rtt != null ? ` (${rtt} ms)` : ''}`,
      isErr ? 'warn' : ''
    );
  });
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

function ttfbSeverityClass(ms) {
  if (ms == null) return '';
  if (ms < 500) return 'ok';
  if (ms <= 1000) return 'warn';
  return 'bad';
}

function percentile(arr, p) {
  if (!arr.length) return null;
  const sorted = [...arr].sort((a, b) => a - b);
  if (sorted.length === 1) return sorted[0];
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] * (1 - (idx - lo)) + sorted[hi] * (idx - lo);
}

function renderLatencyDashboard() {
  const pts = state.latencies;
  const ttfbs = pts.map((p) => p.ttfb_ms).filter((v) => v != null && v > 0);
  const durs = pts.map((p) => p.duration_ms).filter((v) => v != null && v > 0);

  const avgTtfb = ttfbs.length
    ? Math.round(ttfbs.reduce((a, b) => a + b, 0) / ttfbs.length)
    : null;
  const p95Ttfb = ttfbs.length ? Math.round(percentile(ttfbs, 95)) : null;
  const avgDur = durs.length
    ? Math.round(durs.reduce((a, b) => a + b, 0) / durs.length)
    : null;
  const avgToolRtt = state.toolRtts.length
    ? Math.round(state.toolRtts.reduce((a, b) => a + b, 0) / state.toolRtts.length)
    : null;

  el('lat-avg-ttfb').textContent = avgTtfb != null ? `${fmt(avgTtfb)}` : '—';
  el('lat-p95-ttfb').textContent = p95Ttfb != null ? `${fmt(p95Ttfb)}` : '—';
  el('lat-avg-dur').textContent = avgDur != null ? `${fmt(avgDur)}` : '—';

  const setupStr = state.setupLatencyMs != null ? `${Math.round(state.setupLatencyMs)}` : '—';
  const toolStr = avgToolRtt != null ? `${avgToolRtt}` : '—';
  el('lat-setup').textContent = `${setupStr} / ${toolStr}`;

  const lastTtfb = ttfbs.length ? Math.round(ttfbs[ttfbs.length - 1]) : null;
  el('lat-last').textContent = lastTtfb != null ? `${fmt(lastTtfb)} ms` : '—';

  const svg = el('latency-chart');
  if (!pts.length) {
    svg.innerHTML = '';
    el('latency-max').textContent = '0 ms';
    return;
  }
  const w = 260;
  const h = 64;
  const maxVal = Math.max(...ttfbs, 1000);
  el('latency-max').textContent = `${fmt(Math.round(Math.max(...ttfbs, 0)))} ms`;

  const n = pts.length;
  const slotW = w / Math.max(n, 8);
  const barW = Math.max(4, Math.min(18, slotW * 0.68));
  const y500 = (h - (500 / maxVal) * (h - 10)).toFixed(1);
  const y1000 = (h - (1000 / maxVal) * (h - 10)).toFixed(1);

  const bars = pts
    .map((p, idx) => {
      const val = p.ttfb_ms ?? 0;
      const barH = Math.max(3, (val / maxVal) * (h - 10));
      const x = (idx * slotW + (slotW - barW) / 2).toFixed(1);
      const y = (h - barH).toFixed(1);
      const color =
        val < 500 ? 'var(--ok)' : val <= 1000 ? 'var(--warn)' : 'var(--bad)';
      return `<rect x="${x}" y="${y}" width="${barW.toFixed(1)}" height="${barH.toFixed(1)}" rx="2" fill="${color}">
        <title>Turn ${p.turn}: TTFB ${Math.round(val)} ms · Duration ${Math.round(p.duration_ms || 0)} ms</title>
      </rect>`;
    })
    .join('');

  svg.innerHTML = `
    <line x1="0" y1="${y500}" x2="${w}" y2="${y500}" stroke="#3fb95055" stroke-dasharray="3,3" stroke-width="1"/>
    <line x1="0" y1="${y1000}" x2="${w}" y2="${y1000}" stroke="#d2992255" stroke-dasharray="3,3" stroke-width="1"/>
    ${bars}
  `;
}

// --------------------------------------------------------------- token panel

function renderTurn(payload) {
  const s = payload.scalars;
  const c = payload.cost || {};
  const lat = payload.latency || {};
  state.turns.push({ prompt: s.prompt, compression: c.compression_event });

  if (lat.setup_latency_ms != null && state.setupLatencyMs == null) {
    state.setupLatencyMs = lat.setup_latency_ms;
  }
  if (lat.ttfb_ms != null || lat.turn_duration_ms != null) {
    state.latencies.push({
      turn: payload.turn + 1,
      ttfb_ms: lat.ttfb_ms,
      duration_ms: lat.turn_duration_ms,
      tool_rtt_ms: lat.tool_round_trip_ms,
    });
    renderLatencyDashboard();
  }

  const ttfbText = lat.ttfb_ms != null ? `${Math.round(lat.ttfb_ms)} ms` : '—';
  const durText = lat.turn_duration_ms != null ? `${Math.round(lat.turn_duration_ms)} ms` : '—';
  const toolRttHtml =
    lat.tool_round_trip_ms != null
      ? `<span class="lat-pill">🔧 Tool RTT: ${Math.round(lat.tool_round_trip_ms)} ms</span>`
      : '';

  const row = document.createElement('div');
  row.className = 'turn';
  row.innerHTML = `
    <div class="turn-head">
      <strong>Turn ${payload.turn + 1}</strong>
      <span class="turn-total">${fmt(s.total)} tokens</span>
    </div>
    <div class="turn-latency-bar">
      <span class="lat-pill ${ttfbSeverityClass(lat.ttfb_ms)}">⚡ TTFB: ${ttfbText}</span>
      <span class="lat-pill">⏱ Duration: ${durText}</span>
      ${toolRttHtml}
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
  refreshTelemetryFromServer();

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
  // Tell the server what language this user is in, so the call opens in it
  // rather than in the server's default.
  const locale = encodeURIComponent(navigator.language || '');
  const ws = new WebSocket(`${proto}://${window.location.host}/ws?locale=${locale}`);
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
        if (msg.setup_latency_ms != null) {
          state.setupLatencyMs = msg.setup_latency_ms;
          renderLatencyDashboard();
          logEvent('telemetry', `session setup ${Math.round(msg.setup_latency_ms)} ms`);
        }
        logEvent('session', `${msg.tools.length} tool(s) available`);
        if (msg.language) {
          logEvent(
            'language',
            msg.language_mode === 'follow_user'
              ? `starts in ${msg.language}, follows your language`
              : `pinned to ${msg.language}`
          );
        }
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
        renderToolCall(msg);
        break;
      case 'tool_result':
        renderToolResult(msg);
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
  // The rates come from the server; creating a context before they land would
  // bake in the fallback and resample everything.
  await metaReady;
  state.playCtx = new AudioContext({ sampleRate: audio.outputRate });
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
  state.micCtx = new AudioContext({ sampleRate: audio.inputRate });
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

function applyTelemetrySnapshot(t) {
  if (!t) return;
  if (t.dashboard_url && el('gcp-dashboard-link')) {
    el('gcp-dashboard-link').href = t.dashboard_url;
  }
  const agg = t.aggregates || {};
  if (state.setupLatencyMs == null && agg.avg_setup_latency_ms != null) {
    state.setupLatencyMs = agg.avg_setup_latency_ms;
  }
  // If no live session turns have been recorded in this browser tab yet, seed
  // the chart from gemini-live-telemetry's in-memory MetricsStore.
  if (!state.latencies.length && Array.isArray(t.turns) && t.turns.length) {
    state.latencies = t.turns
      .filter((item) => item.ttfb_ms != null || item.duration_ms != null)
      .map((item, idx) => ({
        turn: item.turn_number || idx + 1,
        ttfb_ms: item.ttfb_ms,
        duration_ms: item.duration_ms,
      }));
  }
  if (!state.toolRtts.length && Array.isArray(t.tool_calls) && t.tool_calls.length) {
    state.toolRtts = t.tool_calls
      .map((tc) => tc.round_trip_ms)
      .filter((v) => v != null && v > 0);
  }
  renderLatencyDashboard();
}

async function refreshTelemetryFromServer() {
  try {
    const res = await fetch('/api/telemetry');
    if (res.ok) {
      const data = await res.json();
      applyTelemetrySnapshot(data);
    }
  } catch {
    // non-fatal
  }
}

async function loadMeta() {
  try {
    const res = await fetch('/api/config');
    const data = await res.json();
    if (data.audio) {
      audio.inputRate = data.audio.input_sample_rate;
      audio.outputRate = data.audio.output_sample_rate;
    }
    if (data.telemetry) {
      applyTelemetrySnapshot(data.telemetry);
    }
    const agentName = data.config.agent?.name || 'Ananya';
    const agentGender = data.config.agent?.gender || 'female';
    const voiceName = data.config.speech?.voice_name || 'Kore';
    state.agentName = agentName;
    el('model-name').textContent = `${agentName} (${agentGender} · ${voiceName}) · ${data.config.model.name}`;
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
metaReady = loadMeta();
