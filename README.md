# Gemini Live — bidirectional audio boilerplate

A production-shaped starting point for a **Gemini Live** voice agent on **Vertex AI**, with three things most samples leave out:

1. **Honest token accounting** — per-turn and per-session, broken down by modality, with the context-rent math that explains your bill.
2. **Intelligent tool curation** — point it at a large MCP toolset and it selects a good small subset, because tools are re-billed on every turn.
3. **A domain policy** for grounded search, with an explicit statement of what it can and cannot enforce.

Talk to it in a browser: mic in, audio out, transcripts and live token costs on screen.

---

## Quick start (macOS / Linux, running locally)

Requires Python 3.11+. [`uv`](https://docs.astral.sh/uv/) will fetch a suitable
Python for you if your system one is too old, which on macOS it usually is.

```bash
# 0. Install uv, if you do not have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Install this project
uv sync --extra dev

# 2. Point at your Google Cloud project and authenticate
export GOOGLE_CLOUD_PROJECT=your-project-id
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com --project "$GOOGLE_CLOUD_PROJECT"

# 3. Confirm everything is reachable before debugging anything else
uv run glive doctor

# 4. Go
uv run glive serve
```

Then open **<http://localhost:8080>** and click *Connect*. The browser will ask
for microphone permission the first time.

> [!TIP]
> Running on your own machine needs no TLS and no tunnel: browsers treat
> `localhost` as a secure context, so the microphone just works.

<details>
<summary><b>Running on a remote machine instead?</b></summary>

Chrome only grants microphone access in a [secure
context](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts).
A plain-HTTP page served from a remote hostname is **not** one, so
`getUserMedia()` fails, often silently.

Forward the port and use localhost, which *is* trusted:

```bash
ssh -L 8080:localhost:8080 your-devbox.example.com
```

Then open `http://localhost:8080` on your laptop — not the remote hostname. The
UI detects this mistake and shows a banner. Alternatively run
`./scripts/gen_cert.sh` and serve over HTTPS with a self-signed certificate.

</details>

---

## Why this exists

### Every turn is billed for the entire context

This is the single most important thing to understand about Live API cost. A turn is not charged for the new audio you just spoke — it is charged for **the whole resident context**: conversation history, system instruction, and every tool declaration, re-read from scratch, every single turn.

Two consequences drive the design of this repo:

- **Tool declarations are a recurring charge, not a fixed one.** A 6,000-token tool payload over a 30-turn conversation is ~180,000 billed prompt tokens. Loading "all the tools from my MCP servers" is therefore not free, and it is not good practice.
- **Context compression is a cost control**, not just a quality one. It is on by default.

The UI surfaces this directly as **context rent** (the prompt tokens you re-pay each turn) and **rent ratio** (what fraction of a turn's bill was just re-reading history). When the rent ratio climbs toward 1.0, you are paying almost entirely to re-send the past.

### Tools are frozen at session setup

The Live client protocol accepts exactly four message types — `setup`, `clientContent`, `realtimeInput`, `toolResponse`. There is **no tool-update message**. Whatever tool set you connect with is what you carry, and re-pay for, until you tear down the socket.

So the only lever is choosing a good small set *before* connecting. Google's own guidance is to keep the active set to **10–20 tools** (the hard API cap is 128); selection accuracy degrades as the count grows, on top of the cost.

That is what the curator does: it takes your full MCP toolset, ranks every tool against what your agent is actually *for*, and packs the best ones under a token budget.

```bash
uv run glive tools explain
```

shows exactly what was selected, what was dropped and why, what it costs per turn, and what it will cost over a conversation.

---

## Configuration

Everything lives in [`config/config.yaml`](config/config.yaml), validated by Pydantic. The schema sets `extra="forbid"` — a misspelled key is a loud error, not a silently ignored setting. `${VAR}` and `${VAR:-default}` interpolate from the environment and `.env`.

[`config/config.minimal.yaml`](config/config.minimal.yaml) shows the true minimum: just your GCP project.

The four things the original brief asked for:

| Requirement | Where |
|---|---|
| MCP toolsets / tool restriction | `tools.mcp`, `tools.curation`, `tools.pinned`, `tools.exclude`, `tools.allow` |
| Domain restriction for search | `search.domains.allow`, `search.domains.deny`, `search.google_search.exclude_domains` |
| Gemini Live model | `model.name` |
| Other Live flags | `speech`, `vad`, `transcription`, `thinking`, `session`, `media` |

### Tool selection

```yaml
tools:
  curation:
    mode: auto          # rank + pack automatically
    purpose: "Answer questions about our internal API docs and file bugs."
    max_tools: 20
    budget_tokens: 6000
  pinned:  ["*__file_bug"]     # always include
  exclude: ["*__delete_*"]     # never include
  mcp:
    - name: docs
      transport: stdio
      command: npx
      args: ["-y", "@modelcontextprotocol/server-docs"]
```

`purpose` is the text tool descriptions are ranked against — the more specific it is, the better the selection. Leave it empty and it falls back to your system instruction.

Set `mode: manual` if you would rather hand-maintain `tools.allow`.

---

## Domain restriction: what it actually guarantees

> [!CAUTION]
> **The allow-list is advisory. The deny-list is enforced.**
>
> The Live API has **no allow-list field**. `types.GoogleSearch` exposes only `exclude_domains`, a deny-list, and only on Vertex.

An allow-list here is enforced in three layers, and you should know the strength of each:

| Layer | Strength |
|---|---|
| `google_search.exclude_domains` (fed by `domains.deny`) | **Enforced server-side.** Real. |
| Generated system-instruction rules | Advisory — the model usually complies, but may not. |
| Post-hoc audit of grounding citations | Detects violations. Does **not** prevent them. |

The gap that matters: grounding metadata can arrive *after* audio playback has already started, so in the default `audit` mode the model may finish speaking a non-approved source before the violation is logged. Violations land in `logs/violations.jsonl` and appear in the UI.

If that is unacceptable for your use case, opt into buffering:

```yaml
search:
  enforcement: strict_buffered   # holds audio until citations clear; adds latency
```

Citations whose real hostname cannot be recovered — Vertex returns opaque redirect URIs — are reported as `UNKNOWN` rather than being quietly counted as allowed.

---

## Token reporting

After every turn you get the full `usageMetadata` breakdown: prompt, cached, response, tool-use, thinking, and total, each with its per-modality split (AUDIO vs TEXT), plus a running session total.

> [!NOTE]
> **The API reports exactly four per-modality arrays** — prompt, cache, response, and tool-use. There is no `thoughts_tokens_details` and no `total_tokens_details`. Thinking tokens are a scalar only, and any "total by modality" figure is **derived** by summing the four real arrays. This repo labels it as derived so it is never mistaken for a server-reported number.

### The counting trap

`usageMetadata` is **cumulative within a turn and resets at each `turnComplete`**. Most messages restate a running total, so the obvious implementation —

```python
total += response.usage_metadata.total_token_count   # WRONG
```

— over-counts badly. The correct algorithm is *last-wins within a turn → commit on `turnComplete` → sum committed turns*, which is what [`TokenAccountant`](src/gemini_live/usage/accountant.py) does.

There is one genuinely unsettled detail: sources disagree on whether the *response* counter is cumulative or a per-message delta. Rather than guess, the accountant tracks both interpretations and `accounting_mode: auto` picks based on observed monotonicity. To see which your model actually emits:

```bash
uv run glive calibrate-usage
```

Everything is logged to `logs/usage.jsonl`; analyse a past session offline with `uv run glive usage-report`.

---

## CLI

| Command | Purpose | Needs GCP |
|---|---|:-:|
| `glive serve` | Run the web UI and Live relay | yes |
| `glive validate-config` | Validate and pretty-print the resolved config | no |
| `glive doctor` | Check credentials, model access, and MCP connectivity | yes |
| `glive tools explain` | Show the curated tool set, drop reasons, and projected cost | yes¹ |
| `glive usage-report` | Offline analysis of `logs/usage.jsonl` | no |
| `glive selftest` | Headless TEXT session exercising the tool loop and token reporting | yes |
| `glive calibrate-usage` | Empirically determine the token accounting mode | yes |
| `glive calibrate-tools` | Probe how session resumption interacts with a changed tool list | yes |

¹ Falls back to estimated token costs if `count_tokens` and embeddings are unreachable, so it still produces useful output offline.

---

## Layout

```
src/gemini_live/
  settings/    config schema, ${VAR} loader, per-model capability filtering
  usage/       token accounting, cost model, optional pricing
  tools/       MCP client, schema compaction, cost metering, curation, dispatch
  search/      domain policy and grounding-citation audit
  live/        Live client, connect config, session runner
  server/      FastAPI app, WebSocket relay, browser UI
```

## Development

```bash
uv run pytest -v          # full suite, runs offline with no GCP credentials
uv run ruff check src tests
uv run mypy src
```

## Known limitations

- **Vertex AI only.** The Gemini Developer API path is deliberately not wired up; `exclude_domains` is rejected there anyway.
- **Search allow-listing is best-effort** in the default mode. See above.
- **Adaptive tool re-selection is off by default.** Changing tools requires a reconnect, and how session resumption handles a changed tool list is undocumented — `glive calibrate-tools` is there to find out for your model.
- **Dollar estimates are off by default.** A stale rate card is worse than no number; fill in `usage.pricing` with current rates if you want them.
