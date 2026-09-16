#!/usr/bin/env python3
"""Sample script to test the Cloud Run MCP tools during a Gemini Live conversation.

Usage:
    # Against the running local web server (ws://127.0.0.1:8080/ws):
    uv run python scripts/test_mcp_conversation.py

    # With a custom prompt:
    uv run python scripts/test_mcp_conversation.py \
        "Check service status and create a high priority ticket for CUST-1001"
"""

from __future__ import annotations

import asyncio
import json
import sys

import websockets

DEFAULT_PROMPT = (
    "Look up customer Rajesh Sharma in the CRM and tell me his company and annual "
    "contract value. Also calculate the monthly EMI for a 25 lakh rupee loan at "
    "8.5 percent interest for 60 months, and check if any services are degraded."
)


async def run_test(prompt: str, ws_url: str = "ws://127.0.0.1:8080/ws?locale=en-IN") -> None:
    print(f"Connecting to {ws_url} ...")
    async with websockets.connect(ws_url) as ws:
        # 1. Wait for session ready event
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") == "connected":
                print(f"[CONNECTED] Model: {msg.get('model')}")
                print(f"[TOOLS LOADED] {msg.get('tools')}\n")
                break

        # 2. Send user prompt
        print(f"[USER PROMPT]\n  {prompt}\n")
        await ws.send(json.dumps({"type": "text", "text": prompt}))

        # 3. Stream tool calls, results, audio bytes, and spoken transcript
        transcript_fragments: list[str] = []
        audio_bytes_received = 0

        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=45.0)
            if isinstance(raw, bytes):
                audio_bytes_received += len(raw)
                continue

            msg = json.loads(raw)
            etype = msg.get("type")

            if etype == "tool_call":
                print(f"  -> [CLOUD RUN MCP CALL]   {msg.get('names')}")
            elif etype == "tool_result":
                print(
                    f"  <- [CLOUD RUN MCP RESULT] {msg.get('names')} "
                    f"(errors={msg.get('errors')})"
                )
            elif etype == "transcript" and msg.get("role") == "model":
                transcript_fragments.append(msg.get("text", ""))
            elif etype == "usage_turn" and transcript_fragments:
                # Wait for the final turn where the model has spoken its answer
                full_text = "".join(transcript_fragments).strip()
                scalars = msg.get("scalars", {})
                cost = msg.get("cost", {})
                print("\n[MODEL SPOKEN ANSWER]")
                print(f"  {full_text}")
                print(f"\n[AUDIO STREAMED] {audio_bytes_received:,} PCM bytes (24 kHz)")
                print(
                    f"[TOKEN USAGE]    total={scalars.get('total')} "
                    f"(prompt={scalars.get('prompt')}, response={scalars.get('response')}) | "
                    f"tool_decl_tokens={cost.get('tool_declaration_tokens')}"
                )
                break

        # 4. Cleanly hang up the call
        await ws.send(json.dumps({"type": "end_call"}))


if __name__ == "__main__":
    user_prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_PROMPT
    asyncio.run(run_test(user_prompt))
