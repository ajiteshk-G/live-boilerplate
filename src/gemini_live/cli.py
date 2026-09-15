"""``glive`` command line interface."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from .live.client import build_client
from .live.events import CollectingSink
from .live.runner import LiveSessionRunner
from .pipeline import ToolPipeline
from .settings.capabilities import filter_for_model
from .settings.loader import ConfigError, load_config
from .settings.schema import AppConfig, redact

app = typer.Typer(add_completion=False, help="Gemini Live boilerplate (Vertex AI)")
console = Console()

DEFAULT_CONFIG = "config/config.yaml"


def _load(path: str) -> AppConfig:
    try:
        cfg = load_config(path)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    cfg, warnings = filter_for_model(cfg)
    for warning in warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    return cfg


def _setup_logging(cfg: AppConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.app.log_level.upper(), logging.INFO),
        format="%(levelname)-7s %(name)s: %(message)s",
    )


# --------------------------------------------------------------------------- config


@app.command("validate-config")
def validate_config(config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c")) -> None:
    """Parse and validate the config, printing the resolved tree (secrets masked)."""
    cfg = _load(config)
    console.print_json(json.dumps(redact(cfg)))
    console.print("\n[green]Configuration is valid.[/green]")


@app.command()
def doctor(config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c")) -> None:
    """Check auth, project, and model reachability before you debug anything else."""
    cfg = _load(config)
    table = Table("check", "result", box=None)

    table.add_row("project", cfg.vertex.project or "[red]unset[/red]")
    table.add_row("location", cfg.vertex.location)
    table.add_row("model", cfg.model.name)

    try:
        client = build_client(cfg)
        table.add_row("client", "[green]created[/green]")
    except Exception as exc:
        table.add_row("client", f"[red]{exc}[/red]")
        console.print(table)
        raise typer.Exit(1) from exc

    try:
        resp = client.models.count_tokens(
            model=cfg.tools.curation.cost_model, contents="ping"
        )
        table.add_row(
            "vertex reachable", f"[green]yes[/green] (count_tokens -> {resp.total_tokens})"
        )
    except Exception as exc:
        table.add_row("vertex reachable", f"[red]{exc}[/red]")
        console.print(table)
        console.print(
            "\n[yellow]Hint:[/yellow] run [bold]gcloud auth application-default login[/bold] "
            "and ensure the Vertex AI API is enabled on the project."
        )
        raise typer.Exit(1) from exc

    console.print(table)
    console.print("\n[green]Doctor checks passed.[/green]")


# ---------------------------------------------------------------------------- tools


@app.command("tools")
def tools_cmd(
    action: str = typer.Argument("explain"),
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    turns: int = typer.Option(30, "--turns", help="Horizon for projected recurring cost"),
) -> None:
    """Inspect the tool catalog, measured costs, and curation decisions."""
    if action != "explain":
        console.print(f"[red]unknown action {action!r} (expected 'explain')[/red]")
        raise typer.Exit(1)

    cfg = _load(config)
    _setup_logging(cfg)

    async def run() -> None:
        try:
            client = build_client(cfg)
        except Exception as exc:
            console.print(f"[yellow]no Vertex client ({exc}); using estimated costs[/yellow]")
            client = None

        async with ToolPipeline(cfg, client) as pipeline:
            result = await pipeline.build()
            sel = result.selection

            console.print(
                f"\n[bold]Catalog:[/bold] {sel.catalog_size} tools   "
                f"[bold]Purpose:[/bold] {cfg.curation_purpose()[:70]!r}"
            )
            console.print(
                f"[bold]Budget:[/bold] {cfg.tools.curation.budget_tokens} tokens / "
                f"max {cfg.tools.curation.max_tools} tools   "
                f"[bold]Mode:[/bold] {sel.mode}\n"
            )

            chosen = Table(
                "score", "tokens", "tool", "origin", title=f"SELECTED ({len(sel.selected)})"
            )
            for cand in sel.selected:
                chosen.add_row(
                    f"{cand.score:.2f}",
                    str(cand.token_cost),
                    cand.exposed_name + (" [pinned]" if cand.pinned else ""),
                    cand.origin,
                )
            console.print(chosen)

            if sel.dropped:
                dropped = Table(
                    "score", "tokens", "tool", "reason", title=f"DROPPED ({len(sel.dropped)})"
                )
                for cand in sorted(sel.dropped, key=lambda c: -c.score)[:40]:
                    dropped.add_row(
                        f"{cand.score:.2f}",
                        str(cand.token_cost),
                        cand.exposed_name,
                        cand.drop_reason or "",
                    )
                console.print(dropped)

            console.print(
                f"\n[bold]Declaration cost:[/bold] {sel.total_tokens} tokens per turn "
                f"(full catalog would be {sel.catalog_tokens})"
            )
            console.print(
                f"[bold]Projected over {turns} turns:[/bold] "
                f"[cyan]{sel.projected_cost(turns):,}[/cyan] billed prompt tokens "
                f"([green]{sel.saved_tokens * turns:,} saved[/green] vs loading everything)"
            )
            console.print(
                "\n[dim]Declarations sit in the resident context, so they are re-billed "
                "on every turn. Google's guidance is 10-20 active tools.[/dim]"
            )
            for name, err in result.mcp_failures.items():
                console.print(f"[yellow]MCP {name} unavailable:[/yellow] {err}")

    asyncio.run(run())


# ---------------------------------------------------------------------- calibration


@app.command("calibrate-usage")
def calibrate_usage(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    prompts: int = typer.Option(3, "--turns"),
) -> None:
    """Settle empirically whether Live token counters are cumulative or deltas.

    Two internal Google sources disagree about whether ``response_token_count`` is
    a running total within a turn or a per-message delta. This runs a short TEXT
    session and inspects the raw snapshot sequence.
    """
    cfg = _load(config)
    _setup_logging(cfg)
    cfg.model.response_modalities = ["TEXT"]

    async def run() -> None:

        client = build_client(cfg)
        from .live.connect_config import build_live_config

        live_cfg = build_live_config(cfg, None)
        questions = [
            "In one short sentence, what is a vector database?",
            "And in one sentence, why are embeddings useful?",
            "Finally, name one tradeoff of using them.",
        ][:prompts]

        observations: list[list[int]] = []
        async with client.aio.live.connect(model=cfg.model.name, config=live_cfg) as session:
            for question in questions:
                await session.send_realtime_input(text=question)
                seq: list[int] = []
                async for message in session.receive():
                    um = getattr(message, "usage_metadata", None)
                    if um is not None:
                        seq.append(int(getattr(um, "response_token_count", 0) or 0))
                    content = getattr(message, "server_content", None)
                    if content is not None and getattr(content, "turn_complete", False):
                        break
                observations.append(seq)

        table = Table("turn", "snapshots", "response_token_count sequence", "monotonic?")
        verdicts = []
        for i, seq in enumerate(observations):
            mono = all(b >= a for a, b in zip(seq, seq[1:], strict=False))
            verdicts.append(mono)
            table.add_row(
                str(i + 1),
                str(len(seq)),
                " -> ".join(map(str, seq)) or "(none)",
                "[green]yes[/green]" if mono else "[yellow]no[/yellow]",
            )
        console.print(table)

        if all(verdicts):
            console.print(
                "\n[green]VERDICT: cumulative within the turn -> use accounting_mode "
                '"last_wins".[/green]'
            )
        else:
            console.print(
                "\n[yellow]VERDICT: counters reset mid-turn -> use accounting_mode "
                '"delta_sum".[/yellow]'
            )
        console.print(
            '[dim]"auto" mode already applies this rule per turn; set it explicitly '
            "if you want determinism.[/dim]"
        )

    asyncio.run(run())


@app.command("calibrate-tools")
def calibrate_tools(config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c")) -> None:
    """Test whether a resumed session honours a CHANGED tool list.

    Precedence here is undocumented, which is why adaptive re-curation ships
    disabled. This probes it directly.
    """
    cfg = _load(config)
    _setup_logging(cfg)
    cfg.model.response_modalities = ["TEXT"]

    async def run() -> None:
        from google.genai import types

        from .live.connect_config import build_live_config

        client = build_client(cfg)

        def tool(name: str) -> Any:
            return types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=name,
                        description=f"Diagnostic probe {name}. Call it when asked to run {name}.",
                        parameters_json_schema={"type": "object", "properties": {}},
                    )
                ]
            )

        handle: str | None = None
        base = build_live_config(cfg, None)
        base.tools = [tool("probe_alpha")]
        async with client.aio.live.connect(model=cfg.model.name, config=base) as session:
            await session.send_realtime_input(text="Please run probe_alpha now.")
            async for message in session.receive():
                update = getattr(message, "session_resumption_update", None)
                if update is not None and getattr(update, "resumable", False):
                    handle = getattr(update, "new_handle", None)
                content = getattr(message, "server_content", None)
                if content is not None and getattr(content, "turn_complete", False):
                    break

        if not handle:
            console.print(
                "[yellow]No resumption handle was issued; cannot test. "
                "Ensure session.resumption.enabled is true.[/yellow]"
            )
            return

        swapped = build_live_config(cfg, None, resumption_handle=handle)
        swapped.tools = [tool("probe_beta")]
        called: list[str] = []
        async with client.aio.live.connect(model=cfg.model.name, config=swapped) as session:
            await session.send_realtime_input(
                text="Please run probe_beta now. If probe_beta is unavailable, say UNAVAILABLE."
            )
            async for message in session.receive():
                call = getattr(message, "tool_call", None)
                if call is not None:
                    called += [fc.name for fc in call.function_calls]
                    await session.send_tool_response(
                        function_responses=[
                            types.FunctionResponse(id=fc.id, name=fc.name, response={"ok": True})
                            for fc in call.function_calls
                        ]
                    )
                content = getattr(message, "server_content", None)
                if content is not None and getattr(content, "turn_complete", False):
                    break

        console.print(f"\nTools called after resume: {called or '(none)'}")
        if "probe_beta" in called:
            console.print(
                "[green]VERDICT: the new tool list IS honoured on resume. "
                "tools.adaptive.enabled is safe to turn on.[/green]"
            )
        else:
            console.print(
                "[yellow]VERDICT: the new tool list was NOT honoured. "
                "Keep tools.adaptive.enabled = false and reconnect without a handle "
                "when the tool set changes.[/yellow]"
            )

    asyncio.run(run())


# --------------------------------------------------------------------------- runtime


@app.command()
def selftest(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    prompt: str = typer.Option(
        "What time is it right now? Use your tools, then answer in one sentence.",
        "--prompt",
    ),
) -> None:
    """Headless TEXT session that exercises the tool loop and token reporting."""
    cfg = _load(config)
    _setup_logging(cfg)
    cfg.model.response_modalities = ["TEXT"]

    async def run() -> None:
        client = build_client(cfg)
        async with ToolPipeline(cfg, client) as pipeline:
            result = await pipeline.build()
            sink = CollectingSink()
            runner = LiveSessionRunner(cfg, client, result.registry, sink=sink)
            task = asyncio.create_task(runner.run())
            await asyncio.sleep(1.5)
            await runner.uplink.text(prompt)

            for _ in range(120):
                await asyncio.sleep(0.5)
                if sink.of_kind("usage_turn"):
                    break
                if task.done():
                    # The session died. Waiting out the remaining timeout for a
                    # usage event that can never arrive helps nobody.
                    break

            await runner.close()
            task.cancel()
            # Teardown only: whatever the cancelled task raises is irrelevant
            # to the result we already collected.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

            errors = sink.of_kind("error")
            if errors:
                console.print(f"\n[red]Session failed:[/red] {errors[0]['message']}")
                if errors[0].get("hint"):
                    console.print(f"\n{errors[0]['hint']}")
                raise typer.Exit(code=1)

            text = "".join(e["text"] for e in sink.of_kind("text"))
            console.print(f"\n[bold]Response:[/bold] {text or '(none)'}")
            for call in sink.of_kind("tool_call"):
                console.print(f"[bold]Tool called:[/bold] {', '.join(call['names'])}")
            for usage in sink.of_kind("usage_turn"):
                console.print_json(json.dumps(usage))
            if not sink.of_kind("usage_turn"):
                console.print("[yellow]No usage_turn event was produced.[/yellow]")

    asyncio.run(run())


@app.command("usage-report")
def usage_report(
    path: str = typer.Argument("logs/usage.jsonl"),
) -> None:
    """Analyse a recorded usage log offline."""
    log_path = Path(path)
    if not log_path.exists():
        console.print(f"[red]No usage log at {log_path}[/red]")
        raise typer.Exit(1)

    turns = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line.strip() and json.loads(line).get("kind") == "turn"
    ]
    if not turns:
        console.print("[yellow]No committed turns in the log.[/yellow]")
        return

    table = Table("turn", "prompt", "response", "thoughts", "total", "rent %", "mode")
    total = 0
    for rec in turns:
        s = rec["scalars"]
        total += s["total"]
        rent = s["prompt"] / s["total"] if s["total"] else 0
        table.add_row(
            str(rec["turn"] + 1),
            f"{s['prompt']:,}",
            f"{s['response']:,}",
            f"{s['thoughts']:,}",
            f"{s['total']:,}",
            f"{rent:.0%}",
            rec.get("accounting_mode", "?"),
        )
    console.print(table)

    prompts = [r["scalars"]["prompt"] for r in turns]
    console.print(f"\n[bold]Session total:[/bold] {total:,} tokens over {len(turns)} turns")
    console.print(
        f"[bold]Context grew[/bold] {prompts[0]:,} -> {prompts[-1]:,} tokens per turn"
    )
    drops = sum(1 for a, b in zip(prompts, prompts[1:], strict=False) if b < a - 256)
    console.print(f"[bold]Inferred compression events:[/bold] {drops}")


@app.command()
def serve(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    ssl_certfile: str | None = typer.Option(None, "--ssl-certfile"),
    ssl_keyfile: str | None = typer.Option(None, "--ssl-keyfile"),
) -> None:
    """Run the web console."""
    import uvicorn

    cfg = _load(config)
    _setup_logging(cfg)
    from .server.app import create_app

    bind_host = host or cfg.server.host
    bind_port = port or cfg.server.port
    scheme = "https" if ssl_certfile else "http"

    console.print(f"\n[bold]Open {scheme}://localhost:{bind_port}[/bold]")
    if scheme == "http":
        console.print(
            "[yellow]Microphone note:[/yellow] Chrome only grants mic access on a secure "
            "origin. If you are on a remote host, forward the port first:\n"
            f"  [bold]ssh -L {bind_port}:localhost:{bind_port} <host>[/bold]\n"
            "then open localhost (not the remote hostname).\n"
        )

    uvicorn.run(
        create_app(cfg),
        host=bind_host,
        port=bind_port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        log_level=cfg.app.log_level.lower(),
    )


if __name__ == "__main__":
    app()
