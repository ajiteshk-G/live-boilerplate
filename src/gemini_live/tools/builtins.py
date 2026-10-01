"""Built-in local tools.

Kept deliberately small. They exist to give the boilerplate something useful out
of the box and to serve as a template for adding your own local function tools.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..settings.schema import ToolsSection
from .catalog import ToolCandidate


async def _get_current_time(args: dict[str, Any]) -> dict[str, Any]:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    tz_name = str((args or {}).get("timezone") or "UTC").strip() or "UTC"
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, KeyError, ValueError):
        tz = UTC  # type: ignore[assignment]
        tz_name = "UTC"
    now = datetime.now(tz)
    tz_label = now.tzname() or tz_name
    return {
        "iso8601": now.isoformat(),
        "human": f"{now.strftime('%A, %d %B %Y at %H:%M')} {tz_label}",
        "timezone": tz_name,
    }


_BUILTINS: dict[str, dict[str, Any]] = {
    "get_current_time": {
        "description": (
            "Get the current date and time (defaults to UTC, or specify an IANA timezone). "
            "Invocation Condition: Invoke this tool when the user asks for the current "
            "time, date, or day of the week."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        "Optional IANA timezone name (e.g. 'America/New_York', 'Asia/Kolkata')."
                    ),
                }
            },
        },
        "invoke": _get_current_time,
    },
}


def build_builtin_candidates(tools_cfg: ToolsSection) -> list[ToolCandidate]:
    out: list[ToolCandidate] = []
    for name, spec in _BUILTINS.items():
        cfg = tools_cfg.builtins.get(name)
        if cfg is None or not cfg.enabled:
            continue
        out.append(
            ToolCandidate(
                exposed_name=name,
                description=spec["description"],
                input_schema=dict(spec["schema"]),
                invoke=spec["invoke"],
                origin="builtin",
                raw_name=name,
            )
        )
    return out
