"""Per-model capability matrix for Gemini 3.8 Live models.

``gemini-3.8-live`` is optimized for low-latency dialogue and omits
``thinking_config``, while ``gemini-3.8-live-extended-thinking`` supports
``thinking.level`` (``low`` | ``medium`` | ``high``) and requires non-blocking
function calling (``behavior="NON_BLOCKING"``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from .schema import AppConfig


@dataclass(frozen=True)
class Capabilities:
    thinking_style: str | None  # "level" (3.8-live-extended-thinking) | None (3.8-live)
    default_tool_behavior: str = "NON_BLOCKING"
    requires_non_blocking_tools: bool = False


_MATRIX: list[tuple[str, Capabilities]] = [
    (
        "gemini-3.8-live-extended-thinking",
        Capabilities(
            thinking_style="level",
            default_tool_behavior="NON_BLOCKING",
            requires_non_blocking_tools=True,
        ),
    ),
    (
        "gemini-3.8-live",
        Capabilities(
            thinking_style=None,
            default_tool_behavior="NON_BLOCKING",
            requires_non_blocking_tools=False,
        ),
    ),
]

_DEFAULT = Capabilities(
    thinking_style=None,
    default_tool_behavior="NON_BLOCKING",
    requires_non_blocking_tools=False,
)


def capabilities_for(model_name: str) -> Capabilities:
    for prefix, c in _MATRIX:
        if model_name.startswith(prefix):
            return c
    return _DEFAULT


def filter_for_model(cfg: AppConfig) -> tuple[AppConfig, list[str]]:
    """Return a copy of ``cfg`` with model-incompatible fields removed."""
    caps = capabilities_for(cfg.model.name)
    warnings: list[str] = []
    out = copy.deepcopy(cfg)

    if caps.thinking_style is None and (
        out.thinking.level or out.thinking.include_thoughts
    ):
        warnings.append(f"{cfg.model.name} does not support thinking config; dropping it.")
        out.thinking.level = None
        out.thinking.include_thoughts = False

    if caps.requires_non_blocking_tools and out.tools.behavior == "BLOCKING":
        warnings.append(
            f"{cfg.model.name} requires asynchronous function calling "
            "(behavior='NON_BLOCKING'); overriding tools.behavior='BLOCKING'."
        )
        out.tools.behavior = "NON_BLOCKING"

    return out, warnings
