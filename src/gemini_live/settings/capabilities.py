"""Per-model capability matrix.

Live models differ in which config fields they accept. Rather than hard-failing
on a config that is valid for a sibling model, unsupported fields are dropped
with a warning so one config file can be shared across models.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from .schema import AppConfig


@dataclass(frozen=True)
class Capabilities:
    thinking_style: str | None  # "level" (3.x) | "budget" (2.5) | None
    supports_affective_dialog: bool
    supports_proactivity: bool
    supports_language_code: bool


_MATRIX: list[tuple[str, Capabilities]] = [
    # Most specific prefixes first.
    (
        "gemini-3",
        Capabilities(
            thinking_style="level",
            supports_affective_dialog=False,
            supports_proactivity=False,
            supports_language_code=False,
        ),
    ),
    (
        "gemini-live-2.5",
        Capabilities(
            thinking_style="budget",
            supports_affective_dialog=True,
            supports_proactivity=True,
            supports_language_code=False,
        ),
    ),
    (
        "gemini-2.5",
        Capabilities(
            thinking_style="budget",
            supports_affective_dialog=True,
            supports_proactivity=True,
            supports_language_code=True,
        ),
    ),
]

_DEFAULT = Capabilities(
    thinking_style="budget",
    supports_affective_dialog=False,
    supports_proactivity=False,
    supports_language_code=True,
)


def capabilities_for(model_name: str) -> Capabilities:
    from dataclasses import replace

    caps = _DEFAULT
    for prefix, c in _MATRIX:
        if model_name.startswith(prefix):
            caps = c
            break
    if "native-audio" in model_name and caps.supports_language_code:
        caps = replace(caps, supports_language_code=False)
    return caps


def filter_for_model(cfg: AppConfig) -> tuple[AppConfig, list[str]]:
    """Return a copy of ``cfg`` with model-incompatible fields removed."""
    caps = capabilities_for(cfg.model.name)
    warnings: list[str] = []
    out = copy.deepcopy(cfg)

    if caps.thinking_style == "level" and out.thinking.budget is not None:
        warnings.append(
            f"{cfg.model.name} uses thinking.level, not thinking.budget; dropping "
            f"thinking.budget={out.thinking.budget}."
        )
        out.thinking.budget = None
    if caps.thinking_style == "budget" and out.thinking.level is not None:
        warnings.append(
            f"{cfg.model.name} uses thinking.budget, not thinking.level; dropping "
            f"thinking.level={out.thinking.level!r}."
        )
        out.thinking.level = None
    if caps.thinking_style is None and (out.thinking.level or out.thinking.budget is not None):
        warnings.append(f"{cfg.model.name} does not support thinking config; dropping it.")
        out.thinking.level = None
        out.thinking.budget = None

    if not caps.supports_language_code and out.speech.language_code:
        # The field itself is not accepted, but the requirement still is -- as a
        # rule in the system instruction, which is Google's documented
        # workaround for native-audio models.
        if out.speech.enforce_language_in_system_instruction:
            warnings.append(
                f"{cfg.model.name} detects language automatically and ignores "
                f"speech.language_code={out.speech.language_code!r}; enforcing it through "
                "the system instruction instead."
            )
        else:
            warnings.append(
                f"{cfg.model.name} ignores speech.language_code="
                f"{out.speech.language_code!r} and "
                "speech.enforce_language_in_system_instruction is false, so the spoken "
                "language is left entirely to the model."
            )
            out.speech.language_code = None

    return out, warnings
