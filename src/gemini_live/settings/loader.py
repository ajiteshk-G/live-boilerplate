"""YAML config loading with ``${ENV_VAR}`` interpolation and friendly errors."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from .schema import AppConfig

# ${VAR} or ${VAR:-default}
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(Exception):
    """Raised when a config file is missing, malformed, or fails validation."""


def _interpolate(node: Any) -> Any:
    """Recursively expand environment variables inside a parsed YAML tree.

    A string that is *entirely* one placeholder resolves to ``None`` when the
    variable is unset and no default was given. That matters: it lets Pydantic
    report "vertex.project is required" rather than silently accepting an empty
    string as a valid project ID.
    """
    if isinstance(node, str):
        whole = _VAR.fullmatch(node)
        if whole:
            return os.environ.get(whole.group(1), whole.group(2))
        return _VAR.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), node)
    if isinstance(node, dict):
        return {k: _interpolate(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate(v) for v in node]
    return node


def _render_error(exc: ValidationError, path: Path) -> str:
    lines = [f"Invalid configuration in {path}:", ""]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  {loc}: {err['msg']}")
    lines.append("")
    lines.append("Tip: unknown keys are rejected on purpose, so a typo shows up here")
    lines.append("rather than being silently ignored at runtime.")
    return "\n".join(lines)


def load_config(path: str | Path, *, env_file: str | Path | None = ".env") -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    if env_file and Path(env_file).exists():
        load_dotenv(env_file)
    else:
        load_dotenv()

    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {path}:\n{exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")

    try:
        return AppConfig.model_validate(_interpolate(raw))
    except ValidationError as exc:
        raise ConfigError(_render_error(exc, path)) from exc
