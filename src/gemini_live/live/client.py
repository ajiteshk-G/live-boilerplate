"""Vertex AI client construction."""

from __future__ import annotations

from typing import Any

from ..settings.schema import AppConfig


def build_client(cfg: AppConfig) -> Any:
    """Create a ``genai.Client`` bound to Vertex AI.

    Authentication uses Application Default Credentials, so run
    ``gcloud auth application-default login`` first.
    """
    from google import genai
    from google.genai import types

    http_options = (
        types.HttpOptions(api_version=cfg.vertex.api_version)
        if cfg.vertex.api_version
        else None
    )
    return genai.Client(
        vertexai=True,
        project=cfg.vertex.project,
        location=cfg.vertex.location,
        http_options=http_options,
    )
