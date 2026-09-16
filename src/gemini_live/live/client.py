"""Vertex AI client construction with automatic gcloud CLI credential fallback."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from typing import Any

import google.auth
import google.auth.credentials

from ..settings.schema import AppConfig

log = logging.getLogger(__name__)


class _GcloudCliCredentials(google.auth.credentials.Credentials):
    """Self-refreshing credentials backed by ``gcloud auth print-access-token``.

    Cloudtop and multi-account dev setups often have an active ``gcloud`` CLI
    account (e.g. an altostrat admin) that has Vertex AI permissions on the
    target project while ``application_default_credentials.json`` points to a
    different corporate identity. Using the active ``gcloud`` token with
    ``quota_project_id`` avoids 1008 Permission Denied drops on Live sockets.
    """

    def __init__(self, quota_project_id: str | None = None) -> None:
        super().__init__()
        self._quota_project_id = quota_project_id
        self.refresh(None)

    @property
    def quota_project_id(self) -> str | None:
        return self._quota_project_id

    def refresh(self, request: Any) -> None:  # noqa: ARG002
        gcloud_bin = shutil.which("gcloud")
        if not gcloud_bin:
            raise RuntimeError("gcloud binary not found on PATH")
        token = subprocess.check_output(
            [gcloud_bin, "auth", "print-access-token"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        if not token:
            raise RuntimeError("gcloud auth print-access-token returned empty token")
        self.token = token
        # gcloud access tokens are valid for ~1 hour; refresh proactively after 45m.
        self.expiry = datetime.now(UTC).replace(tzinfo=None) + timedelta(minutes=45)


def _resolve_credentials(project: str | None) -> Any:
    """Prefer active gcloud CLI credentials on workstations; fall back to ADC."""
    # On Cloud Run (K_SERVICE set), use metadata server ADC directly.
    if not os.environ.get("K_SERVICE") and shutil.which("gcloud"):
        try:
            return _GcloudCliCredentials(quota_project_id=project)
        except Exception as exc:
            log.debug("gcloud CLI credentials unavailable (%s); falling back to ADC", exc)
    try:
        creds, _ = google.auth.default(quota_project_id=project)
        return creds
    except Exception:
        creds, _ = google.auth.default()
        return creds


def build_client(cfg: AppConfig) -> Any:
    """Create a ``genai.Client`` bound to Vertex AI."""
    from google import genai
    from google.genai import types

    http_options = (
        types.HttpOptions(api_version=cfg.vertex.api_version)
        if cfg.vertex.api_version
        else None
    )
    credentials = _resolve_credentials(cfg.vertex.project)
    return genai.Client(
        vertexai=True,
        project=cfg.vertex.project,
        location=cfg.vertex.location,
        credentials=credentials,
        http_options=http_options,
    )
