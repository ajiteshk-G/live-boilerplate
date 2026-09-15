"""Pydantic models describing the boilerplate's YAML configuration.

Every knob the application exposes lives here. ``extra="forbid"`` is set on the
root model on purpose: in a config-driven application a silently-ignored
misspelled key is the most expensive class of bug, so we fail loudly instead.
"""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .voices import (
    LIVE_VOICES,
    canonical_voice,
    is_supported_language,
    resolve_client_locale,
)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- app


class AppSection(_Base):
    name: str = "gemini-live-boilerplate"
    log_level: str = "INFO"


class VertexSection(_Base):
    project: str | None = None
    location: str = "us-central1"
    api_version: str | None = "v1beta1"

    @model_validator(mode="after")
    def _require_project(self) -> VertexSection:
        if not self.project:
            raise ValueError(
                "vertex.project is required. Set GOOGLE_CLOUD_PROJECT in your environment "
                "or .env file, or hardcode it in the config."
            )
        return self


ResponseModality = Literal["AUDIO", "TEXT"]


def _default_modalities() -> list[ResponseModality]:
    return ["AUDIO"]


class ModelSection(_Base):
    name: str = "gemini-live-2.5-flash-native-audio"
    response_modalities: list[ResponseModality] = Field(default_factory=_default_modalities)
    system_instruction: str = "You are a helpful, concise voice assistant."
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None


class SpeechSection(_Base):
    """Voice (timbre) and language (accent) of the spoken response.

    These are independent: a voice is not tied to a locale. The Indian accent
    comes from ``language_code``, not from ``voice_name``.
    """

    voice_name: str | None = "Kore"

    language_mode: Literal["follow_user", "pinned"] = "follow_user"
    """Whether the agent adapts to the user's language or stays in one.

    ``follow_user`` -- reply in whatever language the user speaks, switching
    mid-conversation as they do. ``language_code`` is then only the language the
    call opens in, and the fallback when the user's language is unclear.

    ``pinned`` -- always speak ``language_code``, whatever the user does.
    """

    language_code: str | None = "en-IN"

    use_client_locale: bool = True
    """Let the browser's locale choose the opening language.

    The web client passes ``navigator.language``; if it names a language the
    Live API supports it replaces ``language_code`` for that session. An
    unsupported or missing locale falls back to ``language_code``.
    """

    enforce_language_in_system_instruction: bool = True
    """Also state the language rule in the system instruction.

    Required for native-audio models, which ignore ``language_code`` outright.
    With ``language_mode: follow_user`` this is also what licenses the model to
    switch languages, so turning it off leaves language entirely to the model.
    """

    @field_validator("voice_name")
    @classmethod
    def _known_voice(cls, v: str | None) -> str | None:
        if v is None:
            return None
        canonical = canonical_voice(v)
        if canonical is None:
            raise ValueError(
                f"speech.voice_name {v!r} is not a Live API voice. "
                f"Run `glive voices` for the list of {len(LIVE_VOICES)}."
            )
        return canonical

    @field_validator("language_code")
    @classmethod
    def _known_language(cls, v: str | None) -> str | None:
        if v is None:
            return None
        code = v.strip().replace("_", "-")
        if not is_supported_language(code):
            raise ValueError(
                f"speech.language_code {v!r} is not a Live API language. "
                "Run `glive voices` for the list (e.g. en-IN, hi-IN, en-US)."
            )
        return code


class TranscriptionSection(_Base):
    input: bool = True
    output: bool = True


class VadSection(_Base):
    enabled: bool = True
    start_sensitivity: str = "START_SENSITIVITY_HIGH"
    end_sensitivity: str = "END_SENSITIVITY_LOW"
    prefix_padding_ms: int | None = 20
    silence_duration_ms: int | None = 400


class ThinkingSection(_Base):
    level: Literal["minimal", "low", "medium", "high"] | None = None
    budget: int | None = None
    include_thoughts: bool = False


class CompressionSection(_Base):
    enabled: bool = True
    trigger_tokens: int = 25600
    sliding_window_target_tokens: int = 12800


class ResumptionSection(_Base):
    enabled: bool = True


class ReconnectSection(_Base):
    """How hard to try when a live socket drops mid-call.

    Reconnects only happen when `resumption.enabled` is true; without a
    resumption handle a reconnect would silently lose the conversation.
    """

    max_attempts: int = 3
    """Consecutive failures tolerated. The counter resets once a reconnected
    session actually produces output, so a long call is not capped at this
    many drops in total."""

    initial_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 8.0


class SessionSection(_Base):
    context_window_compression: CompressionSection = CompressionSection()
    resumption: ResumptionSection = ResumptionSection()
    reconnect: ReconnectSection = ReconnectSection()


class MediaSection(_Base):
    resolution: str | None = "MEDIA_RESOLUTION_LOW"


# ------------------------------------------------------------------------- usage


class UsageAlertsSection(_Base):
    warn_prompt_tokens_per_turn: int | None = 30000
    warn_rent_ratio: float | None = 0.90
    warn_session_total: int | None = 500000


class PricingSection(_Base):
    """Optional per-1k-token rates. Disabled by default; stale rates are worse than none."""

    text_input_per_1k: float = 0.0
    audio_input_per_1k: float = 0.0
    text_output_per_1k: float = 0.0
    audio_output_per_1k: float = 0.0
    thinking_per_1k: float = 0.0
    currency: str = "USD"


class UsageSection(_Base):
    enabled: bool = True
    accounting_mode: Literal["auto", "last_wins", "delta_sum"] = "auto"
    log_path: str | None = "./logs/usage.jsonl"
    log_raw_snapshots: bool = True
    show_in_ui: bool = True
    alerts: UsageAlertsSection = UsageAlertsSection()
    pricing: PricingSection | None = None


# ------------------------------------------------------------------------- tools


class McpServerConfig(_Base):
    name: str
    enabled: bool = True
    transport: Literal["stdio", "streamable_http", "sse"] = "stdio"
    # stdio
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    # http / sse
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    # common
    startup_timeout_s: float = 20.0
    tools: list[str] = Field(default_factory=list)
    """Per-server allow-list. Empty means "all tools from this server"."""

    @model_validator(mode="after")
    def _check_transport(self) -> McpServerConfig:
        if self.transport == "stdio" and not self.command:
            raise ValueError(f"mcp server {self.name!r}: transport 'stdio' requires 'command'")
        if self.transport in ("streamable_http", "sse") and not self.url:
            raise ValueError(
                f"mcp server {self.name!r}: transport {self.transport!r} requires 'url'"
            )
        return self


class CurationSection(_Base):
    mode: Literal["auto", "manual"] = "auto"
    purpose: str = ""
    max_tools: int = 20
    budget_tokens: int = 6000
    min_score: float = 0.25
    dedupe_threshold: float = 0.92
    max_desc_chars: int = 200
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    cost_model: str = "gemini-2.5-flash"
    """Non-Live model used purely to measure declaration token cost via count_tokens."""

    @field_validator("max_tools")
    @classmethod
    def _cap(cls, v: int) -> int:
        if v > 128:
            raise ValueError("tools.curation.max_tools cannot exceed the API limit of 128")
        return v


class BuiltinToolConfig(_Base):
    enabled: bool = True


class ToolsSection(_Base):
    curation: CurationSection = CurationSection()
    pinned: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    allow: list[str] = Field(default_factory=list)
    builtins: dict[str, BuiltinToolConfig] = Field(default_factory=dict)
    mcp: list[McpServerConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _manual_needs_allow(self) -> ToolsSection:
        if self.curation.mode == "manual" and not self.allow:
            raise ValueError(
                "tools.curation.mode == 'manual' requires a non-empty tools.allow list. "
                "Use mode 'auto' for automatic curation."
            )
        return self


# ------------------------------------------------------------------------ search


class GoogleSearchSection(_Base):
    enabled: bool = True
    exclude_domains: list[str] = Field(default_factory=list)


class SearchDomains(_Base):
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)

    @field_validator("allow", "deny", mode="after")
    @classmethod
    def _normalize(cls, v: list[str]) -> list[str]:
        return [normalize_domain(d) for d in v if d and d.strip()]


class SearchSection(_Base):
    google_search: GoogleSearchSection = GoogleSearchSection()
    domains: SearchDomains = SearchDomains()
    inject_domain_rules_into_system_instruction: bool = True
    log_violations: str | None = "./logs/violations.jsonl"


class ServerSection(_Base):
    host: str = "127.0.0.1"
    port: int = 8080
    cors_origins: list[str] = Field(default_factory=list)
    """Empty means "derive from `port`" - see `_default_cors`. Set explicitly
    to allow other origins."""

    @model_validator(mode="after")
    def _default_cors(self) -> ServerSection:
        # Hardcoding a port here would silently break CORS for anyone who
        # changes `port`, which is a confusing failure to debug.
        if not self.cors_origins:
            self.cors_origins = [
                f"http://localhost:{self.port}",
                f"http://127.0.0.1:{self.port}",
            ]
        return self


# --------------------------------------------------------------------------- root


class AppConfig(_Base):
    app: AppSection = AppSection()
    vertex: VertexSection
    model: ModelSection = ModelSection()
    speech: SpeechSection = SpeechSection()
    transcription: TranscriptionSection = TranscriptionSection()
    vad: VadSection = VadSection()
    thinking: ThinkingSection = ThinkingSection()
    session: SessionSection = SessionSection()
    media: MediaSection = MediaSection()
    usage: UsageSection = UsageSection()
    tools: ToolsSection = ToolsSection()
    search: SearchSection = SearchSection()
    server: ServerSection = ServerSection()

    @model_validator(mode="after")
    def _coherence(self) -> AppConfig:
        if len(self.model.response_modalities) != 1:
            raise ValueError(
                "The Live API supports exactly one response modality per session "
                f"(got {self.model.response_modalities!r}). Use ['AUDIO'] or ['TEXT']."
            )
        if self.thinking.level and self.thinking.budget is not None:
            raise ValueError(
                "Set thinking.level (Gemini 3.x models) OR thinking.budget (2.5 models), not both."
            )
        if self.search.google_search.enabled and self.search.domains.allow:
            # Not an error, but the user must understand what they are getting.
            object.__setattr__(self, "_allowlist_is_advisory", True)
        return self

    def curation_purpose(self) -> str:
        """The text the curator ranks tools against."""
        return self.tools.curation.purpose.strip() or self.model.system_instruction.strip()

    def for_client_locale(self, raw_locale: str | None) -> AppConfig:
        """A copy of this config that opens in the caller's language.

        Returns ``self`` unchanged when the feature is off, the locale is
        missing or unsupported, or it already matches. Each connection gets its
        own copy so one user's language never leaks into another's session.
        """
        if not self.speech.use_client_locale:
            return self
        resolved = resolve_client_locale(raw_locale)
        if resolved is None or resolved == self.speech.language_code:
            return self
        clone = self.model_copy(deep=True)
        clone.speech.language_code = resolved
        return clone



def normalize_domain(raw: str) -> str:
    """``https://Cloud.Google.com/vertex-ai/`` -> ``cloud.google.com``."""
    raw = raw.strip()
    host = urlparse(raw if "//" in raw else f"//{raw}").hostname or raw
    return host.lower().strip(".").removeprefix("www.")


def redact(cfg: AppConfig) -> dict[str, Any]:
    """Config as a dict with obvious secrets masked, for display."""
    data = cfg.model_dump(mode="json")
    for server in data.get("tools", {}).get("mcp", []):
        for key in list(server.get("headers", {})):
            server["headers"][key] = "***"
        for key in list(server.get("env", {})):
            server["env"][key] = "***"
    return data
