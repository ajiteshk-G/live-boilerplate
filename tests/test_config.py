"""Config loading: ${VAR} interpolation, typo rejection, and coherence rules."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gemini_live.settings.capabilities import capabilities_for, filter_for_model
from gemini_live.settings.loader import ConfigError, load_config
from gemini_live.settings.schema import AppConfig, redact

MINIMAL = "vertex:\n  project: p\n"


@pytest.fixture
def write_config(tmp_path):
    def _write(text: str) -> str:
        path = tmp_path / "config.yaml"
        path.write_text(text)
        return str(path)

    return _write


# ------------------------------------------------------------ interpolation


def test_env_var_is_interpolated(write_config, monkeypatch):
    monkeypatch.setenv("MY_PROJECT", "proj-123")
    cfg = load_config(write_config("vertex:\n  project: ${MY_PROJECT}\n"), env_file=None)
    assert cfg.vertex.project == "proj-123"


def test_default_is_used_when_var_unset(write_config, monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    cfg = load_config(
        write_config("vertex:\n  project: p\n  location: ${NOPE:-europe-west4}\n"), env_file=None
    )
    assert cfg.vertex.location == "europe-west4"


def test_env_var_wins_over_default(write_config, monkeypatch):
    monkeypatch.setenv("LOC", "asia-south1")
    cfg = load_config(
        write_config("vertex:\n  project: p\n  location: ${LOC:-europe-west4}\n"), env_file=None
    )
    assert cfg.vertex.location == "asia-south1"


def test_embedded_var_is_substituted_in_place(write_config, monkeypatch):
    monkeypatch.setenv("TOKEN", "s3cret")
    cfg = load_config(
        write_config(
            "vertex:\n  project: p\n"
            "tools:\n  mcp:\n"
            "    - name: docs\n      transport: streamable_http\n"
            "      url: https://x/mcp\n"
            "      headers:\n        Authorization: Bearer ${TOKEN}\n"
        ),
        env_file=None,
    )
    assert cfg.tools.mcp[0].headers["Authorization"] == "Bearer s3cret"


def test_unset_whole_value_var_reports_a_required_field(write_config, monkeypatch):
    """It must resolve to None, not the empty string.

    An empty string would sail past validation and fail much later as a baffling
    auth error against project "".
    """
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(ConfigError) as exc:
        load_config(write_config("vertex:\n  project: ${GOOGLE_CLOUD_PROJECT}\n"), env_file=None)
    assert "vertex.project is required" in str(exc.value)


# ------------------------------------------------------------------- errors


def test_missing_file_is_a_clear_error():
    with pytest.raises(ConfigError, match="not found"):
        load_config("/nonexistent/config.yaml")


def test_malformed_yaml_is_reported(write_config):
    with pytest.raises(ConfigError, match="Could not parse YAML"):
        load_config(write_config("vertex:\n  project: [unclosed\n"), env_file=None)


def test_non_mapping_root_is_rejected(write_config):
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(write_config("- a\n- b\n"), env_file=None)


def test_typo_in_a_key_is_rejected_loudly(write_config):
    """The entire reason for extra='forbid'. A silently ignored misspelling is
    the most expensive class of config bug."""
    with pytest.raises(ConfigError) as exc:
        load_config(write_config("vertex:\n  project: p\n  locaton: us-central1\n"), env_file=None)
    assert "locaton" in str(exc.value)


def test_error_message_explains_the_strictness(write_config):
    with pytest.raises(ConfigError) as exc:
        load_config(write_config("vertex:\n  project: p\nbogus: 1\n"), env_file=None)
    assert "typo" in str(exc.value)


# -------------------------------------------------------------- coherence


def test_non_audio_response_modality_is_rejected():
    """Gemini 3.8 Live models are native-audio and only support ['AUDIO']."""
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {"vertex": {"project": "p"}, "model": {"response_modalities": ["TEXT"]}}
        )


def test_thinking_level_minimal_is_rejected():
    """Gemini 3.8 Live Extended Thinking only supports low, medium, and high."""
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {"vertex": {"project": "p"}, "thinking": {"level": "minimal"}}
        )


def test_manual_curation_requires_an_allow_list():
    with pytest.raises(ValueError, match="requires a non-empty tools.allow"):
        AppConfig.model_validate(
            {"vertex": {"project": "p"}, "tools": {"curation": {"mode": "manual"}}}
        )


def test_max_tools_above_the_api_cap_is_rejected():
    with pytest.raises(ValueError, match="128"):
        AppConfig.model_validate(
            {"vertex": {"project": "p"}, "tools": {"curation": {"max_tools": 200}}}
        )


def test_custom_vocabulary_above_the_api_cap_is_rejected():
    with pytest.raises(ValueError, match="1000"):
        AppConfig.model_validate(
            {
                "vertex": {"project": "p"},
                "transcription": {"custom_vocabulary": [f"term-{i}" for i in range(1001)]},
            }
        )


def test_stdio_server_requires_a_command():
    with pytest.raises(ValueError, match="requires 'command'"):
        AppConfig.model_validate(
            {"vertex": {"project": "p"}, "tools": {"mcp": [{"name": "x", "transport": "stdio"}]}}
        )


def test_http_server_requires_a_url():
    with pytest.raises(ValueError, match="requires 'url'"):
        AppConfig.model_validate(
            {
                "vertex": {"project": "p"},
                "tools": {"mcp": [{"name": "x", "transport": "streamable_http"}]},
            }
        )


# --------------------------------------------------------------- behaviour


def test_domains_are_normalized_at_load_time():
    cfg = AppConfig.model_validate(
        {
            "vertex": {"project": "p"},
            "search": {"domains": {"allow": ["https://GitHub.com/x", "www.Example.COM"]}},
        }
    )
    assert cfg.search.domains.allow == ["github.com", "example.com"]


def test_curation_purpose_falls_back_to_the_system_instruction():
    cfg = AppConfig.model_validate(
        {"vertex": {"project": "p"}, "model": {"system_instruction": "Help with billing."}}
    )
    assert cfg.curation_purpose() == "Help with billing."


def test_explicit_purpose_wins():
    cfg = AppConfig.model_validate(
        {
            "vertex": {"project": "p"},
            "model": {"system_instruction": "Be nice."},
            "tools": {"curation": {"purpose": "Query the docs corpus."}},
        }
    )
    assert cfg.curation_purpose() == "Query the docs corpus."


def test_redact_masks_credentials():
    cfg = AppConfig.model_validate(
        {
            "vertex": {"project": "p"},
            "tools": {
                "mcp": [
                    {
                        "name": "docs",
                        "transport": "streamable_http",
                        "url": "https://x/mcp",
                        "headers": {"Authorization": "Bearer real-secret"},
                        "env": {"API_KEY": "real-key"},
                    }
                ]
            },
        }
    )
    data = redact(cfg)
    server = data["tools"]["mcp"][0]

    assert server["headers"]["Authorization"] == "***"
    assert server["env"]["API_KEY"] == "***"
    assert "real-secret" not in str(data)
    assert "real-key" not in str(data)
    assert server["url"] == "https://x/mcp"  # non-secret detail preserved


# ------------------------------------------------------------ capabilities


@pytest.mark.parametrize(
    ("model", "style", "requires_non_blocking"),
    [
        ("gemini-3.8-live", None, False),
        ("gemini-3.8-live-extended-thinking", "level", True),
    ],
)
def test_thinking_style_by_model_family(model, style, requires_non_blocking):
    caps = capabilities_for(model)
    assert caps.thinking_style == style
    assert caps.requires_non_blocking_tools is requires_non_blocking


def test_default_model_is_gemini_3_8_live():
    cfg = AppConfig.model_validate({"vertex": {"project": "p"}})
    assert cfg.model.name == "gemini-3.8-live"


def test_gemini_3_8_live_drops_thinking_config_with_warning():
    cfg = AppConfig.model_validate(
        {
            "vertex": {"project": "p"},
            "model": {"name": "gemini-3.8-live"},
            "thinking": {"level": "low"},
        }
    )
    filtered, warnings = filter_for_model(cfg)

    assert filtered.thinking.level is None
    assert len(warnings) == 1
    assert "does not support thinking config" in warnings[0]


def test_gemini_3_8_live_extended_thinking_coerces_blocking_tools():
    cfg = AppConfig.model_validate(
        {
            "vertex": {"project": "p"},
            "model": {"name": "gemini-3.8-live-extended-thinking"},
            "thinking": {"level": "low"},
            "tools": {"behavior": "BLOCKING"},
        }
    )
    filtered, warnings = filter_for_model(cfg)

    assert filtered.thinking.level == "low"
    assert filtered.tools.behavior == "NON_BLOCKING"
    assert len(warnings) == 1


def test_filtering_does_not_mutate_the_original_config():
    cfg = AppConfig.model_validate(
        {
            "vertex": {"project": "p"},
            "model": {"name": "gemini-3.8-live"},
            "thinking": {"level": "low"},
        }
    )
    filter_for_model(cfg)
    assert cfg.thinking.level == "low"


def test_compatible_config_produces_no_warnings():
    cfg = AppConfig.model_validate(
        {
            "vertex": {"project": "p"},
            "model": {"name": "gemini-3.8-live"},
            "speech": {"language_code": "en-IN"},
        }
    )
    filtered, warnings = filter_for_model(cfg)

    assert warnings == []
    assert filtered.speech.language_code == "en-IN"


def test_shipped_reference_config_is_valid(monkeypatch):
    """config/config.yaml must stay in sync with the schema and emit zero warnings."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    cfg = load_config("config/config.yaml", env_file=None)
    assert cfg.vertex.project == "test-project"
    _, warnings = filter_for_model(cfg)
    assert warnings == []


def test_shipped_minimal_config_is_valid(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    cfg = load_config("config/config.minimal.yaml", env_file=None)
    assert cfg.model.response_modalities == ["AUDIO"]


# ------------------------------------------------------------------ cors / ports


def test_cors_origins_default_follows_the_configured_port():
    """A hardcoded port here would silently break CORS for anyone who moves the
    server off 8080, which is a miserable failure to diagnose."""
    cfg = AppConfig.model_validate({"vertex": {"project": "p"}, "server": {"port": 9111}})

    assert cfg.server.cors_origins == [
        "http://localhost:9111",
        "http://127.0.0.1:9111",
    ]


def test_explicit_cors_origins_are_left_alone():
    cfg = AppConfig.model_validate(
        {
            "vertex": {"project": "p"},
            "server": {"port": 9111, "cors_origins": ["https://example.com"]},
        }
    )

    assert cfg.server.cors_origins == ["https://example.com"]


# ------------------------------------------------------- no aspirational knobs

# These keys were once declared and documented (or belonged to legacy 2.5 models)
# and are now rejected by `extra="forbid"`.


@pytest.mark.parametrize(
    "payload",
    [
        {"session": {"max_duration_minutes": 30}},
        {"media": {"enable_video_input": True}},
        {"tools": {"adaptive": {"enabled": True}}},
        {"usage": {"report": {"per_turn": True}}},
        {"search": {"enforcement": "strict_buffered"}},
        {"thinking": {"budget": 1024}},
        {"speech": {"enforce_language_in_system_instruction": True}},
    ],
)
def test_removed_dead_keys_are_rejected_not_silently_ignored(payload):
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"vertex": {"project": "p"}, **payload})
