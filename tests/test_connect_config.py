"""Building the LiveConnectConfig.

Pure function, so it is fully testable offline. The recurring failure mode it
guards against is sending an explicit None for an unset field, which the backend
rejects with an unhelpful error.
"""

from __future__ import annotations

from gemini_live.live.connect_config import build_live_config, build_system_instruction
from gemini_live.search.domain_policy import DomainPolicy
from gemini_live.settings.schema import AppConfig
from gemini_live.tools.catalog import ToolCandidate
from gemini_live.tools.registry import ToolRegistry


def make_cfg(**overrides) -> AppConfig:
    base = {"vertex": {"project": "p"}}
    base.update(overrides)
    return AppConfig.model_validate(base)


def make_registry(*names: str) -> ToolRegistry:
    async def noop(args):
        return {}

    return ToolRegistry(
        [
            ToolCandidate(
                exposed_name=n,
                description="d",
                input_schema={"type": "object", "properties": {}},
                invoke=noop,
                origin="mcp:test",
            )
            for n in names
        ]
    )


# ------------------------------------------------------- system instruction


def test_domain_rules_are_injected():
    cfg = make_cfg(
        model={"system_instruction": "Be helpful."},
        search={"domains": {"allow": ["cloud.google.com"]}},
    )
    policy = DomainPolicy(cfg.search.domains.allow, cfg.search.domains.deny)
    text = build_system_instruction(cfg, policy)

    assert "Be helpful." in text
    assert "cloud.google.com" in text


def test_injection_can_be_disabled():
    cfg = make_cfg(
        model={"system_instruction": "Be helpful."},
        search={
            "domains": {"allow": ["cloud.google.com"]},
            "inject_domain_rules_into_system_instruction": False,
        },
    )
    policy = DomainPolicy(cfg.search.domains.allow, cfg.search.domains.deny)
    assert build_system_instruction(cfg, policy) == "Be helpful."


def test_no_rules_appended_when_no_domains_configured():
    cfg = make_cfg(model={"system_instruction": "Be helpful."})
    assert build_system_instruction(cfg, DomainPolicy()) == "Be helpful."


# -------------------------------------------------------------- core fields


def test_response_modality_is_mapped():
    live = build_live_config(make_cfg())
    assert [str(m.value) for m in live.response_modalities] == ["AUDIO"]


def test_unset_sampling_fields_are_omitted_not_none():
    """Sending an explicit null is a reliable way to get a 400."""
    live = build_live_config(make_cfg())
    assert live.temperature is None
    assert live.top_p is None
    assert live.max_output_tokens is None


def test_configured_sampling_fields_are_passed_through():
    live = build_live_config(make_cfg(model={"temperature": 0.4, "top_p": 0.9}))
    assert live.temperature == 0.4
    assert live.top_p == 0.9


def test_voice_is_configured():
    live = build_live_config(make_cfg(speech={"voice_name": "Puck"}))
    assert live.speech_config.voice_config.prebuilt_voice_config.voice_name == "Puck"


def test_no_speech_config_when_voice_is_null():
    assert build_live_config(make_cfg(speech={"voice_name": None})).speech_config is None


def test_transcription_toggles():
    both = build_live_config(make_cfg())
    assert both.input_audio_transcription is not None
    assert both.output_audio_transcription is not None

    neither = build_live_config(make_cfg(transcription={"input": False, "output": False}))
    assert neither.input_audio_transcription is None
    assert neither.output_audio_transcription is None


# --------------------------------------------------------------------- vad


def test_vad_enabled_carries_sensitivities():
    live = build_live_config(make_cfg())
    vad = live.realtime_input_config.automatic_activity_detection

    assert vad.disabled is False
    assert vad.silence_duration_ms == 400
    assert vad.prefix_padding_ms == 20


def test_vad_disabled_sets_only_the_disable_flag():
    live = build_live_config(make_cfg(vad={"enabled": False}))
    vad = live.realtime_input_config.automatic_activity_detection

    assert vad.disabled is True
    # Sensitivities are meaningless when VAD is off and must not be sent.
    assert vad.silence_duration_ms is None


# ---------------------------------------------------------------- thinking


def test_no_thinking_config_by_default():
    assert build_live_config(make_cfg()).thinking_config is None


def test_thinking_budget_is_applied():
    live = build_live_config(make_cfg(thinking={"budget": 1024}))
    assert live.thinking_config.thinking_budget == 1024


def test_include_thoughts_alone_produces_a_config():
    live = build_live_config(make_cfg(thinking={"include_thoughts": True}))
    assert live.thinking_config.include_thoughts is True


# ----------------------------------------------------------------- session


def test_compression_is_enabled_by_default():
    """It is a cost control: every turn re-bills the whole context."""
    comp = build_live_config(make_cfg()).context_window_compression
    assert comp.trigger_tokens == 25600
    assert comp.sliding_window.target_tokens == 12800


def test_compression_can_be_disabled():
    cfg = make_cfg(session={"context_window_compression": {"enabled": False}})
    assert build_live_config(cfg).context_window_compression is None


def test_resumption_handle_is_passed_on_reconnect():
    live = build_live_config(make_cfg(), resumption_handle="handle-abc")
    assert live.session_resumption.handle == "handle-abc"


def test_fresh_session_has_a_null_resumption_handle():
    live = build_live_config(make_cfg())
    assert live.session_resumption is not None
    assert live.session_resumption.handle is None


def test_resumption_omitted_when_disabled():
    cfg = make_cfg(session={"resumption": {"enabled": False}})
    assert build_live_config(cfg).session_resumption is None


# -------------------------------------------------------------------- tools


def test_function_declarations_are_included():
    cfg = make_cfg(search={"google_search": {"enabled": False}})
    live = build_live_config(cfg, make_registry("a__x", "b__y"))

    decls = live.tools[0].function_declarations
    assert {d.name for d in decls} == {"a__x", "b__y"}


def test_google_search_tool_is_added_when_enabled():
    live = build_live_config(make_cfg())
    assert any(getattr(t, "google_search", None) is not None for t in live.tools)


def test_exclude_domains_reach_the_search_tool():
    """This is the one domain control that is genuinely server-enforced."""
    cfg = make_cfg(search={"google_search": {"exclude_domains": ["reddit.com"]}})
    live = build_live_config(cfg)

    search = next(t.google_search for t in live.tools if getattr(t, "google_search", None))
    assert search.exclude_domains == ["reddit.com"]


def test_search_tool_omitted_when_disabled():
    cfg = make_cfg(search={"google_search": {"enabled": False}})
    assert build_live_config(cfg).tools is None


def test_functions_and_search_coexist():
    live = build_live_config(make_cfg(), make_registry("a__x"))
    assert len(live.tools) == 2


def test_no_tools_key_when_nothing_is_configured():
    """An empty tools list is not the same as omitting it, and the API cares."""
    cfg = make_cfg(search={"google_search": {"enabled": False}})
    assert build_live_config(cfg, make_registry()).tools is None


def test_media_resolution_is_applied():
    live = build_live_config(make_cfg(media={"resolution": "MEDIA_RESOLUTION_MEDIUM"}))
    assert live.media_resolution.value == "MEDIA_RESOLUTION_MEDIUM"
