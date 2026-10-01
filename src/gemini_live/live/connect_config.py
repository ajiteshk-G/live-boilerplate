"""Translate ``AppConfig`` into a ``types.LiveConnectConfig``.

Kept as a pure function so it is trivially testable without any network access.
Fields are only set when actually configured -- sending an explicit ``None`` is a
reliable way to get rejected by the backend.
"""

from __future__ import annotations

from typing import Any

from ..search.domain_policy import DomainPolicy
from ..settings.capabilities import capabilities_for
from ..settings.schema import AppConfig, normalize_domain
from ..settings.voices import agent_identity_directive, language_directive
from ..tools.registry import ToolRegistry

TOOL_RETRY_DIRECTIVE = (
    "TOOL CALL & RETRY POLICY: When a tool returns no results or an error, "
    "tell the user the result before calling the tool again with adjusted "
    "parameters, and ask if they would like to try a different option. "
    "Do not sweep through alternative parameter values automatically unless "
    "the user explicitly asks you to check multiple options. "
    "Hard cap: Never issue more than two consecutive function calls without "
    "saying something back to the user."
)


def build_system_instruction(cfg: AppConfig, policy: DomainPolicy) -> str:
    """Assemble the system instruction following the Gemini 3.8 Live best-practice order:
    1. Persona & Agent Identity
    2. Conversational & Tool Retry / Anti-Sweeping Rules
    3. Topic, Language, and Domain Guardrails
    """
    text = cfg.model.system_instruction.strip()
    if cfg.agent.enforce_in_system_instruction and (
        "agent" in cfg.model_fields_set or cfg.speech.language_code is not None
    ):
        text = f"{text}\n{agent_identity_directive(cfg.agent.name, cfg.agent.gender)}"
    if cfg.tools.enforce_retry_policy_in_system_instruction and (
        "tools" in cfg.model_fields_set or cfg.speech.language_code is not None
    ):
        text = f"{text}\n{TOOL_RETRY_DIRECTIVE}"
    if cfg.model.talk_only_about and cfg.model.talk_only_about.strip():
        topic = cfg.model.talk_only_about.strip()
        text = (
            f"{text}\n"
            f'TOPIC RESTRICTION: Talk ONLY about "{topic}". '
            "Do not answer questions or engage in conversation about any other topic. "
            f'If the user asks about anything outside "{topic}", politely decline and '
            f'remind them that you can only assist with "{topic}".'
        )
    # Gemini 3.8 Live detects language automatically from audio and steers the
    # opening language and regional accent through the system instruction.
    if cfg.speech.language_code:
        text = (
            f"{text}\n"
            + language_directive(
                cfg.speech.language_code,
                follow_user=cfg.speech.language_mode == "follow_user",
                default_accent=cfg.speech.default_accent,
            )
        )
    if cfg.search.inject_domain_rules_into_system_instruction:
        rules = policy.system_instruction_rules()
        if rules:
            text = f"{text}\n{rules}"
    return text


def build_live_config(
    cfg: AppConfig,
    registry: ToolRegistry | None = None,
    *,
    policy: DomainPolicy | None = None,
    resumption_handle: str | None = None,
) -> Any:
    from google.genai import types

    policy = policy or DomainPolicy(cfg.search.domains.allow, cfg.search.domains.deny)
    kwargs: dict[str, Any] = {
        "response_modalities": [types.Modality(m) for m in cfg.model.response_modalities],
        "system_instruction": types.Content(
            parts=[types.Part(text=build_system_instruction(cfg, policy))]
        ),
    }

    if cfg.model.temperature is not None:
        kwargs["temperature"] = cfg.model.temperature
    if cfg.model.top_p is not None:
        kwargs["top_p"] = cfg.model.top_p
    if cfg.model.max_output_tokens is not None:
        kwargs["max_output_tokens"] = cfg.model.max_output_tokens

    # --- voice -------------------------------------------------------------
    if cfg.speech.voice_name:
        kwargs["speech_config"] = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=cfg.speech.voice_name
                )
            )
        )

    # --- transcription -----------------------------------------------------
    if cfg.transcription.input:
        in_tx_kwargs: dict[str, Any] = {}
        if cfg.transcription.language_codes:
            in_tx_kwargs["language_codes"] = list(cfg.transcription.language_codes)
        if cfg.transcription.custom_vocabulary:
            in_tx_kwargs["custom_vocabulary"] = list(cfg.transcription.custom_vocabulary)
        kwargs["input_audio_transcription"] = types.AudioTranscriptionConfig(**in_tx_kwargs)
    if cfg.transcription.output:
        out_tx_kwargs: dict[str, Any] = {}
        if cfg.transcription.language_codes:
            out_tx_kwargs["language_codes"] = list(cfg.transcription.language_codes)
        kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig(**out_tx_kwargs)

    # --- voice activity detection -----------------------------------------
    detection: dict[str, Any] = {"disabled": not cfg.vad.enabled}
    if cfg.vad.enabled:
        detection["start_of_speech_sensitivity"] = types.StartSensitivity(
            cfg.vad.start_sensitivity
        )
        detection["end_of_speech_sensitivity"] = types.EndSensitivity(cfg.vad.end_sensitivity)
        if cfg.vad.prefix_padding_ms is not None:
            detection["prefix_padding_ms"] = cfg.vad.prefix_padding_ms
        if cfg.vad.silence_duration_ms is not None:
            detection["silence_duration_ms"] = cfg.vad.silence_duration_ms
    kwargs["realtime_input_config"] = types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(**detection)
    )
    if cfg.vad.explicit_vad_signal:
        kwargs["explicit_vad_signal"] = True

    # --- thinking ----------------------------------------------------------
    thinking: dict[str, Any] = {}
    if cfg.thinking.level is not None:
        thinking["thinking_level"] = cfg.thinking.level
    if cfg.thinking.include_thoughts:
        thinking["include_thoughts"] = True
    if thinking:
        kwargs["thinking_config"] = types.ThinkingConfig(**thinking)

    # --- session lifetime --------------------------------------------------
    # Context-window compression is a COST control as much as a quality one:
    # every turn re-bills the whole resident context.
    comp = cfg.session.context_window_compression
    if comp.enabled:
        kwargs["context_window_compression"] = types.ContextWindowCompressionConfig(
            trigger_tokens=comp.trigger_tokens,
            sliding_window=types.SlidingWindow(
                target_tokens=comp.sliding_window_target_tokens
            ),
        )
    if cfg.session.resumption.enabled:
        res_kwargs: dict[str, Any] = {"handle": resumption_handle}
        if cfg.session.resumption.transparent:
            res_kwargs["transparent"] = True
        kwargs["session_resumption"] = types.SessionResumptionConfig(**res_kwargs)
    if cfg.session.initial_history_in_client_content:
        kwargs["history_config"] = types.HistoryConfig(
            initial_history_in_client_content=True
        )

    # --- media -------------------------------------------------------------
    if cfg.media.resolution:
        kwargs["media_resolution"] = types.MediaResolution(cfg.media.resolution)

    # --- tools -------------------------------------------------------------
    caps = capabilities_for(cfg.model.name)
    tool_behavior = cfg.tools.behavior or caps.default_tool_behavior
    tools: list[Any] = (
        list(registry.declarations(behavior=tool_behavior)) if registry else []
    )
    if cfg.search.google_search.enabled:
        search_kwargs: dict[str, Any] = {}
        excluded = list(
            dict.fromkeys(
                normalize_domain(d)
                for d in (
                    *(cfg.search.google_search.exclude_domains or []),
                    *(cfg.search.domains.deny or []),
                )
                if d
            )
        )
        if excluded:
            # Real server-side deny-list. Vertex-only; the Developer API rejects it.
            search_kwargs["exclude_domains"] = excluded
        tools.append(types.Tool(google_search=types.GoogleSearch(**search_kwargs)))
    if tools:
        kwargs["tools"] = tools

    return types.LiveConnectConfig(**kwargs)
