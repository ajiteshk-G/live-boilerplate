"""Voice and language selection.

The distinction under test: a voice is a timbre, the language code is what makes
the model sound Indian -- and native-audio models only honour the latter through
the system instruction.
"""

from __future__ import annotations

import pytest

from gemini_live.live.connect_config import build_live_config, build_system_instruction
from gemini_live.search.domain_policy import DomainPolicy
from gemini_live.settings.schema import AppConfig
from gemini_live.settings.voices import (
    CORE_VOICES,
    INDIAN_LOCALES,
    LIVE_LANGUAGES,
    LIVE_VOICES,
    canonical_voice,
    is_supported_language,
    language_directive,
    language_label,
)


def make_cfg(**over):
    base = {"vertex": {"project": "p"}}
    base.update(over)
    return AppConfig.model_validate(base)


# -------------------------------------------------------------------- catalog


def test_catalog_matches_the_documented_counts():
    assert len(LIVE_VOICES) == 30
    assert len(LIVE_LANGUAGES) == 70
    # The eight legacy voices work on every Live model, native audio or not.
    assert set(LIVE_VOICES) >= CORE_VOICES
    assert len(CORE_VOICES) == 8


def test_every_indian_locale_resolves_to_a_supported_language():
    for code in INDIAN_LOCALES:
        assert is_supported_language(code), code


def test_voice_names_are_normalised_not_rejected_on_case():
    assert canonical_voice("kore") == "Kore"
    assert canonical_voice("  ZEPHYR ") == "Zephyr"
    assert canonical_voice("Siri") is None


def test_language_label_keeps_the_region():
    assert language_label("en-IN") == "English (India)"
    assert language_label("hi-IN") == "Hindi (India)"
    assert language_label("en") == "English"
    assert language_label("pt-BR") == "Portuguese (BR)"


# ------------------------------------------------------------------ validation


def test_config_rejects_a_voice_that_does_not_exist():
    with pytest.raises(ValueError, match="not a Live API voice"):
        make_cfg(speech={"voice_name": "Ramesh"})


def test_config_rejects_a_language_that_does_not_exist():
    with pytest.raises(ValueError, match="not a Live API language"):
        make_cfg(speech={"language_code": "xx-YY"})


def test_regional_variants_of_supported_languages_are_accepted():
    assert make_cfg(speech={"language_code": "hi-IN"}).speech.language_code == "hi-IN"
    assert make_cfg(speech={"language_code": "en_IN"}).speech.language_code == "en-IN"


def test_the_shipped_default_is_indian_english():
    cfg = make_cfg()
    assert cfg.speech.language_code == "en-IN"
    assert cfg.speech.enforce_language_in_system_instruction is True


# -------------------------------------------------------------- wiring to live


def test_indian_language_reaches_the_system_instruction():
    cfg = make_cfg(
        model={"name": "gemini-live-2.5-flash-native-audio"},
        speech={"language_code": "en-IN"},
    )
    text = build_system_instruction(cfg, DomainPolicy())

    assert "ENGLISH (INDIA)" in text
    assert "Indian accent" in text


def test_language_directive_only_claims_an_accent_for_indian_locales():
    assert "Indian accent" in language_directive("hi-IN")
    assert "Indian accent" not in language_directive("en-US")


def test_no_directive_when_enforcement_is_disabled():
    cfg = make_cfg(
        speech={"language_code": "en-IN", "enforce_language_in_system_instruction": False}
    )
    assert "ENGLISH (INDIA)" not in build_system_instruction(cfg, DomainPolicy())


def test_native_audio_models_are_not_sent_the_unsupported_language_field():
    """Sending language_code to a native-audio model is a config error, even
    though the config still carries it for the system-instruction path."""
    cfg = make_cfg(
        model={"name": "gemini-live-2.5-flash-native-audio"},
        speech={"voice_name": "Kore", "language_code": "en-IN"},
    )
    live = build_live_config(cfg)

    assert live.speech_config.voice_config.prebuilt_voice_config.voice_name == "Kore"
    assert live.speech_config.language_code is None


def test_models_that_support_language_code_receive_it():
    cfg = make_cfg(
        model={"name": "gemini-2.5-flash"},
        speech={"voice_name": "Leda", "language_code": "hi-IN"},
    )
    live = build_live_config(cfg)

    assert live.speech_config.language_code == "hi-IN"
