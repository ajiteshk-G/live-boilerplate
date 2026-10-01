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


# -------------------------------------------------------------- wiring to live


def test_pinned_mode_puts_the_indian_language_in_the_system_instruction():
    cfg = make_cfg(
        speech={"language_code": "en-IN", "language_mode": "pinned"},
    )
    text = build_system_instruction(cfg, DomainPolicy())

    assert "ENGLISH (INDIA)" in text
    assert "Indian accent" in text


def test_follow_user_mode_tells_the_model_to_switch_with_the_user():
    """The default: en-IN is only where the call opens."""
    cfg = make_cfg(
        speech={"language_code": "en-IN"},
    )
    text = build_system_instruction(cfg, DomainPolicy())

    assert "SPEAK THE USER'S LANGUAGE" in text
    assert "Open the conversation in English (India)" in text
    assert "switch with them immediately" in text
    # The pinning wording would contradict switching, so it must be absent.
    assert "Speak only English (India)" not in text
    assert "UNMISTAKABLY" not in text


def test_follow_user_is_the_default_mode():
    assert make_cfg().speech.language_mode == "follow_user"


def test_follow_user_still_asks_for_an_indian_accent_when_it_speaks_english():
    assert "Indian accent" in language_directive("en-IN", follow_user=True)
    assert "Indian accent" not in language_directive("de-DE", follow_user=True)


def test_language_directive_only_claims_an_accent_for_indian_locales():
    assert "Indian accent" in language_directive("hi-IN")
    assert "Indian accent" not in language_directive("en-US")


def test_no_directive_when_language_code_is_null():
    cfg = make_cfg(speech={"language_code": None})
    text = build_system_instruction(cfg, DomainPolicy())
    assert "ENGLISH (INDIA)" not in text
    assert "SPEAK THE USER'S LANGUAGE" not in text


def test_gemini_3_8_live_does_not_send_speech_config_language_code():
    """Gemini 3.8 Live ignores SpeechConfig.language_code on the wire and uses
    the system instruction + transcription.language_codes instead."""
    cfg = make_cfg(
        speech={"voice_name": "Kore", "language_code": "en-IN"},
    )
    live = build_live_config(cfg)

    assert live.speech_config.voice_config.prebuilt_voice_config.voice_name == "Kore"
    assert live.speech_config.language_code is None


# ------------------------------------------------------------- client locale


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("hi-IN", "hi-IN"),
        ("en-US", "en-US"),
        ("fr", "fr"),
        ("pt_BR", "pt-BR"),
        ("ta-in", "ta-IN"),
        # Accept-Language style lists: take the first, most-preferred entry.
        ("hi-IN,hi;q=0.9,en;q=0.8", "hi-IN"),
        # Script subtags are dropped; only the region matters for the accent.
        ("zh-Hans-CN", "zh-CN"),
    ],
)
def test_browser_locales_are_normalised(raw, expected):
    from gemini_live.settings.voices import resolve_client_locale

    assert resolve_client_locale(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "xx-YY", "123", "klingon"])
def test_unusable_locales_fall_back_rather_than_reaching_the_api(raw):
    from gemini_live.settings.voices import resolve_client_locale

    assert resolve_client_locale(raw) is None


def test_session_opens_in_the_callers_language():
    cfg = make_cfg(speech={"language_code": "en-IN"})
    session = cfg.for_client_locale("hi-IN")

    assert session.speech.language_code == "hi-IN"
    # One user's language must never leak into the shared config.
    assert cfg.speech.language_code == "en-IN"


def test_unsupported_caller_locale_keeps_the_configured_default():
    cfg = make_cfg(speech={"language_code": "en-IN"})

    assert cfg.for_client_locale("xx-YY") is cfg
    assert cfg.for_client_locale(None) is cfg


def test_client_locale_can_be_switched_off():
    cfg = make_cfg(speech={"language_code": "en-IN", "use_client_locale": False})

    assert cfg.for_client_locale("hi-IN") is cfg


def test_the_callers_language_reaches_the_system_instruction():
    """End to end: a Hindi browser should get a session that opens in Hindi."""
    cfg = make_cfg(
        speech={"language_code": "en-IN"},
    ).for_client_locale("hi-IN")

    text = build_system_instruction(cfg, DomainPolicy())
    assert "Open the conversation in Hindi (India)" in text


# ------------------------------------------------------- agent name & gender


def test_every_voice_has_an_assigned_gender():
    from gemini_live.settings.voices import FEMALE_VOICES, MALE_VOICES, VOICE_GENDERS

    assert set(VOICE_GENDERS.keys()) == set(LIVE_VOICES.keys())
    assert len(FEMALE_VOICES) == 14
    assert len(MALE_VOICES) == 16
    assert FEMALE_VOICES.isdisjoint(MALE_VOICES)


def test_mixing_female_agent_with_male_voice_is_rejected():
    with pytest.raises(ValueError, match="Gender mismatch.*Male and female cannot be mixed"):
        make_cfg(
            agent={"name": "Ananya", "gender": "female"},
            speech={"voice_name": "Puck"},
        )


def test_mixing_male_agent_with_female_voice_is_rejected():
    with pytest.raises(ValueError, match="Gender mismatch.*Male and female cannot be mixed"):
        make_cfg(
            agent={"name": "Aarav", "gender": "male"},
            speech={"voice_name": "Kore"},
        )


def test_setting_male_agent_without_voice_selects_male_voice():
    cfg = make_cfg(agent={"name": "Aarav", "gender": "male"})
    assert cfg.speech.voice_name == "Puck"


def test_setting_male_voice_without_agent_aligns_agent_gender():
    cfg = make_cfg(speech={"voice_name": "Fenrir"})
    assert cfg.agent.gender == "male"
    assert cfg.agent.name == "Aarav"


def test_system_instruction_enforces_agent_name_and_anti_mixing_rules():
    female_cfg = make_cfg(agent={"name": "Ananya", "gender": "female"})
    female_prompt = build_system_instruction(female_cfg, DomainPolicy())
    assert 'Your name is "Ananya"' in female_prompt
    assert "strictly FEMALE" in female_prompt
    assert "DO NOT MIX MALE AND FEMALE" in female_prompt
    assert "karungi" in female_prompt

    male_cfg = make_cfg(agent={"name": "Aarav", "gender": "male"})
    male_prompt = build_system_instruction(male_cfg, DomainPolicy())
    assert 'Your name is "Aarav"' in male_prompt
    assert "strictly MALE" in male_prompt
    assert "DO NOT MIX MALE AND FEMALE" in male_prompt
    assert "karunga" in male_prompt


