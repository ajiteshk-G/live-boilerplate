"""Voice and language catalogs for the Live API.

Two things the Live API treats very differently, and that are easy to confuse:

* **Voice** (``speech.voice_name``) is a *timbre*, not a locale. There is no
  "Indian voice": every prebuilt voice speaks every supported language. Picking
  ``Kore`` does not make the model speak English, and picking a different voice
  will not give you an Indian accent.
* **Language** (``speech.language_code``) is what actually decides the language
  and regional accent -- ``en-IN`` is Indian English, ``hi-IN`` is Hindi.

The second catch is that **native-audio models ignore ``language_code``**: they
detect and switch language on their own. Google's documented workaround is to
put the requirement in the system instruction instead, which is what
:func:`language_directive` builds.

Sources:
  https://ai.google.dev/gemini-api/docs/live-guide#change-voice-and-language
  https://ai.google.dev/gemini-api/docs/speech-generation#voices
  https://cloud.google.com/vertex-ai/generative-ai/docs/live-api/configure-language-voice
"""

from __future__ import annotations

# --------------------------------------------------------------------- voices

LIVE_VOICES: dict[str, str] = {
    "Zephyr": "Bright",
    "Puck": "Upbeat",
    "Charon": "Informative",
    "Kore": "Firm",
    "Fenrir": "Excitable",
    "Leda": "Youthful",
    "Orus": "Firm",
    "Aoede": "Breezy",
    "Callirrhoe": "Easy-going",
    "Autonoe": "Bright",
    "Enceladus": "Breathy",
    "Iapetus": "Clear",
    "Umbriel": "Easy-going",
    "Algieba": "Smooth",
    "Despina": "Smooth",
    "Erinome": "Clear",
    "Algenib": "Gravelly",
    "Rasalgethi": "Informative",
    "Laomedeia": "Upbeat",
    "Achernar": "Soft",
    "Alnilam": "Firm",
    "Schedar": "Even",
    "Gacrux": "Mature",
    "Pulcherrima": "Forward",
    "Achird": "Friendly",
    "Zubenelgenubi": "Casual",
    "Vindemiatrix": "Gentle",
    "Sadachbia": "Lively",
    "Sadaltager": "Knowledgeable",
    "Sulafat": "Warm",
}
"""All 30 prebuilt voices, mapped to the characteristic Google documents."""

CORE_VOICES: frozenset[str] = frozenset(
    {"Puck", "Charon", "Kore", "Fenrir", "Aoede", "Leda", "Orus", "Zephyr"}
)
"""The eight voices every Live model accepts.

The other 22 arrived with native audio. Staying inside this set keeps one config
file valid across both model families.
"""

_VOICES_BY_LOWER = {name.lower(): name for name in LIVE_VOICES}


def canonical_voice(name: str) -> str | None:
    """``"kore"`` -> ``"Kore"``. None if the voice does not exist."""
    return _VOICES_BY_LOWER.get(name.strip().lower())


# ------------------------------------------------------------------ languages

LIVE_LANGUAGES: dict[str, str] = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "as": "Assamese",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "gl": "Galician",
    "gu": "Gujarati",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "iw": "Hebrew",
    "ja": "Japanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "or": "Odia",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "zu": "Zulu",
}
"""The 70 languages the Live API supports, keyed by their base BCP-47 code.

A regional variant of any of these (``en-IN``, ``hi-IN``, ``pt-BR``, ...) is also
accepted and is what you want whenever the accent matters.
"""

INDIAN_LOCALES: dict[str, str] = {
    "en-IN": "English (India)",
    "hi-IN": "Hindi (India)",
    "bn-IN": "Bengali (India)",
    "gu-IN": "Gujarati (India)",
    "kn-IN": "Kannada (India)",
    "ml-IN": "Malayalam (India)",
    "mr-IN": "Marathi (India)",
    "or-IN": "Odia (India)",
    "pa-IN": "Punjabi (India)",
    "ta-IN": "Tamil (India)",
    "te-IN": "Telugu (India)",
    "ur-IN": "Urdu (India)",
    "as-IN": "Assamese (India)",
}
"""Indian locales, split out because they are the ones this boilerplate defaults to."""


def base_language(code: str) -> str:
    """``"en-IN"`` -> ``"en"``."""
    return code.strip().replace("_", "-").split("-", 1)[0].lower()


def is_supported_language(code: str) -> bool:
    return base_language(code) in LIVE_LANGUAGES


def language_label(code: str) -> str:
    """A human name for a code: ``"en-IN"`` -> ``"English (India)"``."""
    normalized = code.strip().replace("_", "-")
    if normalized in INDIAN_LOCALES:
        return INDIAN_LOCALES[normalized]
    name = LIVE_LANGUAGES.get(base_language(normalized))
    if name is None:
        return normalized
    parts = normalized.split("-", 1)
    return f"{name} ({parts[1].upper()})" if len(parts) == 2 else name


def language_directive(code: str) -> str:
    """The system-instruction sentence that pins the spoken language.

    Native-audio models ignore ``speech.language_code`` and choose a language
    themselves, so this is the only lever that works on them. The wording
    follows Google's own recommendation ("RESPOND IN <LANGUAGE>. YOU MUST
    RESPOND UNMISTAKABLY IN <LANGUAGE>."), plus an accent line, because the
    language alone does not pin the regional accent.
    """
    label = language_label(code)
    region = (code.strip().replace("_", "-").split("-", 1) + [""])[1].upper()
    lines = [
        f"RESPOND IN {label.upper()}. YOU MUST RESPOND UNMISTAKABLY IN {label.upper()}.",
        f"Speak only {label} unless the user explicitly asks for another language.",
    ]
    if region == "IN":
        lines.append(
            "Use a natural Indian accent and Indian conversational phrasing "
            "(Indian names, places, and number formats should sound native)."
        )
    return "\n".join(lines)
