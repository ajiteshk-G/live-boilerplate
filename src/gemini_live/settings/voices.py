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

VOICE_GENDERS: dict[str, str] = {
    "Zephyr": "female",
    "Puck": "male",
    "Charon": "male",
    "Kore": "female",
    "Fenrir": "male",
    "Leda": "female",
    "Orus": "male",
    "Aoede": "female",
    "Callirrhoe": "female",
    "Autonoe": "female",
    "Enceladus": "male",
    "Iapetus": "male",
    "Umbriel": "male",
    "Algieba": "male",
    "Despina": "female",
    "Erinome": "female",
    "Algenib": "male",
    "Rasalgethi": "male",
    "Laomedeia": "female",
    "Achernar": "female",
    "Alnilam": "male",
    "Schedar": "male",
    "Gacrux": "female",
    "Pulcherrima": "female",
    "Achird": "male",
    "Zubenelgenubi": "male",
    "Vindemiatrix": "female",
    "Sadachbia": "male",
    "Sadaltager": "male",
    "Sulafat": "female",
}
"""Every prebuilt voice mapped to its vocal gender ('female' or 'male')."""

FEMALE_VOICES: frozenset[str] = frozenset(
    name for name, gender in VOICE_GENDERS.items() if gender == "female"
)
MALE_VOICES: frozenset[str] = frozenset(
    name for name, gender in VOICE_GENDERS.items() if gender == "male"
)

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


def voice_gender(name: str | None) -> str | None:
    """Return ``'female'`` or ``'male'`` for a prebuilt voice name, or None."""
    if not name:
        return None
    canon = canonical_voice(name)
    return VOICE_GENDERS.get(canon) if canon else None


def agent_identity_directive(name: str, gender: str) -> str:
    """Build system-instruction rules enforcing agent name and strict gender consistency."""
    clean_gender = gender.strip().lower()
    clean_name = name.strip() or ("Ananya" if clean_gender == "female" else "Aarav")
    if clean_gender == "female":
        return (
            f'AGENT IDENTITY & GENDER: Your name is "{clean_name}" and your gender is '
            f'strictly FEMALE. Always introduce yourself as "{clean_name}" when greeting '
            "the user or when asked your name.\n"
            "CRITICAL - DO NOT MIX MALE AND FEMALE: You must NEVER mix male and female "
            "personas, pronouns, or grammatical forms. In languages with gendered grammar "
            "(especially Hindi, Hinglish, Marathi, Gujarati, Punjabi, Urdu, and other Indian "
            "languages), you MUST strictly and consistently use FEMALE first-person verb "
            "conjugations, participles, and adjectives for yourself in EVERY sentence "
            '(for example, in Hindi/Hinglish ALWAYS say "karungi", "dekhti hoon", '
            '"bata rahi hoon", "check karti hoon", "main samajh gayi" — NEVER use male forms '
            'like "karunga", "dekhta hoon", "bata raha hoon", or "main samajh gaya"). '
            "Maintain your female persona and grammar even when speaking to a male user."
        )
    return (
        f'AGENT IDENTITY & GENDER: Your name is "{clean_name}" and your gender is '
        f'strictly MALE. Always introduce yourself as "{clean_name}" when greeting '
        "the user or when asked your name.\n"
        "CRITICAL - DO NOT MIX MALE AND FEMALE: You must NEVER mix male and female "
        "personas, pronouns, or grammatical forms. In languages with gendered grammar "
        "(especially Hindi, Hinglish, Marathi, Gujarati, Punjabi, Urdu, and other Indian "
        "languages), you MUST strictly and consistently use MALE first-person verb "
        "conjugations, participles, and adjectives for yourself in EVERY sentence "
        '(for example, in Hindi/Hinglish ALWAYS say "karunga", "dekhta hoon", '
        '"bata raha hoon", "check karta hoon", "main samajh gaya" — NEVER use female forms '
        'like "karungi", "dekhti hoon", "bata rahi hoon", or "main samajh gayi"). '
        "Maintain your male persona and grammar even when speaking to a female user."
    )


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


def language_directive(
    code: str,
    *,
    follow_user: bool = False,
    default_accent: str | None = None,
) -> str:
    """The system-instruction rule that decides what language and accent are spoken.

    Native-audio models ignore ``speech.language_code`` and choose a language
    themselves, so this is the only lever that works on them.

    * ``follow_user=True`` makes ``code`` the opening language and fallback;
      the model immediately switches to whatever language the user speaks while
      keeping ``default_accent`` (Indian accent and tone) as its vocal persona.
    * ``follow_user=False`` pins one language.
    """
    label = language_label(code)
    region = (code.strip().replace("_", "-").split("-", 1) + [""])[1].upper()
    use_indian_accent = (
        (default_accent or "").strip().lower() == "indian" or region == "IN"
    )
    accent_line = (
        "DEFAULT ACCENT AND VOCAL TONE: Always speak with a natural, warm Indian accent "
        "and Indian conversational cadence/tone as your default vocal persona across all "
        "languages you speak (including English, Hindi, Hinglish, and any regional or "
        "international language). Indian names, places, and number formats (lakhs/crores) "
        "must sound native."
    )

    if follow_user:
        lines = [
            f"Open the conversation in {label}.",
            "SPEAK THE USER'S LANGUAGE: identify the language the user is speaking "
            "and reply in that same language. If they switch language mid-conversation, "
            "switch with them immediately and without being asked, and stay in the new "
            "language until they change again.",
            "Never ask the user which language they would like; infer it from what they "
            f"say. If their language is unclear, use {label}.",
            "Match their script and register too: reply in the script they used, and "
            "keep mixed-language speech (like Hinglish) mixed rather than translating it away.",
        ]
        if use_indian_accent:
            lines.append(accent_line)
        return "\n".join(lines)

    lines = [
        f"RESPOND IN {label.upper()}. YOU MUST RESPOND UNMISTAKABLY IN {label.upper()}.",
        f"Speak only {label} unless the user explicitly asks for another language.",
    ]
    if use_indian_accent:
        lines.append(accent_line)
    return "\n".join(lines)


def resolve_client_locale(raw: str | None) -> str | None:
    """Normalise a browser ``navigator.language`` value into a usable code.

    Returns None for anything unusable -- missing, malformed, or a language the
    Live API does not support -- so the caller can fall back to the configured
    default rather than handing the API something it will reject.
    """
    if not raw:
        return None
    # navigator.language can arrive as a q-weighted list ("hi-IN,hi;q=0.9").
    first = raw.split(",")[0].split(";")[0].strip().replace("_", "-")
    if not first or len(first) > 12:
        return None
    parts = first.split("-")
    if not parts[0].isalpha():
        return None
    if not is_supported_language(first):
        return None
    language = parts[0].lower()
    if len(parts) == 1:
        return language
    # Keep only the region subtag: "zh-Hans-CN" -> "zh-CN".
    region = parts[-1]
    if len(region) == 2 and region.isalpha():
        return f"{language}-{region.upper()}"
    return language

