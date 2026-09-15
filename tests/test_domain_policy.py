"""Domain allow/deny policy.

Security-critical: a matching bug here means a "restricted" agent quietly
sourcing from anywhere. The subdomain-boundary cases are the ones that bite.
"""

from __future__ import annotations

import pytest

from gemini_live.search.domain_policy import DomainPolicy, Verdict
from gemini_live.settings.schema import normalize_domain

# ------------------------------------------------------------------ normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("github.com", "github.com"),
        ("GitHub.COM", "github.com"),
        ("https://github.com", "github.com"),
        ("https://github.com/some/path?q=1", "github.com"),
        ("www.github.com", "github.com"),
        ("  github.com  ", "github.com"),
        ("github.com.", "github.com"),
    ],
)
def test_normalize_domain(raw, expected):
    assert normalize_domain(raw) == expected


# -------------------------------------------------------------- allow-listing


def test_exact_domain_is_allowed():
    policy = DomainPolicy(allow=["github.com"])
    assert policy.verdict("https://github.com/x").verdict is Verdict.ALLOWED


def test_subdomain_of_allowed_domain_is_allowed():
    policy = DomainPolicy(allow=["github.com"])
    assert policy.verdict("https://docs.github.com/en").verdict is Verdict.ALLOWED


def test_suffix_lookalike_is_denied():
    """The attack this anchoring exists to stop.

    A naive ``host.endswith("github.com")`` would happily allow
    ``evilgithub.com``. Matching is anchored on a dot boundary instead.
    """
    policy = DomainPolicy(allow=["github.com"])
    decision = policy.verdict("https://evilgithub.com/malware")
    assert decision.verdict is Verdict.DENIED
    assert decision.domain == "evilgithub.com"


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com.attacker.net/x",  # allowed domain as a *prefix*
        "https://notgithub.com",
        "https://github.co",
    ],
)
def test_other_lookalikes_are_denied(url):
    policy = DomainPolicy(allow=["github.com"])
    assert policy.verdict(url).verdict is Verdict.DENIED


def test_unlisted_domain_is_denied_when_allow_list_present():
    policy = DomainPolicy(allow=["cloud.google.com"])
    assert policy.verdict("https://stackoverflow.com/q/1").verdict is Verdict.DENIED


def test_everything_allowed_when_no_allow_list():
    policy = DomainPolicy()
    decision = policy.verdict("https://anything.example")
    assert decision.verdict is Verdict.ALLOWED
    assert "no allow-list" in decision.reason


# --------------------------------------------------------------- deny-listing


def test_deny_beats_allow():
    """Deny is the enforceable layer, so it must win any conflict."""
    policy = DomainPolicy(allow=["example.com"], deny=["bad.example.com"])
    assert policy.verdict("https://bad.example.com/x").verdict is Verdict.DENIED
    assert policy.verdict("https://good.example.com/x").verdict is Verdict.ALLOWED


def test_deny_applies_to_subdomains():
    policy = DomainPolicy(deny=["reddit.com"])
    assert policy.verdict("https://old.reddit.com/r/x").verdict is Verdict.DENIED


# ----------------------------------------------------------------- unknowns


@pytest.mark.parametrize(
    "uri",
    [
        "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC123",
        "https://example.com/grounding-api-redirect/xyz",
    ],
)
def test_redirect_uris_are_unknown_not_allowed(uri):
    """Vertex returns opaque redirect URIs for grounded citations.

    Their hostname says nothing about the real source. Reporting ALLOWED would
    silently wave through disallowed content; reporting DENIED would cry wolf on
    every citation. UNKNOWN is the only honest answer.
    """
    policy = DomainPolicy(allow=["github.com"])
    decision = policy.verdict(uri)
    assert decision.verdict is Verdict.UNKNOWN
    assert decision.domain is None


def test_domain_metadata_is_preferred_over_redirect_uri():
    """When Vertex does populate `web.domain`, it beats the useless URI."""
    policy = DomainPolicy(allow=["github.com"])
    decision = policy.verdict(
        "docs.github.com",
        fallback="https://vertexaisearch.cloud.google.com/grounding-api-redirect/Z",
    )
    assert decision.verdict is Verdict.ALLOWED
    assert decision.domain == "docs.github.com"


def test_falls_back_to_uri_when_domain_missing():
    policy = DomainPolicy(allow=["github.com"])
    decision = policy.verdict(None, fallback="https://docs.github.com/en")
    assert decision.verdict is Verdict.ALLOWED


@pytest.mark.parametrize("value", [None, "", "   ", "not a url at all"])
def test_unparseable_sources_are_unknown(value):
    policy = DomainPolicy(allow=["github.com"])
    assert policy.verdict(value).verdict in (Verdict.UNKNOWN, Verdict.DENIED)


def test_no_source_at_all_is_unknown():
    policy = DomainPolicy(allow=["github.com"])
    assert policy.verdict(None).verdict is Verdict.UNKNOWN


# ------------------------------------------------------- system instruction


def test_no_rules_emitted_when_policy_inactive():
    assert DomainPolicy().system_instruction_rules() == ""


def test_rules_mention_every_configured_domain():
    policy = DomainPolicy(allow=["cloud.google.com", "ai.google.dev"], deny=["reddit.com"])
    rules = policy.system_instruction_rules()
    assert "cloud.google.com" in rules
    assert "ai.google.dev" in rules
    assert "reddit.com" in rules


def test_rules_tell_the_model_what_to_do_when_it_cannot_comply():
    """Without this, the model tends to answer from another source anyway."""
    rules = DomainPolicy(allow=["cloud.google.com"]).system_instruction_rules()
    assert "say so" in rules.lower()


def test_policy_normalizes_its_inputs():
    """Config may contain URLs or mixed case; matching must still work."""
    policy = DomainPolicy(allow=["https://GitHub.com/"], deny=["WWW.Reddit.com"])
    assert policy.allow == ["github.com"]
    assert policy.deny == ["reddit.com"]
    assert policy.verdict("https://docs.github.com").verdict is Verdict.ALLOWED
