"""Grounding citation audit."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from gemini_live.search.domain_policy import DomainPolicy, Verdict
from gemini_live.search.grounding_filter import GroundingFilter


@dataclass
class FakeWeb:
    uri: str | None = None
    title: str | None = None
    domain: str | None = None


@dataclass
class FakeChunk:
    web: FakeWeb | None = None


@dataclass
class FakeGrounding:
    grounding_chunks: list[FakeChunk] = field(default_factory=list)
    web_search_queries: list[str] = field(default_factory=list)


def test_no_grounding_metadata_is_a_clean_empty_result():
    audit = GroundingFilter(DomainPolicy(allow=["github.com"])).audit(None)
    assert audit.citations == []
    assert audit.clean


def test_allowed_citation_is_clean():
    grounding = FakeGrounding(
        grounding_chunks=[
            FakeChunk(FakeWeb(uri="https://docs.github.com/x", domain="docs.github.com"))
        ],
        web_search_queries=["how to fork"],
    )
    audit = GroundingFilter(DomainPolicy(allow=["github.com"])).audit(grounding)

    assert audit.clean
    assert audit.citations[0].verdict == Verdict.ALLOWED.value
    assert audit.search_queries == ["how to fork"]


def test_disallowed_citation_is_flagged_as_a_violation():
    grounding = FakeGrounding(
        grounding_chunks=[FakeChunk(FakeWeb(uri="https://evil.example/x", domain="evil.example"))]
    )
    audit = GroundingFilter(DomainPolicy(allow=["github.com"])).audit(grounding)

    assert not audit.clean
    assert len(audit.violations) == 1
    assert audit.as_event()["violations"] == 1


def test_opaque_redirect_is_an_unknown_not_a_pass():
    """Vertex returns redirect URIs. Counting those as clean would make the
    whole audit meaningless."""
    grounding = FakeGrounding(
        grounding_chunks=[
            FakeChunk(
                FakeWeb(uri="https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC")
            )
        ]
    )
    audit = GroundingFilter(DomainPolicy(allow=["github.com"])).audit(grounding)

    assert len(audit.unknowns) == 1
    assert audit.violations == []
    assert not audit.clean  # unknowns still deny a clean bill of health


def test_domain_field_is_preferred_over_the_redirect_uri():
    grounding = FakeGrounding(
        grounding_chunks=[
            FakeChunk(
                FakeWeb(
                    uri="https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC",
                    domain="docs.github.com",
                )
            )
        ]
    )
    audit = GroundingFilter(DomainPolicy(allow=["github.com"])).audit(grounding)

    assert audit.clean
    assert audit.citations[0].domain == "docs.github.com"


def test_chunks_without_web_data_are_skipped():
    grounding = FakeGrounding(grounding_chunks=[FakeChunk(web=None)])
    assert GroundingFilter(DomainPolicy()).audit(grounding).citations == []


def test_mixed_results_are_classified_independently():
    grounding = FakeGrounding(
        grounding_chunks=[
            FakeChunk(FakeWeb(uri="https://docs.github.com/a", domain="docs.github.com")),
            FakeChunk(FakeWeb(uri="https://evil.example/b", domain="evil.example")),
            FakeChunk(
                FakeWeb(uri="https://vertexaisearch.cloud.google.com/grounding-api-redirect/C")
            ),
        ]
    )
    audit = GroundingFilter(DomainPolicy(allow=["github.com"])).audit(grounding)

    verdicts = [c.verdict for c in audit.citations]
    assert verdicts == [Verdict.ALLOWED.value, Verdict.DENIED.value, Verdict.UNKNOWN.value]


def test_violations_are_written_to_the_log(tmp_path):
    path = tmp_path / "violations.jsonl"
    grounding = FakeGrounding(
        grounding_chunks=[FakeChunk(FakeWeb(uri="https://evil.example/x", domain="evil.example"))],
        web_search_queries=["q"],
    )
    GroundingFilter(DomainPolicy(allow=["github.com"]), violations_log=path).audit(
        grounding, turn_index=7
    )

    record = json.loads(path.read_text().strip())
    assert record["turn"] == 7
    assert record["domain"] == "evil.example"
    assert record["verdict"] == Verdict.DENIED.value
    assert record["queries"] == ["q"]


def test_clean_audits_write_nothing(tmp_path):
    path = tmp_path / "violations.jsonl"
    grounding = FakeGrounding(
        grounding_chunks=[
            FakeChunk(FakeWeb(uri="https://docs.github.com/x", domain="docs.github.com"))
        ]
    )
    GroundingFilter(DomainPolicy(allow=["github.com"]), violations_log=path).audit(grounding)

    assert not path.exists() or path.read_text() == ""


def test_unconfigured_policy_allows_everything():
    grounding = FakeGrounding(
        grounding_chunks=[
            FakeChunk(FakeWeb(uri="https://anything.example", domain="anything.example"))
        ]
    )
    assert GroundingFilter(DomainPolicy()).audit(grounding).clean
