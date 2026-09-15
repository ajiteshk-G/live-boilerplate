"""Audit grounding citations against the domain policy."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .domain_policy import DomainPolicy, Verdict

log = logging.getLogger(__name__)


@dataclass
class Citation:
    title: str | None
    uri: str | None
    domain: str | None
    verdict: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "uri": self.uri,
            "domain": self.domain,
            "verdict": self.verdict,
            "reason": self.reason,
        }


@dataclass
class AuditResult:
    citations: list[Citation] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)

    @property
    def violations(self) -> list[Citation]:
        return [c for c in self.citations if c.verdict == Verdict.DENIED.value]

    @property
    def unknowns(self) -> list[Citation]:
        return [c for c in self.citations if c.verdict == Verdict.UNKNOWN.value]

    @property
    def clean(self) -> bool:
        return not self.violations and not self.unknowns

    def as_event(self) -> dict[str, Any]:
        return {
            "citations": [c.as_dict() for c in self.citations],
            "queries": self.search_queries,
            "violations": len(self.violations),
            "unknowns": len(self.unknowns),
        }


class GroundingFilter:
    """Classifies grounding chunks and records policy violations."""

    def __init__(self, policy: DomainPolicy, *, violations_log: str | Path | None = None) -> None:
        self._policy = policy
        self._log_path = Path(violations_log) if violations_log else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def audit(self, grounding_metadata: Any, *, turn_index: int = -1) -> AuditResult:
        result = AuditResult()
        if grounding_metadata is None:
            return result

        queries = getattr(grounding_metadata, "web_search_queries", None) or []
        result.search_queries = [str(q) for q in queries]

        for chunk in getattr(grounding_metadata, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web is None:
                continue
            # `domain` is Vertex-only and best-effort; fall back to the URI.
            domain_hint = getattr(web, "domain", None)
            uri = getattr(web, "uri", None)
            decision = self._policy.verdict(domain_hint, fallback=uri)
            result.citations.append(
                Citation(
                    title=getattr(web, "title", None),
                    uri=uri,
                    domain=decision.domain,
                    verdict=decision.verdict.value,
                    reason=decision.reason,
                )
            )

        if result.violations or result.unknowns:
            self._record(result, turn_index)
        return result

    def _record(self, result: AuditResult, turn_index: int) -> None:
        for citation in result.violations + result.unknowns:
            log.warning(
                "grounding policy: %s source %s (%s)",
                citation.verdict,
                citation.domain or citation.uri,
                citation.reason,
            )
        if not self._log_path:
            return
        try:
            with self._log_path.open("a") as fh:
                for citation in result.violations + result.unknowns:
                    fh.write(
                        json.dumps(
                            {
                                "turn": turn_index,
                                "queries": result.search_queries,
                                **citation.as_dict(),
                            }
                        )
                        + "\n"
                    )
        except OSError as exc:
            log.debug("could not write violations log: %s", exc)
