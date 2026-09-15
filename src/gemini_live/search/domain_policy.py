"""Domain allow/deny policy.

HONEST SCOPE
------------
``types.GoogleSearch`` exposes only ``exclude_domains`` -- a deny-list, Vertex
only. There is no allow-list field anywhere in the API. So an "allow-list" here
is enforced as:

1. a real server-side deny-list (``exclude_domains``),
2. generated system-instruction rules (advisory -- the model may ignore them),
3. this post-hoc audit of grounding citations.

Layer 3 cannot prevent the model from *speaking* a non-approved source, because
grounding metadata may arrive after audio playback has already begun. Treat it as
an audit and alerting control, not a hard block.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlparse

from ..settings.schema import normalize_domain

# Vertex returns redirect URLs for grounded citations; their hostname tells us
# nothing about the underlying source.
_REDIRECT_HOSTS = (
    "vertexaisearch.cloud.google.com",
    "grounding-api-redirect",
)


class Verdict(StrEnum):
    ALLOWED = "ALLOWED"
    DENIED = "DENIED"
    UNKNOWN = "UNKNOWN"
    """The real hostname could not be determined (e.g. an unresolved redirect).

    Reported honestly rather than defaulting to ALLOWED (which would silently
    pass disallowed sources) or DENIED (which would cry wolf on every citation).
    """


@dataclass(frozen=True)
class DomainDecision:
    verdict: Verdict
    domain: str | None
    reason: str


def _host_matches(host: str, domain: str) -> bool:
    """Anchored on a dot boundary so ``evilgithub.com`` does not match ``github.com``."""
    return host == domain or host.endswith("." + domain)


class DomainPolicy:
    def __init__(self, allow: list[str] | None = None, deny: list[str] | None = None) -> None:
        self.allow = [normalize_domain(d) for d in (allow or [])]
        self.deny = [normalize_domain(d) for d in (deny or [])]

    @property
    def active(self) -> bool:
        return bool(self.allow or self.deny)

    def extract_host(self, url_or_domain: str | None) -> str | None:
        if not url_or_domain:
            return None
        raw = url_or_domain.strip()
        if any(marker in raw for marker in _REDIRECT_HOSTS):
            return None
        host = urlparse(raw if "//" in raw else f"//{raw}").hostname
        if not host:
            return None
        return host.lower().strip(".").removeprefix("www.")

    def verdict(self, url_or_domain: str | None, *, fallback: str | None = None) -> DomainDecision:
        host = self.extract_host(url_or_domain) or self.extract_host(fallback)
        if host is None:
            return DomainDecision(
                Verdict.UNKNOWN,
                None,
                "source hostname could not be determined (redirect URI without domain metadata)",
            )

        for denied in self.deny:
            if _host_matches(host, denied):
                return DomainDecision(Verdict.DENIED, host, f"matches deny entry {denied!r}")

        if not self.allow:
            return DomainDecision(Verdict.ALLOWED, host, "no allow-list configured")

        for allowed in self.allow:
            if _host_matches(host, allowed):
                return DomainDecision(Verdict.ALLOWED, host, f"matches allow entry {allowed!r}")

        return DomainDecision(Verdict.DENIED, host, "not present in the configured allow-list")

    def system_instruction_rules(self) -> str:
        """Advisory rules injected into the system instruction."""
        if not self.active:
            return ""
        lines = ["", "SOURCE POLICY:"]
        if self.allow:
            lines.append(
                "- Only use information from these approved domains: "
                + ", ".join(self.allow)
                + "."
            )
            lines.append(
                "- If the answer is not available from those domains, say so plainly "
                "instead of citing another source."
            )
        if self.deny:
            lines.append("- Never use information from: " + ", ".join(self.deny) + ".")
        lines.append("- State which source you used when you rely on a web result.")
        return "\n".join(lines)
