from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class SourcePolicyError(ValueError):
    """Raised when a data source is not approved for the requested collection mode."""


@dataclass(frozen=True)
class SourcePolicy:
    source_id: str
    name: str
    status: str
    bulk_allowed: bool
    automated_allowed: bool
    note: str


SOURCE_POLICIES = {
    "balldontlie": SourcePolicy(
        "balldontlie", "BALLDONTLIE MLB API", "allowed", True, True,
        "Reviewed 2026-09-09: official API only, authorized subscription and rate limits required. https://www.balldontlie.io/terms.html",
    ),
    "retrosheet": SourcePolicy(
        "retrosheet",
        "Retrosheet",
        "allowed",
        True,
        True,
        "Bulk historical data allowed with the required Retrosheet notice.",
    ),
    "chadwick_register": SourcePolicy(
        "chadwick_register",
        "Chadwick Register",
        "allowed",
        True,
        True,
        "ODC Attribution License; preserve attribution.",
    ),
    "nws": SourcePolicy(
        "nws",
        "US National Weather Service API",
        "allowed",
        True,
        True,
        "Open government weather data; identify the client and respect rate limits.",
    ),
    "data_go_kr_kspo_results": SourcePolicy(
        "data_go_kr_kspo_results",
        "KSPO match results via data.go.kr",
        "allowed",
        False,
        True,
        "Free official API; results arrive 14 days after games and contain no odds.",
    ),
    "user_input": SourcePolicy(
        "user_input",
        "User-provided odds and lineup input",
        "allowed",
        False,
        False,
        "Keep private inputs local and out of version control.",
    ),
    "mlb_statsapi": SourcePolicy(
        "mlb_statsapi",
        "MLB Stats API",
        "allowed",
        False,
        True,
        "Personal-use connection requested by user 2026-09-10. Not an open-data license or confirmation of MLB permission. Cached, bounded requests only; no access-control bypass.",
    ),
    "baseball_savant": SourcePolicy(
        "baseball_savant",
        "Baseball Savant",
        "blocked",
        False,
        False,
        "No automated collection without separately verified permission.",
    ),
    "the_odds_api": SourcePolicy(
        "the_odds_api",
        "The Odds API",
        "blocked",
        False,
        False,
        "Licensed service, not an open-data source; excluded by this project's policy.",
    ),
    "sportsbook_scrape": SourcePolicy(
        "sportsbook_scrape",
        "Sportsbook scraping",
        "blocked",
        False,
        False,
        "Scraping sportsbook pages is outside the approved source policy.",
    ),
}


def _apply_config_policy() -> None:
    """Make config/data_sources.json the enforcement source of truth."""
    config_path = Path(__file__).resolve().parents[2] / "config" / "data_sources.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = set(config.get("allowed", []))
    blocked = set(config.get("blocked", []))
    overlap = allowed & blocked
    if overlap:
        raise RuntimeError(f"source policy lists overlap: {sorted(overlap)}")
    unknown = allowed - set(SOURCE_POLICIES)
    if unknown:
        raise RuntimeError(f"source policy metadata missing: {sorted(unknown)}")
    for source_id, policy in tuple(SOURCE_POLICIES.items()):
        status = "allowed" if source_id in allowed else "blocked"
        SOURCE_POLICIES[source_id] = SourcePolicy(
            policy.source_id,
            policy.name,
            status,
            policy.bulk_allowed if status == "allowed" else False,
            policy.automated_allowed if status == "allowed" else False,
            policy.note,
        )


_apply_config_policy()


def require_approved_source(
    source_id: str, *, automated: bool = False, bulk: bool = False
) -> SourcePolicy:
    """Validate a source before an importer performs collection."""
    policy = SOURCE_POLICIES.get(source_id)
    if policy is None:
        raise SourcePolicyError(f"unreviewed data source: {source_id}")
    if policy.status != "allowed":
        raise SourcePolicyError(f"blocked data source: {policy.name}. {policy.note}")
    if automated and not policy.automated_allowed:
        raise SourcePolicyError(f"automated collection is not allowed for {policy.name}")
    if bulk and not policy.bulk_allowed:
        raise SourcePolicyError(f"bulk collection is not allowed for {policy.name}")
    return policy
