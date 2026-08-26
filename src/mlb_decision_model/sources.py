from __future__ import annotations

from dataclasses import dataclass


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
        "blocked",
        False,
        False,
        "No automated collection without separate written authorization from MLB.",
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
