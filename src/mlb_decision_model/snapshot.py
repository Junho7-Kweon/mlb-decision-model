from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


MAX_IMAGE_BYTES = 8 * 1024 * 1024
IMAGE_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


@dataclass(frozen=True)
class SnapshotPick:
    event_id: str
    name: str
    probability: float
    odds: float


def validate_picks(raw_picks: list[dict[str, Any]]) -> list[SnapshotPick]:
    if len(raw_picks) < 2:
        raise ValueError("at least two picks are required")
    picks: list[SnapshotPick] = []
    seen: set[str] = set()
    for raw in raw_picks:
        event_id = str(raw.get("event_id") or raw.get("name", "")).strip()
        name = str(raw.get("name", "")).strip()
        probability = float(raw.get("probability", 0))
        odds = float(raw.get("odds", 0))
        if not event_id:
            raise ValueError("each pick requires a game/event name")
        if not name or name in seen:
            raise ValueError("pick names must be non-empty and unique")
        if not 0.001 <= probability <= 0.999:
            raise ValueError(f"probability must be between 0.001 and 0.999: {name}")
        if not 1.01 <= odds <= 100.0:
            raise ValueError(f"decimal odds must be between 1.01 and 100: {name}")
        seen.add(name)
        picks.append(SnapshotPick(event_id, name, probability, odds))
    return picks


def decode_image(data_url: str) -> tuple[bytes, str]:
    match = re.fullmatch(r"data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=\r\n]+)", data_url)
    if not match:
        raise ValueError("snapshot must be a PNG, JPEG, or WebP data URL")
    try:
        content = base64.b64decode(match.group(2), validate=True)
    except binascii.Error as exc:
        raise ValueError("snapshot image is not valid base64") from exc
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError("snapshot image must be between 1 byte and 8 MB")
    return content, IMAGE_TYPES[match.group(1)]


def save_snapshot(payload: dict[str, Any], private_root: Path) -> dict[str, Any]:
    picks = validate_picks(list(payload.get("picks", [])))
    captured_at = str(payload.get("captured_at", "")).strip()
    try:
        timestamp = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("captured_at must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError("captured_at must include a timezone")

    private_root.mkdir(parents=True, exist_ok=True)
    snapshot_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:10]}"
    image_path = None
    data_url = str(payload.get("image_data_url", ""))
    if data_url:
        content, suffix = decode_image(data_url)
        image_path = private_root / f"{snapshot_id}{suffix}"
        image_path.write_bytes(content)

    record = {
        "snapshot_id": snapshot_id,
        "source_id": "user_input",
        "captured_at": timestamp.isoformat(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "image_file": image_path.name if image_path else None,
        "picks": [asdict(pick) for pick in picks],
    }
    with (private_root / "odds_snapshots.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
