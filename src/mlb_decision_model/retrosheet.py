from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from .sources import SourcePolicyError, require_approved_source


EXPECTED_CSVS = {
    "allplayers.csv",
    "gameinfo.csv",
    "teamstats.csv",
    "batting.csv",
    "pitching.csv",
    "fielding.csv",
    "plays.csv",
}
ALLOWED_HOSTS = {"retrosheet.org", "www.retrosheet.org"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_bundle(bundle: Path, source_url: str) -> dict[str, object]:
    """Validate a manually downloaded Retrosheet ZIP and return provenance."""
    require_approved_source("retrosheet", automated=False, bulk=True)
    host = (urlparse(source_url).hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise SourcePolicyError(f"Retrosheet bundle URL must use an official host: {host}")
    if not bundle.is_file() or not zipfile.is_zipfile(bundle):
        raise ValueError(f"not a readable ZIP archive: {bundle}")

    with zipfile.ZipFile(bundle) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        names = {PurePosixPath(item.filename).name.lower() for item in members}
        missing = sorted(EXPECTED_CSVS - names)
        if missing:
            raise ValueError(f"Retrosheet bundle is missing required CSV files: {missing}")
        for item in members:
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive member: {item.filename}")

    return {
        "source_id": "retrosheet",
        "source_url": source_url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_version": "bundle-through-2025",
        "archive_name": bundle.name,
        "archive_sha256": _sha256(bundle),
        "required_files": sorted(EXPECTED_CSVS),
    }


def extract_bundle(bundle: Path, destination: Path, source_url: str) -> Path:
    """Safely extract CSV members and write a provenance manifest."""
    manifest = inspect_bundle(bundle, source_url)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        for item in archive.infolist():
            if item.is_dir() or not item.filename.lower().endswith(".csv"):
                continue
            output = destination / PurePosixPath(item.filename).name
            with archive.open(item) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
    manifest_path = destination / "provenance.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and extract a manually downloaded official Retrosheet CSV ZIP"
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args()
    if args.extract_to:
        manifest_path = extract_bundle(args.bundle, args.extract_to, args.source_url)
        print(f"validated and extracted; provenance: {manifest_path}")
    else:
        print(json.dumps(inspect_bundle(args.bundle, args.source_url), indent=2))


if __name__ == "__main__":
    main()
