#!/usr/bin/env python3
"""Download selected files from the Runs N' Poses Zenodo record with MD5 verification.

Uses the Zenodo REST API so the **version** record id is explicit (concept DOIs
resolve to multiple versions; file URLs must target a specific version).

Default record id **18366081** matches Zenodo version **10.5281/zenodo.18366081**
(concept **10.5281/zenodo.14794785**).

Examples::

    python scripts/download_runs_n_poses_zenodo.py --out data/runs_n_poses/zenodo \\
        --files annotations.csv,all_similarity_scores.parquet,inputs.json

    python scripts/download_runs_n_poses_zenodo.py --out data/runs_n_poses/zenodo \\
        --all-metadata-only   # writes manifest for all files, downloads none
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_RECORD_ID = "18366081"
ZENODO_API_RECORD = "https://zenodo.org/api/records/{record_id}"


def _fetch_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "tps-diffusion-model-zenodo-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 — fixed API URL
        return json.loads(resp.read().decode("utf-8"))


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "tps-diffusion-model-zenodo-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310
        with tmp.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record-id",
        default=DEFAULT_RECORD_ID,
        help=f"Zenodo **version** record numeric id (default: {DEFAULT_RECORD_ID}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/runs_n_poses/zenodo"),
        help="Output directory for files and manifest.",
    )
    parser.add_argument(
        "--files",
        type=str,
        default="annotations.csv,all_similarity_scores.parquet,inputs.json",
        help="Comma-separated list of Zenodo file keys to download.",
    )
    parser.add_argument(
        "--all-metadata-only",
        action="store_true",
        help="Write manifest.json for all remote files but do not download.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if destination exists and MD5 matches manifest.",
    )
    args = parser.parse_args()

    api_url = ZENODO_API_RECORD.format(record_id=args.record_id)
    try:
        meta = _fetch_json(api_url)
    except urllib.error.HTTPError as e:
        print(f"HTTP error fetching {api_url}: {e}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Network error fetching {api_url}: {e}", file=sys.stderr)
        return 1

    files = meta.get("files") or []
    if not files:
        print("No files in Zenodo record metadata.", file=sys.stderr)
        return 1

    out_root: Path = args.out.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.json"
    manifest: dict[str, Any] = {
        "zenodo_record_id": args.record_id,
        "doi": meta.get("doi"),
        "conceptdoi": meta.get("conceptdoi"),
        "files": [],
    }

    want: set[str] = set()
    if not args.all_metadata_only:
        want = {k.strip() for k in args.files.split(",") if k.strip()}

    for f in files:
        key = f.get("key")
        checksum = f.get("checksum") or ""
        md5_expected = checksum.split(":", 1)[-1].strip() if checksum else ""
        entry = {
            "key": key,
            "size": f.get("size"),
            "checksum": checksum,
            "md5": md5_expected,
            "url": (f.get("links") or {}).get("self"),
        }
        manifest["files"].append(entry)

        if args.all_metadata_only:
            continue
        if key not in want:
            continue
        url = entry["url"]
        if not url or not key:
            print(f"Missing download URL for {key!r}", file=sys.stderr)
            return 1
        dest = out_root / key
        if args.skip_existing and dest.is_file() and md5_expected:
            got = _md5_file(dest)
            if got.lower() == md5_expected.lower():
                print(f"[skip] {key} (existing MD5 OK)")
                continue
        print(f"[get] {key} -> {dest}")
        try:
            _download(url, dest)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            print(f"Download failed for {key}: {e}", file=sys.stderr)
            return 1
        if md5_expected:
            got = _md5_file(dest)
            if got.lower() != md5_expected.lower():
                print(
                    f"MD5 mismatch for {key}: expected {md5_expected}, got {got}",
                    file=sys.stderr,
                )
                return 1

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
