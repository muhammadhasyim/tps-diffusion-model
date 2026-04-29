#!/usr/bin/env python3
"""Download selected ATLAS protein-only MD bundles into a local cache."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

from genai_tps.data.atlas import (  # noqa: E402
    AtlasDownloadRecord,
    atlas_api_url,
    atlas_static_url,
    cache_zip_path,
    download_atlas_zip,
    read_ids_file,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download selected ATLAS protein ZIP bundles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/atlas/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/atlas/manifests/download_manifest.json"))
    parser.add_argument("--dataset", type=str, default="ATLAS", choices=["ATLAS", "chameleon", "DPF"])
    parser.add_argument("--url-source", type=str, default="static", choices=["static", "api"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    atlas_ids = read_ids_file(args.ids_file, limit=args.limit)
    if not atlas_ids:
        raise SystemExit(f"No ATLAS IDs found in {args.ids_file}")

    if args.dry_run:
        records = []
        for atlas_id in atlas_ids:
            url = (
                atlas_api_url(atlas_id, dataset=args.dataset)
                if args.url_source == "api"
                else atlas_static_url(atlas_id, dataset=args.dataset)
            )
            path = cache_zip_path(args.raw_dir, atlas_id)
            print(f"[dry-run] {atlas_id}: {url} -> {path}")
            records.append(
                AtlasDownloadRecord(
                    atlas_id=atlas_id,
                    url=url,
                    path=path,
                    size_bytes=0,
                    sha256="",
                    status="dry-run",
                )
            )
        write_manifest(args.manifest, records)
        return

    records: list[AtlasDownloadRecord] = []
    max_workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                download_atlas_zip,
                atlas_id,
                raw_dir=args.raw_dir,
                dataset=args.dataset,
                url_source=args.url_source,
                overwrite=args.overwrite,
            ): atlas_id
            for atlas_id in atlas_ids
        }
        for future in as_completed(futures):
            atlas_id = futures[future]
            record = future.result()
            records.append(record)
            if record.status == "failed":
                print(f"[download] {atlas_id}: failed: {record.error}", file=sys.stderr)
            else:
                print(f"[download] {atlas_id}: {record.status} ({record.size_bytes} bytes)")

    records.sort(key=lambda record: record.atlas_id)
    write_manifest(args.manifest, records)
    n_failed = sum(record.status == "failed" for record in records)
    print(f"[download] Manifest: {args.manifest}")
    if n_failed:
        raise SystemExit(f"{n_failed} ATLAS downloads failed.")


if __name__ == "__main__":
    main()
