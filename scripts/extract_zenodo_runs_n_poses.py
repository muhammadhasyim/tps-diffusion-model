#!/usr/bin/env python3
"""Extract a Runs N' Poses Zenodo tarball (e.g. ground_truth.tar.gz) idempotently.

Writes ``.extracted_ok`` in the output directory with the archive path and MD5
so re-running skips work already done. Use ``--force`` to unpack again (does
not delete existing files; may overwrite paths from the tarball).

Example::

    python scripts/extract_zenodo_runs_n_poses.py \\
        --archive data/runs_n_poses/zenodo/ground_truth.tar.gz \\
        --out data/runs_n_poses/extracted/ground_truth
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path

MARKER_NAME = ".extracted_ok"


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _marker_content(archive: Path, digest: str) -> str:
    return f"{archive.resolve()}\n{digest}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Path to .tar.gz from Zenodo (e.g. ground_truth.tar.gz).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/runs_n_poses/extracted/ground_truth"),
        help="Directory to extract into (created if missing).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore .extracted_ok and run extraction again.",
    )
    args = parser.parse_args()

    archive = args.archive.expanduser().resolve()
    out = args.out.expanduser().resolve()

    if not archive.is_file():
        print(f"ERROR: archive not found: {archive}", file=sys.stderr)
        return 1

    digest = _md5_file(archive)
    marker = out / MARKER_NAME
    expected = _marker_content(archive, digest)

    if out.is_dir() and marker.is_file() and not args.force:
        if marker.read_text(encoding="utf-8") == expected:
            print(f"[skip] already extracted to {out} (marker matches archive MD5)")
            return 0

    out.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive} -> {out}", flush=True)
    try:
        with tarfile.open(archive, "r:gz") as tf:
            # Python 3.12+: reject path traversal in tar members
            tf.extractall(out, filter="data")  # type: ignore[call-arg]
    except TypeError:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(out)
    except (OSError, tarfile.TarError) as exc:
        print(f"ERROR: extraction failed: {exc}", file=sys.stderr)
        return 1

    marker.write_text(expected, encoding="utf-8")
    print(f"Wrote marker {marker}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
