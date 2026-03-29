#!/usr/bin/env python3
"""Thin wrapper around upstream ``papers/runs-n-poses/similarity_scoring.py``.

The upstream script expects to be run **from the runs-n-poses repository root**
with PLINDER data, Foldseek parquets under ``scoring/``, and ``new_pdb_ids.txt``
already prepared (see ``input_preparation.ipynb`` and
``docs/runs_n_poses_reproduction.md``).

It validates common environment variables, logs interpreter and key package
versions, then invokes ``similarity_scoring.py`` with the same ``argv`` tail.

Example::

    export OST_COMPOUNDS_CHEMLIB=/path/to/compounds.chemlib
    # PLINDER must be configured (see plinder ``get_config()``).
    cd papers/runs-n-poses && # ensure scoring/ layout exists
    python ../../scripts/run_skrinjar_full_similarity_batch.py 8cq9
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_runs_n_poses_root() -> Path:
    return _repo_root() / "papers" / "runs-n-poses"


def _log(msg: str) -> None:
    print(f"[skrinjar-batch] {msg}", flush=True)


def _run_version_snippet() -> dict[str, str]:
    code = (
        "import importlib, sys; out = {};\n"
        "for name in ('rdkit', 'plinder', 'pandas', 'numpy'):\n"
        "    try:\n"
        "        m = importlib.import_module(name)\n"
        "        out[name] = getattr(m, '__version__', str(m))\n"
        "    except Exception as e:\n"
        "        out[name] = f'missing:{e}'\n"
        "print(__import__('json').dumps(out))"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip())
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-n-poses-root",
        type=Path,
        default=None,
        help=f"Path to runs-n-poses clone (default: {_default_runs_n_poses_root()})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be run, then exit 0.",
    )
    parser.add_argument(
        "pdb_id",
        nargs="?",
        help="Four-letter PDB id passed to similarity_scoring.py (positional).",
    )
    args, tail = parser.parse_known_args()

    root = (args.runs_n_poses_root or _default_runs_n_poses_root()).expanduser().resolve()
    script = root / "similarity_scoring.py"
    if not script.is_file():
        _log(f"ERROR: missing {script} (init submodule: git submodule update --init papers/runs-n-poses)")
        return 1

    ost = os.environ.get("OST_COMPOUNDS_CHEMLIB")
    if not ost:
        _log("WARNING: OST_COMPOUNDS_CHEMLIB unset — OpenStructure may fail on exotic ligands.")
    else:
        p = Path(ost)
        if not p.is_file():
            _log(f"WARNING: OST_COMPOUNDS_CHEMLIB path does not exist: {p}")

    foldseek = os.environ.get("FOLDSEEK_BIN", "foldseek")
    _log(f"FOLDSEEK_BIN effective: {foldseek!r} (PATH lookup unless absolute)")

    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _log(f"CUDA_VISIBLE_DEVICES={cuda!r}")

    if not (root / "new_pdb_ids.txt").is_file():
        _log(
            "WARNING: new_pdb_ids.txt not found in runs-n-poses root — "
            "SimilarityScorer will fail at import time."
        )

    pdb_id = args.pdb_id
    if tail and tail[0] and not tail[0].startswith("-"):
        if pdb_id is not None:
            _log("ERROR: pass pdb_id either as positional or in tail, not both.")
            return 1
        pdb_id = tail[0]
        tail = tail[1:]
    if pdb_id is None:
        parser.error("pdb_id is required (e.g. 8cq9)")

    manifest = {
        "utc_iso": datetime.now(timezone.utc).isoformat(),
        "runs_n_poses_root": str(root),
        "pdb_id": pdb_id,
        "python": sys.executable,
        "env_ost_compounds_chemlib": ost,
        "foldseek_bin": foldseek,
        "cuda_visible_devices": cuda,
        "packages": _run_version_snippet(),
        "argv": [str(script), pdb_id, *tail],
    }
    manifest_path = root / "scoring" / "batch_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _log(f"Wrote manifest {manifest_path}")

    cmd = [sys.executable, str(script), pdb_id, *tail]
    _log(f"cwd={root}")
    _log(f"cmd={' '.join(cmd)}")
    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, cwd=str(root), check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
