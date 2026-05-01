"""Validate OneOPES HREX smoke outputs (exchange logs, timing, PLUMED trees).

Used by :file:`scripts/smoke/validate_oneopes_repex_smoke.py` and unit tests.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

__all__ = [
    "validate_exchange_log_two_replica_rows",
    "validate_exchange_log_multi_rows",
    "validate_repex_smoke_directory",
]


def _finite_float(s: str) -> bool:
    try:
        return math.isfinite(float(s))
    except (TypeError, ValueError):
        return False


def validate_exchange_log_two_replica_rows(rows: list[dict[str, str]]) -> list[str]:
    """Return human-readable errors for two-replica ``exchange_log.csv`` data rows."""
    errs: list[str] = []
    required = (
        "md_step",
        "u00",
        "u01",
        "u10",
        "u11",
        "log_accept",
        "accepted",
        "rng_u",
    )
    for i, row in enumerate(rows):
        for k in required:
            if k not in row or row[k] is None or str(row[k]).strip() == "":
                errs.append(f"row {i}: missing column {k!r}")
        for k in ("u00", "u01", "u10", "u11", "log_accept", "rng_u"):
            if k in row and str(row[k]).strip() != "" and not _finite_float(str(row[k])):
                errs.append(f"row {i}: non-finite {k}={row[k]!r}")
        if "accepted" in row and str(row["accepted"]).strip() not in ("0", "1"):
            errs.append(f"row {i}: accepted must be 0 or 1, got {row['accepted']!r}")
    return errs


def validate_exchange_log_multi_rows(rows: list[dict[str, str]]) -> list[str]:
    """Return errors for multi-replica (``n`` >= 3) ``exchange_log.csv`` data rows."""
    errs: list[str] = []
    required = (
        "md_step",
        "phase_mod_2",
        "pair_i",
        "pair_j",
        "u_ii",
        "u_jj",
        "u_ij",
        "u_ji",
        "log_accept",
        "accepted",
        "rng_u",
    )
    for i, row in enumerate(rows):
        for k in required:
            if k not in row or row[k] is None or str(row[k]).strip() == "":
                errs.append(f"row {i}: missing column {k!r}")
        for k in ("u_ii", "u_jj", "u_ij", "u_ji", "log_accept", "rng_u"):
            if k in row and str(row[k]).strip() != "" and not _finite_float(str(row[k])):
                errs.append(f"row {i}: non-finite {k}={row[k]!r}")
        if "accepted" in row and str(row["accepted"]).strip() not in ("0", "1"):
            errs.append(f"row {i}: accepted must be 0 or 1, got {row['accepted']!r}")
        for k in ("phase_mod_2", "pair_i", "pair_j"):
            if k in row and str(row[k]).strip() != "":
                try:
                    int(row[k])
                except ValueError:
                    errs.append(f"row {i}: {k} must be int-like, got {row[k]!r}")
    return errs


def _load_repex_config(out_root: Path) -> dict[str, Any] | None:
    p = out_root / "repex_config.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"_parse_error": str(e)}


def validate_repex_smoke_directory(
    out_root: Path,
    *,
    require_barrier_timing: bool = True,
    gpu_monitor_log: Path | None = None,
) -> list[str]:
    """Validate a completed OneOPES REPEX smoke under *out_root*.

    Parameters
    ----------
    out_root
        Run directory containing ``exchange_log.csv`` (and usually
        ``repex_config.json``, ``repNNN/``, ``eval_scratch/``).
    require_barrier_timing
        When ``True``, require non-empty ``barrier_timing.jsonl``.
    gpu_monitor_log
        Optional path to a GPU monitor log; when set, file must exist and contain
        at least one data line after comment headers.
    """
    errs: list[str] = []
    root = out_root.expanduser().resolve()
    if not root.is_dir():
        return [f"not a directory: {root}"]

    cfg = _load_repex_config(root)
    if cfg is None:
        errs.append(f"missing {root / 'repex_config.json'}")
        n_replicas: int | None = None
    elif isinstance(cfg, dict) and "_parse_error" in cfg:
        errs.append(f"repex_config.json: {cfg['_parse_error']}")
        n_replicas = None
    else:
        assert isinstance(cfg, dict)
        try:
            n_replicas = int(cfg["n_replicas"])
        except (KeyError, TypeError, ValueError):
            n_replicas = None
            errs.append("repex_config.json: missing or invalid n_replicas")

    ex_path = root / "exchange_log.csv"
    if not ex_path.is_file():
        errs.append(f"missing {ex_path}")
        return errs

    text = ex_path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        errs.append("exchange_log.csv: expected header and at least one data row")
        return errs

    with ex_path.open(newline="", encoding="utf-8") as fcsv:
        reader = csv.DictReader(fcsv)
        fieldnames = reader.fieldnames or ()
        rows = list(reader)

    if not rows:
        errs.append("exchange_log.csv: no data rows after header")
        return errs

    inferred_multi = "pair_i" in fieldnames
    if n_replicas is None:
        effective_n = 99 if inferred_multi else 2
    else:
        effective_n = int(n_replicas)

    if effective_n >= 3 or inferred_multi:
        if "pair_i" not in fieldnames:
            errs.append("exchange_log.csv: expected multi-replica header with pair_i")
        else:
            errs.extend(validate_exchange_log_multi_rows(rows))
    else:
        if "u00" not in fieldnames:
            errs.append("exchange_log.csv: expected two-replica header with u00")
        else:
            errs.extend(validate_exchange_log_two_replica_rows(rows))

    barrier = root / "barrier_timing.jsonl"
    if require_barrier_timing:
        if not barrier.is_file():
            errs.append(f"missing {barrier}")
        elif barrier.stat().st_size == 0:
            errs.append("barrier_timing.jsonl is empty")
        else:
            ok_line = False
            for raw in barrier.read_text(encoding="utf-8", errors="replace").splitlines():
                s = raw.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict) and math.isfinite(float(obj.get("elapsed_s", 0.0))):
                        ok_line = True
                        break
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
            if not ok_line:
                errs.append("barrier_timing.jsonl: no parseable JSON lines with finite elapsed_s")

    rep_pat = re.compile(r"^rep\d{3}$")
    for child in sorted(root.iterdir()):
        if child.is_dir() and rep_pat.match(child.name):
            opes = child / "opes_states"
            if not opes.is_dir():
                errs.append(f"missing opes_states under {child}")
                continue
            if not (opes / "KERNELS").is_file() and not any(opes.glob("KERNELS*")):
                errs.append(f"{opes}: expected KERNELS or KERNELS_*")
            if not (opes / "STATE").is_file():
                errs.append(f"{opes}: missing STATE")

    scratch = root / "eval_scratch"
    if scratch.is_dir():
        for sub in sorted(scratch.iterdir()):
            if not sub.is_dir():
                continue
            deck = sub / "plumed_opes.dat"
            if not deck.is_file():
                continue
            body = deck.read_text(encoding="utf-8", errors="replace")
            scratch_marker = str(sub.resolve())
            if scratch_marker not in body and str(sub) not in body:
                if "/eval_scratch/" not in body.replace("\\", "/") and "eval_scratch" not in body:
                    errs.append(
                        f"{deck}: expected relocated scratch paths "
                        f"(evaluator deck should reference eval_scratch tree)"
                    )

    if gpu_monitor_log is not None:
        gpath = gpu_monitor_log.expanduser().resolve()
        if not gpath.is_file():
            errs.append(f"missing gpu monitor log {gpath}")
        else:
            data_lines = 0
            for ln in gpath.read_text(encoding="utf-8", errors="replace").splitlines():
                if ln.startswith("#") or not ln.strip():
                    continue
                data_lines += 1
            if data_lines == 0:
                errs.append(f"{gpath}: no non-comment data lines")

    return errs
