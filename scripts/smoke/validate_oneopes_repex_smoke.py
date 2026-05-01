#!/usr/bin/env python3
"""Validate OneOPES HREX smoke output directory (exchange log, timing, PLUMED)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from genai_tps.simulation.oneopes_repex_smoke_validate import validate_repex_smoke_directory


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "out_root",
        type=Path,
        help="Run directory (contains exchange_log.csv, repex_config.json, repNNN/)",
    )
    p.add_argument(
        "--no-barrier-timing",
        action="store_true",
        help="Do not require barrier_timing.jsonl",
    )
    p.add_argument(
        "--gpu-monitor-log",
        type=Path,
        default=None,
        help="Optional path to gpu_monitor_*.log from run_gpu_monitor.sh",
    )
    args = p.parse_args()
    errs = validate_repex_smoke_directory(
        args.out_root,
        require_barrier_timing=not args.no_barrier_timing,
        gpu_monitor_log=args.gpu_monitor_log,
    )
    if errs:
        for e in errs:
            print(e, file=sys.stderr)
        return 1
    print("OK:", args.out_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
