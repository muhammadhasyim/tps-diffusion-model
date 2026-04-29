#!/usr/bin/env python3
"""Multi-batch OPES-TPS orchestrator.

Manages batched restarts of run_opes_tps.py across sandbox sessions.
Each batch gets its own output directory with isolated WDSM samples and logs.
OPES bias state is chained via --opes-restart from the previous batch.

Usage (run inside a sandbox with genai-tps installed):

    python /mnt/shared/26-04-22-tps-diffusion/tps-diffusion-model/examples/orchestrator.py \
        --base-dir /mnt/shared/26-04-22-tps-diffusion/opes_wdsm_12k_long \
        --target-steps 12000 \
        --steps-per-batch 1500

After all batches complete, run with --merge-only to assemble the final dataset:

    python orchestrator.py \
        --base-dir /mnt/shared/26-04-22-tps-diffusion/opes_wdsm_12k_long \
        --merge-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def find_batches(base_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted list of (batch_num, batch_dir) for existing batch_NNN dirs."""
    batches = []
    for d in sorted(base_dir.iterdir()):
        m = re.match(r"batch_(\d+)$", d.name)
        if m and d.is_dir():
            batches.append((int(m.group(1)), d))
    return batches


def count_wdsm_samples(batch_dir: Path) -> int:
    wdsm_dir = batch_dir / "wdsm_samples"
    if not wdsm_dir.is_dir():
        return 0
    return len([f for f in wdsm_dir.iterdir() if f.name.startswith("wdsm_step_")])


def get_latest_opes_state(batch_dir: Path) -> Path | None:
    final = batch_dir / "opes_state_final.json"
    if final.is_file():
        return final
    states_dir = batch_dir / "opes_states"
    if states_dir.is_dir():
        latest = states_dir / "opes_state_latest.json"
        if latest.is_file():
            return latest
        checkpoints = sorted(states_dir.glob("opes_state_*.json"))
        checkpoints = [c for c in checkpoints if "latest" not in c.name]
        if checkpoints:
            return checkpoints[-1]
    return None


def get_total_steps(base_dir: Path) -> int:
    total = 0
    for _, batch_dir in find_batches(base_dir):
        total += count_wdsm_samples(batch_dir)
    return total


def promote_batch1(base_dir: Path) -> None:
    """Move flat batch-1 outputs into batch_001/ directory."""
    batch1_dir = base_dir / "batch_001"
    if batch1_dir.is_dir():
        return

    wdsm_dir = base_dir / "wdsm_samples"
    if not wdsm_dir.is_dir():
        return

    print(f"[orchestrator] Promoting batch 1 outputs to {batch1_dir}")
    batch1_dir.mkdir()

    items_to_move = [
        "wdsm_samples", "opes_states", "opes_state_final.json",
        "tps_steps.jsonl", "cv_values.json", "opes_tps_summary.json",
        "shooting_log.txt", "trajectory_checkpoints",
        "run_stdout.log", "run_stderr.log", "run_pid.txt",
    ]
    for item in items_to_move:
        src = base_dir / item
        if src.exists():
            dst = batch1_dir / item
            shutil.move(str(src), str(dst))
            print(f"  Moved {item}")

    boltz_dirs = list(base_dir.glob("boltz_results_*"))
    for bd in boltz_dirs:
        dst = batch1_dir / bd.name
        shutil.move(str(bd), str(dst))
        print(f"  Moved {bd.name}")


def run_batch(
    base_dir: Path,
    batch_num: int,
    steps: int,
    opes_restart: Path | None,
    yaml_path: Path,
    script_path: Path,
) -> Path:
    batch_dir = base_dir / f"batch_{batch_num:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script_path),
        "--yaml", str(yaml_path),
        "--out", str(batch_dir),
        "--diffusion-steps", "32",
        "--shoot-rounds", str(steps),
        "--bias-cv", "ligand_rmsd,ligand_pocket_dist",
        "--opes-barrier", "5.0",
        "--opes-biasfactor", "10.0",
        "--opes-pace", "1",
        "--save-wdsm-data-every", "1",
        "--save-trajectory-every", "100",
        "--save-opes-state-every", "500",
        "--progress-every", "250",
        "--inference-dtype", "bfloat16",
    ]
    if opes_restart is not None:
        cmd.extend(["--opes-restart", str(opes_restart)])

    print(f"\n{'='*60}")
    print(f"  BATCH {batch_num:03d}: {steps} steps")
    if opes_restart:
        print(f"  OPES restart: {opes_restart}")
    print(f"  Output: {batch_dir}")
    print(f"{'='*60}\n")

    stdout_log = batch_dir / "run_stdout.log"
    stderr_log = batch_dir / "run_stderr.log"

    with open(stdout_log, "w") as out_f, open(stderr_log, "w") as err_f:
        proc = subprocess.run(cmd, stdout=out_f, stderr=err_f)

    print(f"[orchestrator] Batch {batch_num:03d} exited with code {proc.returncode}")

    n_samples = count_wdsm_samples(batch_dir)
    opes_state = get_latest_opes_state(batch_dir)
    print(f"[orchestrator] WDSM samples: {n_samples}, OPES state: {opes_state}")

    return batch_dir


def merge_wdsm(base_dir: Path, min_samples: int = 100) -> None:
    """Merge all batch WDSM samples into merged/wdsm_all/ with global step numbering.

    Batches with fewer than ``min_samples`` WDSM files are skipped to exclude
    interrupted runs that may have inconsistent OPES state chains.
    """
    merged_dir = base_dir / "merged"
    wdsm_all = merged_dir / "wdsm_all"
    wdsm_all.mkdir(parents=True, exist_ok=True)

    batches = find_batches(base_dir)
    global_step = 0
    total_copied = 0

    for batch_num, batch_dir in batches:
        wdsm_dir = batch_dir / "wdsm_samples"
        if not wdsm_dir.is_dir():
            continue
        files = sorted(wdsm_dir.glob("wdsm_step_*.npz"))
        if len(files) < min_samples:
            print(f"[merge] Batch {batch_num:03d}: SKIPPED ({len(files)} < {min_samples} samples)")
            continue
        print(f"[merge] Batch {batch_num:03d}: {len(files)} samples (global offset {global_step})")

        for f in files:
            m = re.search(r"wdsm_step_(\d+)\.npz$", f.name)
            if not m:
                continue
            local_step = int(m.group(1))
            g_step = global_step + local_step
            dst = wdsm_all / f"wdsm_step_{g_step:08d}.npz"
            shutil.copy2(str(f), str(dst))
            total_copied += 1

        global_step += len(files)

    print(f"[merge] Total: {total_copied} samples merged into {wdsm_all}")

    jsonl_merged = merged_dir / "tps_steps_merged.jsonl"
    global_step = 0
    with open(jsonl_merged, "w") as out_f:
        for batch_num, batch_dir in batches:
            jsonl = batch_dir / "tps_steps.jsonl"
            if not jsonl.is_file():
                continue
            n_in_batch = count_wdsm_samples(batch_dir)
            with open(jsonl) as in_f:
                for line in in_f:
                    try:
                        entry = json.loads(line)
                        entry["global_step"] = global_step + entry.get("step", 0)
                        entry["batch"] = batch_num
                        out_f.write(json.dumps(entry) + "\n")
                    except json.JSONDecodeError:
                        continue
            global_step += n_in_batch

    print(f"[merge] Merged TPS log: {jsonl_merged}")

    logws = []
    for f in sorted(wdsm_all.glob("wdsm_step_*.npz")):
        d = np.load(f)
        logws.append(float(d["logw"]))
    if logws:
        logws = np.array(logws)
        w = np.exp(logws - logws.max())
        w /= w.sum()
        n_eff = 1.0 / np.sum(w**2)
        analysis = {
            "n_samples": len(logws),
            "n_eff": float(n_eff),
            "n_eff_fraction": float(n_eff / len(logws)),
            "logw_mean": float(logws.mean()),
            "logw_std": float(logws.std()),
            "logw_min": float(logws.min()),
            "logw_max": float(logws.max()),
            "max_weight_fraction": float(w.max()),
        }
        analysis_path = merged_dir / "logw_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"[merge] logw analysis: N_eff={n_eff:.1f} ({n_eff/len(logws)*100:.1f}%), "
              f"std={logws.std():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Multi-batch OPES-TPS orchestrator")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, default=12000)
    parser.add_argument("--steps-per-batch", type=int, default=1500)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument(
        "--yaml", type=Path,
        default=Path("/mnt/shared/26-04-22-tps-diffusion/tps-diffusion-model/inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml"),
    )
    parser.add_argument(
        "--script", type=Path,
        default=Path("/mnt/shared/26-04-22-tps-diffusion/tps-diffusion-model/scripts/run_opes_tps.py"),
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge_wdsm(base_dir)
        return

    promote_batch1(base_dir)

    batches = find_batches(base_dir)
    total_steps = get_total_steps(base_dir)
    print(f"[orchestrator] Found {len(batches)} existing batches, {total_steps} total steps")

    if total_steps >= args.target_steps:
        print(f"[orchestrator] Target {args.target_steps} already reached. Run --merge-only to assemble.")
        return

    opes_restart = None
    if batches:
        last_batch_dir = batches[-1][1]
        opes_restart = get_latest_opes_state(last_batch_dir)
        if opes_restart is None:
            print(f"[orchestrator] WARNING: No OPES state found in {last_batch_dir}")

    next_batch_num = (batches[-1][0] + 1) if batches else 1
    remaining = args.target_steps - total_steps
    steps_this_batch = min(remaining, args.steps_per_batch)

    print(f"[orchestrator] Next batch: {next_batch_num:03d}, steps: {steps_this_batch}, "
          f"remaining: {remaining}")

    run_batch(
        base_dir, next_batch_num, steps_this_batch,
        opes_restart, args.yaml, args.script,
    )

    total_after = get_total_steps(base_dir)
    print(f"\n[orchestrator] Total steps after this batch: {total_after}/{args.target_steps}")

    if total_after >= args.target_steps:
        print("[orchestrator] Target reached! Running merge...")
        merge_wdsm(base_dir)
    else:
        print(f"[orchestrator] {args.target_steps - total_after} steps remaining. "
              f"Launch orchestrator again in a new sandbox.")


if __name__ == "__main__":
    main()
