#!/usr/bin/env python3
"""Fully automated multi-batch OPES-TPS pipeline.

Runs LOCALLY on the user's machine. Orchestrates remote GPU sandboxes via
the modal-tools CLI to reach a target step count without manual intervention.

Each iteration:
  1. Checks total WDSM samples on the shared volume
  2. If target reached → runs merge and exits
  3. Creates a fresh A100 sandbox (3h timeout)
  4. Installs boltz + genai-tps
  5. Promotes batch-1 flat outputs if needed
  6. Runs orchestrator.py for one batch (blocking)
  7. Terminates sandbox
  8. Loops back to step 1

Usage:
    python examples/auto_pipeline.py --target-steps 12000

    # Resume after interruption (just rerun — it's idempotent):
    python examples/auto_pipeline.py --target-steps 12000

    # Dry run (check state without launching):
    python examples/auto_pipeline.py --target-steps 12000 --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MODAL_TOOLS = "modal-tools"
BASE_DIR = "/mnt/shared/26-04-22-tps-diffusion/opes_wdsm_12k_long"
TPS_REPO = "/mnt/shared/26-04-22-tps-diffusion/tps-diffusion-model"
IMAGE = "chemistry-py3.12-gpu"
GPU = "A100-80GB"
MEMORY = 32768
SANDBOX_TIMEOUT = 10800
STEPS_PER_BATCH = 1500
POLL_INTERVAL = 120  # seconds between progress checks


def run_cli(args: list[str], timeout: int = 60) -> dict:
    result = subprocess.run(
        [MODAL_TOOLS] + args,
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"modal-tools {' '.join(args)} failed:\n{result.stderr}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"raw": result.stdout.strip()}


def sandbox_exec(sandbox_id: str, code: str, timeout: int = 300) -> dict:
    result = subprocess.run(
        [MODAL_TOOLS, "sandbox", "execute-python", sandbox_id,
         "--code", code, "--timeout", str(timeout), "--json"],
        capture_output=True, text=True, timeout=timeout + 60,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"stdout": result.stdout, "stderr": result.stderr}


def create_sandbox() -> str:
    print(f"\n[pipeline] Creating {GPU} sandbox ({SANDBOX_TIMEOUT}s timeout)...")
    result = run_cli([
        "sandbox", "create",
        "--image", IMAGE,
        "--gpu", GPU,
        "--memory", str(MEMORY),
        "--timeout", str(SANDBOX_TIMEOUT),
        "--json",
    ], timeout=120)
    sid = result.get("sandbox_id")
    if not sid:
        raise RuntimeError(f"Failed to create sandbox: {result}")
    print(f"[pipeline] Sandbox created: {sid}")
    return sid


def terminate_sandbox(sandbox_id: str) -> None:
    print(f"[pipeline] Terminating sandbox {sandbox_id}...")
    try:
        run_cli(["sandbox", "terminate", sandbox_id], timeout=30)
    except Exception as e:
        print(f"[pipeline] Warning: terminate failed: {e}")


def install_deps(sandbox_id: str) -> None:
    print("[pipeline] Installing boltz...")
    r = sandbox_exec(sandbox_id, f"""
import subprocess
r1 = subprocess.run(["pip", "install", "-e", "{TPS_REPO}/boltz"],
                    capture_output=True, text=True, timeout=600)
print(f"boltz: rc={{r1.returncode}}")
if r1.returncode != 0:
    print(r1.stderr[-500:])
""", timeout=600)
    print(f"  {r.get('stdout', '').strip()[:200]}")

    print("[pipeline] Installing genai-tps...")
    r = sandbox_exec(sandbox_id, f"""
import subprocess
r2 = subprocess.run(["pip", "install", "-e", "{TPS_REPO}/[boltz]"],
                    capture_output=True, text=True, timeout=600)
print(f"genai-tps: rc={{r2.returncode}}")
if r2.returncode != 0:
    print(r2.stderr[-500:])
""", timeout=600)
    print(f"  {r.get('stdout', '').strip()[:200]}")


def get_total_steps(sandbox_id: str) -> int:
    r = sandbox_exec(sandbox_id, f"""
import os, re
base = "{BASE_DIR}"
total = 0
# Check batch_NNN dirs
for d in sorted(os.listdir(base)):
    m = re.match(r"batch_(\\d+)$", d)
    if m and os.path.isdir(os.path.join(base, d)):
        wdsm = os.path.join(base, d, "wdsm_samples")
        if os.path.isdir(wdsm):
            total += len([f for f in os.listdir(wdsm) if f.startswith("wdsm_step_")])
# Also check flat wdsm_samples (pre-promotion batch 1)
flat = os.path.join(base, "wdsm_samples")
if os.path.isdir(flat):
    total += len([f for f in os.listdir(flat) if f.startswith("wdsm_step_")])
print(total)
""", timeout=30)
    try:
        return int(r.get("stdout", "0").strip())
    except ValueError:
        return 0


def launch_orchestrator_background(sandbox_id: str, target: int) -> str:
    """Launch orchestrator as a background process. Returns the PID."""
    code = f"""
import subprocess, sys, os

cmd = [
    sys.executable, "{TPS_REPO}/examples/orchestrator.py",
    "--base-dir", "{BASE_DIR}",
    "--target-steps", "{target}",
    "--steps-per-batch", "{STEPS_PER_BATCH}",
]

log_out = open("{BASE_DIR}/_pipeline_batch_stdout.log", "w")
log_err = open("{BASE_DIR}/_pipeline_batch_stderr.log", "w")
proc = subprocess.Popen(cmd, stdout=log_out, stderr=log_err, start_new_session=True)
print(proc.pid)
"""
    r = sandbox_exec(sandbox_id, code, timeout=60)
    pid = r.get("stdout", "").strip()
    print(f"[pipeline] Orchestrator launched as background PID: {pid}")
    return pid


def poll_batch_progress(sandbox_id: str, pid: str) -> tuple[int, bool]:
    """Check WDSM count and whether the process is still running."""
    code = f"""
import os, re, subprocess

result = subprocess.run(["ps", "-p", "{pid}", "-o", "pid", "--no-headers"],
                        capture_output=True, text=True)
alive = bool(result.stdout.strip())

base = "{BASE_DIR}"
total = 0
for d in sorted(os.listdir(base)):
    m = re.match(r"batch_(\\d+)$", d)
    if m and os.path.isdir(os.path.join(base, d)):
        wdsm = os.path.join(base, d, "wdsm_samples")
        if os.path.isdir(wdsm):
            total += len([f for f in os.listdir(wdsm) if f.startswith("wdsm_step_")])
flat = os.path.join(base, "wdsm_samples")
if os.path.isdir(flat):
    total += len([f for f in os.listdir(flat) if f.startswith("wdsm_step_")])

print(f"{{total}} {{alive}}")
"""
    r = sandbox_exec(sandbox_id, code, timeout=30)
    stdout = r.get("stdout", "0 False").strip()
    parts = stdout.split()
    try:
        total = int(parts[0])
        alive = parts[1] == "True"
    except (IndexError, ValueError):
        total = 0
        alive = False
    return total, alive


def run_orchestrator_batch(sandbox_id: str, target: int) -> None:
    """Launch orchestrator in background and poll until done or sandbox nears timeout."""
    print(f"[pipeline] Launching orchestrator (target={target}, batch_size={STEPS_PER_BATCH})...")
    pid = launch_orchestrator_background(sandbox_id, target)

    start_time = time.time()
    max_wait = SANDBOX_TIMEOUT - 600  # stop polling 10 min before sandbox timeout
    prev_total = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            print(f"[pipeline] Approaching sandbox timeout ({elapsed:.0f}s/{max_wait:.0f}s). Ending batch.")
            break

        time.sleep(POLL_INTERVAL)

        try:
            total, alive = poll_batch_progress(sandbox_id, pid)
        except Exception as e:
            print(f"[pipeline] Poll error: {e}")
            continue

        rate = (total - prev_total) / (POLL_INTERVAL / 60) if prev_total > 0 else 0
        print(f"[pipeline] Progress: {total} steps | alive={alive} | "
              f"+{total - prev_total} since last poll ({rate:.1f}/min) | "
              f"{elapsed/60:.0f}min elapsed")
        prev_total = total

        if not alive:
            print(f"[pipeline] Orchestrator process exited. Final steps: {total}")
            break

    # Read batch logs
    try:
        r = sandbox_exec(sandbox_id, f"""
with open("{BASE_DIR}/_pipeline_batch_stdout.log") as f:
    lines = f.readlines()
for l in lines[-15:]:
    print(l.rstrip())
""", timeout=15)
        log_out = r.get("stdout", "")
        if log_out.strip():
            print(f"[pipeline] Batch stdout (last 15 lines):\n{log_out}")
    except Exception:
        pass


def run_merge(sandbox_id: str) -> None:
    print("[pipeline] Running final merge...")
    code = f"""
import subprocess, sys
cmd = [
    sys.executable, "{TPS_REPO}/examples/orchestrator.py",
    "--base-dir", "{BASE_DIR}",
    "--merge-only",
]
proc = subprocess.run(cmd, timeout=600)
print(f"merge exit code: {{proc.returncode}}")
"""
    r = sandbox_exec(sandbox_id, code, timeout=660)
    print(f"[pipeline] Merge output:\n{r.get('stdout', '')[-1000:]}")


def find_topo_npz(sandbox_id: str) -> str:
    """Find the topology NPZ from batch_001's Boltz results."""
    r = sandbox_exec(sandbox_id, f"""
import glob
hits = sorted(glob.glob("{BASE_DIR}/batch_001/boltz_results_*/processed/structures/*.npz"))
if not hits:
    hits = sorted(glob.glob("{BASE_DIR}/batch_005/boltz_results_*/processed/structures/*.npz"))
print(hits[0] if hits else "NOT_FOUND")
""", timeout=15)
    return r.get("stdout", "NOT_FOUND").strip()


TRAINING_DIR = f"{BASE_DIR}/wdsm_training"
EVAL_DIR = f"{BASE_DIR}/evaluation"
MAX_LOG_RATIO = 8.0
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 4
TRAIN_LR = 3e-6
TRAIN_BETA = 0.01


def run_assemble(sandbox_id: str) -> None:
    """Assemble merged WDSM samples into a single training NPZ."""
    topo = find_topo_npz(sandbox_id)
    print(f"[pipeline] Assembling training NPZ (max_log_ratio={MAX_LOG_RATIO})...")
    print(f"[pipeline] Topology NPZ: {topo}")
    code = f"""
import subprocess, sys
cmd = [
    sys.executable, "{TPS_REPO}/scripts/assemble_wdsm_dataset.py",
    "--wdsm-dir", "{BASE_DIR}/merged/wdsm_all",
    "--topo-npz", "{topo}",
    "--output", "{BASE_DIR}/merged/training_data.npz",
    "--max-log-ratio", "{MAX_LOG_RATIO}",
]
proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
print(proc.stdout[-1500:])
if proc.returncode != 0:
    print("STDERR:", proc.stderr[-500:])
"""
    r = sandbox_exec(sandbox_id, code, timeout=660)
    print(f"[pipeline] Assemble output:\n{r.get('stdout', '')[-1000:]}")


def run_split(sandbox_id: str) -> None:
    """Split assembled NPZ into train/val."""
    print("[pipeline] Splitting into train/val (80/20)...")
    code = f"""
import subprocess, sys
cmd = [
    sys.executable, "{TPS_REPO}/scripts/split_wdsm_dataset.py",
    "--data", "{BASE_DIR}/merged/training_data.npz",
    "--output-dir", "{BASE_DIR}/merged",
    "--val-fraction", "0.2",
    "--seed", "42",
]
proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
print(proc.stdout)
if proc.returncode != 0:
    print("STDERR:", proc.stderr[-500:])
"""
    r = sandbox_exec(sandbox_id, code, timeout=360)
    print(f"[pipeline] Split output:\n{r.get('stdout', '')[-500:]}")


def run_training(sandbox_id: str) -> None:
    """Launch WDSM training as background process, poll until done or timeout."""
    print(f"[pipeline] Launching WDSM training ({TRAIN_EPOCHS} epochs, bs={TRAIN_BATCH_SIZE})...")

    code = f"""
import subprocess, sys, os
os.makedirs("{TRAINING_DIR}", exist_ok=True)
cmd = [
    sys.executable, "{TPS_REPO}/scripts/train_weighted_dsm.py",
    "--yaml", "{TPS_REPO}/inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml",
    "--data", "{BASE_DIR}/merged/train.npz",
    "--val-data", "{BASE_DIR}/merged/val.npz",
    "--out", "{TRAINING_DIR}",
    "--epochs", "{TRAIN_EPOCHS}",
    "--batch-size", "{TRAIN_BATCH_SIZE}",
    "--learning-rate", "{TRAIN_LR}",
    "--beta", "{TRAIN_BETA}",
    "--max-grad-norm", "1.0",
    "--checkpoint-every", "2",
    "--device", "cuda",
]
log_out = open("{TRAINING_DIR}/pipeline_stdout.log", "w")
log_err = open("{TRAINING_DIR}/pipeline_stderr.log", "w")
proc = subprocess.Popen(cmd, stdout=log_out, stderr=log_err, start_new_session=True)
print(proc.pid)
"""
    r = sandbox_exec(sandbox_id, code, timeout=60)
    pid = r.get("stdout", "").strip()
    print(f"[pipeline] Training launched as PID {pid}")

    start_time = time.time()
    max_wait = SANDBOX_TIMEOUT - 600

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            print(f"[pipeline] Approaching sandbox timeout. Training may need continuation.")
            break

        time.sleep(POLL_INTERVAL)

        try:
            check = sandbox_exec(sandbox_id, f"""
import subprocess, os
result = subprocess.run(["ps", "-p", "{pid}", "-o", "pid", "--no-headers"],
                        capture_output=True, text=True)
alive = bool(result.stdout.strip())
# Check latest epoch from log
ckpts = sorted([f for f in os.listdir("{TRAINING_DIR}") if f.startswith("boltz2_wdsm_epoch_")])
latest_ckpt = ckpts[-1] if ckpts else "none"
final = os.path.exists("{TRAINING_DIR}/boltz2_wdsm_final.pt")
print(f"alive={{alive}} latest_ckpt={{latest_ckpt}} final={{final}}")
""", timeout=20)
            status = check.get("stdout", "").strip()
            print(f"[pipeline] Training: {status} | {elapsed/60:.0f}min elapsed")

            if "final=True" in status:
                print("[pipeline] Training complete (final checkpoint saved).")
                break
            if "alive=False" in status:
                print("[pipeline] Training process exited.")
                break
        except Exception as e:
            print(f"[pipeline] Training poll error: {e}")

    try:
        r = sandbox_exec(sandbox_id, f"""
with open("{TRAINING_DIR}/pipeline_stdout.log") as f:
    lines = f.readlines()
for l in lines[-20:]:
    print(l.rstrip())
""", timeout=15)
        print(f"[pipeline] Training log tail:\n{r.get('stdout', '')[-1500:]}")
    except Exception:
        pass


def run_evaluation(sandbox_id: str) -> None:
    """Launch evaluation comparing baseline vs fine-tuned model."""
    print("[pipeline] Launching evaluation (baseline vs fine-tuned)...")

    code = f"""
import subprocess, sys, os
os.makedirs("{EVAL_DIR}", exist_ok=True)
cmd = [
    sys.executable, "{TPS_REPO}/scripts/evaluate_wdsm_model.py",
    "--yaml", "{TPS_REPO}/inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml",
    "--finetuned-checkpoint", "{TRAINING_DIR}/boltz2_wdsm_final.pt",
    "--out", "{EVAL_DIR}",
    "--n-samples", "200",
    "--diffusion-steps", "32",
    "--device", "cuda",
]
log_out = open("{EVAL_DIR}/pipeline_stdout.log", "w")
log_err = open("{EVAL_DIR}/pipeline_stderr.log", "w")
proc = subprocess.Popen(cmd, stdout=log_out, stderr=log_err, start_new_session=True)
print(proc.pid)
"""
    r = sandbox_exec(sandbox_id, code, timeout=60)
    pid = r.get("stdout", "").strip()
    print(f"[pipeline] Evaluation launched as PID {pid}")

    start_time = time.time()
    max_wait = SANDBOX_TIMEOUT - 600

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            print(f"[pipeline] Approaching sandbox timeout for evaluation.")
            break

        time.sleep(POLL_INTERVAL)

        try:
            check = sandbox_exec(sandbox_id, f"""
import subprocess, os
result = subprocess.run(["ps", "-p", "{pid}", "-o", "pid", "--no-headers"],
                        capture_output=True, text=True)
alive = bool(result.stdout.strip())
report = os.path.exists("{EVAL_DIR}/evaluation_report.json")
dashboard = os.path.exists("{EVAL_DIR}/evaluation_dashboard.png")
print(f"alive={{alive}} report={{report}} dashboard={{dashboard}}")
""", timeout=20)
            status = check.get("stdout", "").strip()
            print(f"[pipeline] Evaluation: {status} | {elapsed/60:.0f}min elapsed")

            if "report=True" in status:
                print("[pipeline] Evaluation complete!")
                break
            if "alive=False" in status:
                print("[pipeline] Evaluation process exited.")
                break
        except Exception as e:
            print(f"[pipeline] Eval poll error: {e}")

    try:
        r = sandbox_exec(sandbox_id, f"""
with open("{EVAL_DIR}/pipeline_stdout.log") as f:
    lines = f.readlines()
for l in lines[-25:]:
    print(l.rstrip())
""", timeout=15)
        print(f"[pipeline] Evaluation log tail:\n{r.get('stdout', '')[-2000:]}")
    except Exception:
        pass


def run_post_processing(sandbox_id: str) -> None:
    """Run the full post-processing pipeline: assemble → split → train → evaluate."""
    print(f"\n{'='*60}")
    print(f"  POST-PROCESSING PIPELINE")
    print(f"{'='*60}\n")

    run_assemble(sandbox_id)
    run_split(sandbox_id)
    terminate_sandbox(sandbox_id)

    print("\n[pipeline] Creating A100 sandbox for training...")
    train_sid = create_sandbox()
    install_deps(train_sid)
    run_training(train_sid)

    has_final = False
    try:
        r = sandbox_exec(train_sid, f"""
import os
print(os.path.exists("{TRAINING_DIR}/boltz2_wdsm_final.pt"))
""", timeout=10)
        has_final = "True" in r.get("stdout", "")
    except Exception:
        pass

    terminate_sandbox(train_sid)

    if has_final:
        print("\n[pipeline] Creating A100 sandbox for evaluation...")
        eval_sid = create_sandbox()
        install_deps(eval_sid)
        run_evaluation(eval_sid)
        terminate_sandbox(eval_sid)
    else:
        print("[pipeline] Training did not produce final checkpoint. Skipping evaluation.")


def main():
    parser = argparse.ArgumentParser(description="Automated multi-batch OPES-TPS pipeline")
    parser.add_argument("--target-steps", type=int, default=12000)
    parser.add_argument("--max-batches", type=int, default=15,
                        help="Safety limit on total batches to prevent runaway")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check state and print plan without launching")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  OPES-TPS Automated Pipeline")
    print(f"  Target: {args.target_steps} steps")
    print(f"  Batch size: {STEPS_PER_BATCH} steps")
    print(f"  GPU: {GPU}, Timeout: {SANDBOX_TIMEOUT}s")
    print(f"{'='*60}")

    batch_count = 0
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 3

    while batch_count < args.max_batches:
        sandbox_id = None
        try:
            sandbox_id = create_sandbox()
            install_deps(sandbox_id)

            total = get_total_steps(sandbox_id)
            print(f"\n[pipeline] Current total steps: {total}/{args.target_steps}")

            if total >= args.target_steps:
                print("[pipeline] Target reached! Running merge + post-processing...")
                run_merge(sandbox_id)
                run_post_processing(sandbox_id)
                print(f"\n{'='*60}")
                print(f"  FULL PIPELINE COMPLETE")
                print(f"  Total steps: {total}")
                print(f"  Batches used: {batch_count}")
                print(f"{'='*60}")
                return

            remaining = args.target_steps - total
            est_batches = (remaining + STEPS_PER_BATCH - 1) // STEPS_PER_BATCH
            est_hours = est_batches * 3.3
            print(f"[pipeline] Remaining: {remaining} steps (~{est_batches} batches, ~{est_hours:.0f}h)")

            if args.dry_run:
                print("[pipeline] Dry run — exiting without launching batch")
                terminate_sandbox(sandbox_id)
                return

            consecutive_errors = 0
            batch_count += 1
            print(f"\n[pipeline] === BATCH {batch_count} ===")
            run_orchestrator_batch(sandbox_id, args.target_steps)

            total_after = get_total_steps(sandbox_id)
            print(f"[pipeline] Steps after batch: {total_after}/{args.target_steps}")

            terminate_sandbox(sandbox_id)
            sandbox_id = None

            if total_after >= args.target_steps:
                sandbox_id = create_sandbox()
                install_deps(sandbox_id)
                run_merge(sandbox_id)
                run_post_processing(sandbox_id)
                print(f"\n{'='*60}")
                print(f"  FULL PIPELINE COMPLETE")
                print(f"  Total steps: {total_after}")
                print(f"  Batches used: {batch_count}")
                print(f"{'='*60}")
                return

            print(f"[pipeline] Batch {batch_count} done. Waiting 30s before next...")
            time.sleep(30)

        except KeyboardInterrupt:
            print("\n[pipeline] Interrupted by user")
            if sandbox_id:
                terminate_sandbox(sandbox_id)
            return
        except Exception as e:
            consecutive_errors += 1
            print(f"\n[pipeline] ERROR ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}")
            if sandbox_id:
                try:
                    terminate_sandbox(sandbox_id)
                except Exception:
                    pass
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"[pipeline] {MAX_CONSECUTIVE_ERRORS} consecutive errors — aborting.")
                return
            print("[pipeline] Waiting 60s before retry...")
            time.sleep(60)

    print(f"[pipeline] Max batches ({args.max_batches}) reached without completing target")


if __name__ == "__main__":
    main()
