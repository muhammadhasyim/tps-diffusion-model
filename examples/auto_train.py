#!/usr/bin/env python3
"""Automated multi-sandbox WDSM training with checkpoint resumption.

Runs LOCALLY. Each iteration:
  1. Creates A100 sandbox
  2. Installs deps
  3. Finds latest checkpoint on volume
  4. Launches training with --resume-from
  5. Polls until done or sandbox nears timeout
  6. Repeats until target epochs reached

Usage:
    PYTHONUNBUFFERED=1 nohup python -u examples/auto_train.py > auto_train.log 2>&1 &
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

MODAL_TOOLS = "modal-tools"
IMAGE = "chemistry-py3.12-gpu"
GPU = "A100-80GB"
MEMORY = 32768
SANDBOX_TIMEOUT = 10800
POLL_INTERVAL = 120

TPS_REPO = "/mnt/shared/26-04-22-tps-diffusion/tps-diffusion-model"
OUT = "/mnt/shared/26-04-22-tps-diffusion/openmm_opes_md"
TRAIN_DIR = f"{OUT}/wdsm_training_quotient"
TARGET_EPOCHS = 1000


def run_cli(args, timeout=60):
    r = subprocess.run([MODAL_TOOLS] + args, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed: {r.stderr[:300]}")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"raw": r.stdout.strip()}


def sandbox_exec(sid, code, timeout=300):
    r = subprocess.run(
        [MODAL_TOOLS, "sandbox", "execute-python", sid, "--code", code, "--timeout", str(timeout), "--json"],
        capture_output=True, text=True, timeout=timeout + 60)
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"stdout": r.stdout, "stderr": r.stderr}


def create_sandbox():
    print(f"[train] Creating {GPU} sandbox...", flush=True)
    r = run_cli(["sandbox", "create", "--image", IMAGE, "--gpu", GPU,
                 "--memory", str(MEMORY), "--timeout", str(SANDBOX_TIMEOUT), "--json"], timeout=120)
    sid = r.get("sandbox_id")
    print(f"[train] Sandbox: {sid}", flush=True)
    return sid


def install_deps(sid):
    print("[train] Installing deps...", flush=True)
    sandbox_exec(sid, f"""
import subprocess
subprocess.run(["pip", "install",
    "-e", "{TPS_REPO}/boltz",
    "-e", "{TPS_REPO}/[boltz]"],
    capture_output=True, text=True, timeout=300)
print("ok")
""", timeout=360)


def get_training_state(sid):
    r = sandbox_exec(sid, f"""
import os, re, json
d = "{TRAIN_DIR}"
if not os.path.isdir(d):
    print("0 none false")
else:
    ckpts = sorted([f for f in os.listdir(d) if re.match(r"boltz2_wdsm_epoch_\\d+\\.pt", f)])
    latest = ckpts[-1] if ckpts else "none"
    m = re.search(r"epoch_(\\d+)", latest) if latest != "none" else None
    epoch_num = int(m.group(1)) if m else 0
    # Check for early stopping
    summary_path = os.path.join(d, "training_summary.json")
    converged = False
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            converged = summary.get("stopped_early", False)
        except Exception:
            pass
    print(f"{{epoch_num}} {{latest}} {{converged}}")
""", timeout=15)
    parts = r.get("stdout", "0 none false").strip().split()
    epoch_num = int(parts[0]) if parts else 0
    latest_ckpt = parts[1] if len(parts) > 1 else "none"
    converged = parts[2] == "True" if len(parts) > 2 else False
    return epoch_num, latest_ckpt, converged


def launch_training(sid, resume_ckpt, start_epoch):
    remaining = TARGET_EPOCHS - start_epoch + 1
    print(f"[train] Launching: start_epoch={start_epoch}, remaining={remaining}, "
          f"resume={'from ' + resume_ckpt if resume_ckpt != 'none' else 'fresh'}", flush=True)

    resume_args = ""
    if resume_ckpt != "none":
        resume_args = f'"--resume-from", "{TRAIN_DIR}/{resume_ckpt}", "--start-epoch", "{start_epoch}",'

    code = f"""
import subprocess, sys, os
os.makedirs("{TRAIN_DIR}", exist_ok=True)
cmd = [
    sys.executable, "{TPS_REPO}/scripts/train_weighted_dsm.py",
    "--yaml", "{TPS_REPO}/inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml",
    "--data", "{OUT}/train_remapped.npz",
    "--val-data", "{OUT}/val_remapped.npz",
    "--out", "{TRAIN_DIR}",
    "--epochs", "{TARGET_EPOCHS}", "--batch-size", "4", "--learning-rate", "1e-5",
    "--beta", "0", "--max-grad-norm", "1.0",
    "--loss-type", "quotient",
    "--checkpoint-every", "10", "--diffusion-steps", "16",
    "--lr-schedule", "cosine", "--lr-warmup-epochs", "5", "--lr-min", "1e-8",
    "--early-stopping-patience", "50",
    "--recycling-steps", "1", "--device", "cuda",
    {resume_args}
]
log_out = open("{TRAIN_DIR}/run_stdout.log", "{'a' if resume_ckpt != 'none' else 'w'}")
log_err = open("{TRAIN_DIR}/run_stderr.log", "{'a' if resume_ckpt != 'none' else 'w'}")
proc = subprocess.Popen(cmd, stdout=log_out, stderr=log_err, start_new_session=True)
print(proc.pid)
"""
    r = sandbox_exec(sid, code, timeout=60)
    pid = r.get("stdout", "").strip()
    print(f"[train] PID: {pid}", flush=True)
    return pid


def poll_progress(sid, pid):
    r = sandbox_exec(sid, f"""
import subprocess, os, re
alive = bool(subprocess.run(["ps", "-p", "{pid}", "-o", "pid", "--no-headers"],
             capture_output=True, text=True).stdout.strip())
d = "{TRAIN_DIR}"
with open(f"{{d}}/run_stdout.log") as f:
    lines = f.readlines()
epochs = [l.strip() for l in lines if "[Epoch" in l]
ckpts = sorted([f for f in os.listdir(d) if re.match(r"boltz2_wdsm_epoch_\\d+\\.pt", f)])
csv = f"{{d}}/training_log.csv"
n_batches = sum(1 for _ in open(csv)) - 1 if os.path.exists(csv) else 0
last_epoch = epochs[-1] if epochs else "none"
print(f"alive={{alive}} ckpts={{len(ckpts)}} batches={{n_batches}} last={{last_epoch}}")
""", timeout=15)
    return r.get("stdout", "").strip()


def main():
    print(f"{'='*60}")
    print(f"  WDSM Auto-Training Pipeline")
    print(f"  Target: {TARGET_EPOCHS} epochs")
    print(f"{'='*60}", flush=True)

    max_rounds = 200
    for round_num in range(1, max_rounds + 1):
        sid = None
        try:
            sid = create_sandbox()
            install_deps(sid)

            epoch_done, latest_ckpt, converged = get_training_state(sid)
            print(f"[train] Current state: {epoch_done}/{TARGET_EPOCHS} epochs, latest: {latest_ckpt}, converged: {converged}", flush=True)

            if converged:
                print(f"[train] Model converged via early stopping at epoch {epoch_done}!", flush=True)
                run_cli(["sandbox", "terminate", sid], timeout=30)
                print(f"\n{'='*60}")
                print(f"  TRAINING CONVERGED (early stopping)")
                print(f"  Final epoch: {epoch_done}")
                print(f"{'='*60}", flush=True)
                return

            if epoch_done >= TARGET_EPOCHS:
                print(f"[train] All {TARGET_EPOCHS} epochs complete!", flush=True)
                run_cli(["sandbox", "terminate", sid], timeout=30)
                return

            start_epoch = epoch_done + 1
            pid = launch_training(sid, latest_ckpt, start_epoch)

            start_time = time.time()
            max_wait = SANDBOX_TIMEOUT - 600

            while True:
                elapsed = time.time() - start_time
                if elapsed > max_wait:
                    print(f"[train] Approaching timeout ({elapsed:.0f}s). Ending round.", flush=True)
                    break

                time.sleep(POLL_INTERVAL)

                try:
                    status = poll_progress(sid, pid)
                    print(f"[train] Round {round_num} | {elapsed/60:.0f}min | {status}", flush=True)

                    if "alive=False" in status:
                        print("[train] Training process exited.", flush=True)
                        break
                except Exception as e:
                    print(f"[train] Poll error: {e}", flush=True)

            epoch_done, latest_ckpt, converged = get_training_state(sid)
            print(f"[train] After round {round_num}: {epoch_done}/{TARGET_EPOCHS} epochs, converged: {converged}", flush=True)

            if converged:
                print(f"\n{'='*60}")
                print(f"  TRAINING CONVERGED (early stopping at epoch {epoch_done})")
                print(f"{'='*60}", flush=True)
                try:
                    run_cli(["sandbox", "terminate", sid], timeout=30)
                except Exception:
                    pass
                return

            try:
                run_cli(["sandbox", "terminate", sid], timeout=30)
            except Exception:
                pass

            if epoch_done >= TARGET_EPOCHS:
                print(f"\n{'='*60}")
                print(f"  TRAINING COMPLETE: {TARGET_EPOCHS} epochs")
                print(f"{'='*60}", flush=True)
                return

            print(f"[train] Waiting 30s before next round...", flush=True)
            time.sleep(30)

        except KeyboardInterrupt:
            print("\n[train] Interrupted", flush=True)
            if sid:
                try:
                    run_cli(["sandbox", "terminate", sid], timeout=30)
                except Exception:
                    pass
            return
        except Exception as e:
            print(f"[train] ERROR: {e}", flush=True)
            if sid:
                try:
                    run_cli(["sandbox", "terminate", sid], timeout=30)
                except Exception:
                    pass
            time.sleep(60)

    print(f"[train] Max rounds ({max_rounds}) reached.")


if __name__ == "__main__":
    main()
