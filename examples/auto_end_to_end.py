#!/usr/bin/env python3
"""Fully automated end-to-end pipeline: MD replicas → merge → train → evaluate.

Runs LOCALLY. Orchestrates the entire workflow without manual intervention:
  Phase 1: Monitor 3 MD replicas until 100k structures each (300k total)
  Phase 2: Merge all replica data into a single training NPZ with atom remapping
  Phase 3: Train quotient-space WDSM on the merged 300k dataset
  Phase 4: Run compare_models.py evaluation

Usage:
    PYTHONUNBUFFERED=1 nohup python -u examples/auto_end_to_end.py > auto_e2e.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import subprocess
import time

MODAL_TOOLS = "modal-tools"
IMAGE = "chemistry-py3.12-gpu"
GPU = "A100-80GB"
MEMORY = 32768
SANDBOX_TIMEOUT = 10800
POLL_INTERVAL = 120

TPS_REPO = "/mnt/shared/26-04-22-tps-diffusion/tps-diffusion-model"
SYS_DIR = "/mnt/shared/26-04-22-tps-diffusion/openmm_opes_md"
REPLICA_BASE = "/mnt/shared/26-04-22-tps-diffusion/openmm_opes_md_replicas"
MERGED_DIR = "/mnt/shared/26-04-22-tps-diffusion/openmm_opes_md_merged_300k"
TRAIN_DIR = "/mnt/shared/26-04-22-tps-diffusion/openmm_opes_md_merged_300k/wdsm_training_quotient"
EVAL_DIR = "/mnt/shared/26-04-22-tps-diffusion/openmm_opes_md_merged_300k/evaluation"
CARTESIAN_CKPT = "/mnt/shared/26-04-22-tps-diffusion/openmm_opes_md/wdsm_training_v6/boltz2_wdsm_final.pt"

N_REPLICAS = 3
TARGET_PER_REPLICA = 100_000
TARGET_EPOCHS = 1000


def run_cli(args, timeout=60):
    r = subprocess.run([MODAL_TOOLS] + args, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"CLI failed: {r.stderr[:300]}")
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
    r = run_cli(["sandbox", "create", "--image", IMAGE, "--gpu", GPU,
                 "--memory", str(MEMORY), "--timeout", str(SANDBOX_TIMEOUT), "--json"], timeout=120)
    return r.get("sandbox_id")


def install_deps(sid):
    sandbox_exec(sid, f"""
import subprocess
subprocess.run(["pip", "install", "-e", "{TPS_REPO}/boltz", "-e", "{TPS_REPO}/[boltz]"],
    capture_output=True, text=True, timeout=300)
print("ok")
""", timeout=360)


# ============================================================
# PHASE 1: Check MD replica progress
# ============================================================

def check_all_replicas(sid):
    """Returns dict of {replica_id: n_structures}."""
    r = sandbox_exec(sid, f"""
import os, numpy as np
counts = {{}}
for i in range(3):
    d = "{REPLICA_BASE}/replica_{{i}}/wdsm_samples"
    total = 0
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith('.npz'):
                try:
                    total += np.load(os.path.join(d, f))['coords'].shape[0]
                except:
                    pass
    counts[i] = total
print(counts)
""", timeout=60)
    try:
        return eval(r.get("stdout", "{}").strip())
    except Exception:
        return {0: 0, 1: 0, 2: 0}


def wait_for_replicas():
    """Phase 1: Wait until all replicas have enough structures."""
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Waiting for MD replicas ({TARGET_PER_REPLICA:,} × {N_REPLICAS})")
    print(f"{'='*60}", flush=True)

    while True:
        try:
            sid = create_sandbox()
            install_deps(sid)
            counts = check_all_replicas(sid)
            run_cli(["sandbox", "terminate", sid], timeout=30)

            total = sum(counts.values())
            all_done = all(v >= TARGET_PER_REPLICA for v in counts.values())
            print(f"[phase1] Replicas: {counts} | Total: {total:,} | Done: {all_done}", flush=True)

            if all_done:
                return counts
            if total >= TARGET_PER_REPLICA * N_REPLICAS * 0.9:
                print("[phase1] 90%+ reached, proceeding anyway", flush=True)
                return counts

        except Exception as e:
            print(f"[phase1] Error: {e}", flush=True)

        time.sleep(300)


# ============================================================
# PHASE 2: Merge all replicas + remap to Boltz ordering
# ============================================================

def merge_replicas(sid):
    """Phase 2: Merge all replica data into training NPZ."""
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Merging replicas into training dataset")
    print(f"{'='*60}", flush=True)

    r = sandbox_exec(sid, f"""
import os, sys, numpy as np
sys.path.insert(0, "{TPS_REPO}/src/python")

REPLICA_BASE = "{REPLICA_BASE}"
MERGED = "{MERGED_DIR}"
SYS = "{SYS_DIR}"
os.makedirs(MERGED, exist_ok=True)

# Load atom mapping
mapping = np.load(f"{{SYS}}/boltz_to_7xlp_heavy_mapping.npy")
n_boltz = len(mapping)
TARGET = 2784

# Collect all replica data
all_coords, all_logw, all_cv = [], [], []
for i in range(3):
    d = f"{{REPLICA_BASE}}/replica_{{i}}/wdsm_samples"
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if not f.endswith('.npz'):
            continue
        data = np.load(os.path.join(d, f))
        coords_batch = data['coords']
        logw_batch = data['logw']
        cv_batch = data['cv']

        for j in range(len(coords_batch)):
            omm_coords = coords_batch[j]
            boltz_coords = np.zeros((TARGET, 3), dtype=np.float32)
            for bi in range(n_boltz):
                idx = int(mapping[bi])
                if 0 <= idx < omm_coords.shape[0]:
                    boltz_coords[bi] = omm_coords[idx]
            all_coords.append(boltz_coords)
            all_logw.append(logw_batch[j])
            all_cv.append(cv_batch[j])

        if len(all_coords) % 10000 == 0:
            print(f"  Loaded {{len(all_coords):,}} structures...", flush=True)

N = len(all_coords)
coords = np.array(all_coords, dtype=np.float32)
logw = np.array(all_logw, dtype=np.float64)

# Clip weights (same spirit as assemble_wdsm_dataset --max-log-ratio)
max_lr = 8.0
log_mean = np.log(np.mean(np.exp(logw - logw.max()))) + logw.max()
logw_clipped = np.minimum(logw, log_mean + max_lr)

# Atom mask
atom_mask = np.zeros((N, TARGET), dtype=np.float32)
for bi in range(n_boltz):
    if mapping[bi] >= 0:
        atom_mask[:, bi] = 1.0

# Train/val split
rng = np.random.default_rng(42)
indices = rng.permutation(N)
split = int(0.8 * N)
train_idx, val_idx = indices[:split], indices[split:]

np.savez(f"{{MERGED}}/train.npz", coords=coords[train_idx], logw=logw_clipped[train_idx], atom_mask=atom_mask[train_idx])
np.savez(f"{{MERGED}}/val.npz", coords=coords[val_idx], logw=logw_clipped[val_idx], atom_mask=atom_mask[val_idx])

w = np.exp(logw_clipped - logw_clipped.max()); w /= w.sum()
neff = 1.0 / np.sum(w**2)

print(f"\\nMerged: {{N:,}} structures -> train={{len(train_idx):,}} val={{len(val_idx):,}}")
print(f"N_eff={{neff:.0f}} ({{neff/N*100:.1f}}%)")
print(f"logw: mean={{logw_clipped.mean():.3f}} std={{logw_clipped.std():.3f}}")
print("MERGE COMPLETE")
""", timeout=600)
    print(f"[phase2] {r.get('stdout', '')[-500:]}", flush=True)


# ============================================================
# PHASE 3: Train quotient-space WDSM
# ============================================================

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
    converged = False
    summary = os.path.join(d, "training_summary.json")
    if os.path.exists(summary):
        try:
            with open(summary) as f:
                converged = json.load(f).get("stopped_early", False)
        except:
            pass
    print(f"{{epoch_num}} {{latest}} {{converged}}")
""", timeout=15)
    parts = r.get("stdout", "0 none false").strip().split()
    epoch_num = int(parts[0]) if parts else 0
    latest_ckpt = parts[1] if len(parts) > 1 else "none"
    converged = parts[2] == "True" if len(parts) > 2 else False
    return epoch_num, latest_ckpt, converged


def run_training():
    """Phase 3: Train quotient WDSM until convergence."""
    print(f"\n{'='*60}")
    print(f"  PHASE 3: Quotient-space WDSM training (up to {TARGET_EPOCHS} epochs)")
    print(f"{'='*60}", flush=True)

    for round_num in range(1, 200):
        sid = None
        try:
            sid = create_sandbox()
            install_deps(sid)

            epoch_done, latest_ckpt, converged = get_training_state(sid)
            print(f"[phase3] Round {round_num}: epoch {epoch_done}/{TARGET_EPOCHS}, converged={converged}", flush=True)

            if converged:
                print("[phase3] Early stopping triggered!", flush=True)
                run_cli(["sandbox", "terminate", sid], timeout=30)
                return True
            if epoch_done >= TARGET_EPOCHS:
                run_cli(["sandbox", "terminate", sid], timeout=30)
                return True

            start_epoch = epoch_done + 1
            resume_args = ""
            if latest_ckpt != "none":
                resume_args = f'"--resume-from", "{TRAIN_DIR}/{latest_ckpt}", "--start-epoch", "{start_epoch}",'

            code = f"""
import subprocess, sys, os
os.makedirs("{TRAIN_DIR}", exist_ok=True)
cmd = [sys.executable, "{TPS_REPO}/scripts/train_weighted_dsm.py",
    "--yaml", "{TPS_REPO}/inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml",
    "--data", "{MERGED_DIR}/train.npz", "--val-data", "{MERGED_DIR}/val.npz",
    "--out", "{TRAIN_DIR}", "--epochs", "{TARGET_EPOCHS}", "--batch-size", "4",
    "--learning-rate", "1e-5", "--beta", "0", "--max-grad-norm", "1.0",
    "--loss-type", "quotient", "--checkpoint-every", "10", "--diffusion-steps", "16",
    "--recycling-steps", "1", "--device", "cuda",
    "--lr-schedule", "cosine", "--lr-warmup-epochs", "5", "--lr-min", "1e-8",
    "--early-stopping-patience", "50",
    {resume_args}
]
log_out = open("{TRAIN_DIR}/run_stdout.log", "{'a' if latest_ckpt != 'none' else 'w'}")
log_err = open("{TRAIN_DIR}/run_stderr.log", "{'a' if latest_ckpt != 'none' else 'w'}")
proc = subprocess.Popen(cmd, stdout=log_out, stderr=log_err, start_new_session=True)
print(proc.pid)
"""
            r = sandbox_exec(sid, code, timeout=60)
            pid = r.get("stdout", "").strip()
            print(f"[phase3] Training PID: {pid}", flush=True)

            start_time = time.time()
            while time.time() - start_time < SANDBOX_TIMEOUT - 600:
                time.sleep(POLL_INTERVAL)
                try:
                    status = sandbox_exec(sid, f"""
import subprocess, os, re
alive = bool(subprocess.run(["ps", "-p", "{pid}", "-o", "pid", "--no-headers"],
             capture_output=True, text=True).stdout.strip())
d = "{TRAIN_DIR}"
with open(f"{{d}}/run_stdout.log") as f:
    lines = f.readlines()
epochs = [l.strip() for l in lines if "[Epoch" in l]
ckpts = sorted([f for f in os.listdir(d) if re.match(r"boltz2_wdsm_epoch_\\d+\\.pt", f)])
last = epochs[-1] if epochs else "none"
print(f"alive={{alive}} ckpts={{len(ckpts)}} last={{last}}")
""", timeout=15).get("stdout", "").strip()
                    elapsed = (time.time() - start_time) / 60
                    print(f"[phase3] {elapsed:.0f}min | {status}", flush=True)
                    if "alive=False" in status:
                        break
                except Exception:
                    pass

            epoch_done, latest_ckpt, converged = get_training_state(sid)
            print(f"[phase3] After round {round_num}: epoch {epoch_done}, converged={converged}", flush=True)

            try:
                run_cli(["sandbox", "terminate", sid], timeout=30)
            except Exception:
                pass

            if converged:
                return True
            if epoch_done >= TARGET_EPOCHS:
                return True

            time.sleep(30)

        except KeyboardInterrupt:
            if sid:
                try: run_cli(["sandbox", "terminate", sid], timeout=30)
                except: pass
            return False
        except Exception as e:
            print(f"[phase3] ERROR: {e}", flush=True)
            if sid:
                try: run_cli(["sandbox", "terminate", sid], timeout=30)
                except: pass
            time.sleep(60)

    return False


# ============================================================
# PHASE 4: Evaluation
# ============================================================

def run_evaluation():
    """Phase 4: Run compare_models.py."""
    print(f"\n{'='*60}")
    print(f"  PHASE 4: Model Comparison Evaluation")
    print(f"{'='*60}", flush=True)

    sid = create_sandbox()
    install_deps(sid)

    # Find best quotient checkpoint
    r = sandbox_exec(sid, f"""
import os
d = "{TRAIN_DIR}"
best = os.path.join(d, "boltz2_wdsm_best.pt")
final = os.path.join(d, "boltz2_wdsm_final.pt")
if os.path.exists(best):
    print(best)
elif os.path.exists(final):
    print(final)
else:
    ckpts = sorted([f for f in os.listdir(d) if f.endswith('.pt')])
    print(os.path.join(d, ckpts[-1]) if ckpts else "NONE")
""", timeout=15)
    quotient_ckpt = r.get("stdout", "NONE").strip()
    print(f"[phase4] Quotient checkpoint: {quotient_ckpt}", flush=True)

    if quotient_ckpt == "NONE":
        print("[phase4] No checkpoint found, skipping evaluation", flush=True)
        run_cli(["sandbox", "terminate", sid], timeout=30)
        return

    code = f"""
import subprocess, sys, os
os.makedirs("{EVAL_DIR}", exist_ok=True)
cmd = [sys.executable, "{TPS_REPO}/scripts/compare_models.py",
    "--yaml", "{TPS_REPO}/inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml",
    "--training-dir", "{SYS_DIR}/wdsm_samples",
    "--cartesian-checkpoint", "{CARTESIAN_CKPT}",
    "--quotient-checkpoint", "{quotient_ckpt}",
    "--out", "{EVAL_DIR}", "--n-samples", "200", "--diffusion-steps", "32", "--device", "cuda"]
log_out = open("{EVAL_DIR}/run_stdout.log", "w")
log_err = open("{EVAL_DIR}/run_stderr.log", "w")
proc = subprocess.Popen(cmd, stdout=log_out, stderr=log_err, start_new_session=True)
print(proc.pid)
"""
    r = sandbox_exec(sid, code, timeout=60)
    pid = r.get("stdout", "").strip()
    print(f"[phase4] Evaluation PID: {pid}", flush=True)

    start_time = time.time()
    while time.time() - start_time < SANDBOX_TIMEOUT - 600:
        time.sleep(120)
        try:
            r = sandbox_exec(sid, f"""
import os
report = os.path.exists("{EVAL_DIR}/comparison_report.json")
print(f"report={{report}}")
""", timeout=10)
            status = r.get("stdout", "").strip()
            print(f"[phase4] {status}", flush=True)
            if "report=True" in status:
                break
        except Exception:
            pass

    try:
        run_cli(["sandbox", "terminate", sid], timeout=30)
    except Exception:
        pass

    print("[phase4] Evaluation complete!", flush=True)


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"{'='*60}")
    print(f"  END-TO-END PIPELINE")
    print(f"  MD replicas → Merge → Train → Evaluate")
    print(f"{'='*60}\n", flush=True)

    # Phase 1: Wait for MD replicas
    counts = wait_for_replicas()
    total = sum(counts.values())
    print(f"\n[main] Phase 1 complete: {total:,} structures from {N_REPLICAS} replicas", flush=True)

    # Phase 2: Merge
    sid = create_sandbox()
    install_deps(sid)
    merge_replicas(sid)
    run_cli(["sandbox", "terminate", sid], timeout=30)

    # Phase 3: Train
    converged = run_training()
    print(f"\n[main] Phase 3 complete: converged={converged}", flush=True)

    # Phase 4: Evaluate
    run_evaluation()

    print(f"\n{'='*60}")
    print(f"  FULL PIPELINE COMPLETE")
    print(f"  MD data: {total:,} structures")
    print(f"  Training: quotient-space WDSM")
    print(f"  Results: {EVAL_DIR}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
