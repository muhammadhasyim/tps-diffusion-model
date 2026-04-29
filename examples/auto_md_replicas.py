#!/usr/bin/env python3
"""Launch and manage 3 parallel OPES-MD replicas for 1M+ training structures.

Each replica runs on its own A100 sandbox with a different random seed and
initial coordinate perturbation, auto-restarting until 100k+ structures
are accumulated per replica (300k+ total).

Usage:
    PYTHONUNBUFFERED=1 nohup python -u examples/auto_md_replicas.py > auto_md_replicas.log 2>&1 &
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
N_REPLICAS = 3
TARGET_PER_REPLICA = 100_000
TPS_REPO = "/mnt/shared/26-04-22-tps-diffusion/tps-diffusion-model"
SYS_DIR = "/mnt/shared/26-04-22-tps-diffusion/openmm_opes_md"
BASE_OUT = "/mnt/shared/26-04-22-tps-diffusion/openmm_opes_md_replicas"


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


def get_replica_count(sid, replica_id):
    r = sandbox_exec(sid, f"""
import os
d = "{BASE_OUT}/replica_{replica_id}/wdsm_samples"
if os.path.isdir(d):
    import numpy as np
    total = 0
    for f in os.listdir(d):
        if f.endswith('.npz'):
            total += np.load(os.path.join(d, f))['coords'].shape[0]
    print(total)
else:
    print(0)
""", timeout=30)
    try:
        return int(r.get("stdout", "0").strip())
    except ValueError:
        return 0


def launch_replica(sid, replica_id):
    seed = 42 + replica_id * 1000
    perturb = 0.1 * (replica_id + 1)

    code = f"""
import subprocess, sys, os
os.makedirs("{BASE_OUT}/replica_{replica_id}/wdsm_samples", exist_ok=True)
os.makedirs("{BASE_OUT}/replica_{replica_id}/opes_states", exist_ok=True)

script = '''
import sys, os, time
sys.path.insert(0, "{TPS_REPO}/src/python")
import numpy as np
from openmm.app import PDBFile, Simulation
from openmm import XmlSerializer, LangevinIntegrator, Platform, unit
from genai_tps.simulation import OPESBias

OUT = "{BASE_OUT}/replica_{replica_id}"
SYS = "{SYS_DIR}"

np.random.seed({seed})

with open(f"{{SYS}}/system_with_fzc.xml") as f:
    system = XmlSerializer.deserialize(f.read())
pdb = PDBFile(f"{{SYS}}/cleaned_with_fzc.pdb")
idx = np.load(f"{{SYS}}/atom_indices.npz")
fzc_heavy, protein_ca, all_heavy = idx["fzc_heavy"], idx["protein_ca"], idx["all_heavy"]

integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
integrator.setRandomNumberSeed({seed})
platform = Platform.getPlatformByName("CUDA")
sim = Simulation(pdb.topology, system, integrator, platform, {{"CudaPrecision": "mixed"}})

# Perturb initial positions
import openmm
pos = pdb.positions
perturbed = [openmm.Vec3(p[0] + np.random.normal(0, {perturb})*unit.angstrom,
                          p[1] + np.random.normal(0, {perturb})*unit.angstrom,
                          p[2] + np.random.normal(0, {perturb})*unit.angstrom)
             for p in pos]
sim.context.setPositions(perturbed)
sim.minimizeEnergy(maxIterations=1000)

ref_pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
fzc_ref = ref_pos[fzc_heavy]
pocket_ca = protein_ca[np.array([np.linalg.norm(ref_pos[i] - fzc_ref.mean(axis=0)) < 6.0 for i in protein_ca])]

# Restart OPES if previous state exists
opes_path = f"{{OUT}}/opes_state_final.json"
if not os.path.exists(opes_path):
    ckpt_dir = f"{{OUT}}/opes_states"
    if os.path.isdir(ckpt_dir):
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.json')])
        if ckpts:
            opes_path = os.path.join(ckpt_dir, ckpts[-1])
if os.path.exists(opes_path):
    opes = OPESBias.load_state(opes_path)
    print(f"[R{replica_id}] OPES restarted from {{opes_path}}: {{opes.n_kernels}} kernels", flush=True)
else:
    kbt = 8.314e-3 * 300.0
    opes = OPESBias(ndim=2, kbt=kbt, barrier=5.0, biasfactor=10.0, pace=500,
                    fixed_sigma=np.array([0.3, 0.5]))

def compute_cv():
    pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    fzc = pos[fzc_heavy]
    rmsd = np.sqrt(np.mean(np.sum((fzc - fzc_ref)**2, axis=1)))
    dist = np.linalg.norm(fzc.mean(axis=0) - pos[pocket_ca].mean(axis=0))
    return np.array([rmsd, dist], dtype=np.float64)

def get_heavy():
    return sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)[all_heavy].astype(np.float32)

# Count existing structures
existing = 0
for f in os.listdir(f"{{OUT}}/wdsm_samples"):
    if f.endswith(".npz"):
        existing += np.load(f"{{OUT}}/wdsm_samples/{{f}}")["coords"].shape[0]

N_STEPS = 5_000_000
DEPOSIT_PACE = 500
BATCH_SIZE = 1000

sim.step(5000)  # equilibrate
print(f"[R{replica_id}] seed={seed} perturb={perturb} existing={{existing}}", flush=True)

n_saved = 0
file_idx = len([f for f in os.listdir(f"{{OUT}}/wdsm_samples") if f.endswith(".npz")])
batch_c, batch_w, batch_cv = [], [], []
t0 = time.time()

for step in range(5000 + DEPOSIT_PACE, N_STEPS + 1, DEPOSIT_PACE):
    sim.step(DEPOSIT_PACE)
    cv = compute_cv()
    opes.update(cv, step)
    coords = get_heavy()
    logw = float(opes.evaluate(cv)) / opes.kbt
    batch_c.append(coords); batch_w.append(logw); batch_cv.append(cv)
    n_saved += 1
    if len(batch_c) >= BATCH_SIZE:
        np.savez_compressed(f"{{OUT}}/wdsm_samples/batch_{{file_idx:04d}}.npz",
                           coords=np.array(batch_c, dtype=np.float32),
                           logw=np.array(batch_w, dtype=np.float64),
                           cv=np.array(batch_cv, dtype=np.float64))
        file_idx += 1; batch_c, batch_w, batch_cv = [], [], []
    if step % 1000000 == 0:
        if batch_c:
            np.savez_compressed(f"{{OUT}}/wdsm_samples/batch_{{file_idx:04d}}.npz",
                               coords=np.array(batch_c, dtype=np.float32),
                               logw=np.array(batch_w, dtype=np.float64),
                               cv=np.array(batch_cv, dtype=np.float64))
            file_idx += 1; batch_c, batch_w, batch_cv = [], [], []
        elapsed = time.time() - t0
        print(f"[R{replica_id}] {{step:,}}/{{N_STEPS:,}} | {{step/elapsed:.0f}} sps | saved={{n_saved+existing:,}}", flush=True)
        opes.save_state(f"{{OUT}}/opes_states/opes_{{step:010d}}.json",
                       bias_cv="ligand_rmsd,ligand_pocket_dist",
                       bias_cv_names=["ligand_rmsd", "ligand_pocket_dist"])

if batch_c:
    np.savez_compressed(f"{{OUT}}/wdsm_samples/batch_{{file_idx:04d}}.npz",
                       coords=np.array(batch_c, dtype=np.float32),
                       logw=np.array(batch_w, dtype=np.float64),
                       cv=np.array(batch_cv, dtype=np.float64))

opes.save_state(f"{{OUT}}/opes_state_final.json",
               bias_cv="ligand_rmsd,ligand_pocket_dist",
               bias_cv_names=["ligand_rmsd", "ligand_pocket_dist"])
print(f"[R{replica_id}] Done: {{n_saved+existing:,}} total structures", flush=True)
'''

script_path = f"{BASE_OUT}/replica_{replica_id}/run_md.py"
with open(script_path, "w") as f:
    f.write(script)

log_out = open(f"{BASE_OUT}/replica_{replica_id}/stdout.log", "w")
log_err = open(f"{BASE_OUT}/replica_{replica_id}/stderr.log", "w")
proc = subprocess.Popen([sys.executable, script_path], stdout=log_out, stderr=log_err, start_new_session=True)
print(proc.pid)
"""
    r = sandbox_exec(sid, code, timeout=60)
    pid = r.get("stdout", "").strip()
    return pid


def main():
    print(f"{'='*60}")
    print(f"  3-Replica OPES-MD Pipeline")
    print(f"  Target: {TARGET_PER_REPLICA:,} structures × {N_REPLICAS} replicas = {TARGET_PER_REPLICA*N_REPLICAS:,}")
    print(f"{'='*60}", flush=True)

    max_rounds = 50
    for round_num in range(1, max_rounds + 1):
        print(f"\n--- Round {round_num} ---", flush=True)

        sandboxes = []
        for i in range(N_REPLICAS):
            try:
                sid = create_sandbox()
                print(f"[R{i}] Sandbox: {sid}", flush=True)

                sandbox_exec(sid, f"""
import subprocess
subprocess.run(["pip", "install",
    "-e", "{TPS_REPO}/boltz", "-e", "{TPS_REPO}/[boltz]"],
    capture_output=True, text=True, timeout=300)
print("ok")
""", timeout=360)

                pid = launch_replica(sid, i)
                print(f"[R{i}] PID: {pid}", flush=True)
                sandboxes.append((i, sid, pid))
            except Exception as e:
                print(f"[R{i}] ERROR creating: {e}", flush=True)

        if not sandboxes:
            print("No sandboxes created. Waiting 60s...", flush=True)
            time.sleep(60)
            continue

        start = time.time()
        max_wait = SANDBOX_TIMEOUT - 600

        while time.time() - start < max_wait:
            time.sleep(120)
            for i, sid, pid in sandboxes:
                try:
                    r = sandbox_exec(sid, f"""
import subprocess, os, numpy as np
alive = bool(subprocess.run(["ps", "-p", "{pid}", "-o", "pid", "--no-headers"],
             capture_output=True, text=True).stdout.strip())
d = "{BASE_OUT}/replica_{i}/wdsm_samples"
total = 0
if os.path.isdir(d):
    for f in os.listdir(d):
        if f.endswith('.npz'):
            total += np.load(os.path.join(d, f))['coords'].shape[0]
print(f"alive={{alive}} n={{total}}")
""", timeout=20)
                    status = r.get("stdout", "").strip()
                    elapsed = (time.time() - start) / 60
                    print(f"[R{i}] {elapsed:.0f}min | {status}", flush=True)
                except Exception:
                    pass

            all_done = True
            for i, sid, pid in sandboxes:
                try:
                    n = get_replica_count(sid, i)
                    if n < TARGET_PER_REPLICA:
                        all_done = False
                except Exception:
                    all_done = False
            if all_done:
                break

        for i, sid, pid in sandboxes:
            try:
                run_cli(["sandbox", "terminate", sid], timeout=30)
            except Exception:
                pass

        total_all = 0
        for i in range(N_REPLICAS):
            try:
                sid_check = create_sandbox()
                sandbox_exec(sid_check, "import subprocess; subprocess.run(['pip','install','-e','/mnt/shared/26-04-22-tps-diffusion/tps-diffusion-model/boltz','-e','/mnt/shared/26-04-22-tps-diffusion/tps-diffusion-model/[boltz]'], capture_output=True, text=True, timeout=300)", timeout=360)
                n = get_replica_count(sid_check, i)
                total_all += n
                print(f"[R{i}] Total: {n:,}", flush=True)
                run_cli(["sandbox", "terminate", sid_check], timeout=30)
                break
            except Exception:
                pass

        print(f"Combined total: ~{total_all*N_REPLICAS:,}+ structures", flush=True)

        if total_all >= TARGET_PER_REPLICA:
            print(f"\n{'='*60}")
            print(f"  ALL REPLICAS COMPLETE!")
            print(f"{'='*60}", flush=True)
            return

        print(f"Waiting 30s before next round...", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
