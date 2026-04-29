#!/usr/bin/env python3
"""OPES-biased OpenMM MD for ground-truth WDSM training data collection.

Runs Langevin dynamics with AMBER14/GBn2 + GAFF2 ligand parameterization,
biased by OPES on (ligand_rmsd, ligand_pocket_dist). Saves structures in
Boltz atom ordering + logw for direct use with train_weighted_dsm.py.

Example::

    python scripts/run_openmm_opes_md.py \
        --topo-npz /mnt/shared/.../structures/system.npz \
        --frame-npz /mnt/shared/.../structures/system.npz \
        --out /mnt/shared/.../openmm_opes_md \
        --n-steps 5000000 \
        --save-every 1000 \
        --opes-barrier 5.0 \
        --opes-biasfactor 10.0 \
        --deposit-pace 500 \
        --temperature 300.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))


def main() -> None:
    parser = argparse.ArgumentParser(description="OPES-biased OpenMM MD for WDSM data collection")
    parser.add_argument("--topo-npz", type=Path, required=True, help="Boltz processed structures NPZ")
    parser.add_argument("--frame-npz", type=Path, default=None, help="NPZ with initial coords (default: same as topo)")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index in frame-npz")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--n-steps", type=int, default=5_000_000, help="Total MD steps (default: 5M = 10ns at 2fs)")
    parser.add_argument("--save-every", type=int, default=1000, help="Save structure every N steps (default: 1000 = 2ps)")
    parser.add_argument("--deposit-pace", type=int, default=500, help="OPES kernel deposit every N steps")
    parser.add_argument("--opes-barrier", type=float, default=5.0, help="OPES barrier (kJ/mol)")
    parser.add_argument("--opes-biasfactor", type=float, default=10.0)
    parser.add_argument("--opes-sigma", type=str, default="0.3,0.5", help="Per-dim kernel sigma (Angstrom)")
    parser.add_argument("--opes-restart", type=Path, default=None, help="Resume OPES from saved state")
    parser.add_argument("--temperature", type=float, default=300.0, help="Langevin temperature (K)")
    parser.add_argument("--pocket-radius", type=float, default=6.0)
    parser.add_argument("--platform", type=str, default="CUDA", choices=["CUDA", "OpenCL", "CPU"])
    parser.add_argument("--minimize-steps", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--save-opes-every", type=int, default=50000)
    parser.add_argument("--ligand-smiles", type=str, default=None, help="chain:SMILES (e.g. B:CC...)")
    parser.add_argument("--mol-dir", type=Path, default=None, help="Boltz CCD mol dir for SMILES lookup")
    args = parser.parse_args()

    from genai_tps.analysis.boltz_npz_export import load_topo, npz_to_pdb
    from genai_tps.backends.boltz.collective_variables import PoseCVIndexer
    from genai_tps.enhanced_sampling.opes_bias import OPESBias
    from genai_tps.rl.fes_teacher import (
        OpenMMTeacher,
        build_openmm_indices_for_boltz_atoms,
        load_build_md_simulation_from_pdb,
    )

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    wdsm_dir = out / "wdsm_samples"
    wdsm_dir.mkdir(parents=True, exist_ok=True)
    opes_dir = out / "opes_states"
    opes_dir.mkdir(parents=True, exist_ok=True)

    structure, n_struct = load_topo(args.topo_npz)
    n_s = int(n_struct)
    print(f"[MD-OPES] Loaded topology: {n_s} atoms", flush=True)

    pdb_path = out / "initial_structure.pdb"
    frame_npz = args.frame_npz or args.topo_npz
    npz_to_pdb(frame_npz, structure, n_struct, pdb_path, frame_idx=args.frame_idx)
    print(f"[MD-OPES] Wrote initial PDB: {pdb_path}", flush=True)

    ref_data = np.load(str(frame_npz))
    if "coords" in ref_data:
        ref_coords = ref_data["coords"][args.frame_idx][:n_s].astype(np.float32)
    else:
        ref_coords = ref_data["atom_coords"][args.frame_idx][:n_s].astype(np.float32)
    indexer = PoseCVIndexer(structure, ref_coords, pocket_radius=args.pocket_radius)
    print(f"[MD-OPES] PoseCVIndexer: {len(indexer.ligand_indices)} ligand atoms, "
          f"{len(indexer.pocket_ca_indices)} pocket Cas", flush=True)

    ligand_smiles = None
    if args.ligand_smiles:
        parts = args.ligand_smiles.split(":", 1)
        ligand_smiles = {parts[0]: parts[1]}
    elif args.mol_dir:
        from genai_tps.enhanced_sampling.openmm_cv import _detect_ligand_smiles
        ligand_smiles = _detect_ligand_smiles(structure, args.mol_dir)

    kbt_kjmol = 8.314e-3 * args.temperature
    sigma = [float(s) for s in args.opes_sigma.split(",")]

    if args.opes_restart:
        print(f"[MD-OPES] Restarting OPES from {args.opes_restart}", flush=True)
        opes = OPESBias.load_state(args.opes_restart)
    else:
        opes = OPESBias(
            ndim=2,
            kbt=kbt_kjmol,
            barrier=args.opes_barrier,
            biasfactor=args.opes_biasfactor,
            pace=args.deposit_pace,
            fixed_sigma=np.array(sigma),
        )

    print(f"[MD-OPES] OPES: barrier={opes.barrier:.1f} biasfactor={opes.biasfactor:.1f} "
          f"kbt={opes.kbt:.3f} kJ/mol sigma={sigma}", flush=True)

    print(f"[MD-OPES] Building OpenMM simulation (AMBER14/GBn2, {args.platform})...", flush=True)
    build_md = load_build_md_simulation_from_pdb()
    sim, meta = build_md(
        pdb_path,
        platform_name=args.platform,
        temperature_k=args.temperature,
        ligand_smiles=ligand_smiles,
    )
    print(f"[MD-OPES] Platform: {meta['platform_used']}", flush=True)

    omm_idx = build_openmm_indices_for_boltz_atoms(structure, sim.topology)
    n_boltz = int(structure.atoms.shape[0])

    if args.minimize_steps > 0:
        print(f"[MD-OPES] Minimizing ({args.minimize_steps} steps)...", flush=True)
        sim.minimizeEnergy(maxIterations=args.minimize_steps)

    def get_coords_boltz_order():
        state = sim.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        coords = np.zeros((n_boltz, 3), dtype=np.float32)
        for i in range(n_boltz):
            coords[i] = pos[int(omm_idx[i])] * 10.0
        return coords

    def compute_cv(coords):
        x = torch.from_numpy(coords[:n_s]).float().unsqueeze(0)
        from types import SimpleNamespace
        snap = SimpleNamespace(_tensor_coords_gpu=x)
        from genai_tps.backends.boltz.collective_variables import (
            ligand_pose_rmsd, ligand_pocket_distance,
        )
        rmsd = float(ligand_pose_rmsd(snap, indexer))
        dist = float(ligand_pocket_distance(snap, indexer))
        return np.array([rmsd, dist], dtype=np.float64)

    print(f"\n{'='*60}")
    print(f"  OpenMM OPES-MD: {args.n_steps:,} steps ({args.n_steps * 2e-6:.1f} ns at 2fs)")
    print(f"  Save every {args.save_every} steps | Deposit every {args.deposit_pace} steps")
    print(f"  Temperature: {args.temperature} K")
    print(f"{'='*60}\n", flush=True)

    cv_log = []
    n_saved = 0
    t0 = time.time()

    for step in range(1, args.n_steps + 1):
        sim.step(1)

        if step % args.deposit_pace == 0:
            coords = get_coords_boltz_order()
            cv = compute_cv(coords)
            opes.update(cv, step)

        if step % args.save_every == 0:
            coords = get_coords_boltz_order()
            cv = compute_cv(coords)
            logw = float(opes.evaluate(cv)) / opes.kbt

            n_saved += 1
            npz_path = wdsm_dir / f"wdsm_step_{n_saved:08d}.npz"
            np.savez_compressed(
                str(npz_path),
                coords=coords,
                cv=cv,
                logw=np.float64(logw),
                md_step=np.int64(step),
            )
            cv_log.append({"step": step, "cv": cv.tolist(), "logw": logw})

        if step % args.save_opes_every == 0:
            opes.save_state(
                opes_dir / f"opes_state_{step:010d}.json",
                bias_cv="ligand_rmsd,ligand_pocket_dist",
                bias_cv_names=["ligand_rmsd", "ligand_pocket_dist"],
            )

        if step % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = step / elapsed
            eta = (args.n_steps - step) / rate
            coords = get_coords_boltz_order()
            cv = compute_cv(coords)
            print(f"[MD-OPES] step {step:>10,}/{args.n_steps:,} | "
                  f"{rate:.0f} steps/s | ETA {eta/60:.1f}min | "
                  f"CV=[{cv[0]:.3f}, {cv[1]:.3f}] | "
                  f"kernels={opes.n_kernels} | saved={n_saved}",
                  flush=True)

    opes.save_state(
        out / "opes_state_final.json",
        bias_cv="ligand_rmsd,ligand_pocket_dist",
        bias_cv_names=["ligand_rmsd", "ligand_pocket_dist"],
    )

    with open(out / "cv_log.json", "w") as f:
        json.dump(cv_log, f)

    elapsed = time.time() - t0
    summary = {
        "total_steps": args.n_steps,
        "total_time_s": elapsed,
        "n_saved": n_saved,
        "n_kernels": opes.n_kernels,
        "temperature_k": args.temperature,
        "opes_barrier": opes.barrier,
        "opes_biasfactor": opes.biasfactor,
    }
    with open(out / "md_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  MD Complete: {args.n_steps:,} steps in {elapsed:.0f}s ({args.n_steps/elapsed:.0f} steps/s)")
    print(f"  Saved {n_saved} structures to {wdsm_dir}")
    print(f"  OPES: {opes.n_kernels} kernels, {opes.counter} depositions")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
