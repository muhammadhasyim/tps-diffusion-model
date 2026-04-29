#!/usr/bin/env python3
"""Export OPES-TPS trajectory checkpoints into a ReweightedStructureDataset NPZ.

Reads trajectory checkpoint NPZs and the converged OPES state, computes
per-structure log importance weights, and writes the (coords, logw, atom_mask)
NPZ expected by ``weighted_dsm/dataset.py``.

Example::

    python scripts/export_opes_tps_dataset.py \
        --opes-dir /mnt/shared/.../opes_tps_mek1_fzc_500 \
        --output /mnt/shared/.../mek1_fzc_wdsm_data.npz \
        --pocket-radius 8.0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OPES-TPS data to weighted DSM NPZ.")
    parser.add_argument("--opes-dir", type=Path, required=True, help="OPES-TPS output directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ path.")
    parser.add_argument("--topo-npz", type=Path, default=None, help="Override topology NPZ.")
    parser.add_argument("--pocket-radius", type=float, default=8.0)
    args = parser.parse_args()

    from genai_tps.analysis.boltz_npz_export import load_topo
    from genai_tps.backends.boltz.collective_variables import (
        PoseCVIndexer,
        ligand_pocket_distance,
        ligand_pose_rmsd,
    )
    from genai_tps.enhanced_sampling.opes_bias import OPESBias
    from genai_tps.weighted_dsm.diagnostics import effective_sample_size, weight_statistics

    opes_dir = Path(args.opes_dir)

    # Auto-detect topology
    if args.topo_npz:
        topo_npz = Path(args.topo_npz)
    else:
        candidates = sorted(opes_dir.rglob("processed/structures/*.npz"))
        if not candidates:
            candidates = sorted(opes_dir.rglob("boltz_results_*/processed/structures/*.npz"))
        if not candidates:
            print("No topology NPZ found. Pass --topo-npz explicitly.", file=sys.stderr)
            sys.exit(1)
        topo_npz = candidates[0]
    print(f"[export] Topology: {topo_npz}", flush=True)

    structure, n_struct = load_topo(topo_npz)
    n_struct = int(n_struct)
    ref_coords = np.asarray(structure.atoms["coords"], dtype=np.float64)[:n_struct]
    indexer = PoseCVIndexer(structure, ref_coords, pocket_radius=args.pocket_radius)
    print(f"[export] n_struct={n_struct}, ligand_atoms={len(indexer.ligand_idx)}, "
          f"pocket_ca={len(indexer.pocket_ca_idx)}", flush=True)

    # Load OPES bias
    opes_state_path = opes_dir / "opes_state_final.json"
    if not opes_state_path.is_file():
        print(f"OPES state not found: {opes_state_path}", file=sys.stderr)
        sys.exit(1)
    opes = OPESBias.load_state(opes_state_path)
    print(f"[export] OPES: {opes.n_kernels} kernels, {opes.counter} depositions, kbt={opes.kbt}", flush=True)

    # Find trajectory checkpoints
    ckpt_dir = opes_dir / "trajectory_checkpoints"
    pattern = re.compile(r"tps_mc_step_(\d+)\.npz$")
    ckpt_files = []
    for f in sorted(ckpt_dir.glob("tps_mc_step_*.npz")):
        m = pattern.search(f.name)
        if m:
            ckpt_files.append((int(m.group(1)), f))
    ckpt_files.sort(key=lambda x: x[0])
    print(f"[export] Found {len(ckpt_files)} trajectory checkpoints", flush=True)

    if not ckpt_files:
        print("No trajectory checkpoints found.", file=sys.stderr)
        sys.exit(1)

    # Extract last frames and compute weights
    coords_list = []
    logw_list = []
    cv_list = []

    for step, ckpt_path in ckpt_files:
        data = np.load(ckpt_path)
        all_coords = data["coords"]  # (n_frames, n_atoms_padded, 3)
        last_frame = all_coords[-1]  # (n_atoms_padded, 3)

        n_padded = last_frame.shape[0]
        if n_padded < n_struct:
            print(f"  step {step}: coords have {n_padded} atoms < n_struct={n_struct}, skipping",
                  file=sys.stderr)
            continue
        frame = last_frame.astype(np.float32)  # keep full padded shape

        # Compute CVs from the real atoms only
        x_tensor = torch.from_numpy(frame[:n_struct]).float().unsqueeze(0)
        snap = SimpleNamespace(_tensor_coords_gpu=x_tensor)
        rmsd = ligand_pose_rmsd(snap, indexer)
        pocket_dist = ligand_pocket_distance(snap, indexer)
        cv = np.array([rmsd, pocket_dist], dtype=np.float64)

        # Compute log importance weight
        bias_val = float(opes.evaluate(cv))
        logw = bias_val / opes.kbt

        coords_list.append(frame)
        logw_list.append(logw)
        cv_list.append(cv)

        print(f"  step {step:5d}: rmsd={rmsd:.4f}A  pkt_d={pocket_dist:.4f}A  "
              f"V={bias_val:.4f}  logw={logw:.4f}", flush=True)

    if not coords_list:
        print("No valid samples extracted.", file=sys.stderr)
        sys.exit(1)

    coords_array = np.stack(coords_list, axis=0)       # (N, M_padded, 3)
    logw_array = np.array(logw_list, dtype=np.float64)  # (N,)
    n_padded = coords_array.shape[1]
    atom_mask = np.zeros((len(coords_list), n_padded), dtype=np.float32)
    atom_mask[:, :n_struct] = 1.0  # real atoms = 1, padding = 0

    n_eff = effective_sample_size(logw_array)
    stats = weight_statistics(logw_array)

    print(f"\n[export] Dataset: {len(coords_list)} samples, {n_struct} atoms each", flush=True)
    print(f"[export] N_eff={n_eff:.2f} ({n_eff/len(coords_list)*100:.1f}%)", flush=True)
    print(f"[export] logw range: [{logw_array.min():.4f}, {logw_array.max():.4f}]", flush=True)
    print(f"[export] max_weight_fraction={stats['max_weight_fraction']:.4f}", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, coords=coords_array, logw=logw_array, atom_mask=atom_mask)
    print(f"[export] Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
