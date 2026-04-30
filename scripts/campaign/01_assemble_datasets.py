#!/usr/bin/env python3
"""Stage 1 — Assemble per-step WDSM shards into training NPZ files.

Merges the per-step ``wdsm_step_*.npz`` shards produced by Stage 0
(``00_generate_reference_data.py``) into single ``ReweightedStructureDataset``
NPZ files, one per diagnostic system.

Key operations:
  - Stacks ``coords`` and ``logw`` across all shards in step order.
  - Reads ``atom_mask`` from the topology NPZ so that padded atoms are masked.
  - Applies optional log-weight clipping.
  - Reports N_eff diagnostics so you can judge sampling quality before training.

Pipeline role:
    00_generate_reference_data  →  01_assemble_datasets  →  02_finetune_boltz2

Outputs::

    outputs/campaign/case{1,2,3}/
        training_dataset.npz        # coords (N,M,3), logw (N,), atom_mask (N,M)
        training_dataset_val.npz    # 10% held-out split (optional)
        openmm_opes_md/opes_states/fes_reweighted_2d.dat  # if --plumed-fes (2 CV COLVAR)
        openmm_opes_md/opes_states/fes_reweighted_3d.dat  # same flag if COLVAR has lig_contacts
        01_assembly_log.json        # per-system N_eff / statistics

Example::

    python scripts/campaign/01_assemble_datasets.py \\
        --out outputs/campaign \\
        --max-log-ratio 10.0 \\
        --val-fraction 0.1

    # Same run, plus PLUMED COLVAR → reweighted FES (non-fatal on failure unless
    # ``--plumed-fes-strict``):
    python scripts/campaign/01_assemble_datasets.py --out outputs/campaign --plumed-fes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

_CASE_NAMES = [
    "case1_mek1_fzc_novel",
    "case2_cdk2_atp_wildtype",
    "case3_cdk2_atp_packed",
]


def _plumed_fes_kwargs_from_colvar(colvar: Path, sigma_arg: str) -> dict:
    """Choose reweighting targets from COLVAR ``#! FIELDS`` (delegates to plumed_colvar_fes)."""
    from genai_tps.simulation.plumed_colvar_fes import reweighting_kwargs_from_colvar_path

    return dict(reweighting_kwargs_from_colvar_path(colvar, sigma_arg=sigma_arg))


def _find_topo_npz(md_out_dir: Path) -> Path:
    """Locate the topology NPZ written by the Boltz prep pass in Stage 0."""
    candidates = sorted(md_out_dir.rglob("boltz_prep/**/processed/structures/*.npz"))
    if not candidates:
        candidates = sorted(md_out_dir.rglob("processed/structures/*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No topology NPZ found under {md_out_dir}. "
            "Re-run Stage 0 or pass --topo-npz explicitly."
        )
    return candidates[0]


def _resolve_campaign_out_root(raw: Path) -> Path:
    """Resolve ``--out`` for relative paths without doubling ``scripts/campaign/``.

    If the shell cwd is ``scripts/campaign/`` and the user passes
    ``--out scripts/campaign/outputs/...`` (meant as repo-relative), naive
    ``Path.cwd() / out`` becomes ``.../scripts/campaign/scripts/campaign/...``.
    We pick the first candidate that looks like a Stage-0 output tree.
    """
    expanded = raw.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    cwd_root = (Path.cwd() / expanded).resolve()
    repo_root = (_REPO_ROOT / expanded).resolve()

    def _looks_like_stage0_root(p: Path) -> bool:
        if not p.is_dir():
            return False
        return any((p / name / "openmm_opes_md").is_dir() for name in _CASE_NAMES)

    if _looks_like_stage0_root(cwd_root):
        return cwd_root
    if _looks_like_stage0_root(repo_root):
        return repo_root
    return cwd_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: Assemble OPES-MD shards into training NPZ files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/campaign"),
                        help="Root campaign output directory (same as Stage 0 --out).")
    parser.add_argument("--cases", type=str, default="1,2,3",
                        help="Comma-separated case numbers to process.")
    parser.add_argument("--max-log-ratio", type=float, default=None,
                        help="Clip log-weights exceeding this value (reduces variance).")
    parser.add_argument("--min-step", type=int, default=0,
                        help="Discard shards with mc_step < this (removes equilibration).")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction held out as validation set (0 = no split).")
    parser.add_argument(
        "--plumed-fes",
        action="store_true",
        help=(
            "After assembling each case, run PLUMED FES_from_Reweighting.py on "
            "openmm_opes_md/opes_states/COLVAR (skip if missing). Failures log a "
            "warning unless --plumed-fes-strict."
        ),
    )
    parser.add_argument(
        "--plumed-fes-strict",
        action="store_true",
        help="With --plumed-fes, raise if COLVAR is missing or the FES script fails.",
    )
    parser.add_argument(
        "--plumed-fes-temperature",
        type=float,
        default=300.0,
        help="Kelvin passed to PLUMED reweighting script.",
    )
    parser.add_argument(
        "--plumed-fes-sigma",
        type=str,
        default="0.3,0.5",
        help="KDE sigma string for PLUMED FES (comma-separated, Å).",
    )
    parser.add_argument(
        "--plumed-fes-skiprows",
        type=int,
        default=0,
        help="COLVAR burn-in rows for PLUMED FES.",
    )
    parser.add_argument(
        "--plumed-fes-blocks",
        type=int,
        default=1,
        help="Block count for PLUMED FES uncertainty.",
    )
    args = parser.parse_args()

    from genai_tps.simulation.dataset_assembly import assemble_wdsm_from_directory, save_assembled_npz
    from genai_tps.training.diagnostics import effective_sample_size, weight_statistics

    out_root = _resolve_campaign_out_root(args.out)
    print(f"[01] Campaign output root: {out_root}", flush=True)
    selected_cases = {int(c.strip()) for c in args.cases.split(",")}
    assembly_log = []

    for idx, name in enumerate(_CASE_NAMES, start=1):
        if idx not in selected_cases:
            print(f"\n[01] Skipping {name} (not in --cases {args.cases})")
            continue

        print(f"\n{'='*60}")
        print(f"  Case {idx}: {name}")
        print(f"{'='*60}", flush=True)

        md_out = out_root / name / "openmm_opes_md"
        wdsm_dir = md_out / "wdsm_samples"
        if not wdsm_dir.is_dir():
            raise FileNotFoundError(
                f"WDSM shards directory not found: {wdsm_dir}\n"
                "Run Stage 0 first."
            )

        shard_count = len(list(wdsm_dir.glob("wdsm_step_*.npz")))
        print(f"  [01] Found {shard_count} shards in {wdsm_dir}", flush=True)

        topo_npz = _find_topo_npz(md_out)
        print(f"  [01] Topology NPZ: {topo_npz}", flush=True)

        assembled = assemble_wdsm_from_directory(
            wdsm_dir,
            topo_npz=topo_npz,
            max_log_ratio=args.max_log_ratio,
            min_step=args.min_step,
        )

        n_samples = assembled.coords.shape[0]
        n_atoms = assembled.coords.shape[1]
        n_eff = effective_sample_size(assembled.logw)
        stats = weight_statistics(assembled.logw)

        print(f"  [01] Assembled: {n_samples} samples, {n_atoms} atoms", flush=True)
        print(f"  [01] N_eff={n_eff:.1f} ({n_eff/n_samples*100:.1f}%)", flush=True)
        print(f"  [01] logw range: [{assembled.logw.min():.3f}, {assembled.logw.max():.3f}]", flush=True)
        print(f"  [01] max_weight_fraction={stats['max_weight_fraction']:.4f}", flush=True)

        case_out = out_root / name
        out_npz = case_out / "training_dataset.npz"
        save_assembled_npz(out_npz, assembled)
        print(f"  [01] Saved: {out_npz}", flush=True)

        if args.plumed_fes:
            from genai_tps.simulation.plumed_colvar_fes import run_fes_from_reweighting_script

            colvar = md_out / "opes_states" / "COLVAR"
            if not colvar.is_file():
                msg = f"No COLVAR at {colvar}; skip PLUMED FES."
                if args.plumed_fes_strict:
                    raise FileNotFoundError(msg)
                print(f"  [01] {msg}", flush=True)
            else:
                try:
                    fes_kw = _plumed_fes_kwargs_from_colvar(
                        colvar, args.plumed_fes_sigma
                    )
                    outfile = fes_kw["outfile"]
                    run_fes_from_reweighting_script(
                        colvar_path=colvar,
                        outfile=outfile,
                        temperature_k=args.plumed_fes_temperature,
                        sigma=fes_kw["sigma"],
                        cv_names=fes_kw["cv_names"],
                        grid_min=None,
                        grid_max=None,
                        grid_bin=fes_kw["grid_bin"],
                        skiprows=args.plumed_fes_skiprows,
                        blocks=args.plumed_fes_blocks,
                    )
                    print(f"  [01] PLUMED FES: {outfile}", flush=True)
                except Exception as exc:
                    if args.plumed_fes_strict:
                        raise
                    print(f"  [01] PLUMED FES failed (non-fatal): {exc}", flush=True)

        # Optional train/val split
        val_path = None
        if args.val_fraction > 0:
            split_dir = case_out / "dataset_split"
            split_dir.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(42)
            indices = rng.permutation(n_samples)
            split = int((1.0 - args.val_fraction) * n_samples)
            train_idx, val_idx = indices[:split], indices[split:]
            train_npz = split_dir / "train.npz"
            val_npz = split_dir / "val.npz"
            np.savez_compressed(
                train_npz,
                coords=assembled.coords[train_idx],
                logw=assembled.logw[train_idx],
                atom_mask=assembled.atom_mask[train_idx],
            )
            np.savez_compressed(
                val_npz,
                coords=assembled.coords[val_idx],
                logw=assembled.logw[val_idx],
                atom_mask=assembled.atom_mask[val_idx],
            )
            val_path = str(val_npz)
            print(f"  [01] Train/val split: {train_npz.name} / {val_npz.name}", flush=True)

        assembly_log.append({
            "case": name,
            "n_shards": shard_count,
            "n_samples": int(n_samples),
            "n_atoms": int(n_atoms),
            "n_eff": float(n_eff),
            "n_eff_fraction": float(n_eff / n_samples),
            "logw_min": float(assembled.logw.min()),
            "logw_max": float(assembled.logw.max()),
            "max_weight_fraction": float(stats["max_weight_fraction"]),
            "out_npz": str(out_npz),
            "val_npz": val_path,
        })

    log_path = out_root / "01_assembly_log.json"
    with open(log_path, "w") as fh:
        json.dump(assembly_log, fh, indent=2)
    print(f"\n[01] Assembly log: {log_path}", flush=True)
    print("[01] Stage 1 complete. Run 02_finetune_boltz2.py next.", flush=True)


if __name__ == "__main__":
    main()
