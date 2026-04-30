#!/usr/bin/env python3
"""Stage 3 — Generate large structure ensembles from baseline and fine-tuned models.

For each combination of (diagnostic system) × (model variant), generates N
structures by running the Boltz-2 reverse-diffusion forward pass.  Both the
baseline (unmodified) Boltz-2 and every fine-tuned checkpoint from Stage 2
are evaluated.

Checkpoints trained with ``true-quotient`` loss use the quotient-space ODE sampler
(:math:`P_x`-projected velocities); all others use Boltz default sampling, inferred
from ``training_summary.json`` beside each checkpoint (or from the variant folder name).

Output format:
  - ``generated_structures.npz``: coords ``(N, M, 3)`` Å, atom_mask ``(N, M)``
  - ``pdbs/sample_{i:05d}.pdb``:  individual PDB files (optional, --export-pdbs)

These ensembles feed into:
  - Stage 4 (memorization evaluation)
  - Stage 5 (quality / FES evaluation)

Pipeline role:
    02_finetune_boltz2  →  03_generate_ensembles  →  04_evaluate_memorization
                                                  →  05_evaluate_quality

Outputs::

    outputs/campaign/case{1,2,3}/
        baseline/generated_structures.npz
        sft_cartesian/generated_structures.npz
        sft_true-quotient/generated_structures.npz
        (each dir also has pdbs/ if --export-pdbs)

Example::

    # 100 samples per variant, quick check:
    python scripts/campaign/03_generate_ensembles.py \\
        --out outputs/campaign --n-samples 100 --diffusion-steps 32

    # Full production run (500 samples, export PDBs):
    python scripts/campaign/03_generate_ensembles.py \\
        --out outputs/campaign --n-samples 500 --diffusion-steps 200 --export-pdbs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

from genai_tps.utils.compute_device import (
    maybe_set_torch_cuda_current_device,
    parse_torch_device,
)

_CASE_YAMLS = {
    "case1_mek1_fzc_novel":    "inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml",
    "case2_cdk2_atp_wildtype": "inputs/tps_diagnostic/case2_cdk2_atp_wildtype.yaml",
    "case3_cdk2_atp_packed":   "inputs/tps_diagnostic/case3_cdk2_atp_packed.yaml",
}


def _generate(
    *,
    yaml_path: Path,
    cache: Path,
    prep_dir: Path,
    checkpoint: Path | None,
    n_samples: int,
    diffusion_steps: int,
    recycling_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(coords, atom_mask)`` arrays for one model variant.

    Parameters
    ----------
    checkpoint:
        Path to a ``.pt`` state-dict file, or ``None`` for baseline Boltz-2.

    Returns
    -------
    coords : np.ndarray, shape (N, M, 3)
    atom_mask : np.ndarray, shape (N, M)
    """
    from genai_tps.backends.boltz.inference import (
        build_boltz_inference_session,
        quotient_space_sampling_for_checkpoint,
    )
    from genai_tps.evaluation.tps_runner import generate_structures

    quotient_space_sampling = quotient_space_sampling_for_checkpoint(checkpoint)
    bundle = build_boltz_inference_session(
        yaml_path=yaml_path,
        cache=cache,
        boltz_prep_dir=prep_dir,
        device=device,
        diffusion_steps=diffusion_steps,
        recycling_steps=recycling_steps,
        kernels=False,
        quotient_space_sampling=quotient_space_sampling,
    )

    if checkpoint is not None:
        state = torch.load(str(checkpoint), map_location=device)
        bundle.model.load_state_dict(state)
        print(f"    [gen] Loaded checkpoint: {checkpoint.name}", flush=True)
    else:
        print(f"    [gen] Using baseline Boltz-2 (no checkpoint)", flush=True)

    coords = generate_structures(
        bundle.core,
        n_samples=n_samples,
        device=device,
    ).astype(np.float32)
    atom_mask_row = bundle.atom_mask.detach().cpu().numpy().astype(np.float32)
    if atom_mask_row.ndim == 2:
        atom_mask_row = atom_mask_row[0]
    atom_mask = np.broadcast_to(atom_mask_row[None], (coords.shape[0], atom_mask_row.shape[0])).copy()

    return coords, atom_mask


def _export_pdbs(
    coords: np.ndarray,
    topo_npz: Path,
    out_dir: Path,
) -> None:
    """Write one PDB per generated structure."""
    from genai_tps.io.boltz_npz_export import load_topo, npz_to_pdb

    structure, n_struct = load_topo(topo_npz)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame in enumerate(coords):
        tmp_npz = out_dir / f"_tmp_sample_{idx:05d}.npz"
        np.savez(tmp_npz, coords=frame[None])
        npz_to_pdb(
            tmp_npz,
            structure,
            n_struct,
            out_dir / f"sample_{idx:05d}.pdb",
            frame_idx=0,
        )
        tmp_npz.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3: Generate structure ensembles from baseline + SFT models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/campaign"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--cases", type=str, default="1,2,3")
    parser.add_argument("--variants", type=str, default="baseline,sft_cartesian,sft_true-quotient",
                        help="Comma-separated model variants to generate. Use 'baseline' for "
                             "unmodified Boltz-2; 'sft_<loss_type>' for fine-tuned checkpoints.")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of structures to generate per variant.")
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device: cpu, cuda, or cuda:N (default: cuda).",
    )
    parser.add_argument("--export-pdbs", action="store_true",
                        help="Write individual PDB files in addition to the NPZ.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip if generated_structures.npz already exists.")
    args = parser.parse_args()

    cache = Path(args.cache).expanduser() if args.cache else Path.home() / ".boltz"
    out_root = args.out.expanduser().resolve()
    if torch.cuda.is_available():
        device = parse_torch_device(args.device)
        maybe_set_torch_cuda_current_device(device)
    else:
        device = torch.device("cpu")
    selected_cases = {int(c.strip()) for c in args.cases.split(",")}
    variants = [v.strip() for v in args.variants.split(",")]
    case_names = list(_CASE_YAMLS.keys())
    gen_log = []

    for idx, name in enumerate(case_names, start=1):
        if idx not in selected_cases:
            continue

        yaml_path = _REPO_ROOT / _CASE_YAMLS[name]
        case_dir = out_root / name

        for variant in variants:
            print(f"\n{'='*60}")
            print(f"  Case {idx}: {name}  |  variant={variant}")
            print(f"{'='*60}", flush=True)

            variant_dir = case_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            out_npz = variant_dir / "generated_structures.npz"

            if args.resume and out_npz.is_file():
                print(f"  [03] Exists: {out_npz}; skipping.", flush=True)
                gen_log.append({"case": name, "variant": variant, "status": "skipped_resume"})
                continue

            # Resolve checkpoint
            checkpoint: Path | None = None
            if variant != "baseline":
                ckpt_path = case_dir / variant / "boltz2_wdsm_final.pt"
                if not ckpt_path.is_file():
                    # Try best checkpoint
                    ckpt_path = case_dir / variant / "boltz2_wdsm_best.pt"
                if not ckpt_path.is_file():
                    print(f"  [03] WARNING: No checkpoint found for {variant}; skipping.", flush=True)
                    gen_log.append({"case": name, "variant": variant, "status": "no_checkpoint"})
                    continue
                checkpoint = ckpt_path

            prep_dir = variant_dir / "boltz_prep"
            coords, atom_mask = _generate(
                yaml_path=yaml_path,
                cache=cache,
                prep_dir=prep_dir,
                checkpoint=checkpoint,
                n_samples=args.n_samples,
                diffusion_steps=args.diffusion_steps,
                recycling_steps=args.recycling_steps,
                device=device,
            )

            np.savez_compressed(out_npz, coords=coords, atom_mask=atom_mask)
            print(f"  [03] Saved {len(coords)} structures → {out_npz}", flush=True)

            # Optionally write PDBs
            if args.export_pdbs:
                # Find topology NPZ from any adjacent prep directory
                topo_candidates = sorted(variant_dir.rglob("processed/structures/*.npz"))
                if not topo_candidates:
                    topo_candidates = sorted(
                        (case_dir / "openmm_opes_md").rglob("processed/structures/*.npz")
                    )
                if topo_candidates:
                    pdb_dir = variant_dir / "pdbs"
                    _export_pdbs(coords, topo_candidates[0], pdb_dir)
                    print(f"  [03] PDBs exported to {pdb_dir}", flush=True)
                else:
                    print(f"  [03] WARNING: No topology NPZ found; skipping PDB export.", flush=True)

            gen_log.append({
                "case": name,
                "variant": variant,
                "status": "complete",
                "n_samples": int(len(coords)),
                "n_atoms": int(coords.shape[1]),
                "out_npz": str(out_npz),
            })

    log_path = out_root / "03_generation_log.json"
    with open(log_path, "w") as fh:
        json.dump(gen_log, fh, indent=2)
    print(f"\n[03] Generation log: {log_path}", flush=True)
    print("[03] Stage 3 complete. Run 04_evaluate_memorization.py and 05_evaluate_quality.py.", flush=True)


if __name__ == "__main__":
    main()
