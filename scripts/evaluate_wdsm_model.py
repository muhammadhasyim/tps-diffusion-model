#!/usr/bin/env python3
"""Evaluate fine-tuned WDSM model vs baseline Boltz-2.

Generates N structures from each model, computes CVs, and compares
distributions with KS tests and visualization.

Example::

    python scripts/evaluate_wdsm_model.py \
        --yaml inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml \
        --finetuned-checkpoint /mnt/shared/.../wdsm_training/boltz2_wdsm_final.pt \
        --out /mnt/shared/.../evaluation \
        --n-samples 500 \
        --diffusion-steps 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from genai_tps.utils.compute_device import (
    maybe_set_torch_cuda_current_device,
    parse_torch_device,
)

from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.inference import (
    build_boltz_inference_session,
    quotient_space_sampling_for_checkpoint,
)
from genai_tps.backends.boltz.collective_variables import compute_cvs
from evaluation_dashboard import generate_wdsm_evaluation_dashboard as generate_dashboard
from genai_tps.evaluation.tps_runner import generate_structures


def main():
    parser = argparse.ArgumentParser(description="Evaluate WDSM fine-tuned vs baseline Boltz-2.")
    parser.add_argument("--yaml", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device: cpu, cuda, or cuda:N (default: cuda).",
    )
    args = parser.parse_args()

    yaml_path = args.yaml or (_REPO_ROOT / "inputs" / "tps_diagnostic" / "case1_mek1_fzc_novel.yaml")
    cache = Path(args.cache).expanduser() if args.cache else Path.home() / ".boltz"
    if torch.cuda.is_available():
        device = parse_torch_device(args.device)
        maybe_set_torch_cuda_current_device(device)
    else:
        device = torch.device("cpu")
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  WDSM Evaluation: Baseline vs Fine-tuned")
    print(f"  Samples per model: {args.n_samples}")
    print(f"  Diffusion steps: {args.diffusion_steps}")
    print(f"{'='*60}\n")

    print("[eval] Building Boltz session (baseline)...", flush=True)
    bundle = build_boltz_inference_session(
        yaml_path=yaml_path,
        cache=cache,
        boltz_prep_dir=out / "baseline_prep",
        device=device,
        diffusion_steps=args.diffusion_steps,
        recycling_steps=args.recycling_steps,
        kernels=False,
        quotient_space_sampling=False,
    )
    model, core, atom_mask, nck = bundle.model, bundle.core, bundle.atom_mask, bundle.network_kwargs
    topo_npz = bundle.topo_npz
    if topo_npz is None:
        raise RuntimeError("No topology NPZ found under processed structures.")

    ref_x = core.sample_initial_noise()
    for step in range(core.num_sampling_steps):
        ref_x, *_ = core.single_forward_step(ref_x, step)
    reference_coords = ref_x[0].detach().cpu().numpy()
    atom_mask_np = atom_mask.cpu().numpy()

    print("\n[eval] Generating baseline structures...", flush=True)
    baseline_structs = generate_structures(core, args.n_samples, device)
    np.savez(out / "baseline_structures.npz", coords=baseline_structs)

    print("[eval] Computing baseline CVs...", flush=True)
    baseline_cvs = compute_cvs(baseline_structs, reference_coords, atom_mask_np, topo_npz)

    print("\n[eval] Loading fine-tuned checkpoint...", flush=True)
    state_dict = torch.load(str(args.finetuned_checkpoint), map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    qs_ft = quotient_space_sampling_for_checkpoint(args.finetuned_checkpoint)
    print(f"[eval] quotient_space_sampling (fine-tuned): {qs_ft}", flush=True)
    core_ft = BoltzSamplerCore(
        model.structure_module,
        atom_mask,
        nck,
        multiplicity=1,
        compile_model=core.compile_model,
        n_fixed_point=core.n_fixed_point,
        inference_dtype=core.inference_dtype,
        quotient_space_sampling=qs_ft,
    )
    core_ft.build_schedule(args.diffusion_steps)

    print("[eval] Generating fine-tuned structures...", flush=True)
    finetuned_structs = generate_structures(core_ft, args.n_samples, device)
    np.savez(out / "finetuned_structures.npz", coords=finetuned_structs)

    print("[eval] Computing fine-tuned CVs...", flush=True)
    finetuned_cvs = compute_cvs(finetuned_structs, reference_coords, atom_mask_np, topo_npz)

    print("\n[eval] Generating comparison dashboard...", flush=True)
    ks_results = generate_dashboard(baseline_cvs, finetuned_cvs, out, args.n_samples, args.n_samples)

    report = {
        "n_baseline": args.n_samples,
        "n_finetuned": args.n_samples,
        "finetuned_checkpoint": str(args.finetuned_checkpoint),
        "diffusion_steps": args.diffusion_steps,
        "quotient_space_sampling_finetuned": qs_ft,
        "ks_tests": ks_results,
        "baseline_summary": {k: {"mean": float(v[np.isfinite(v)].mean()), "std": float(v[np.isfinite(v)].std()),
                                  "min": float(v[np.isfinite(v)].min()), "max": float(v[np.isfinite(v)].max())}
                             for k, v in baseline_cvs.items() if np.any(np.isfinite(v))},
        "finetuned_summary": {k: {"mean": float(v[np.isfinite(v)].mean()), "std": float(v[np.isfinite(v)].std()),
                                   "min": float(v[np.isfinite(v)].min()), "max": float(v[np.isfinite(v)].max())}
                              for k, v in finetuned_cvs.items() if np.any(np.isfinite(v))},
    }
    with open(out / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Evaluation Complete")
    print(f"  Report: {out / 'evaluation_report.json'}")
    print(f"  Dashboard: {out / 'evaluation_dashboard.png'}")
    for cv, ks in ks_results.items():
        sig = "***" if ks["p_value"] < 0.001 else "**" if ks["p_value"] < 0.01 else "*" if ks["p_value"] < 0.05 else ""
        print(f"  {cv}: KS={ks['ks_statistic']:.3f} p={ks['p_value']:.3g} {sig}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
