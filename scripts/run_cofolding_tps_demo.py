#!/usr/bin/env python3
"""Transition-path–style sampling on a **co-folding** Boltz-2 input (two chains).

This script:

1. Preprocesses a Boltz YAML (default: ``examples/cofolding_multimer_msa_empty.yaml``)
   — a classic **heterodimer** setup (two proteins), the same biological idea as
   studying association in co-folding benchmarks (cf. barnase–barstar–style
   problems), using Boltz’s own multimer-style example with ``msa: empty``.
2. Runs the **Boltz-2 trunk** (pairformer + diffusion conditioning) once.
3. Builds :class:`~genai_tps.backends.boltz.gpu_core.BoltzSamplerCore` on the real
   ``AtomDiffusion`` module and rolls out **diffusion-time trajectories** (noise →
   structure), then runs **OpenPathSampling** one-way shooting with Metropolis
   acceptance (fixed-length TPS ensemble in diffusion noise scale ``sigma``).

**Requirements:** GPU strongly recommended; ``pip install -e ./boltz`` and
``pip install -e ".[boltz,dev]"``; first run downloads Boltz-2 weights and mol
tarballs into ``--cache`` (same layout as ``boltz predict``).

**cuEquivariance (optional):** Boltz’s fast pairformer kernels need NVIDIA’s
``cuequivariance_torch`` packages. They are **not** in Boltz’s base install;
use ``pip install -e "./boltz[cuda]"`` (see Boltz’s README). This script defaults
to ``use_kernels=False`` so it runs without them; pass ``--kernels`` after
installing ``boltz[cuda]``.

Example::

    python scripts/run_cofolding_tps_demo.py \\
      --out ./cofolding_tps_out \\
      --diffusion-steps 32 \\
      --shoot-rounds 5

Output::

    ``trajectory_summary.json`` — metadata, per-frame sigmas, coordinate norms.
    ``shooting_log.txt`` — OPS MC steps (one line per round; use ``tail -f`` to watch live).
    Each line includes ``min_1_r`` = :math:`\\min(1,\\pi_\\mathrm{new}/\\pi_\\mathrm{old})`
    for the factorized path density in ``path_probability.compute_log_path_prob`` (see
    that module for relation to OPS ``metropolis_acceptance``). Use ``--progress-every N``
    for periodic rate/ETA lines on stderr during long runs.
    ``coords_trajectory.npz`` — optional stacked coordinates (CPU) for inspection.

    With ``--cartoon-every N``, every ``N`` MC steps writes
    ``cartoon_snapshots/last_frame_000001.png``, ``last_frame_000002.png``, … (6-digit
    sequence) and refreshes ``last_frame_cartoon.png`` (requires ``pymol`` on ``PATH`` for a
    **subprocess** render; in-process PyMOL next to CUDA can segfault).
    Ray-traced PyMOL cartoons are slow; use ``--cartoon-width``, ``--cartoon-height``,
    ``--cartoon-dpi`` to trade quality for speed.
    Use ``--log-path-prob-every 0`` (default) for long runs; ``N=1`` recomputes full-path
    log densities every step and is very slow.

    For long jobs, prefer ``--save-trajectory-every N`` (writes ``trajectory_checkpoints/*.npz``,
    no PyMOL) and run ``scripts/visualize_cofolding_trajectory.py`` afterward for PDB/cartoon.
    Use ``--cartoon-every 0`` (default) to disable in-loop PyMOL.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from openpathsampling.engines.trajectory import Trajectory

from genai_tps.backends.boltz.boltz2_trunk import boltz2_trunk_to_network_kwargs
from genai_tps.backends.boltz.bridge import snapshot_from_gpu
from genai_tps.backends.boltz.engine import BoltzDiffusionEngine
from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.snapshot import BoltzSnapshot
from genai_tps.backends.boltz.snapshot import boltz_snapshot_descriptor
from genai_tps.backends.boltz.snapshot import snapshot_frame_numpy_copy
from genai_tps.backends.boltz.tps_sampling import run_tps_path_sampling


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _last_frame_coords_numpy(traj: Trajectory) -> np.ndarray:
    return snapshot_frame_numpy_copy(traj[-1])


def _visualize_helpers():
    scripts_dir = _repo_root() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import visualize_cofolding_trajectory as viz  # noqa: PLC0415

    return viz.write_last_frame_boltz2_pdb, viz.render_cartoon_png_pymol


def _cartoon_checkpoint_callback(
    structures_npz: Path,
    work_root: Path,
    *,
    cartoon_width: int = 1600,
    cartoon_height: int = 1600,
    cartoon_dpi: int = 150,
    use_ray: bool = False,
    prefer_software_gl: bool = True,
) -> Callable[[int, Trajectory], None]:
    write_pdb, render_png = _visualize_helpers()
    snap_dir = work_root / "cartoon_snapshots"
    tmp_pdb = snap_dir / "_last_for_cartoon.pdb"
    latest = work_root / "last_frame_cartoon.png"
    seq = 0

    def cb(step: int, traj: Trajectory) -> None:
        nonlocal seq
        snap_dir.mkdir(parents=True, exist_ok=True)
        coords = _last_frame_coords_numpy(traj)
        write_pdb(structures_npz, coords, tmp_pdb)
        seq += 1
        out_png = snap_dir / f"last_frame_{seq:06d}.png"
        try:
            # Subprocess PyMOL: in-process pymol+cmd loads OpenGL next to CUDA and can segfault.
            render_png(
                tmp_pdb,
                out_png,
                width=cartoon_width,
                height=cartoon_height,
                dpi=cartoon_dpi,
                force_subprocess=True,
                use_ray=use_ray,
                prefer_software_gl=prefer_software_gl,
            )
        except Exception as e:
            seq -= 1
            print(f"[TPS] cartoon at MC step {step} failed: {e}", file=sys.stderr, flush=True)
            return
        shutil.copyfile(out_png, latest)
        print(
            f"[TPS] cartoon #{seq} (MC step {step}) -> {out_png.name}",
            file=sys.stderr,
            flush=True,
        )

    return cb


def _initial_trajectory(core: BoltzSamplerCore, n_steps: int | None = None) -> Trajectory:
    """One full forward diffusion path (noise → coordinates)."""
    x = core.sample_initial_noise()
    n = n_steps if n_steps is not None else core.num_sampling_steps
    snaps = []
    sig0 = float(core.schedule[0].sigma_tm)
    snaps.append(snapshot_from_gpu(x, 0, None, None, None, sig0, center_mean_before_step=None))
    for step in range(n):
        x, eps, rr, tr, meta = core.single_forward_step(x, step)
        snaps.append(
            snapshot_from_gpu(
                x,
                step + 1,
                eps,
                rr,
                tr,
                float(meta["sigma_t"]),
                center_mean_before_step=meta.get("center_mean"),
            )
        )
    return Trajectory(snaps)


_npz_all_nonfinite_error_count = 0


def _validate_coords_array_for_npz(
    arr: np.ndarray,
    *,
    label: str,
    out_npz: Path,
    mc_step: int | None = None,
) -> bool:
    """Return True if ``arr`` should be written; else log and return False."""
    global _npz_all_nonfinite_error_count
    if not np.any(np.isfinite(arr)):
        _npz_all_nonfinite_error_count += 1
        n = _npz_all_nonfinite_error_count
        if n <= 3 or (mc_step is not None and mc_step % 500 == 0):
            print(
                f"[TPS] ERROR: {label} has no finite coordinates; refusing to write {out_npz}. "
                "Dynamics may have diverged (NaN), or GPU readback failed."
                + (f" (failure #{n})" if n > 1 else ""),
                file=sys.stderr,
                flush=True,
            )
        return False
    n_fin = int(np.isfinite(arr).sum())
    n_tot = int(arr.size)
    if n_fin < n_tot:
        print(
            f"[TPS] Warning: {label} is only {100.0 * n_fin / n_tot:.1f}% finite "
            f"({n_fin}/{n_tot} values).",
            file=sys.stderr,
            flush=True,
        )
    return True


def _trajectory_to_npz(traj: Trajectory, out_npz: Path) -> bool:
    """Write stacked coords; return True if file was written."""
    stack = []
    for snap in traj:
        if not isinstance(snap, BoltzSnapshot) or snap.tensor_coords is None:
            continue
        stack.append(snapshot_frame_numpy_copy(snap))
    if not stack:
        return False
    arr = np.stack(stack, axis=0)
    if not _validate_coords_array_for_npz(
        arr, label=f"coords_trajectory ({out_npz.name})", out_npz=out_npz, mc_step=None
    ):
        return False
    np.savez_compressed(out_npz, coords=arr, frame_indices=np.arange(arr.shape[0]))
    return True


def _write_tps_checkpoint_npz(traj: Trajectory, out_npz: Path, *, mc_step: int) -> bool:
    """Save accepted path coordinates for post-processing (PyMOL, viz) outside the TPS loop.

    Refuses to write if stacked coordinates contain no finite values (avoids silent NaN dumps).
    Returns True if ``out_npz`` was written.
    """
    stack = []
    for snap in traj:
        if not isinstance(snap, BoltzSnapshot) or snap.tensor_coords is None:
            continue
        stack.append(snapshot_frame_numpy_copy(snap))
    if not stack:
        return False
    arr = np.stack(stack, axis=0)
    if not _validate_coords_array_for_npz(
        arr,
        label=f"trajectory checkpoint (mc_step={mc_step})",
        out_npz=out_npz,
        mc_step=mc_step,
    ):
        return False
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        coords=arr,
        frame_indices=np.arange(arr.shape[0], dtype=np.int32),
        mc_step=np.int64(mc_step),
    )
    return True


def _trajectory_checkpoint_callback(work_root: Path) -> Callable[[int, Trajectory], None]:
    """Periodic NPZ checkpoints (no PyMOL); visualize later with ``visualize_cofolding_trajectory.py``."""

    def cb(mc_step: int, traj: Trajectory) -> None:
        ck = work_root / "trajectory_checkpoints"
        ck.mkdir(parents=True, exist_ok=True)
        tagged = ck / f"tps_mc_step_{mc_step:08d}.npz"
        wrote_ckpt = _write_tps_checkpoint_npz(traj, tagged, mc_step=mc_step)
        latest = ck / "latest.npz"
        if wrote_ckpt:
            shutil.copyfile(tagged, latest)
        last_only = ck / "last_frame_only.npz"
        snap = traj[-1]
        if wrote_ckpt and isinstance(snap, BoltzSnapshot) and snap.tensor_coords is not None:
            lf = snapshot_frame_numpy_copy(snap)
            if np.any(np.isfinite(lf)):
                np.savez_compressed(
                    last_only,
                    coords=lf.reshape(1, *lf.shape),
                    mc_step=np.int64(mc_step),
                )
        if wrote_ckpt:
            print(
                f"[TPS] trajectory checkpoint MC step {mc_step} -> {tagged.name} (latest.npz updated)",
                file=sys.stderr,
                flush=True,
            )

    return cb


def _run_exact_jacobian_diagnostic(
    core: "BoltzSamplerCore",
    init_traj: "Trajectory",
    network_kwargs: dict,
    work_root: Path,
    diffusion: "torch.nn.Module",
    chunk_size: int = 64,
) -> None:
    """Compute exact vs scalar Jacobian log-determinants for diagnostic logging.

    For each step of init_traj (where eps_used is available), computes:
      - scalar_logdet  = log_det_jacobian_step(alpha_i, M)  [fast, O(1)]
      - exact_logdet   = compute_log_det_jacobian_exact(...)  [slow, O(M/chunk) fwd passes]

    Results are written to <work_root>/exact_jacobian_diagnostic.json.

    Theory reference: docs/tps_diffusion_theory.tex, Sections 4.3--4.4.
    Note: the scalar approximation is CORRECT for TPS acceptance ratios
    (alpha_i is schedule-dependent, not path-dependent), so these should
    differ but the difference cancels in the acceptance ratio.
    """
    from genai_tps.backends.boltz.path_probability import (
        compute_log_det_jacobian_exact,
        log_det_jacobian_step,
    )
    from genai_tps.backends.boltz.snapshot import BoltzSnapshot

    print("[TPS] --exact-jacobian: computing diagnostic...", flush=True)
    results = []
    n_steps = len(init_traj) - 1
    for step_idx in range(n_steps):
        snap_next = init_traj[step_idx + 1]
        if not isinstance(snap_next, BoltzSnapshot) or snap_next.eps_used is None:
            continue
        sch = core.schedule[step_idx]
        t_hat = float(sch.t_hat)
        sigma_t = float(sch.sigma_t)
        step_scale = float(core.diffusion.step_scale)
        alpha = step_scale * (sigma_t - t_hat) / t_hat
        n_atoms = int(snap_next.eps_used.shape[1])
        scalar_ld = float(log_det_jacobian_step(alpha, n_atoms))

        x_noisy = snap_next.tensor_coords
        if x_noisy is None:
            results.append({
                "step": step_idx,
                "alpha": alpha,
                "n_atoms": n_atoms,
                "scalar_logdet": scalar_ld,
                "exact_logdet": None,
                "delta": None,
                "note": "tensor_coords unavailable",
            })
            continue

        try:
            exact_ld = float(
                compute_log_det_jacobian_exact(
                    score_model=diffusion.net,
                    x_noisy=x_noisy,
                    t_hat=t_hat,
                    network_kwargs=network_kwargs,
                    alpha_i=alpha,
                    sigma_data=float(getattr(diffusion, "sigma_data", 16.0)),
                    chunk_size=chunk_size,
                )
            )
            delta = exact_ld - scalar_ld
        except Exception as exc:
            print(
                f"[TPS] exact Jacobian step {step_idx} failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            results.append({
                "step": step_idx,
                "alpha": alpha,
                "n_atoms": n_atoms,
                "scalar_logdet": scalar_ld,
                "exact_logdet": None,
                "delta": None,
                "note": str(exc),
            })
            continue

        results.append({
            "step": step_idx,
            "alpha": alpha,
            "n_atoms": n_atoms,
            "scalar_logdet": scalar_ld,
            "exact_logdet": exact_ld,
            "delta": delta,
        })
        print(
            f"[TPS] Jacobian diagnostic step {step_idx}: "
            f"scalar={scalar_ld:.4f}  exact={exact_ld:.4f}  delta={delta:.4f}",
            flush=True,
        )

    out_path = work_root / "exact_jacobian_diagnostic.json"
    out_path.write_text(json.dumps({"steps": results}, indent=2))
    print(f"[TPS] exact Jacobian diagnostic written to {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Co-folding TPS-style demo on Boltz-2")
    parser.add_argument(
        "--yaml",
        type=Path,
        default=None,
        help="Boltz YAML (default: examples/cofolding_multimer_msa_empty.yaml)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cofolding_tps_out"),
        help="Output directory for work and artifacts",
    )
    parser.add_argument("--cache", type=Path, default=None, help="Boltz cache (~/.boltz)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=32,
        help="Number of reverse diffusion steps (smaller = faster demo)",
    )
    parser.add_argument("--shoot-rounds", type=int, default=5)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Print TPS progress every N MC steps to stderr (0 = quiet). "
            "Also tail -f shooting_log.txt to watch each step."
        ),
    )
    parser.add_argument(
        "--cartoon-every",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Every N MC steps, write cartoon_snapshots/last_frame_000001.png, "
            "last_frame_000002.png, … (6-digit sequence) and copy the latest to "
            "last_frame_cartoon.png. Renders via ``pymol -cqx`` subprocess (needs ``pymol`` on "
            "PATH in this env; pip/conda pymol-open-source). 0 = off."
        ),
    )
    parser.add_argument(
        "--cartoon-width",
        type=int,
        default=1600,
        metavar="PX",
        help="PyMOL cartoon PNG width in pixels (smaller = faster ray trace).",
    )
    parser.add_argument(
        "--cartoon-height",
        type=int,
        default=1600,
        metavar="PX",
        help="PyMOL cartoon PNG height in pixels (smaller = faster ray trace).",
    )
    parser.add_argument(
        "--cartoon-dpi",
        type=int,
        default=150,
        metavar="DPI",
        help="PyMOL png dpi (lower = faster, smaller effective resolution).",
    )
    parser.add_argument(
        "--cartoon-ray",
        action="store_true",
        help=(
            "Use PyMOL ray-traced png (prettier; often SIGSEGV in headless/OpenGL setups). "
            "Default is OpenGL framebuffer only (more reliable during long TPS runs)."
        ),
    )
    parser.add_argument(
        "--cartoon-native-gl",
        action="store_true",
        help=(
            "Do not set LIBGL_ALWAYS_SOFTWARE=1 for PyMOL subprocess (default: software Mesa "
            "for headless stability). Use only if cartoons work on your display stack."
        ),
    )
    parser.add_argument(
        "--save-trajectory-every",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Every N MC steps, write trajectory_checkpoints/tps_mc_step_########.npz (full path, "
            "CPU) plus latest.npz and last_frame_only.npz — no PyMOL. Post-process later, e.g. "
            "python scripts/visualize_cofolding_trajectory.py --npz .../latest.npz --last-frame-out ... "
            "and optional --render-cartoon. 0 = off."
        ),
    )
    parser.add_argument(
        "--log-path-prob-every",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Compute path log-probability diagnostics every N MC steps (0 = never; "
            "much faster). Use N=1 only when debugging — each call walks the full path on GPU."
        ),
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        default=False,
        help=(
            "Use only forward shooting moves (no backward re-noising, no global reshuffles). "
            "Forward shooting always accepts reactive paths so this is a fast baseline. "
            "Default: False (use forward + backward + global reshuffle for full ergodicity)."
        ),
    )
    parser.add_argument(
        "--reshuffle-probability",
        type=float,
        default=0.1,
        metavar="P",
        help=(
            "Fraction of MC steps devoted to global reshuffle moves (draw a completely "
            "fresh path from the prior; always accepts when reactive). "
            "Ignored when --forward-only is set. Default: 0.1."
        ),
    )
    parser.add_argument(
        "--kernels",
        action="store_true",
        help=(
            "Use Boltz cuEquivariance pairformer kernels (requires "
            "'pip install -e \"./boltz[cuda]\"'). Default: off, pure PyTorch."
        ),
    )
    parser.add_argument(
        "--exact-jacobian",
        action="store_true",
        default=False,
        dest="exact_jacobian",
        help=(
            "Before starting TPS, run a diagnostic that computes the exact Jacobian "
            "log-determinant (via chunked forward-mode AD) for each step of the initial "
            "trajectory and compares it to the scalar approximation.  Results are "
            "written to <out>/exact_jacobian_diagnostic.json.  This is O(3M/chunk_size) "
            "forward passes per diffusion step and can be slow for large systems.  "
            "Has no effect on the TPS sampling itself (the scalar approximation is "
            "correct for MCMC acceptance ratios)."
        ),
    )
    args = parser.parse_args()

    repo = _repo_root()
    yaml_path = args.yaml or (repo / "examples" / "cofolding_multimer_msa_empty.yaml")
    if not yaml_path.is_file():
        print(f"Input YAML not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
        from boltz.data.types import Manifest
        from boltz.main import (
            Boltz2DiffusionParams,
            BoltzSteeringParams,
            MSAModuleArgs,
            PairformerArgsV2,
            check_inputs,
            download_boltz2,
            process_inputs,
        )
        from boltz.model.models.boltz2 import Boltz2
    except ImportError as e:
        print(
            "Boltz is required: pip install -e ./boltz from the repo root.\n",
            e,
            file=sys.stderr,
        )
        sys.exit(1)

    cache = Path(args.cache).expanduser() if args.cache else Path.home() / ".boltz"
    cache.mkdir(parents=True, exist_ok=True)
    mol_dir = cache / "mols"
    download_boltz2(cache)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: running on CPU will be very slow.", file=sys.stderr)

    work_root = args.out.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    boltz_run_dir = work_root / f"boltz_results_{yaml_path.stem}"
    boltz_run_dir.mkdir(parents=True, exist_ok=True)

    data_list = check_inputs(yaml_path)
    process_inputs(
        data=data_list,
        out_dir=boltz_run_dir,
        ccd_path=cache / "ccd.pkl",
        mol_dir=mol_dir,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        use_msa_server=False,
        boltz2=True,
        preprocessing_threads=1,
    )

    manifest = Manifest.load(boltz_run_dir / "processed" / "manifest.json")
    if not manifest.records:
        print("No records in manifest after preprocessing.", file=sys.stderr)
        sys.exit(1)

    processed_dir = boltz_run_dir / "processed"
    dm = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        mol_dir=mol_dir,
        num_workers=0,
        constraints_dir=processed_dir / "constraints"
        if (processed_dir / "constraints").exists()
        else None,
        template_dir=processed_dir / "templates" if (processed_dir / "templates").exists() else None,
        extra_mols_dir=processed_dir / "mols" if (processed_dir / "mols").exists() else None,
    )
    loader = dm.predict_dataloader()
    batch = next(iter(loader))
    batch = dm.transfer_batch_to_device(batch, device, dataloader_idx=0)

    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True)
    steering = BoltzSteeringParams()
    steering.fk_steering = False
    steering.physical_guidance_update = False
    steering.contact_guidance_update = False

    ckpt = cache / "boltz2_conf.ckpt"
    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.diffusion_steps,
        "diffusion_samples": 1,
        "max_parallel_samples": None,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    model = Boltz2.load_from_checkpoint(
        str(ckpt),
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=args.kernels,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering),
    )
    model.to(device)
    model.eval()

    atom_mask, network_kwargs = boltz2_trunk_to_network_kwargs(
        model, batch, recycling_steps=args.recycling_steps
    )
    for k, v in list(network_kwargs.items()):
        if hasattr(v, "to"):
            network_kwargs[k] = v.to(device)
    if isinstance(network_kwargs.get("feats"), dict):
        network_kwargs["feats"] = {
            fk: fv.to(device) if hasattr(fv, "to") else fv
            for fk, fv in network_kwargs["feats"].items()
        }

    diffusion = model.structure_module
    core = BoltzSamplerCore(diffusion, atom_mask, network_kwargs, multiplicity=1)
    core.build_schedule(args.diffusion_steps)
    n_atoms = int(atom_mask.shape[1])

    torch.manual_seed(0)
    init_traj = _initial_trajectory(core)

    # ------------------------------------------------------------------
    # Optional: exact Jacobian diagnostic (--exact-jacobian)
    # ------------------------------------------------------------------
    if args.exact_jacobian:
        _run_exact_jacobian_diagnostic(
            core=core,
            init_traj=init_traj,
            network_kwargs=network_kwargs,
            work_root=work_root,
            diffusion=diffusion,
        )

    descriptor = boltz_snapshot_descriptor(n_atoms=n_atoms)
    engine = BoltzDiffusionEngine(
        core,
        descriptor,
        options={"n_frames_max": core.num_sampling_steps + 4},
    )

    shoot_log = work_root / "shooting_log.txt"

    # Log the active move scheme to stderr before starting
    _fwd_only = bool(args.forward_only)
    _reshuffle_p = float(args.reshuffle_probability)
    if _fwd_only:
        _scheme_desc = "forward-only (always accepts reactive paths)"
    else:
        _shoot_w = (1.0 - _reshuffle_p) / 2.0
        _scheme_desc = (
            f"forward ({_shoot_w:.0%}) + backward ({_shoot_w:.0%})"
            + (f" + global-reshuffle ({_reshuffle_p:.0%})" if _reshuffle_p > 0 else "")
        )
    print(f"[TPS] move scheme: {_scheme_desc}", flush=True)
    cartoon_every = max(0, int(args.cartoon_every))
    periodic_cb: Callable[[int, Trajectory], None] | None = None
    if cartoon_every > 0:
        struct_candidates = sorted((processed_dir / "structures").glob("*.npz"))
        if not struct_candidates:
            print(
                "[TPS] No processed/structures/*.npz; --cartoon-every disabled.",
                file=sys.stderr,
                flush=True,
            )
            cartoon_every = 0
        elif not shutil.which("pymol"):
            print(
                "[TPS] No 'pymol' executable on PATH; cartoons use a subprocess PyMOL "
                "to avoid CUDA segfaults. Install e.g. pymol-open-source in this env. "
                "Disabling --cartoon-every.",
                file=sys.stderr,
                flush=True,
            )
            cartoon_every = 0
        else:
            struct_npz = struct_candidates[0]
            if len(struct_candidates) > 1:
                print(
                    f"[TPS] Multiple structure npz files; cartoons use {struct_npz}",
                    file=sys.stderr,
                    flush=True,
                )
            periodic_cb = _cartoon_checkpoint_callback(
                struct_npz.resolve(),
                work_root,
                cartoon_width=max(1, int(args.cartoon_width)),
                cartoon_height=max(1, int(args.cartoon_height)),
                cartoon_dpi=max(1, int(args.cartoon_dpi)),
                use_ray=bool(args.cartoon_ray),
                prefer_software_gl=not bool(args.cartoon_native_gl),
            )

    save_traj_every = max(0, int(args.save_trajectory_every))
    periodic_extra: list[tuple[Callable[[int, Trajectory], None], int]] = []
    if save_traj_every > 0:
        periodic_extra.append((_trajectory_checkpoint_callback(work_root), save_traj_every))

    final_traj, tps_step_log = run_tps_path_sampling(
        engine,
        init_traj,
        args.shoot_rounds,
        shoot_log,
        progress_every=args.progress_every,
        periodic_callback=periodic_cb,
        periodic_every=cartoon_every,
        periodic_callbacks=periodic_extra if periodic_extra else None,
        log_path_prob_every=max(0, int(args.log_path_prob_every)),
        forward_only=bool(args.forward_only),
        reshuffle_probability=float(args.reshuffle_probability),
    )

    frames_meta = []
    x0 = final_traj[0].tensor_coords
    ref = x0.detach().cpu().numpy()[0] if x0 is not None else None
    for i, snap in enumerate(final_traj):
        tc = snap.tensor_coords
        arr = tc.detach().cpu().numpy()[0] if tc is not None else None
        rms = None
        if ref is not None and arr is not None:
            rms = float(np.sqrt(np.mean((arr - ref) ** 2)))
        frames_meta.append(
            {
                "frame": i,
                "step_index": snap.step_index,
                "sigma": snap.sigma,
                "rms_vs_frame0_ang": rms,
                "coord_norm": float(np.linalg.norm(arr)) if arr is not None else None,
            }
        )

    summary = {
        "yaml": str(yaml_path.resolve()),
        "benchmark_note": "Two-chain heterodimer (Boltz multimer-style example; msa: empty).",
        "n_atoms": n_atoms,
        "diffusion_steps": core.num_sampling_steps,
        "sigma0": float(core.schedule[0].sigma_tm),
        "n_frames": len(final_traj),
        "frames": frames_meta,
        "shooting_log": str(shoot_log.resolve()),
        "tps_steps": tps_step_log,
    }
    summary_path = work_root / "trajectory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _trajectory_to_npz(final_traj, work_root / "coords_trajectory.npz")

    print(f"Wrote {summary_path}")
    print(f"Wrote {work_root / 'coords_trajectory.npz'}")
    print(f"Wrote {shoot_log}")
    print("Done. Inspect trajectory_summary.json and coords_trajectory.npz for diffusion-time paths.")


if __name__ == "__main__":
    main()
