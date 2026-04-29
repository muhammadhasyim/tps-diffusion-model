#!/usr/bin/env python3
"""Run harmonic-umbrella enhanced TPS on a Boltz-2 co-folding input.

Harmonic bias on the path CV (last frame):

    V(phi) = 0.5 * kappa * (phi - center)^2

with Metropolis factor exp(-(V_new - V_old) / k_BT), matching OpenMM / PLUMED
umbrella restraints.

Run multiple instances at different ``--umbrella-center`` / ``--spring-constant``
settings, then combine using MBAR (see ``scripts/validate_enhanced_sampling.py``).

Annealing (centers in CV units): ``--annealing-schedule 0:100,2:200,4:300``
runs 100 MC steps at center=0, 200 at center=2, 300 at center=4.

Example::

    python scripts/run_umbrella_tps.py \\
        --out ./umbrella_tps_out \\
        --diffusion-steps 32 \\
        --shoot-rounds 2000 \\
        --umbrella-center 2.0 \\
        --spring-constant 5.0 \\
        --bias-cv rmsd \\
        --save-trajectory-every 10 \\
        --progress-every 50
"""

from __future__ import annotations

import argparse
import json
import os
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
from genai_tps.backends.boltz.collective_variables import (
    radius_of_gyration,
    rmsd_to_reference,
)
from genai_tps.backends.boltz.engine import BoltzDiffusionEngine
from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.snapshot import (
    BoltzSnapshot,
    boltz_snapshot_descriptor,
    snapshot_frame_numpy_copy,
)
from genai_tps.backends.boltz.tps_sampling import run_tps_path_sampling
from genai_tps.simulation import HarmonicUmbrellaBias
from genai_tps.simulation.mbar_analysis import MBARDistributionEstimator
from genai_tps.simulation.openmm_cv import OpenMMLocalMinRMSD


def _write_mbar_samples_latest(work_root: Path, bias: HarmonicUmbrellaBias) -> None:
    """Snapshot umbrella samples to mbar_samples_latest.json (atomic replace)."""
    est = MBARDistributionEstimator()
    est.add_samples_from_bias(bias, burn_in_fraction=0.0)
    tmp = work_root / "mbar_samples_latest.json.tmp"
    final = work_root / "mbar_samples_latest.json"
    est.save_samples(tmp)
    os.replace(tmp, final)


def _mbar_latest_step_callback(work_root: Path, bias: HarmonicUmbrellaBias):
    def cb(mc_step: int, entry: dict) -> None:
        try:
            _write_mbar_samples_latest(work_root, bias)
        except Exception as exc:
            print(
                f"[TPS-UMBRELLA] mbar_samples_latest flush failed at step {mc_step}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    return cb


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_annealing_schedule(schedule_str: str) -> list[tuple[float, int]]:
    """Parse 'center:steps,center:steps,...' into [(umbrella_center, steps), ...]."""
    segments = []
    for part in schedule_str.split(","):
        part = part.strip()
        if ":" not in part:
            raise ValueError(f"Annealing schedule segment must be 'center:steps'; got '{part}'")
        lam_str, steps_str = part.split(":", 1)
        segments.append((float(lam_str), int(steps_str)))
    return segments


def _make_cv_function(
    cv_type: str,
    reference_coords: torch.Tensor | None = None,
    atom_mask: torch.Tensor | None = None,
    topo_npz: Path | None = None,
    openmm_platform: str = "CUDA",
    openmm_max_iter: int = 500,
    mol_dir: Path | None = None,
) -> Callable[[Trajectory], float]:
    """Build a CV function that operates on the last frame of a trajectory.

    Parameters
    ----------
    cv_type:
        ``"openmm"`` -- AMBER14/GBn2 Cα-RMSD to local energy minimum;
                        identical to the CV used by ``watch_rmsd_live.py``.
                        Requires *topo_npz*.
        ``"rmsd"``   -- fast RMSD to the initial structure (proxy CV).
        ``"rg"``     -- fast radius of gyration (proxy CV).
    mol_dir:
        Path to the Boltz CCD molecule directory (``~/.boltz/mols``).  When
        provided, SMILES for non-protein (NONPOLYMER) chains are read from
        the pkl files and passed to OpenMM for GAFF2 parameterisation.
        Required for inputs that contain ligands (e.g. FZC, ATP, MG).
    """

    def _cv_rmsd(traj: Trajectory) -> float:
        snap = traj[-1]
        return rmsd_to_reference(snap, reference_coords, atom_mask)

    def _cv_rg(traj: Trajectory) -> float:
        snap = traj[-1]
        return radius_of_gyration(snap, atom_mask)

    if cv_type == "rmsd":
        if reference_coords is None:
            raise ValueError("RMSD CV requires a reference structure (use the initial trajectory's last frame).")
        return _cv_rmsd
    elif cv_type == "rg":
        return _cv_rg
    elif cv_type == "openmm":
        if topo_npz is None:
            raise ValueError(
                "--bias-cv openmm requires --topo-npz pointing to the Boltz "
                "processed/structures/*.npz file."
            )
        return OpenMMLocalMinRMSD(
            topo_npz=topo_npz,
            platform=openmm_platform,
            max_iter=openmm_max_iter,
            mol_dir=mol_dir,
        )
    else:
        raise ValueError(f"Unknown CV type: {cv_type!r}. Use 'rmsd', 'rg', or 'openmm'.")


def _write_tps_checkpoint_npz(traj: Trajectory, out_npz: Path, *, mc_step: int) -> bool:
    """Save accepted path coordinates for post-processing."""
    stack = []
    for snap in traj:
        if not isinstance(snap, BoltzSnapshot) or snap.tensor_coords is None:
            continue
        stack.append(snapshot_frame_numpy_copy(snap))
    if not stack:
        return False
    arr = np.stack(stack, axis=0)
    if not np.any(np.isfinite(arr)):
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
    def cb(mc_step: int, traj: Trajectory) -> None:
        ck = work_root / "trajectory_checkpoints"
        ck.mkdir(parents=True, exist_ok=True)
        tagged = ck / f"tps_mc_step_{mc_step:08d}.npz"
        wrote = _write_tps_checkpoint_npz(traj, tagged, mc_step=mc_step)
        if wrote:
            latest = ck / "latest.npz"
            shutil.copyfile(tagged, latest)
            snap = traj[-1]
            if isinstance(snap, BoltzSnapshot) and snap.tensor_coords is not None:
                lf = snapshot_frame_numpy_copy(snap)
                if np.any(np.isfinite(lf)):
                    last_only = ck / "last_frame_only.npz"
                    np.savez_compressed(
                        last_only,
                        coords=lf.reshape(1, *lf.shape),
                        mc_step=np.int64(mc_step),
                    )
            print(
                f"[TPS-UMBRELLA] checkpoint MC step {mc_step} -> {tagged.name}",
                file=sys.stderr, flush=True,
            )
    return cb


def _initial_trajectory(core: BoltzSamplerCore, n_steps: int | None = None) -> Trajectory:
    x = core.sample_initial_noise()
    n = n_steps if n_steps is not None else core.num_sampling_steps
    snaps = []
    sig0 = float(core.schedule[0].sigma_tm)
    snaps.append(snapshot_from_gpu(x, 0, None, None, None, sig0, center_mean_before_step=None))
    for step in range(n):
        x, eps, rr, tr, meta = core.single_forward_step(x, step)
        snaps.append(
            snapshot_from_gpu(
                x, step + 1, eps, rr, tr,
                float(meta["sigma_t"]),
                center_mean_before_step=meta.get("center_mean"),
            )
        )
    return Trajectory(snaps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harmonic-umbrella enhanced TPS on Boltz-2 co-folding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--yaml", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("umbrella_tps_out"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--shoot-rounds", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--save-trajectory-every", type=int, default=10)
    parser.add_argument("--log-path-prob-every", type=int, default=0)
    parser.add_argument("--forward-only", action="store_true", default=False)
    parser.add_argument("--reshuffle-probability", type=float, default=0.1)
    parser.add_argument("--kernels", action="store_true")

    umb = parser.add_argument_group("Harmonic umbrella parameters")
    umb.add_argument(
        "--umbrella-center",
        type=float,
        default=0.0,
        help="Harmonic restraint center phi0 (CV units).",
    )
    umb.add_argument(
        "--spring-constant",
        type=float,
        default=1.0,
        help="kappa in V = 0.5*kappa*(phi-center)^2 (energy / CV^2). Default: 1.0",
    )
    umb.add_argument(
        "--kbt",
        type=float,
        default=2.494,
        help="Thermal energy k_B*T in same energy units as kappa (default ~300K kJ/mol).",
    )
    umb.add_argument(
        "--bias-cv", type=str, default="openmm", choices=["rmsd", "rg", "openmm"],
        help=(
            "Collective variable to bias. "
            "'openmm' (default): AMBER14/GBn2 Cα-RMSD to local energy minimum -- "
            "identical to watch_rmsd_live.py; requires --topo-npz. "
            "'rmsd': fast RMSD to initial structure (proxy; no OpenMM needed). "
            "'rg': fast radius of gyration (proxy)."
        ),
    )
    umb.add_argument(
        "--topo-npz", type=Path, default=None,
        help=(
            "Path to Boltz processed/structures/*.npz for PDB conversion. "
            "Required when --bias-cv openmm."
        ),
    )
    umb.add_argument(
        "--openmm-platform", type=str, default="CUDA",
        choices=["CUDA", "OpenCL", "CPU"],
        help="OpenMM platform for minimization (default: CUDA).",
    )
    umb.add_argument(
        "--openmm-max-iter", type=int, default=500,
        help="Max L-BFGS iterations per minimization (default: 500).",
    )
    umb.add_argument(
        "--annealing-schedule", type=str, default=None,
        help="Comma-separated center:steps pairs (e.g. '0:100,2:200,4:300'). Overrides --shoot-rounds and fixed --umbrella-center.",
    )
    umb.add_argument(
        "--burn-in-fraction", type=float, default=0.1,
        help="Fraction of initial samples to discard as burn-in when saving MBAR data.",
    )
    umb.add_argument(
        "--save-mbar-json-every",
        type=int,
        default=100,
        help=(
            "Write mbar_samples_latest.json every N MC steps (burn-in=0 snapshot; "
            "0 disables). Default: 100."
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
        print(f"Boltz is required: pip install -e ./boltz\n{e}", file=sys.stderr)
        sys.exit(1)

    cache = Path(args.cache).expanduser() if args.cache else Path.home() / ".boltz"
    cache.mkdir(parents=True, exist_ok=True)
    mol_dir = cache / "mols"
    download_boltz2(cache)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    work_root = args.out.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    boltz_run_dir = work_root / f"boltz_results_{yaml_path.stem}"
    boltz_run_dir.mkdir(parents=True, exist_ok=True)

    data_list = check_inputs(yaml_path)
    process_inputs(
        data=data_list, out_dir=boltz_run_dir,
        ccd_path=cache / "ccd.pkl", mol_dir=mol_dir,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", use_msa_server=False,
        boltz2=True, preprocessing_threads=1,
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
        mol_dir=mol_dir, num_workers=0,
        constraints_dir=processed_dir / "constraints" if (processed_dir / "constraints").exists() else None,
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
        str(ckpt), strict=True, predict_args=predict_args,
        map_location="cpu", diffusion_process_args=asdict(diffusion_params),
        ema=False, use_kernels=args.kernels,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering),
    )
    model.to(device)
    model.eval()

    atom_mask, network_kwargs = boltz2_trunk_to_network_kwargs(
        model, batch, recycling_steps=args.recycling_steps,
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

    last_snap = init_traj[-1]
    ref_coords = None
    if isinstance(last_snap, BoltzSnapshot) and last_snap.tensor_coords is not None:
        ref_coords = last_snap.tensor_coords[0].clone()

    # Auto-detect topo_npz from the processed structures directory if not set.
    topo_npz = args.topo_npz
    if topo_npz is None and args.bias_cv == "openmm":
        struct_candidates = sorted((processed_dir / "structures").glob("*.npz"))
        if struct_candidates:
            topo_npz = struct_candidates[0]
            if len(struct_candidates) > 1:
                print(
                    f"[TPS-UMBRELLA] Multiple structure npz files; using {topo_npz}",
                    file=sys.stderr, flush=True,
                )
            print(f"[TPS-UMBRELLA] Auto-detected topo_npz: {topo_npz}", flush=True)
        else:
            print(
                "[TPS-UMBRELLA] ERROR: --bias-cv openmm requires --topo-npz but no "
                "processed/structures/*.npz found.",
                file=sys.stderr, flush=True,
            )
            sys.exit(1)

    cv_function = _make_cv_function(
        args.bias_cv,
        reference_coords=ref_coords,
        topo_npz=topo_npz,
        openmm_platform=args.openmm_platform,
        openmm_max_iter=args.openmm_max_iter,
        mol_dir=mol_dir,
    )

    descriptor = boltz_snapshot_descriptor(n_atoms=n_atoms)
    engine = BoltzDiffusionEngine(
        core, descriptor,
        options={"n_frames_max": core.num_sampling_steps + 4},
    )

    shoot_log = work_root / "shooting_log.txt"
    save_traj_every = max(0, int(args.save_trajectory_every))
    periodic_extra: list[tuple[Callable[[int, Trajectory], None], int]] = []
    if save_traj_every > 0:
        periodic_extra.append((_trajectory_checkpoint_callback(work_root), save_traj_every))

    if args.annealing_schedule is not None:
        anneal_schedule = _parse_annealing_schedule(args.annealing_schedule)
        first_c = anneal_schedule[0][0]
        bias = HarmonicUmbrellaBias(
            center=first_c,
            kappa=float(args.spring_constant),
            kbt=float(args.kbt),
        )
    else:
        anneal_schedule = None
        bias = HarmonicUmbrellaBias(
            center=float(args.umbrella_center),
            kappa=float(args.spring_constant),
            kbt=float(args.kbt),
        )

    save_mbar_every = max(0, int(args.save_mbar_json_every))
    periodic_step_mbar: list = []
    if save_mbar_every > 0:
        periodic_step_mbar.append(
            (_mbar_latest_step_callback(work_root, bias), save_mbar_every)
        )
    step_cb_arg = periodic_step_mbar if periodic_step_mbar else None

    if anneal_schedule is not None:
        print(f"[TPS-UMBRELLA] annealing schedule: {anneal_schedule}", flush=True)
        all_step_logs: list[dict] = []

        for seg_idx, (ctr, n_steps) in enumerate(anneal_schedule):
            bias.set_center(ctr)
            print(
                f"[TPS-UMBRELLA] segment {seg_idx+1}/{len(anneal_schedule)}: "
                f"center={ctr:.4g}, steps={n_steps}",
                flush=True,
            )
            final_traj, seg_log = run_tps_path_sampling(
                engine, init_traj, n_steps, shoot_log,
                progress_every=args.progress_every,
                periodic_callbacks=periodic_extra if periodic_extra else None,
                periodic_step_callbacks=step_cb_arg,
                log_path_prob_every=args.log_path_prob_every,
                forward_only=args.forward_only,
                reshuffle_probability=args.reshuffle_probability,
                enhanced_bias=bias,
                cv_function=cv_function,
            )
            all_step_logs.extend(seg_log)
            init_traj = final_traj
    else:
        total_rounds = args.shoot_rounds
        print(
            f"[TPS-UMBRELLA] center={args.umbrella_center:.4g}, kappa={args.spring_constant:.4g}, rounds={total_rounds}",
            flush=True,
        )

        final_traj, all_step_logs = run_tps_path_sampling(
            engine, init_traj, total_rounds, shoot_log,
            progress_every=args.progress_every,
            periodic_callbacks=periodic_extra if periodic_extra else None,
            periodic_step_callbacks=step_cb_arg,
            log_path_prob_every=args.log_path_prob_every,
            forward_only=args.forward_only,
            reshuffle_probability=args.reshuffle_probability,
            enhanced_bias=bias,
            cv_function=cv_function,
        )

    if save_mbar_every > 0:
        try:
            _write_mbar_samples_latest(work_root, bias)
        except Exception as exc:
            print(
                f"[TPS-UMBRELLA] final mbar_samples_latest flush failed: {exc}",
                file=sys.stderr,
                flush=True,
            )

    mbar_est = MBARDistributionEstimator()
    mbar_est.add_samples_from_bias(bias, burn_in_fraction=args.burn_in_fraction)
    mbar_path = work_root / "mbar_samples.json"
    mbar_est.save_samples(mbar_path)
    print(f"[TPS-UMBRELLA] MBAR samples saved to {mbar_path}", flush=True)

    summary = {
        "bias_type": "harmonic_umbrella",
        "umbrella_center": args.umbrella_center,
        "spring_constant": args.spring_constant,
        "kbt": args.kbt,
        "annealing_schedule": args.annealing_schedule,
        "bias_cv": args.bias_cv,
        "total_steps": len(all_step_logs),
        "n_samples_recorded": len(bias.samples),
        "mbar_samples_path": str(mbar_path),
        "mbar_samples_latest_path": str((work_root / "mbar_samples_latest.json").resolve())
        if save_mbar_every > 0
        else None,
        "tps_steps": all_step_logs,
    }
    summary_path = work_root / "umbrella_tps_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[TPS-UMBRELLA] Summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
