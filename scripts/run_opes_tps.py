#!/usr/bin/env python3
"""Run OPES-biased enhanced TPS on a Boltz-2 co-folding input.

Wraps the standard ``run_cofolding_tps_demo.py`` workflow with OPES (On-the-fly
Probability Enhanced Sampling) adaptive bias.  The bias is built on-the-fly
using kernel density estimation with compression, and enters the Metropolis
acceptance as:

    trial.bias *= exp(-(V(cv_new) - V(cv_old)) / kT)

The OPES state is saved periodically for restarts and for post-hoc reweighting.

Example::

    python scripts/run_opes_tps.py \\
        --out ./opes_tps_out \\
        --diffusion-steps 32 \\
        --shoot-rounds 5000 \\
        --opes-barrier 5.0 \\
        --opes-biasfactor 10.0 \\
        --opes-pace 1 \\
        --bias-cv rmsd \\
        --save-trajectory-every 10 \\
        --save-opes-state-every 100 \\
        --progress-every 50
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
from genai_tps.backends.boltz.collective_variables import (
    radius_of_gyration,
    rmsd_to_reference,
)
from genai_tps.backends.boltz.engine import BoltzDiffusionEngine
from genai_tps.enhanced_sampling.openmm_cv import OpenMMLocalMinRMSD
from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.snapshot import (
    BoltzSnapshot,
    boltz_snapshot_descriptor,
    snapshot_frame_numpy_copy,
)
from genai_tps.backends.boltz.tps_sampling import run_tps_path_sampling
from genai_tps.enhanced_sampling import OPESBias


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
        ``"rmsd"``   -- fast RMSD to the initial structure's last frame (proxy CV).
        ``"rg"``     -- fast radius of gyration (proxy CV).
        ``"openmm"`` -- AMBER14/GBn2 Cα-RMSD to local energy minimum; identical
                        to the CV used by ``watch_rmsd_live.py`` and required for
                        a meaningful comparison against the vanilla TPS baseline.
                        Requires *topo_npz* and has ~1--10 s per-call cost
                        (mitigated by coordinate-hash caching).
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
            raise ValueError("RMSD CV requires a reference structure.")
        return _cv_rmsd
    elif cv_type == "rg":
        return _cv_rg
    elif cv_type == "openmm":
        if topo_npz is None:
            raise ValueError(
                "--bias-cv openmm requires --topo-npz pointing to the Boltz "
                "processed/structures/*.npz file."
            )
        openmm_cv = OpenMMLocalMinRMSD(
            topo_npz=topo_npz,
            platform=openmm_platform,
            max_iter=openmm_max_iter,
            mol_dir=mol_dir,
        )
        return openmm_cv
    else:
        raise ValueError(f"Unknown CV type: {cv_type!r}. Use 'rmsd', 'rg', or 'openmm'.")


def _write_tps_checkpoint_npz(traj: Trajectory, out_npz: Path, *, mc_step: int) -> bool:
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
                f"[TPS-OPES] checkpoint MC step {mc_step} -> {tagged.name}",
                file=sys.stderr, flush=True,
            )
    return cb


def _opes_state_checkpoint_callback(
    bias: OPESBias, work_root: Path, every: int,
) -> Callable[[int, Trajectory], None]:
    """Periodic OPES state checkpoints for restart and analysis."""
    def cb(mc_step: int, _traj: Trajectory) -> None:
        if every <= 0 or mc_step % every != 0:
            return
        state_dir = work_root / "opes_states"
        state_dir.mkdir(parents=True, exist_ok=True)
        tagged = state_dir / f"opes_state_{mc_step:08d}.json"
        bias.save_state(tagged)
        latest = state_dir / "opes_state_latest.json"
        shutil.copyfile(tagged, latest)
        print(
            f"[TPS-OPES] OPES state checkpoint MC step {mc_step} "
            f"({bias.n_kernels} kernels, {bias.counter} depositions)",
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
        description="OPES-biased enhanced TPS on Boltz-2 co-folding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--yaml", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("opes_tps_out"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--shoot-rounds", type=int, default=2000)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--save-trajectory-every", type=int, default=10)
    parser.add_argument("--log-path-prob-every", type=int, default=0)
    parser.add_argument("--forward-only", action="store_true", default=False)
    parser.add_argument("--reshuffle-probability", type=float, default=0.1)
    parser.add_argument("--kernels", action="store_true")

    opes_group = parser.add_argument_group("OPES bias parameters")
    opes_group.add_argument(
        "--opes-barrier", type=float, default=5.0,
        help="Estimated free-energy barrier in kT units. Sets initial kernel width. (default: 5.0)",
    )
    opes_group.add_argument(
        "--opes-biasfactor", type=float, default=10.0,
        help="Well-tempering factor gamma. Larger = flatter target. Use 'inf' for uniform. (default: 10.0)",
    )
    opes_group.add_argument(
        "--opes-pace", type=int, default=1,
        help="Deposit a kernel every N MC steps. (default: 1)",
    )
    opes_group.add_argument(
        "--opes-epsilon", type=float, default=None,
        help="Regularization epsilon. Default: exp(-barrier/kT).",
    )
    opes_group.add_argument(
        "--opes-compression-threshold", type=float, default=1.0,
        help="Kernel merge threshold in sigma units. (default: 1.0)",
    )
    opes_group.add_argument(
        "--opes-sigma-min", type=float, default=0.0,
        help="Minimum kernel width. (default: 0.0 = no floor)",
    )
    opes_group.add_argument(
        "--opes-fixed-sigma", type=float, default=None,
        help="Fixed kernel width (overrides adaptive estimation). Default: adaptive.",
    )
    opes_group.add_argument(
        "--opes-explore", action="store_true", default=False,
        help="Use OPES explore mode (unweighted KDE). Default: convergence mode.",
    )
    opes_group.add_argument(
        "--opes-kbt", type=float, default=1.0,
        help="Thermal energy kT. For dimensionless TPS, use 1.0. (default: 1.0)",
    )
    opes_group.add_argument(
        "--opes-restart", type=Path, default=None,
        help="Path to saved OPES state JSON for restart.",
    )
    opes_group.add_argument(
        "--bias-cv", type=str, default="openmm", choices=["rmsd", "rg", "openmm"],
        help=(
            "Collective variable for bias. "
            "'openmm' (default): AMBER14/GBn2 Cα-RMSD to local energy minimum -- "
            "identical to watch_rmsd_live.py; requires --topo-npz. "
            "'rmsd': fast RMSD to initial structure (proxy; no OpenMM needed). "
            "'rg': fast radius of gyration (proxy)."
        ),
    )
    opes_group.add_argument(
        "--topo-npz", type=Path, default=None,
        help=(
            "Path to Boltz processed/structures/*.npz for PDB conversion. "
            "Required when --bias-cv openmm is used."
        ),
    )
    opes_group.add_argument(
        "--openmm-platform", type=str, default="CUDA",
        choices=["CUDA", "OpenCL", "CPU"],
        help="OpenMM platform for minimization (default: CUDA).",
    )
    opes_group.add_argument(
        "--openmm-max-iter", type=int, default=500,
        help="Max L-BFGS iterations per minimization (default: 500).",
    )
    opes_group.add_argument(
        "--save-opes-state-every", type=int, default=100,
        help="Save OPES state every N MC steps. (default: 100)",
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
                    f"[TPS-OPES] Multiple structure npz files; using {topo_npz}",
                    file=sys.stderr, flush=True,
                )
            print(f"[TPS-OPES] Auto-detected topo_npz: {topo_npz}", flush=True)
        else:
            print(
                "[TPS-OPES] ERROR: --bias-cv openmm requires --topo-npz but no "
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

    if args.opes_restart is not None:
        print(f"[TPS-OPES] Restarting from {args.opes_restart}", flush=True)
        bias = OPESBias.load_state(args.opes_restart)
    else:
        biasfactor = float("inf") if args.opes_biasfactor <= 0 else args.opes_biasfactor
        bias = OPESBias(
            kbt=args.opes_kbt,
            barrier=args.opes_barrier,
            biasfactor=biasfactor,
            epsilon=args.opes_epsilon,
            kernel_cutoff=None,
            compression_threshold=args.opes_compression_threshold,
            pace=args.opes_pace,
            sigma_min=args.opes_sigma_min,
            fixed_sigma=args.opes_fixed_sigma,
            explore=args.opes_explore,
        )

    print(
        f"[TPS-OPES] barrier={bias.barrier:.2f} biasfactor={bias.biasfactor:.2f} "
        f"pace={bias.pace} epsilon={bias.epsilon:.2e} "
        f"mode={'explore' if bias.explore else 'convergence'}",
        flush=True,
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

    save_opes_every = max(0, int(args.save_opes_state_every))
    if save_opes_every > 0:
        periodic_extra.append(
            (_opes_state_checkpoint_callback(bias, work_root, save_opes_every), save_opes_every)
        )

    cv_values_log: list[dict] = []

    def _cv_logging_callback(mc_step: int, traj: Trajectory) -> None:
        try:
            cv_val = cv_function(traj)
            cv_values_log.append({"mc_step": mc_step, "cv": cv_val})
        except Exception:
            pass

    periodic_extra.append((_cv_logging_callback, 1))

    final_traj, tps_step_log = run_tps_path_sampling(
        engine, init_traj, args.shoot_rounds, shoot_log,
        progress_every=args.progress_every,
        periodic_callbacks=periodic_extra if periodic_extra else None,
        log_path_prob_every=args.log_path_prob_every,
        forward_only=args.forward_only,
        reshuffle_probability=args.reshuffle_probability,
        enhanced_bias=bias,
        cv_function=cv_function,
    )

    final_state_path = work_root / "opes_state_final.json"
    bias.save_state(final_state_path)
    print(
        f"[TPS-OPES] Final OPES state: {bias.n_kernels} kernels, "
        f"{bias.counter} depositions, zed={bias.zed:.4g}",
        flush=True,
    )

    cv_data_path = work_root / "cv_values.json"
    cv_data = {
        "cv_type": args.bias_cv,
        "cv_values": [entry["cv"] for entry in cv_values_log],
        "mc_steps": [entry["mc_step"] for entry in cv_values_log],
    }
    Path(cv_data_path).write_text(json.dumps(cv_data, indent=2))

    cv_stats = {}
    from genai_tps.enhanced_sampling.openmm_cv import OpenMMLocalMinRMSD
    if isinstance(cv_function, OpenMMLocalMinRMSD):
        cv_stats = cv_function.stats()
        print(
            f"[TPS-OPES] OpenMM CV stats: {cv_stats['n_calls']} calls, "
            f"{cv_stats['cache_hit_rate']:.1%} cache hit rate, "
            f"{cv_stats['n_failures']} failures",
            flush=True,
        )

    summary = {
        "bias_type": "opes_adaptive",
        "opes_barrier": bias.barrier,
        "opes_biasfactor": bias.biasfactor,
        "opes_pace": bias.pace,
        "opes_explore": bias.explore,
        "bias_cv": args.bias_cv,
        "total_steps": len(tps_step_log),
        "n_kernels_final": bias.n_kernels,
        "n_depositions": bias.counter,
        "n_merges": bias._n_merges,
        "zed_final": bias.zed,
        "rct_final": bias.rct,
        "opes_state_path": str(final_state_path),
        "cv_values_path": str(cv_data_path),
        "cv_stats": cv_stats,
        "tps_steps": tps_step_log,
    }
    summary_path = work_root / "opes_tps_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[TPS-OPES] Summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
