#!/usr/bin/env python3
"""FES-guided RL for Boltz-2: OpenMM+OPES teacher vs Boltz student (CV KL surrogate).

Each iteration:

1. Run a short OpenMM Langevin burst with on-the-fly OPES bias (refines target CV density).
2. Roll out Boltz diffusion trajectories; map terminal frames to pose CVs.
3. Update a running KDE over student CVs; compute clipped advantages
   ``log p_target - log p_Boltz`` from the teacher OPES estimate.
4. Apply the DDPO-IS / PPO clipped surrogate (:mod:`genai_tps.rl.training`).

Example::

    pip install -e ".[boltz,dev]" && pip install -e ./boltz
    python scripts/train_fes_guided_boltz.py \\
        --out ./fes_rl_out \\
        --yaml examples/cofolding_multimer_msa_empty.yaml \\
        --n-iters 2 \\
        --rollouts-per-iter 1 \\
        --diffusion-steps 8

Requires OpenMM (+ optional PDBFixer / openff for ligands) and a Boltz-2 checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

from genai_tps.utils.compute_device import (  # noqa: E402
    maybe_set_torch_cuda_current_device,
    parse_torch_device,
)


def _exit_if_rl_excluded() -> None:
    try:
        import genai_tps.rl.config  # noqa: F401
    except ImportError:
        print(
            "This script requires the optional genai_tps.rl package.\n"
            "Restore it with: git checkout HEAD -- src/python/genai_tps/rl",
            file=sys.stderr,
        )
        sys.exit(2)


def _build_boltz_session(
    *,
    yaml_path: Path,
    cache: Path,
    boltz_prep_dir: Path,
    device: torch.device,
    diffusion_steps: int,
    recycling_steps: int,
    kernels: bool,
):
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

    from genai_tps.backends.boltz.boltz2_trunk import boltz2_trunk_to_network_kwargs
    from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore

    cache.mkdir(parents=True, exist_ok=True)
    mol_dir = cache / "mols"
    download_boltz2(cache)

    boltz_run_dir = boltz_prep_dir / f"boltz_prep_{yaml_path.stem}"
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
        raise RuntimeError("No records in manifest after preprocessing.")

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
        "recycling_steps": recycling_steps,
        "sampling_steps": diffusion_steps,
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
        use_kernels=kernels,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering),
    )
    model.to(device)

    atom_mask, network_kwargs = boltz2_trunk_to_network_kwargs(
        model, batch, recycling_steps=recycling_steps
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
    core.build_schedule(diffusion_steps)

    struct_candidates = sorted((processed_dir / "structures").glob("*.npz"))
    topo_npz = struct_candidates[0] if struct_candidates else None

    return model, core, batch, processed_dir, topo_npz, boltz_run_dir


def _write_ref_pdb_from_structure(structure, n_struct: int, out_pdb: Path) -> None:
    """Single-frame PDB from Boltz topology coordinates (for OpenMM system build)."""
    from boltz.data.types import Coords, Interface
    from boltz.data.write.pdb import to_pdb

    fc = np.asarray(structure.atoms["coords"], dtype=np.float32)[: int(n_struct)]
    atoms = structure.atoms.copy()
    atoms["coords"] = fc
    atoms["is_present"] = True
    residues = structure.residues.copy()
    residues["is_present"] = True
    coord_arr = np.array([(x,) for x in fc], dtype=Coords)
    interfaces = np.array([], dtype=Interface)
    new_s = replace(
        structure,
        atoms=atoms,
        residues=residues,
        interfaces=interfaces,
        coords=coord_arr,
    )
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pdb.write_text(to_pdb(new_s, plddts=None, boltz2=True))


def main() -> None:
    _exit_if_rl_excluded()
    parser = argparse.ArgumentParser(description="FES-guided RL (OpenMM+OPES teacher, Boltz student).")
    parser.add_argument("--yaml", type=Path, default=None, help="Boltz input YAML (co-folding).")
    parser.add_argument("--cache", type=Path, default=None, help="Boltz cache dir (default ~/.boltz).")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device: cpu, cuda, or cuda:N (default: cuda).",
    )
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--kernels", action="store_true")
    parser.add_argument("--n-iters", type=int, default=1000)
    parser.add_argument("--md-steps-per-burst", type=int, default=2000)
    parser.add_argument("--md-deposit-pace", type=int, default=10)
    parser.add_argument("--rollouts-per-iter", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--tau-sq", type=float, default=1e-2)
    parser.add_argument("--topo-npz", type=Path, default=None, help="structures/*.npz topology.")
    parser.add_argument("--ref-pdb", type=Path, default=None, help="Heavy-atom PDB for OpenMM (default: write from topo).")
    parser.add_argument(
        "--ligand-smiles-json",
        type=Path,
        default=None,
        help='JSON dict of chain_id → SMILES (same convention as compute_cv_rmsd).',
    )
    parser.add_argument("--openmm-platform", type=str, default="CUDA")
    parser.add_argument("--opes-barrier", type=float, default=5.0)
    parser.add_argument("--opes-biasfactor", type=float, default=10.0)
    parser.add_argument("--opes-kbt", type=float, default=2.494)
    parser.add_argument("--student-kde-window", type=int, default=200)
    parser.add_argument("--advantage-clip", type=float, default=5.0)
    parser.add_argument("--teacher-minimize-steps", type=int, default=0)
    parser.add_argument("--pocket-radius", type=float, default=6.0)
    parser.add_argument("--train-full-model", action="store_true", default=False)
    args = parser.parse_args()

    yaml_path = args.yaml or (_REPO_ROOT / "examples" / "cofolding_multimer_msa_empty.yaml")
    if not yaml_path.is_file():
        print(f"YAML not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from genai_tps.io.boltz_npz_export import load_topo
        from genai_tps.backends.boltz.cache_paths import default_boltz_cache_dir
        from genai_tps.backends.boltz.collective_variables import PoseCVIndexer
        from genai_tps.simulation import OPESBias
        from genai_tps.rl.config import BoltzRLConfig, FESTeacherConfig
        from genai_tps.rl.fes_teacher import OpenMMTeacher, boltz_terminal_pose_cv_numpy
        from genai_tps.rl.rollout import rollout_forward_trajectory
        from genai_tps.rl.student_distribution import BoltzStudentKDE
        from genai_tps.rl.training import fes_guided_trajectory_loss
    except ImportError as e:
        print(f"Import error: {e}", file=sys.stderr)
        sys.exit(1)

    cache = Path(args.cache).expanduser() if args.cache else default_boltz_cache_dir()
    if torch.cuda.is_available():
        device = parse_torch_device(args.device)
        maybe_set_torch_cuda_current_device(device)
    else:
        device = torch.device("cpu")
    work_root = args.out.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    ligand_smiles: dict[str, str] | None = None
    if args.ligand_smiles_json is not None:
        ligand_smiles = json.loads(Path(args.ligand_smiles_json).read_text(encoding="utf-8"))

    model, core, _batch, processed_dir, topo_auto, _prep_dir = _build_boltz_session(
        yaml_path=yaml_path,
        cache=cache,
        boltz_prep_dir=work_root,
        device=device,
        diffusion_steps=args.diffusion_steps,
        recycling_steps=args.recycling_steps,
        kernels=args.kernels,
    )

    topo_npz = Path(args.topo_npz).expanduser().resolve() if args.topo_npz else topo_auto
    if topo_npz is None or not topo_npz.is_file():
        print("Topology NPZ required; pass --topo-npz.", file=sys.stderr)
        sys.exit(1)

    structure, n_struct = load_topo(topo_npz)
    ref_coords = np.asarray(structure.atoms["coords"], dtype=np.float64)[: int(n_struct)]
    indexer = PoseCVIndexer(structure, ref_coords, pocket_radius=float(args.pocket_radius))

    ref_pdb = Path(args.ref_pdb).expanduser().resolve() if args.ref_pdb else work_root / "fes_ref_topology.pdb"
    if args.ref_pdb is None or not ref_pdb.is_file():
        _write_ref_pdb_from_structure(structure, int(n_struct), ref_pdb)

    fes_cfg = FESTeacherConfig(
        md_steps_per_burst=int(args.md_steps_per_burst),
        md_deposit_pace=int(args.md_deposit_pace),
        boltz_rollouts_per_iter=int(args.rollouts_per_iter),
        n_iters=int(args.n_iters),
        opes_barrier=float(args.opes_barrier),
        opes_biasfactor=float(args.opes_biasfactor),
        opes_kbt=float(args.opes_kbt),
        student_kde_window=int(args.student_kde_window),
        advantage_clip=float(args.advantage_clip),
        teacher_minimize_steps=int(args.teacher_minimize_steps),
        pocket_radius=float(args.pocket_radius),
    )
    rl_cfg = BoltzRLConfig(
        learning_rate=float(args.learning_rate),
        clip_range=float(args.clip_range),
        velocity_log_prob_tau_sq=float(args.tau_sq),
    )

    opes = OPESBias(
        ndim=2,
        kbt=float(args.opes_kbt),
        barrier=float(args.opes_barrier),
        biasfactor=float(args.opes_biasfactor),
        pace=1,
    )
    teacher = OpenMMTeacher(
        ref_pdb,
        structure,
        indexer,
        opes,
        ligand_smiles=ligand_smiles,
        platform_name=str(args.openmm_platform),
        temperature_k=300.0,
        logp_kbt=float(args.opes_kbt),
        minimize_steps=int(fes_cfg.teacher_minimize_steps),
    )
    student_kde = BoltzStudentKDE(
        2,
        window=fes_cfg.student_kde_window,
        bandwidth=fes_cfg.student_kde_bandwidth,
    )

    train_full = bool(args.train_full_model)
    params = list(model.parameters()) if train_full else list(model.structure_module.parameters())
    optimizer = Adam(params, lr=rl_cfg.learning_rate)

    for it in range(1, fes_cfg.n_iters + 1):
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        teacher.run_md_burst(fes_cfg.md_steps_per_burst, fes_cfg.md_deposit_pace)

        trajectories: list = []
        cvs: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            core.diffusion.eval()
            for _ in range(fes_cfg.boltz_rollouts_per_iter):
                tr = rollout_forward_trajectory(core, num_steps=args.diffusion_steps)
                trajectories.append(tr)
                cv = boltz_terminal_pose_cv_numpy(tr[-1].x_next, int(n_struct), indexer)
                cvs.append(cv)

        for cv in cvs:
            student_kde.update(cv)

        model.train()
        if not train_full:
            model.structure_module.train()
        core.diffusion.train()

        optimizer.zero_grad()
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        for tr, cv in zip(trajectories, cvs):
            loss = fes_guided_trajectory_loss(
                core,
                tr,
                cv,
                teacher,
                student_kde,
                fes_cfg=fes_cfg,
                rl_cfg=rl_cfg,
            )
            total_loss = total_loss + loss
        total_loss.backward()
        if rl_cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, rl_cfg.max_grad_norm)
        optimizer.step()

        print(
            f"[FES-RL] iter {it}/{fes_cfg.n_iters} loss={float(total_loss.detach().cpu()):.6f} "
            f"openmm_platform={teacher.platform_used}",
            flush=True,
        )
        if it % max(1, fes_cfg.n_iters // 10) == 0 or it == fes_cfg.n_iters:
            ckpt_path = work_root / f"boltz2_fes_rl_iter_{it}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[FES-RL] saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
