#!/usr/bin/env python3
"""Bidirectional FES-guided RL: OpenMM+OPES and Boltz diffusion in a mutual feedback loop.

Each iteration has six steps:

1. **MD burst** -- Run short OpenMM Langevin dynamics with OPES kernel deposition
   (refines the physics-based FES reference).
2. **Boltz rollouts** -- Generate diffusion trajectories and compute terminal CVs.
3. **Generative OPES deposit** -- Inject Boltz terminal CVs into the shared OPES bias
   at reduced weight (``generative_deposit_weight``), so the target density
   incorporates regions the generator has explored.
4. **Disagreement map** -- Score each Boltz sample by
   ``|log p_target(cv) - log p_student(cv)|``.  Identify the highest-disagreement
   structure as the most informative starting point for physics validation.
5. **Warm-start MD** -- (Optional, controlled by ``--disagreement-warmstart``)
   Teleport OpenMM to the Boltz structure with maximal disagreement, minimise, and
   let the next MD burst physically explore that neighbourhood.
6. **PPO update** -- Standard DDPO-IS clipped surrogate loss using the now-updated
   OPES target and student KDE baseline.

The loop is self-correcting: if Boltz proposes implausible structures, the MD burst
at that location produces low ``log_p_target``, yielding a negative advantage that
pushes the generator away.

Example::

    python scripts/train_bidirectional_fes_boltz.py \\
        --out ./bidir_fes_out \\
        --yaml examples/cofolding_multimer_msa_empty.yaml \\
        --n-iters 500 \\
        --rollouts-per-iter 4 \\
        --diffusion-steps 16 \\
        --disagreement-warmstart \\
        --generative-deposit-weight 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bidirectional FES-guided RL (OpenMM <-> Boltz feedback loop).",
    )
    parser.add_argument("--yaml", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--kernels", action="store_true")
    parser.add_argument("--n-iters", type=int, default=500)
    parser.add_argument("--md-steps-per-burst", type=int, default=2000)
    parser.add_argument("--md-deposit-pace", type=int, default=10)
    parser.add_argument("--rollouts-per-iter", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--tau-sq", type=float, default=1e-2)
    parser.add_argument("--topo-npz", type=Path, default=None)
    parser.add_argument("--ref-pdb", type=Path, default=None)
    parser.add_argument("--ligand-smiles-json", type=Path, default=None)
    parser.add_argument("--openmm-platform", type=str, default="CUDA")
    parser.add_argument("--opes-barrier", type=float, default=5.0)
    parser.add_argument("--opes-biasfactor", type=float, default=10.0)
    parser.add_argument("--opes-kbt", type=float, default=2.494)
    parser.add_argument("--student-kde-window", type=int, default=200)
    parser.add_argument("--advantage-clip", type=float, default=5.0)
    parser.add_argument("--teacher-minimize-steps", type=int, default=0)
    parser.add_argument("--pocket-radius", type=float, default=6.0)
    parser.add_argument("--train-full-model", action="store_true", default=False)

    # Bidirectional loop parameters
    parser.add_argument(
        "--generative-deposit-weight", type=float, default=0.2,
        help="OPES kernel height scale for generative-model deposits (0, 1]. Default: 0.2.",
    )
    parser.add_argument(
        "--disagreement-warmstart", action="store_true", default=False,
        help="Teleport OpenMM to highest-disagreement Boltz structure each warm-start iteration.",
    )
    parser.add_argument(
        "--warmstart-minimize-steps", type=int, default=200,
        help="Energy minimisation iterations after warm-start position set. Default: 200.",
    )
    parser.add_argument(
        "--warmstart-fraction", type=float, default=0.5,
        help="Fraction of iterations that perform warm-start (0=never, 1=every). Default: 0.5.",
    )

    args = parser.parse_args()

    yaml_path = args.yaml or (_REPO_ROOT / "examples" / "cofolding_multimer_msa_empty.yaml")
    if not yaml_path.is_file():
        print(f"YAML not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from genai_tps.analysis.boltz_npz_export import load_topo
        from genai_tps.backends.boltz.collective_variables import PoseCVIndexer
        from genai_tps.backends.boltz.session import (
            boltz_prep_run_dir,
            build_boltz_session,
            write_ref_pdb_from_structure,
        )
        from genai_tps.enhanced_sampling.opes_bias import OPESBias
        from genai_tps.rl.config import BoltzRLConfig, FESTeacherConfig
        from genai_tps.rl.fes_teacher import OpenMMTeacher, boltz_terminal_pose_cv_numpy
        from genai_tps.rl.rollout import rollout_forward_trajectory
        from genai_tps.rl.student_distribution import BoltzStudentKDE
        from genai_tps.rl.training import fes_guided_trajectory_loss
    except ImportError as e:
        print(f"Import error: {e}", file=sys.stderr)
        sys.exit(1)

    cache = Path(args.cache).expanduser() if args.cache else Path.home() / ".boltz"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    work_root = args.out.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    ligand_smiles: dict[str, str] | None = None
    if args.ligand_smiles_json is not None:
        ligand_smiles = json.loads(Path(args.ligand_smiles_json).read_text(encoding="utf-8"))

    boltz_run_dir = boltz_prep_run_dir(work_root, yaml_path.stem)
    model, core, _batch, processed_dir, topo_auto, _prep_dir, _ = build_boltz_session(
        yaml_path=yaml_path,
        cache=cache,
        boltz_run_dir=boltz_run_dir,
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
        write_ref_pdb_from_structure(structure, int(n_struct), ref_pdb)

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
        generative_deposit_weight=float(args.generative_deposit_weight),
        disagreement_warmstart=bool(args.disagreement_warmstart),
        warmstart_minimize_steps=int(args.warmstart_minimize_steps),
        warmstart_fraction=float(args.warmstart_fraction),
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

    gen_deposit_counter = 0
    warmstart_every = max(1, int(round(1.0 / fes_cfg.warmstart_fraction))) if fes_cfg.warmstart_fraction > 0 else 0

    train_full = bool(args.train_full_model)
    params = list(model.parameters()) if train_full else list(model.structure_module.parameters())
    optimizer = Adam(params, lr=rl_cfg.learning_rate)

    for it in range(1, fes_cfg.n_iters + 1):
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Step 1: Physics MD burst (deposits OPES kernels from Langevin MD)
        # ------------------------------------------------------------------
        teacher.run_md_burst(fes_cfg.md_steps_per_burst, fes_cfg.md_deposit_pace)

        # ------------------------------------------------------------------
        # Step 2: Boltz rollouts + terminal CVs and coordinates
        # ------------------------------------------------------------------
        trajectories: list = []
        cvs: list[np.ndarray] = []
        terminal_coords: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            core.diffusion.eval()
            for _ in range(fes_cfg.boltz_rollouts_per_iter):
                tr = rollout_forward_trajectory(core, num_steps=args.diffusion_steps)
                trajectories.append(tr)
                cv = boltz_terminal_pose_cv_numpy(tr[-1].x_next, int(n_struct), indexer)
                cvs.append(cv)
                terminal_coords.append(
                    tr[-1].x_next[:, : int(n_struct), :].detach().cpu().numpy()
                )

        # ------------------------------------------------------------------
        # Step 3: Generative OPES deposits (weighted)
        # ------------------------------------------------------------------
        gen_deposits = 0
        for cv in cvs:
            if np.all(np.isfinite(cv)):
                gen_deposit_counter += 1
                teacher.opes.update(
                    cv,
                    gen_deposit_counter,
                    height_scale=fes_cfg.generative_deposit_weight,
                )
                gen_deposits += 1

        # ------------------------------------------------------------------
        # Step 4: Update student KDE + disagreement-driven warm-start
        # ------------------------------------------------------------------
        for cv in cvs:
            student_kde.update(cv)

        warmstarted = False
        max_disagreement = 0.0
        if (
            fes_cfg.disagreement_warmstart
            and warmstart_every > 0
            and it % warmstart_every == 0
        ):
            disagreements = []
            for cv in cvs:
                log_pt = teacher.log_p_target(cv)
                log_ps = student_kde.log_density(cv)
                disagreements.append(abs(log_pt - log_ps))
            best_k = int(np.argmax(disagreements))
            max_disagreement = disagreements[best_k]
            teacher.set_positions_from_boltz(
                terminal_coords[best_k],
                minimize_steps=fes_cfg.warmstart_minimize_steps,
            )
            warmstarted = True

        # ------------------------------------------------------------------
        # Step 5: PPO update
        # ------------------------------------------------------------------
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
            clip_grad_norm_(params, rl_cfg.max_grad_norm)
        optimizer.step()

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        ws_tag = f" warmstart(d={max_disagreement:.3g})" if warmstarted else ""
        print(
            f"[BiDir-FES] iter {it}/{fes_cfg.n_iters} "
            f"loss={float(total_loss.detach().cpu()):.6f} "
            f"gen_deposits={gen_deposits} "
            f"opes_kernels={teacher.opes.n_kernels} "
            f"student_n={len(student_kde._buf)}"
            f"{ws_tag}",
            flush=True,
        )
        if it % max(1, fes_cfg.n_iters // 10) == 0 or it == fes_cfg.n_iters:
            ckpt_path = work_root / f"boltz2_bidir_fes_iter_{it}.pt"
            torch.save(model.state_dict(), ckpt_path)
            teacher.opes.save_state(work_root / f"opes_state_iter_{it}.json")
            print(f"[BiDir-FES] saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
