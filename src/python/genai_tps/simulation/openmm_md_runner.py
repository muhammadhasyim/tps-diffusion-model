"""OPES-biased OpenMM MD driver for Boltz-aligned WDSM ground-truth samples."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def _log_coord_stats_np(tag: str, coords: np.ndarray, *, unit_label: str) -> None:
    """Print finite-ness and axis bounds for a coordinate array (diagnostics)."""
    a = np.asarray(coords, dtype=np.float64)
    finite = bool(np.isfinite(a).all())
    if a.size == 0:
        print(f"[MD-OPES] {tag}: empty array ({unit_label})", flush=True)
        return
    print(
        f"[MD-OPES] {tag}: finite={finite} shape={a.shape} ({unit_label}) "
        f"min=({a[:, 0].min():.4f},{a[:, 1].min():.4f},{a[:, 2].min():.4f}) "
        f"max=({a[:, 0].max():.4f},{a[:, 1].max():.4f},{a[:, 2].max():.4f})",
        flush=True,
    )


def _diagnostic_energy_and_large_forces(
    sim,
    tag: str,
    *,
    large_force_threshold: float = 100_000.0,
    max_print: int = 30,
) -> None:
    """Log potential energy and OpenMM particles with very large forces (OpenMM FAQ)."""
    from math import sqrt

    import openmm.unit as u

    st = sim.context.getState(getEnergy=True, getForces=True)
    pe = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    print(f"[MD-OPES] {tag}: potential_energy={pe:.3f} kJ/mol", flush=True)
    forces = st.getForces().value_in_unit(u.kilojoules_per_mole / u.nanometer)
    n_big = 0
    for i, f in enumerate(forces):
        mag = sqrt(float(f.x) ** 2 + float(f.y) ** 2 + float(f.z) ** 2)
        if mag > large_force_threshold:
            if n_big < max_print:
                print(
                    f"[MD-OPES] {tag}: large_force atom_index={i} |F|={mag:.3e} kJ/mol/nm",
                    flush=True,
                )
            n_big += 1
    if n_big > max_print:
        print(
            f"[MD-OPES] {tag}: ({n_big - max_print} more atoms with |F| > "
            f"{large_force_threshold:g} ...)",
            flush=True,
        )


def _parse_opes_nlist_parameters(value: str) -> tuple[float, float]:
    """Parse ``'a,b'`` into PLUMED ``NLIST_PARAMETERS`` components."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "expected two comma-separated floats, e.g. '4.0,0.4'"
        )
    return float(parts[0]), float(parts[1])


def _require_finite_positions(pos_nm: np.ndarray, *, context_msg: str) -> None:
    a = np.asarray(pos_nm, dtype=np.float64)
    if not np.isfinite(a).all():
        n_bad = int(np.sum(~np.isfinite(a)))
        raise RuntimeError(
            f"[MD-OPES] Non-finite particle positions after {context_msg} "
            f"({n_bad} non-finite values in position array). "
            "See OpenMM NaN FAQ: https://github.com/openmm/openmm/wiki/"
            "Frequently-Asked-Questions#nan"
        )


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
    parser.add_argument(
        "--opes-kernel-cutoff",
        type=float,
        default=None,
        metavar="SIGMA",
        help=(
            "PLUMED OPES_METAD KERNEL_CUTOFF in units of per-CV SIGMA. "
            "If omitted, uses PLUMED's default formula max(3.5, sqrt(...)) "
            "to avoid the 'kernels are truncated too much' warning at typical "
            "BARRIER/BIASFACTOR/T."
        ),
    )
    parser.add_argument(
        "--opes-nlist-parameters",
        type=_parse_opes_nlist_parameters,
        default=None,
        metavar="A,B",
        help=(
            "Optional PLUMED NLIST_PARAMETERS for OPES_METAD neighbor list, "
            "e.g. '4.0,0.4' (comma-separated floats)."
        ),
    )
    parser.add_argument(
        "--opes-mode",
        type=str,
        default="plumed",
        choices=["plumed", "observer"],
        help=(
            "OPES implementation: 'plumed' applies OPES_METAD bias forces through "
            "openmm-plumed (requires a PLUMED build with the 'opes' module — not "
            "included in default conda-forge plumed). 'observer' runs unbiased MD "
            "and updates Python OPESBias only (different physics than PLUMED OPES)."
        ),
    )
    parser.add_argument(
        "--plumed-force-group",
        type=int,
        default=30,
        help="OpenMM force group used for the PLUMED bias energy.",
    )
    parser.add_argument("--temperature", type=float, default=300.0, help="Langevin temperature (K)")
    parser.add_argument("--pocket-radius", type=float, default=6.0)
    parser.add_argument("--platform", type=str, default="CUDA", choices=["CUDA", "OpenCL", "CPU"])
    parser.add_argument("--minimize-steps", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--save-opes-every", type=int, default=50000)
    parser.add_argument("--ligand-smiles", type=str, default=None, help="chain:SMILES (e.g. B:CC...)")
    parser.add_argument("--mol-dir", type=Path, default=None, help="Boltz CCD mol dir for SMILES lookup")
    args = parser.parse_args()

    if args.opes_mode == "plumed":
        from genai_tps.simulation.plumed_kernel import assert_plumed_opes_metad_available

        assert_plumed_opes_metad_available()

    from genai_tps.backends.boltz.collective_variables import PoseCVIndexer
    from genai_tps.io.boltz_npz_export import coords_frame_from_npz, load_topo, npz_to_pdb
    from genai_tps.simulation.openmm_boltz_bridge import (
        boltz_to_plumed_indices,
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
    with np.load(str(frame_npz)) as ref_data:
        ref_coords = coords_frame_from_npz(
            ref_data, frame_idx=args.frame_idx, n_struct=n_s
        )
    _log_coord_stats_np("checkpoint_after_reference_npz", ref_coords, unit_label="Å")
    npz_to_pdb(frame_npz, structure, n_struct, pdb_path, frame_idx=args.frame_idx)
    print(f"[MD-OPES] Wrote initial PDB: {pdb_path}", flush=True)
    if pdb_path.is_file():
        print(f"[MD-OPES] checkpoint_after_pdb_write: path_ok size={pdb_path.stat().st_size} B", flush=True)
    indexer = PoseCVIndexer(structure, ref_coords, pocket_radius=args.pocket_radius)
    print(
        f"[MD-OPES] PoseCVIndexer: {len(indexer.ligand_idx)} ligand atoms, "
        f"{len(indexer.pocket_ca_idx)} pocket Cas",
        flush=True,
    )

    ligand_smiles = None
    if args.ligand_smiles:
        parts = args.ligand_smiles.split(":", 1)
        ligand_smiles = {parts[0]: parts[1]}
    elif args.mol_dir:
        from genai_tps.simulation.openmm_cv import _detect_ligand_smiles

        ligand_smiles = _detect_ligand_smiles(structure, args.mol_dir)

    kbt_kjmol = 8.314e-3 * args.temperature
    sigma = [float(s) for s in args.opes_sigma.split(",")]
    if len(sigma) != 2:
        raise ValueError("--opes-sigma must contain two comma-separated values.")

    opes = None
    if args.opes_mode == "observer":
        from genai_tps.simulation import OPESBias

        if args.opes_restart:
            print(f"[MD-OPES] Restarting observer OPES from {args.opes_restart}", flush=True)
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
        print(
            f"[MD-OPES] Observer OPES: barrier={opes.barrier:.1f} "
            f"biasfactor={opes.biasfactor:.1f} kbt={opes.kbt:.3f} kJ/mol "
            f"sigma={sigma}",
            flush=True,
        )
    else:
        print(
            f"[MD-OPES] PLUMED OPES: barrier={args.opes_barrier:.1f} "
            f"biasfactor={args.opes_biasfactor:.1f} kbt={kbt_kjmol:.3f} kJ/mol "
            f"sigma={sigma}",
            flush=True,
        )

    print(f"[MD-OPES] Building OpenMM simulation (AMBER14/GBn2, {args.platform})...", flush=True)
    build_md = load_build_md_simulation_from_pdb()
    plumed_context: dict[str, object] = {}

    def _add_plumed_force(system, h_topology, h_positions, meta):
        """Build atom selections after hydrogenation and inject PLUMED force."""
        import openmm.unit as u

        from genai_tps.simulation.plumed_opes import (
            add_plumed_opes_to_system,
            generate_plumed_opes_script,
            write_rmsd_reference_pdb,
        )

        h_pos_nm = np.asarray(h_positions.value_in_unit(u.nanometer), dtype=np.float64)
        omm_idx_cb = build_openmm_indices_for_boltz_atoms(
            structure,
            h_topology,
            ref_coords_angstrom=ref_coords,
            omm_positions_nm=h_pos_nm,
        )
        if int(np.min(omm_idx_cb)) < 0:
            raise RuntimeError(
                "[MD-OPES] Invalid Boltz→OpenMM index map before PLUMED setup "
                f"(min={int(np.min(omm_idx_cb))})."
            )

        rmsd_align_boltz = (
            indexer.pocket_ca_idx
            if len(indexer.pocket_ca_idx) >= 3
            else indexer.protein_ca_idx
        )
        if len(rmsd_align_boltz) < 3:
            raise RuntimeError(
                "[MD-OPES] PLUMED ligand RMSD requires at least three protein "
                "C-alpha alignment atoms."
            )
        if len(indexer.pocket_ca_idx) == 0:
            raise RuntimeError(
                "[MD-OPES] PLUMED ligand-pocket distance requires at least one "
                "pocket C-alpha atom."
            )

        ligand_plumed_idx = boltz_to_plumed_indices(indexer.ligand_idx, omm_idx_cb)
        pocket_plumed_idx = boltz_to_plumed_indices(indexer.pocket_ca_idx, omm_idx_cb)
        align_plumed_idx = boltz_to_plumed_indices(rmsd_align_boltz, omm_idx_cb)
        ref_pdb = out / "plumed_rmsd_reference.pdb"
        script_path = out / "plumed_opes.dat"
        write_rmsd_reference_pdb(
            h_topology,
            h_positions,
            ligand_plumed_idx,
            align_plumed_idx,
            ref_pdb,
        )
        script = generate_plumed_opes_script(
            ligand_plumed_idx=ligand_plumed_idx,
            pocket_ca_plumed_idx=pocket_plumed_idx,
            rmsd_reference_pdb=ref_pdb,
            sigma=sigma,
            pace=args.deposit_pace,
            barrier=args.opes_barrier,
            biasfactor=args.opes_biasfactor,
            temperature=args.temperature,
            save_opes_every=args.save_opes_every,
            progress_every=args.progress_every,
            out_dir=opes_dir,
            state_rfile=args.opes_restart,
            kernel_cutoff=args.opes_kernel_cutoff,
            nlist_parameters=args.opes_nlist_parameters,
        )
        script_path.write_text(script, encoding="utf-8")
        force, force_index = add_plumed_opes_to_system(
            system,
            script,
            temperature=args.temperature,
            force_group=args.plumed_force_group,
            restart=args.opes_restart is not None,
        )
        plumed_context.update(
            {
                "omm_idx": omm_idx_cb,
                "force": force,
                "force_index": force_index,
                "force_group": int(args.plumed_force_group),
                "script_path": script_path,
                "reference_pdb": ref_pdb,
            }
        )
        meta["plumed_force_index"] = force_index
        meta["plumed_force_group"] = int(args.plumed_force_group)
        meta["plumed_script"] = str(script_path)
        meta["plumed_reference_pdb"] = str(ref_pdb)

    md_boltz_pose: dict = {}
    if args.mol_dir is not None:
        md_boltz_pose = {
            "boltz_structure": structure,
            "boltz_coords_angstrom": ref_coords,
            "boltz_mol_dir": args.mol_dir.expanduser().resolve(),
            "ligand_pose_policy": "boltz_first",
        }
    sim, meta = build_md(
        pdb_path,
        platform_name=args.platform,
        temperature_k=args.temperature,
        ligand_smiles=ligand_smiles,
        extra_forces=_add_plumed_force if args.opes_mode == "plumed" else None,
        **md_boltz_pose,
    )
    print(f"[MD-OPES] Platform: {meta['platform_used']}", flush=True)

    state0 = sim.context.getState(getPositions=True)
    pos_nm = state0.getPositions(asNumpy=True)
    _log_coord_stats_np("checkpoint_after_build_set_positions", pos_nm, unit_label="nm")
    _diagnostic_energy_and_large_forces(sim, "after_build_set_positions")

    if args.opes_mode == "plumed":
        if "omm_idx" not in plumed_context:
            raise RuntimeError("[MD-OPES] PLUMED setup did not produce an atom map.")
        omm_idx = np.asarray(plumed_context["omm_idx"], dtype=np.int64)
    else:
        omm_idx = build_openmm_indices_for_boltz_atoms(
            structure,
            sim.topology,
            ref_coords_angstrom=ref_coords,
            omm_positions_nm=pos_nm,
        )
    if int(np.min(omm_idx)) < 0:
        raise RuntimeError(
            "[MD-OPES] Invalid Boltz→OpenMM index map (negative OpenMM index). "
            f"min(omm_idx)={int(np.min(omm_idx))}"
        )
    n_boltz = int(structure.atoms.shape[0])

    if args.minimize_steps > 0:
        print(f"[MD-OPES] Minimizing ({args.minimize_steps} steps)...", flush=True)
        sim.minimizeEnergy(maxIterations=args.minimize_steps)

    state_post = sim.context.getState(getPositions=True)
    pos_post_nm = state_post.getPositions(asNumpy=True)
    if args.minimize_steps > 0:
        _log_coord_stats_np("checkpoint_after_minimize", pos_post_nm, unit_label="nm")
        _require_finite_positions(pos_post_nm, context_msg="energy minimization")
        _diagnostic_energy_and_large_forces(sim, "after_minimize")
    else:
        _log_coord_stats_np("checkpoint_skip_minimize", pos_post_nm, unit_label="nm")
        _require_finite_positions(pos_post_nm, context_msg="initial context (minimize_steps=0)")

    def get_coords_boltz_order(state=None):
        import openmm.unit as u

        if state is None:
            state = sim.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(u.angstrom)
        coords = np.zeros((n_boltz, 3), dtype=np.float32)
        for i in range(n_boltz):
            coords[i] = pos[int(omm_idx[i])]
        return coords

    _probe_boltz = get_coords_boltz_order()
    _log_coord_stats_np("checkpoint_first_boltz_ordered_coords", _probe_boltz, unit_label="Å")
    _require_finite_positions(
        _probe_boltz,
        context_msg="first Boltz-ordered coordinate extraction",
    )
    del _probe_boltz

    def compute_cv(coords):
        x = torch.from_numpy(coords[:n_s]).float().unsqueeze(0)
        from types import SimpleNamespace

        snap = SimpleNamespace(_tensor_coords_gpu=x)
        from genai_tps.backends.boltz.collective_variables import ligand_pose_rmsd, ligand_pocket_distance

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

    if args.opes_mode == "observer":
        if opes is None:
            raise RuntimeError("[MD-OPES] Observer mode did not initialize OPESBias.")
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
                print(
                    f"[MD-OPES] step {step:>10,}/{args.n_steps:,} | "
                    f"{rate:.0f} steps/s | ETA {eta/60:.1f}min | "
                    f"CV=[{cv[0]:.3f}, {cv[1]:.3f}] | "
                    f"kernels={opes.n_kernels} | saved={n_saved}",
                    flush=True,
                )

        opes.save_state(
            out / "opes_state_final.json",
            bias_cv="ligand_rmsd,ligand_pocket_dist",
            bias_cv_names=["ligand_rmsd", "ligand_pocket_dist"],
        )
    else:
        import openmm.unit as u

        force_group = int(plumed_context["force_group"])
        groups = {force_group}
        step = 0
        next_progress = int(args.progress_every)
        while step < args.n_steps:
            chunk = min(int(args.save_every), args.n_steps - step)
            sim.step(chunk)
            step += chunk

            state = sim.context.getState(
                getPositions=True,
                getEnergy=True,
                groups=groups,
            )
            coords = get_coords_boltz_order(state)
            cv = compute_cv(coords)
            bias_energy_kj = float(
                state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
            )
            logw = bias_energy_kj / kbt_kjmol

            n_saved += 1
            npz_path = wdsm_dir / f"wdsm_step_{n_saved:08d}.npz"
            np.savez_compressed(
                str(npz_path),
                coords=coords,
                cv=cv,
                logw=np.float64(logw),
                md_step=np.int64(step),
                bias_energy_kj_mol=np.float64(bias_energy_kj),
            )
            cv_log.append(
                {
                    "step": step,
                    "cv": cv.tolist(),
                    "logw": logw,
                    "bias_energy_kj_mol": bias_energy_kj,
                }
            )

            if step >= next_progress or step == args.n_steps:
                elapsed = time.time() - t0
                rate = step / elapsed
                eta = (args.n_steps - step) / rate
                print(
                    f"[MD-OPES] step {step:>10,}/{args.n_steps:,} | "
                    f"{rate:.0f} steps/s | ETA {eta/60:.1f}min | "
                    f"CV=[{cv[0]:.3f}, {cv[1]:.3f}] | "
                    f"bias={bias_energy_kj:.3f} kJ/mol | saved={n_saved}",
                    flush=True,
                )
                while next_progress <= step:
                    next_progress += int(args.progress_every)

    with open(out / "cv_log.json", "w") as f:
        json.dump(cv_log, f)

    elapsed = time.time() - t0
    summary = {
        "total_steps": args.n_steps,
        "total_time_s": elapsed,
        "n_saved": n_saved,
        "opes_mode": args.opes_mode,
        "n_kernels": opes.n_kernels if opes is not None else None,
        "temperature_k": args.temperature,
        "opes_barrier": opes.barrier if opes is not None else args.opes_barrier,
        "opes_biasfactor": (
            opes.biasfactor if opes is not None else args.opes_biasfactor
        ),
        "plumed_force_group": plumed_context.get("force_group"),
        "plumed_script": str(plumed_context.get("script_path"))
        if "script_path" in plumed_context
        else None,
        "plumed_reference_pdb": str(plumed_context.get("reference_pdb"))
        if "reference_pdb" in plumed_context
        else None,
    }
    with open(out / "md_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(
        f"  MD Complete: {args.n_steps:,} steps in {elapsed:.0f}s ({args.n_steps/elapsed:.0f} steps/s)"
    )
    print(f"  Saved {n_saved} structures to {wdsm_dir}")
    if opes is not None:
        print(f"  OPES: {opes.n_kernels} kernels, {opes.counter} depositions")
    else:
        print(f"  OPES: PLUMED force group {plumed_context.get('force_group')}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
