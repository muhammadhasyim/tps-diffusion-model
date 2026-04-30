#!/usr/bin/env python3
"""Run OPES-biased enhanced TPS on a Boltz-2 co-folding input.

Wraps the standard ``run_cofolding_tps_demo.py`` workflow with OPES (On-the-fly
Probability Enhanced Sampling) adaptive bias.  The bias is built on-the-fly
using kernel density estimation with compression, and enters the Metropolis
acceptance as:

    trial.bias *= exp(-(V(cv_new) - V(cv_old)) / kT)

The OPES state is saved periodically for restarts and for post-hoc reweighting.
During the run, ``cv_values.json`` is refreshed every ``--save-cv-json-every`` steps
(default: same as ``--save-opes-state-every``, or 1), and ``tps_steps.jsonl`` receives
one JSON line per MC step for crash-safe postprocessing.

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
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from openpathsampling.engines.trajectory import Trajectory

<<<<<<< Updated upstream
from genai_tps.utils.compute_device import (
    cuda_device_index_for_openmm,
    maybe_set_torch_cuda_current_device,
    parse_torch_device,
)

from genai_tps.backends.boltz.boltz2_trunk import boltz2_trunk_to_network_kwargs
=======
from genai_tps.backends.boltz.session import boltz_results_run_dir, build_boltz_session
from genai_tps.backends.boltz.tps_checkpoint import initial_trajectory, trajectory_checkpoint_callback
>>>>>>> Stashed changes
from genai_tps.backends.boltz.collective_variables import (
    PoseCVIndexer,
    ca_contact_count,
    clash_count,
    contact_order,
    end_to_end_distance,
    lddt_to_reference,
    ligand_pose_rmsd,
    ligand_pocket_distance,
    protein_ligand_contacts,
    protein_ligand_hbond_count,
    radius_of_gyration,
    ramachandran_outlier_fraction,
    rmsd_to_reference,
    shape_acylindricity,
    shape_kappa2,
)
from genai_tps.backends.boltz.engine import BoltzDiffusionEngine
from genai_tps.simulation.openmm_cv import OpenMMEnergy, OpenMMLocalMinRMSD
from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.cache_paths import default_boltz_cache_dir
from genai_tps.backends.boltz.snapshot import (
    BoltzSnapshot,
    boltz_snapshot_descriptor,
    snapshot_frame_numpy_copy,
)
from genai_tps.backends.boltz.tps_sampling import run_tps_path_sampling
from genai_tps.simulation import OPESBias
from genai_tps.evaluation.posebusters import (
    POSEBUSTERS_CV_PREFIX,
    POSEBUSTERS_GPU_CV_PREFIX,
    POSEBUSTERS_GPU_PASS_FRACTION,
    validate_posebusters_bias_cv_names,
    validate_posebusters_gpu_bias_cv_names,
)
from genai_tps.evaluation.tps_runner import (
    atomic_write_json,
    initial_trajectory,
    opes_state_checkpoint_callback,
    trajectory_checkpoint_callback,
)

# All supported single-CV names for --bias-cv
_SINGLE_CV_NAMES = [
    "rmsd",
    "rg",
    "openmm",
    "openmm_energy",
    "contact_order",
    "clash_count",
    "end_to_end",
    "ca_contact_count",
    "shape_kappa2",
    "shape_acylindricity",
    "lddt",
    "ramachandran_outlier",
    "ligand_rmsd",
    "ligand_pocket_dist",
    "ligand_contacts",
    "ligand_hbonds",
    "training_sucos_pocket_qcov",
    "posebusters_pass_fraction",
    POSEBUSTERS_GPU_PASS_FRACTION,
]


def _bias_cv_string_needs_topo_early(bias_cv_raw: str, list_posebusters_cvs: bool) -> bool:
    """Whether resolving ``--topo-npz`` must happen before parsing ``--bias-cv``."""
    if list_posebusters_cvs:
        return True
    low = bias_cv_raw.lower()
    for key in ("openmm", "ligand_", "training_sucos", "posebusters"):
        if key in low:
            return True
    return False


def _cv_names_require_topo_npz(cv_names: list[str]) -> bool:
    static = {
        "openmm",
        "openmm_energy",
        "ligand_rmsd",
        "ligand_pocket_dist",
        "ligand_contacts",
        "ligand_hbonds",
        "training_sucos_pocket_qcov",
        "posebusters_pass_fraction",
    }
    if static & set(cv_names):
        return True
    return any(n.startswith(POSEBUSTERS_CV_PREFIX) for n in cv_names)


def _make_cv_function(
    cv_type: str,
    reference_coords: torch.Tensor | None = None,
    atom_mask: torch.Tensor | None = None,
    topo_npz: Path | None = None,
    openmm_platform: str = "CUDA",
    openmm_max_iter: int = 500,
    openmm_device_index: int | None = None,
    mol_dir: Path | None = None,
    lddt_reference: torch.Tensor | None = None,
    pocket_radius: float = 6.0,
    contact_r0: float = 3.5,
    hbond_cutoff: float = 3.5,
    skrinjar_context: tuple | None = None,
    posebusters_context: dict | None = None,
) -> "Callable[[Trajectory], float]":
    """Build a scalar CV function (single-CV bias, returns float).

    Parameters
    ----------
    cv_type:
        One of: ``"rmsd"``, ``"rg"``, ``"openmm"``, ``"openmm_energy"``,
        ``"contact_order"``, ``"clash_count"``, ``"end_to_end"``,
        ``"ca_contact_count"``, ``"shape_kappa2"``, ``"shape_acylindricity"``, ``"lddt"``,
        ``"ramachandran_outlier"``, ``"ligand_rmsd"``,
        ``"ligand_pocket_dist"``, ``"ligand_contacts"``, ``"ligand_hbonds"``,
        ``"training_sucos_pocket_qcov"``, ``"posebusters_pass_fraction"``,
        ``"posebusters_gpu_pass_fraction"``.
    reference_coords:
        Ca coordinates of the initial structure, shape ``(N, 3)``.
        Required for ``"rmsd"``.
    lddt_reference:
        Reference coordinates for ``"lddt"``.  Falls back to
        ``reference_coords`` when not provided.
    mol_dir:
        Path to the Boltz CCD molecule directory (``~/.boltz/mols``).
        Required for ``"openmm"`` when ligands are present.
    pocket_radius:
        Radius (Å) around the initial ligand COM used to define the binding
        pocket for pose-quality CVs (default 6.0).
    contact_r0:
        Switching-function half-maximum distance (Å) for ``"ligand_contacts"``
        (default 3.5).
    hbond_cutoff:
        Heavy-atom N/O···N/O distance cutoff (Å) for ``"ligand_hbonds"``
        (default 3.5).
    """

    def _cv_rmsd(traj: Trajectory) -> float:
        snap = traj[-1]
        return rmsd_to_reference(snap, reference_coords, atom_mask)

    def _cv_rg(traj: Trajectory) -> float:
        snap = traj[-1]
        return radius_of_gyration(snap, atom_mask)

    def _cv_end_to_end(traj: Trajectory) -> float:
        return end_to_end_distance(traj[-1], atom_mask)

    def _cv_ca_contact_count(traj: Trajectory) -> float:
        return ca_contact_count(traj[-1], atom_mask=atom_mask)

    def _cv_shape_kappa2(traj: Trajectory) -> float:
        return shape_kappa2(traj[-1], atom_mask)

    def _cv_shape_acylindricity(traj: Trajectory) -> float:
        return shape_acylindricity(traj[-1], atom_mask)

    def _cv_contact_order(traj: Trajectory) -> float:
        return contact_order(traj[-1])

    def _cv_clash_count(traj: Trajectory) -> float:
        return float(clash_count(traj[-1]))

    def _cv_ramachandran_outlier(traj: Trajectory) -> float:
        return ramachandran_outlier_fraction(traj[-1])

    _lddt_ref = lddt_reference if lddt_reference is not None else reference_coords

    def _cv_lddt(traj: Trajectory) -> float:
        if _lddt_ref is None:
            raise RuntimeError(
                "--bias-cv lddt requires a reference structure. "
                "Provide --reference-pdb or ensure the initial trajectory "
                "snapshot has valid coordinates."
            )
        return lddt_to_reference(traj[-1], _lddt_ref)

    if cv_type == "rmsd":
        if reference_coords is None:
            raise ValueError("RMSD CV requires a reference structure.")
        return _cv_rmsd
    elif cv_type == "rg":
        return _cv_rg
    elif cv_type == "end_to_end":
        return _cv_end_to_end
    elif cv_type == "ca_contact_count":
        return _cv_ca_contact_count
    elif cv_type == "shape_kappa2":
        return _cv_shape_kappa2
    elif cv_type == "shape_acylindricity":
        return _cv_shape_acylindricity
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
            openmm_device_index=openmm_device_index,
        )
        return openmm_cv
    elif cv_type == "openmm_energy":
        if topo_npz is None:
            raise ValueError(
                "--bias-cv openmm_energy requires --topo-npz pointing to the "
                "Boltz processed/structures/*.npz file."
            )
        openmm_energy_cv = OpenMMEnergy(
            topo_npz=topo_npz,
            platform=openmm_platform,
            mol_dir=mol_dir,
            openmm_device_index=openmm_device_index,
        )
        return openmm_energy_cv
    elif cv_type == "contact_order":
        return _cv_contact_order
    elif cv_type == "clash_count":
        return _cv_clash_count
    elif cv_type == "ramachandran_outlier":
        return _cv_ramachandran_outlier
    elif cv_type == "lddt":
        return _cv_lddt
    elif cv_type in ("ligand_rmsd", "ligand_pocket_dist", "ligand_contacts", "ligand_hbonds"):
        _pose_cv_names = {
            "ligand_rmsd": "ligand_rmsd",
            "ligand_pocket_dist": "ligand_pocket_dist",
            "ligand_contacts": "ligand_contacts",
            "ligand_hbonds": "ligand_hbonds",
        }
        if topo_npz is None:
            raise ValueError(
                f"--bias-cv {cv_type} requires --topo-npz pointing to the Boltz "
                "processed/structures/*.npz file."
            )
        if reference_coords is None:
            raise ValueError(
                f"--bias-cv {cv_type} requires reference coordinates from the "
                "initial trajectory snapshot."
            )
        from genai_tps.io.boltz_npz_export import load_topo  # noqa: PLC0415

        structure, _ = load_topo(Path(topo_npz))
        ref_np = (
            reference_coords.detach().cpu().numpy()
            if isinstance(reference_coords, torch.Tensor)
            else np.asarray(reference_coords)
        )
        indexer = PoseCVIndexer(structure, ref_np, pocket_radius=pocket_radius)
        # Use plain callables on traj[-1].  OPS FunctionCV(traj) maps over all
        # frames and returns shape (n_frames,) — OPES needs a scalar per MC step.
        if cv_type == "ligand_rmsd":

            def _cv_ligand_rmsd(traj: Trajectory) -> float:
                return float(ligand_pose_rmsd(traj[-1], indexer))

            return _cv_ligand_rmsd
        if cv_type == "ligand_pocket_dist":

            def _cv_ligand_pocket(traj: Trajectory) -> float:
                return float(ligand_pocket_distance(traj[-1], indexer))

            return _cv_ligand_pocket
        if cv_type == "ligand_contacts":

            def _cv_ligand_contacts(traj: Trajectory) -> float:
                return float(
                    protein_ligand_contacts(traj[-1], indexer, r0=contact_r0)
                )

            return _cv_ligand_contacts

        def _cv_ligand_hbonds(traj: Trajectory) -> float:
            return float(
                protein_ligand_hbond_count(traj[-1], indexer, cutoff=hbond_cutoff)
            )

        return _cv_ligand_hbonds
    elif cv_type == "training_sucos_pocket_qcov":
        if skrinjar_context is None:
            raise ValueError(
                "--bias-cv training_sucos_pocket_qcov requires Skrinjar scorer "
                "context (internal error: skrinjar_context is None)."
            )
        from genai_tps.backends.boltz.collective_variables import (  # noqa: PLC0415
            make_training_sucos_pocket_qcov_traj_fn,
        )

        scorer, structure, n_struct = skrinjar_context
        return make_training_sucos_pocket_qcov_traj_fn(scorer, structure, int(n_struct))
    elif cv_type == "posebusters_pass_fraction":
        if posebusters_context is None or posebusters_context.get("evaluator") is None:
            raise ValueError(
                "--bias-cv posebusters_pass_fraction requires PoseBusters setup (internal error)."
            )
        from genai_tps.evaluation.posebusters import (  # noqa: PLC0415
            make_posebusters_pass_fraction_traj_fn,
        )

        return make_posebusters_pass_fraction_traj_fn(posebusters_context["evaluator"])
    elif cv_type == POSEBUSTERS_GPU_PASS_FRACTION:
        if posebusters_context is None or posebusters_context.get("gpu_evaluator") is None:
            raise ValueError(
                "--bias-cv posebusters_gpu_pass_fraction requires GPU PoseBusters setup (internal error)."
            )
        from genai_tps.evaluation.posebusters import (  # noqa: PLC0415
            make_posebusters_gpu_pass_fraction_traj_fn,
        )

        return make_posebusters_gpu_pass_fraction_traj_fn(posebusters_context["gpu_evaluator"])
    else:
        raise ValueError(
            f"Unknown CV type: {cv_type!r}. "
            f"Available: {_SINGLE_CV_NAMES}"
        )


def _parse_bias_cv_list(bias_cv_str: str) -> list[str]:
    """Parse comma-separated --bias-cv string into a list of CV names.

    Validates each name against the supported list and raises ``ValueError``
    for unknown names.
    """
    names = [n.strip() for n in bias_cv_str.split(",") if n.strip()]
    for n in names:
        if (
            n in _SINGLE_CV_NAMES
            or n.startswith(POSEBUSTERS_CV_PREFIX)
            or n.startswith(POSEBUSTERS_GPU_CV_PREFIX)
        ):
            continue
        raise ValueError(
            f"Unknown --bias-cv: {n!r}. "
            f"Supported base names: {_SINGLE_CV_NAMES}.  "
            "PoseBusters per-check columns use the posebusters__ prefix (see "
            "--list-posebusters-cvs). GPU-native geometry checks use the "
            "posebusters_gpu__ prefix.  "
            "Separate multiple CVs with commas, e.g. "
            "--bias-cv contact_order,clash_count"
        )
    return names


def _make_multi_cv_function(
    cv_names: list[str],
    reference_coords: torch.Tensor | None = None,
    atom_mask: torch.Tensor | None = None,
    topo_npz: Path | None = None,
    openmm_platform: str = "CUDA",
    openmm_max_iter: int = 500,
    openmm_device_index: int | None = None,
    mol_dir: Path | None = None,
    lddt_reference: torch.Tensor | None = None,
    pocket_radius: float = 6.0,
    contact_r0: float = 3.5,
    hbond_cutoff: float = 3.5,
    skrinjar_context: tuple | None = None,
    posebusters_context: dict | None = None,
) -> "tuple[Callable[[Trajectory], np.ndarray], int]":
    """Build a vector CV function for multi-dimensional OPES bias.

    Returns ``(cv_fn, ndim)`` where ``cv_fn(traj) -> np.ndarray`` of shape
    ``(ndim,)`` and ``ndim = len(cv_names)``.

    For a single CV name, the returned function still returns a scalar
    (``ndim=1``).
    """
    pb_cols = [n for n in cv_names if n.startswith(POSEBUSTERS_CV_PREFIX)]
    gpu_pb_cols = [n for n in cv_names if n.startswith(POSEBUSTERS_GPU_CV_PREFIX)]
    if pb_cols:
        if len(pb_cols) != len(cv_names):
            raise ValueError(
                "posebusters__* CVs cannot be mixed with other names in one --bias-cv list."
            )
        if posebusters_context is None or posebusters_context.get("ordered_columns") is None:
            raise ValueError("internal error: posebusters_context missing ordered_columns")
        from genai_tps.evaluation.posebusters import (  # noqa: PLC0415
            make_posebusters_cached_column_scalar_fns,
        )

        ev = posebusters_context["evaluator"]
        cols: list[str] = posebusters_context["ordered_columns"]
        if len(cols) != len(cv_names):
            raise ValueError(
                f"PoseBusters column count {len(cols)} does not match CV name count {len(cv_names)}."
            )
        scalar_fns = make_posebusters_cached_column_scalar_fns(ev, cols)
    elif gpu_pb_cols:
        if len(gpu_pb_cols) != len(cv_names):
            raise ValueError(
                "posebusters_gpu__* CVs cannot be mixed with other names in one --bias-cv list."
            )
        if posebusters_context is None or posebusters_context.get("gpu_ordered_columns") is None:
            raise ValueError("internal error: posebusters_context missing gpu_ordered_columns")
        from genai_tps.evaluation.posebusters import (  # noqa: PLC0415
            make_posebusters_gpu_cached_column_scalar_fns,
        )

        ev = posebusters_context["gpu_evaluator"]
        cols = posebusters_context["gpu_ordered_columns"]
        if len(cols) != len(cv_names):
            raise ValueError(
                f"GPU PoseBusters column count {len(cols)} does not match CV name count {len(cv_names)}."
            )
        scalar_fns = make_posebusters_gpu_cached_column_scalar_fns(ev, cols)
    else:
        scalar_fns = [
            _make_cv_function(
                name,
                reference_coords=reference_coords,
                atom_mask=atom_mask,
                topo_npz=topo_npz,
                openmm_platform=openmm_platform,
                openmm_max_iter=openmm_max_iter,
                openmm_device_index=openmm_device_index,
                mol_dir=mol_dir,
                lddt_reference=lddt_reference,
                pocket_radius=pocket_radius,
                contact_r0=contact_r0,
                hbond_cutoff=hbond_cutoff,
                skrinjar_context=skrinjar_context if name == "training_sucos_pocket_qcov" else None,
                posebusters_context=(
                    posebusters_context
                    if name in ("posebusters_pass_fraction", POSEBUSTERS_GPU_PASS_FRACTION)
                    else None
                ),
            )
            for name in cv_names
        ]
    ndim = len(scalar_fns)

    if ndim == 1:
        # Scalar path: return float directly for backward compat with 1-D OPES
        fn0 = scalar_fns[0]

        def _scalar_cv(traj: Trajectory) -> float:
            return fn0(traj)

        return _scalar_cv, 1  # type: ignore[return-value]

    def _vector_cv(traj: Trajectory) -> np.ndarray:
        return np.array([fn(traj) for fn in scalar_fns], dtype=np.float64)

    return _vector_cv, ndim


<<<<<<< Updated upstream
=======
def _opes_state_checkpoint_callback(
    bias: OPESBias,
    work_root: Path,
    every: int,
    bias_cv: str,
    bias_cv_names: list[str],
) -> Callable[[int, Trajectory], None]:
    """Periodic OPES state checkpoints for restart and analysis."""
    def cb(mc_step: int, _traj: Trajectory) -> None:
        if every <= 0 or mc_step % every != 0:
            return
        state_dir = work_root / "opes_states"
        state_dir.mkdir(parents=True, exist_ok=True)
        tagged = state_dir / f"opes_state_{mc_step:08d}.json"
        bias.save_state(
            tagged, bias_cv=bias_cv, bias_cv_names=bias_cv_names,
        )
        latest = state_dir / "opes_state_latest.json"
        shutil.copyfile(tagged, latest)
        print(
            f"[TPS-OPES] OPES state checkpoint MC step {mc_step} "
            f"({bias.n_kernels} kernels, {bias.counter} depositions)",
            file=sys.stderr, flush=True,
        )
    return cb

>>>>>>> Stashed changes
def _build_diagnostic_cv_functions(
    names_csv: str | None,
) -> dict[str, "Callable[[Trajectory], float]"] | None:
    """Map comma-separated diagnostic CV names to callables on ``Trajectory``.

    Diagnostic CVs are logged only; they do not enter the OPES bias. Unknown
    names are skipped (warning to stderr). Returns ``None`` if ``names_csv``
    is empty or only whitespace.

    Recognized names match the keys in the internal ``_registry`` (see also
    ``--diagnostic-cvs`` in ``main()``).
    """
    if not names_csv:
        return None
    from genai_tps.backends.boltz.collective_variables import (  # noqa: PLC0415
        ca_contact_count,
        contact_order,
        clash_count,
        end_to_end_distance,
        ramachandran_outlier_fraction,
        radius_of_gyration,
        shape_acylindricity,
        shape_kappa2,
    )
    _registry: dict[str, "Callable"] = {
        "contact_order": lambda traj: contact_order(traj[-1]),
        "clash_count": lambda traj: clash_count(traj[-1]),
        "end_to_end": lambda traj: end_to_end_distance(traj[-1]),
        "ca_contact_count": lambda traj: ca_contact_count(traj[-1]),
        "shape_kappa2": lambda traj: shape_kappa2(traj[-1]),
        "shape_acylindricity": lambda traj: shape_acylindricity(traj[-1]),
        "ramachandran_outlier": lambda traj: ramachandran_outlier_fraction(traj[-1]),
        "rg": lambda traj: radius_of_gyration(traj[-1]),
    }
    result: dict[str, "Callable"] = {}
    for name in [n.strip() for n in names_csv.split(",") if n.strip()]:
        if name in _registry:
            result[name] = _registry[name]
        else:
            print(
                f"[TPS-OPES] WARNING: unknown diagnostic CV '{name}'; skipping. "
                f"Available: {sorted(_registry)}",
                file=sys.stderr, flush=True,
            )
    return result if result else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OPES-biased enhanced TPS on Boltz-2 co-folding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--yaml", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("opes_tps_out"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument(
        "--finetuned-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional WDSM fine-tuned Boltz-2 model state_dict. "
            "When provided, TPS runs against this model instead of the baseline checkpoint."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device: cpu, cuda, or cuda:N (default: cuda).",
    )
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--shoot-rounds", type=int, default=2000)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--save-trajectory-every", type=int, default=10)
    parser.add_argument("--log-path-prob-every", type=int, default=0)
    parser.add_argument("--forward-only", action="store_true", default=False)
    parser.add_argument("--reshuffle-probability", type=float, default=0.1)
    parser.add_argument("--kernels", action="store_true")
    parser.add_argument(
        "--use-msa-server",
        action="store_true",
        help=(
            "Boltz preprocessing: query the ColabFold/MMseqs2 MSA server "
            "(https://api.colabfold.com) for protein chains that use auto MSA. "
            "Your YAML must *not* set msa: empty on those proteins (omit the key "
            "or use msa: path/to/file.a3m). With msa: empty, Boltz stays in "
            "single-sequence mode regardless of this flag."
        ),
    )
    parser.add_argument(
        "--compile-model", action="store_true", default=False,
        help=(
            "Wrap preconditioned_network_forward with torch.compile(mode='reduce-overhead'). "
            "First call incurs ~30s warmup; subsequent calls are 2-3x faster."
        ),
    )
    parser.add_argument(
        "--n-fixed-point", type=int, default=4,
        help=(
            "Fixed-point iterations for the backward-step implicit inversion "
            "(default: 4). Reducing to 2 cuts backward-step cost by ~50%%; "
            "the Hastings ratio absorbs the approximation error."
        ),
    )
    parser.add_argument(
        "--inference-dtype", type=str, default=None,
        choices=["float32", "bfloat16"],
        help=(
            "Dtype for score-network forward passes. 'bfloat16' halves bandwidth "
            "and uses tensor cores on Ampere/Ada GPUs (~1.5-2x on diffusion cost). "
            "Only applied on CUDA. Default: None (float32, no autocast)."
        ),
    )
    parser.add_argument(
        "--diagnostic-cvs", type=str, default=None,
        help=(
            "Comma-separated list of diagnostic CV names to log per MC step "
            "(observation-only; do not affect OPES bias). "
            "Available: contact_order, clash_count, end_to_end, ca_contact_count, "
            "shape_kappa2, shape_acylindricity, ramachandran_outlier, rg. "
            "Example: --diagnostic-cvs contact_order,clash_count"
        ),
    )

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
        "--opes-sigma-min", type=float, default=0.005,
        help=(
            "Minimum kernel width (default: 0.005). Prevents sigma -> 0 when "
            "consecutive accepted paths have identical CV values, which causes the "
            "sigma_0/sigma height amplification to diverge and produce a density "
            "spike of ~1e6. Set to 0.0 to disable the floor."
        ),
    )
    opes_group.add_argument(
        "--opes-fixed-sigma", type=str, default=None,
        help=(
            "Fixed kernel width per dimension (overrides adaptive estimation). "
            "Provide a single float for all dimensions, or a comma-separated list "
            "matching the number of --bias-cv CVs. E.g. '0.1' or '0.1,0.05'. "
            "Default: adaptive."
        ),
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
        "--bias-cv", type=str, default="openmm",
        help=(
            "Collective variable(s) for the OPES bias.  "
            "Single CV: one of {rmsd, rg, openmm, openmm_energy, contact_order, "
            "clash_count, end_to_end, ca_contact_count, shape_kappa2, shape_acylindricity, "
            "lddt, ramachandran_outlier, ligand_rmsd, ligand_pocket_dist, "
            "ligand_contacts, ligand_hbonds}.  "
            "Multi-CV (multi-D OPES): comma-separated list, e.g. "
            "'contact_order,clash_count'.  "
            "'openmm' (default): AMBER14/GBn2 Cα-RMSD to local energy minimum; "
            "requires --topo-npz.  "
            "'openmm_energy': raw AMBER14/GBn2 single-point energy (kJ/mol); "
            "requires --topo-npz.  "
            "'ligand_rmsd': ligand pose RMSD after Kabsch alignment on protein Cα "
            "(BPMD/PLUMED TYPE=OPTIMAL convention); requires --topo-npz.  "
            "'ligand_pocket_dist': COM–COM distance between ligand and pocket Cα; "
            "requires --topo-npz.  "
            "'ligand_contacts': PLUMED COORDINATION-style contact sum (r0=--contact-r0); "
            "requires --topo-npz.  "
            "'ligand_hbonds': N/O···N/O distance proxy for H-bonds (cutoff=--hbond-cutoff); "
            "requires --topo-npz.  "
            "'lddt': requires --reference-pdb or uses initial-trajectory coordinates.  "
            "'training_sucos_pocket_qcov': incremental **proxy** — SuCOS-shape after RDKit "
            "alignment × **geometric** pocket Cα overlap (not PLINDER pocket_qcov); "
            "training ligands are not rotated by Foldseek (u,t) like runs-n-poses "
            "holo scoring (CPU RDKit + optional Foldseek GPU prefilter on complexes); "
            "requires --topo-npz and --skrinjar-training-complexes-dir and/or "
            "--skrinjar-training-ligands-dir (see docs/runs_n_poses_reproduction.md).  "
            "'posebusters_pass_fraction': mean PoseBusters boolean pass rate in [0,1] "
            "(pip install posebusters; see docs/posebusters_opes_cv.md).  "
            "'posebusters_all': expands to all posebusters__* checks (sole token; multi-D OPES).  "
            "'posebusters_gpu_pass_fraction': mean pass rate over the local tensor-native "
            "geometry subset.  "
            "'posebusters_gpu_all': expands to all posebusters_gpu__* checks "
            "(sole token; multi-D OPES).  "
            "Use --list-posebusters-cvs to print the expanded names for your selected backend."
        ),
    )
    opes_group.add_argument(
        "--reference-pdb", type=Path, default=None,
        help=(
            "Path to a reference PDB/coordinate file whose Cα coordinates are "
            "used as the reference for --bias-cv lddt (and optionally --bias-cv rmsd). "
            "Default: initial trajectory last frame."
        ),
    )
    opes_group.add_argument(
        "--topo-npz", type=Path, default=None,
        help=(
            "Path to Boltz processed/structures/*.npz for PDB conversion. "
            "Required when --bias-cv is openmm or openmm_energy."
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
        "--openmm-device-index",
        type=int,
        default=None,
        metavar="N",
        help=(
            "CUDA/OpenCL ordinal for OpenMM bias CVs (DeviceIndex). "
            "Default: same GPU index as --device for cuda:N (or 0 for bare cuda)."
        ),
    )
    opes_group.add_argument(
        "--pocket-radius", type=float, default=6.0,
        help=(
            "Radius (Å) around the initial ligand COM used to define the binding pocket "
            "for pose-quality CVs: ligand_rmsd, ligand_pocket_dist, ligand_contacts, "
            "ligand_hbonds (default: 6.0)."
        ),
    )
    opes_group.add_argument(
        "--contact-r0", type=float, default=3.5,
        help=(
            "Switching-function half-maximum distance (Å) for the ligand_contacts CV. "
            "Corresponds to PLUMED COORDINATION R_0 (default: 3.5)."
        ),
    )
    opes_group.add_argument(
        "--hbond-cutoff", type=float, default=3.5,
        help=(
            "Heavy-atom N/O···N/O distance cutoff (Å) for the ligand_hbonds CV "
            "(standard H-bond donor-acceptor threshold without explicit H; default: 3.5)."
        ),
    )
    skr = parser.add_argument_group(
        "Škrinjar training similarity CV (training_sucos_pocket_qcov)"
    )
    skr.add_argument(
        "--skrinjar-foldseek-db",
        type=Path,
        default=None,
        help="Optional Foldseek target database path for GPU prefilter of training complexes.",
    )
    skr.add_argument(
        "--skrinjar-training-complexes-dir",
        type=Path,
        default=None,
        help="Directory of training holo PDB files (optional; used with Foldseek hits or glob).",
    )
    skr.add_argument(
        "--skrinjar-training-ligands-dir",
        type=Path,
        default=None,
        help="Directory of training ligand .sdf files for ligand-only SuCOS (no pocket_qcov).",
    )
    skr.add_argument(
        "--skrinjar-foldseek-bin",
        type=str,
        default=None,
        help="Foldseek executable (default: FOLDSEEK_BIN env or 'foldseek').",
    )
    skr.add_argument(
        "--skrinjar-top-k", type=int, default=5,
        help="Max training complexes to evaluate after Foldseek prefilter (default: 5).",
    )
    skr.add_argument(
        "--skrinjar-pocket-radius", type=float, default=6.0,
        help="Pocket definition radius (Å) around ligand heavy atoms (default: 6).",
    )
    skr.add_argument(
        "--skrinjar-pocket-qcov-cutoff", type=float, default=2.0,
        help="Cα match cutoff (Å) for pocket_qcov (default: 2).",
    )
    skr.add_argument(
        "--skrinjar-no-gpu",
        action="store_true",
        help="Do not pass --gpu 1 to Foldseek easy-search.",
    )
    skr.add_argument(
        "--skrinjar-no-cache",
        action="store_true",
        help="Disable coordinate-hash cache for the Skrinjar scorer (recompute every MC step).",
    )
    pb = parser.add_argument_group(
        "PoseBusters CV (exact CPU PoseBusters and tensor-first GPU geometric subset)"
    )
    pb.add_argument(
        "--list-posebusters-cvs",
        action="store_true",
        help=(
            "After the initial trajectory, print comma-separated PoseBusters CV names "
            "for the selected backend and current reference settings, then exit."
        ),
    )
    pb.add_argument(
        "--posebusters-backend",
        type=str,
        default="cpu_posebusters",
        choices=["cpu_posebusters", "gpu_fast", "hybrid"],
        help=(
            "Backend for PoseBusters-backed CVs. "
            "'cpu_posebusters' preserves exact upstream PoseBusters via RDKit/files. "
            "'gpu_fast' runs the local tensor-native geometry subset only. "
            "'hybrid' uses the GPU subset for OPES CV values and also runs exact CPU "
            "PoseBusters every --posebusters-cpu-fallback-every evaluations for parity diagnostics."
        ),
    )
    pb.add_argument(
        "--posebusters-mode",
        type=str,
        default="redock_fast",
        choices=[
            "dock",
            "redock",
            "mol",
            "gen",
            "regen",
            "dock_fast",
            "redock_fast",
            "mol_fast",
            "gen_fast",
            "regen_fast",
        ],
        help="PoseBusters YAML config preset (default: redock_fast for cheaper OPES steps).",
    )
    pb.add_argument(
        "--posebusters-reference-ligand-sdf",
        type=Path,
        default=None,
        help="Reference ligand SDF for redock* modes (optional if --posebusters-use-initial-ligand-ref).",
    )
    pb.add_argument(
        "--posebusters-use-initial-ligand-ref",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For redock*: build reference ligand from the initial trajectory frame (default: on). "
            "Disable if you supply --posebusters-reference-ligand-sdf only."
        ),
    )
    pb.add_argument(
        "--posebusters-ligand-chain",
        type=str,
        default=None,
        help="NONPOLYMER chain id for the ligand (default: first nonpolymer chain).",
    )
    pb.add_argument(
        "--posebusters-cpu-fallback-every",
        type=int,
        default=0,
        help=(
            "In --posebusters-backend hybrid mode, run exact CPU PoseBusters every N "
            "evaluations as a parity probe. 0 disables CPU fallback work (default: 0)."
        ),
    )
    wdsm_group = parser.add_argument_group("WDSM data collection")
    wdsm_group.add_argument(
        "--save-wdsm-data-every", type=int, default=0,
        help=(
            "Save terminal-frame coords + OPES logw every N MC steps for weighted "
            "DSM training. 0 disables (default: 0). Each step produces a small NPZ "
            "in --wdsm-data-dir with guaranteed coord/CV/logw consistency."
        ),
    )
    wdsm_group.add_argument(
        "--wdsm-data-dir", type=Path, default=None,
        help="Directory for per-step WDSM NPZ files (default: {out}/wdsm_samples/).",
    )

    opes_group.add_argument(
        "--save-opes-state-every", type=int, default=100,
        help="Save OPES state every N MC steps. (default: 100)",
    )
    opes_group.add_argument(
        "--save-cv-json-every",
        type=int,
        default=None,
        help=(
            "Write cv_values.json every N MC steps (default: same as "
            "--save-opes-state-every if >0, else 1).  Uses the CV from the "
            "TPS step log (no duplicate OpenMM evaluations)."
        ),
    )

    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    yaml_path = args.yaml or (repo / "examples" / "cofolding_multimer_msa_empty.yaml")
    if not yaml_path.is_file():
        print(f"Input YAML not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    try:
        import boltz  # noqa: F401
    except ImportError as e:
        print(f"Boltz is required: pip install -e ./boltz\n{e}", file=sys.stderr)
        sys.exit(1)

<<<<<<< Updated upstream
    cache = Path(args.cache).expanduser() if args.cache else default_boltz_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    mol_dir = cache / "mols"
    download_boltz2(cache)

    if torch.cuda.is_available():
        device = parse_torch_device(args.device)
        maybe_set_torch_cuda_current_device(device)
    else:
        device = torch.device("cpu")
    openmm_gpu_index = (
        args.openmm_device_index
        if args.openmm_device_index is not None
        else cuda_device_index_for_openmm(device)
    )
    work_root = args.out.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    boltz_run_dir = work_root / f"boltz_results_{yaml_path.stem}"
    boltz_run_dir.mkdir(parents=True, exist_ok=True)

    data_list = check_inputs(yaml_path)
    process_inputs(
        data=data_list, out_dir=boltz_run_dir,
        ccd_path=cache / "ccd.pkl", mol_dir=mol_dir,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        use_msa_server=bool(args.use_msa_server),
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
    if args.finetuned_checkpoint is not None:
        finetuned_checkpoint = args.finetuned_checkpoint.expanduser().resolve()
        if not finetuned_checkpoint.is_file():
            print(f"Fine-tuned checkpoint not found: {finetuned_checkpoint}", file=sys.stderr)
            sys.exit(1)
        state = torch.load(str(finetuned_checkpoint), map_location="cpu")
        model.load_state_dict(state)
        print(f"[TPS-OPES] Loaded fine-tuned checkpoint: {finetuned_checkpoint}", flush=True)
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
=======
    cache = Path(args.cache).expanduser() if args.cache else Path.home() / ".boltz"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    work_root = args.out.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    boltz_run_dir = boltz_results_run_dir(work_root, yaml_path.stem)
>>>>>>> Stashed changes
    _dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    inference_dtype = _dtype_map.get(args.inference_dtype) if args.inference_dtype else None
    try:
        model, core, batch, processed_dir, _, boltz_run_dir, _ = build_boltz_session(
            yaml_path=yaml_path,
            cache=cache,
            boltz_run_dir=boltz_run_dir,
            device=device,
            diffusion_steps=args.diffusion_steps,
            recycling_steps=args.recycling_steps,
            kernels=args.kernels,
            use_msa_server=bool(args.use_msa_server),
            model_eval_mode=True,
            sampler_core_extra_kwargs={
                "compile_model": args.compile_model,
                "n_fixed_point": args.n_fixed_point,
                "inference_dtype": inference_dtype,
            },
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    n_atoms = int(core.atom_mask.shape[1])

    torch.manual_seed(0)
    init_traj = initial_trajectory(core)

    last_snap = init_traj[-1]
    ref_coords = None
    if isinstance(last_snap, BoltzSnapshot) and last_snap.tensor_coords is not None:
        ref_coords = last_snap.tensor_coords[0].clone()

    # Optional external reference PDB for lddt (and rmsd override).
    lddt_reference = ref_coords  # default: initial trajectory
    if args.reference_pdb is not None:
        try:
            import mdtraj as md  # noqa: PLC0415
            _traj = md.load(str(args.reference_pdb))
            ca_sel = _traj.top.select("name CA")
            lddt_reference = torch.tensor(
                _traj.xyz[0][ca_sel] * 10.0,  # nm -> Å
                dtype=torch.float32,
            )
            print(
                f"[TPS-OPES] Loaded reference PDB {args.reference_pdb} "
                f"({ca_sel.shape[0]} Cα atoms)",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[TPS-OPES] WARNING: could not load --reference-pdb {args.reference_pdb}: {exc}; "
                "falling back to initial-trajectory coordinates.",
                file=sys.stderr, flush=True,
            )

    # Resolve --topo-npz early when PoseBusters listing/expansion or other CVs need it.
    topo_npz = Path(args.topo_npz).expanduser().resolve() if args.topo_npz else None
    bias_raw = args.bias_cv.strip()
    if topo_npz is None and _bias_cv_string_needs_topo_early(bias_raw, args.list_posebusters_cvs):
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
                "[TPS-OPES] ERROR: --bias-cv / PoseBusters listing requires "
                "--topo-npz or processed/structures/*.npz under the Boltz run.",
                file=sys.stderr, flush=True,
            )
            sys.exit(1)

    posebusters_context: dict | None = None
    bias_cv_work = bias_raw
    if args.list_posebusters_cvs or "posebusters" in bias_raw.lower():
        if topo_npz is None:
            print(
                "[TPS-OPES] ERROR: PoseBusters requires --topo-npz (or auto-detected structures npz).",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)
        from genai_tps.io.boltz_npz_export import load_topo  # noqa: PLC0415
        from genai_tps.evaluation.posebusters import (  # noqa: PLC0415
            GPUPoseBustersEvaluator,
            expand_bias_cv_posebusters_gpu_all,
            expand_posebusters_gpu_all_to_cv_names,
        )

        structure, n_struct = load_topo(Path(topo_npz))
        probe = snapshot_frame_numpy_copy(init_traj[-1])[: int(n_struct)].astype(np.float32)
        scratch_pb = work_root / "posebusters_scratch"
        ref_sdf = (
            Path(args.posebusters_reference_ligand_sdf).expanduser().resolve()
            if args.posebusters_reference_ligand_sdf
            else None
        )
        init_ref = probe if args.posebusters_use_initial_ligand_ref else None
        if str(args.posebusters_mode).startswith("redock") and ref_sdf is None and init_ref is None:
            print(
                "[TPS-OPES] ERROR: redock* PoseBusters modes need "
                "--posebusters-reference-ligand-sdf and/or --posebusters-use-initial-ligand-ref.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)
        pb_kw: dict = {
            "mode": str(args.posebusters_mode),
            "reference_ligand_sdf": ref_sdf,
            "initial_coords_for_reference_ligand": init_ref,
            "ligand_chain": args.posebusters_ligand_chain,
        }
        raw_lower = bias_raw.lower()
        uses_gpu_posebusters = (
            "posebusters_gpu" in raw_lower or str(args.posebusters_backend) in ("gpu_fast", "hybrid")
        )
        uses_cpu_posebusters = (
            "posebusters_pass_fraction" in raw_lower
            or "posebusters_all" in raw_lower
            or POSEBUSTERS_CV_PREFIX in raw_lower
            or str(args.posebusters_backend) == "cpu_posebusters"
        ) and "posebusters_gpu" not in raw_lower

        if uses_gpu_posebusters and uses_cpu_posebusters and not args.list_posebusters_cvs:
            print(
                "[TPS-OPES] ERROR: CPU PoseBusters and GPU PoseBusters CV namespaces cannot be mixed.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)

        if uses_gpu_posebusters:
            expanded, raw_cols, cv_names_from_all = expand_bias_cv_posebusters_gpu_all(bias_raw)
            if args.list_posebusters_cvs:
                names, _ = expand_posebusters_gpu_all_to_cv_names()
                print(",".join(names), flush=True)
                sys.exit(0)
            gpu_ev = GPUPoseBustersEvaluator(
                structure,
                int(n_struct),
                probe,
                pocket_radius=float(args.pocket_radius),
                contact_r0=float(args.contact_r0),
                hbond_cutoff=float(args.hbond_cutoff),
                ligand_chain=args.posebusters_ligand_chain,
                backend_mode=str(args.posebusters_backend),
                cpu_evaluator=None,
                cpu_fallback_every=int(args.posebusters_cpu_fallback_every),
            )
            if str(args.posebusters_backend) == "hybrid":
                try:
                    import posebusters  # noqa: F401, PLC0415
                    from genai_tps.evaluation.posebusters import PoseBustersTrajEvaluator  # noqa: PLC0415

                    gpu_ev.cpu_evaluator = PoseBustersTrajEvaluator(
                        structure,
                        int(n_struct),
                        scratch_pb / "hybrid_cpu",
                        mode=str(args.posebusters_mode),
                        reference_ligand_sdf=ref_sdf,
                        initial_coords_for_reference_ligand=init_ref,
                        ligand_chain=args.posebusters_ligand_chain,
                    )
                except ImportError:
                    print(
                        "[TPS-OPES] WARNING: hybrid GPU PoseBusters requested but upstream "
                        "posebusters is unavailable; continuing without CPU fallback.",
                        file=sys.stderr,
                        flush=True,
                    )
            if raw_cols is not None and cv_names_from_all is not None:
                bias_cv_work = expanded
                posebusters_context = {"gpu_evaluator": gpu_ev, "gpu_ordered_columns": raw_cols}
            elif POSEBUSTERS_GPU_PASS_FRACTION in bias_raw:
                bias_cv_work = bias_raw
                posebusters_context = {"gpu_evaluator": gpu_ev}
            else:
                bias_cv_work = bias_raw
                names_chk, raw_chk = expand_posebusters_gpu_all_to_cv_names()
                parts_chk = [p.strip() for p in bias_cv_work.split(",") if p.strip()]
                if parts_chk != names_chk:
                    print(
                        "[TPS-OPES] ERROR: --bias-cv posebusters_gpu__* list does not match "
                        "the supported GPU check set. Use --bias-cv posebusters_gpu_all "
                        "or run with --list-posebusters-cvs.",
                        file=sys.stderr,
                        flush=True,
                    )
                    sys.exit(1)
                posebusters_context = {"gpu_evaluator": gpu_ev, "gpu_ordered_columns": raw_chk}
        else:
            try:
                import posebusters  # noqa: F401, PLC0415
            except ImportError:
                print(
                    "[TPS-OPES] ERROR: PoseBusters CVs require `pip install posebusters` "
                    "(see environment.yml).",
                    file=sys.stderr,
                    flush=True,
                )
                sys.exit(1)
            from genai_tps.evaluation.posebusters import (  # noqa: PLC0415
                PoseBustersTrajEvaluator,
                expand_bias_cv_posebusters_all,
                expand_posebusters_all_to_cv_names,
            )

            expanded, raw_cols, cv_names_from_all = expand_bias_cv_posebusters_all(
                bias_raw,
                structure,
                int(n_struct),
                scratch_pb,
                probe,
                **pb_kw,
            )
            if args.list_posebusters_cvs:
                names, _ = expand_posebusters_all_to_cv_names(
                    structure, int(n_struct), scratch_pb / "list_probe", probe, **pb_kw
                )
                print(",".join(names), flush=True)
                sys.exit(0)
            if raw_cols is not None and cv_names_from_all is not None:
                bias_cv_work = expanded
                ev = PoseBustersTrajEvaluator(
                    structure,
                    int(n_struct),
                    scratch_pb,
                    mode=str(args.posebusters_mode),
                    reference_ligand_sdf=ref_sdf,
                    initial_coords_for_reference_ligand=init_ref,
                    ligand_chain=args.posebusters_ligand_chain,
                )
                posebusters_context = {"evaluator": ev, "ordered_columns": raw_cols}
            elif "posebusters_pass_fraction" in bias_raw:
                bias_cv_work = bias_raw
                ev = PoseBustersTrajEvaluator(
                    structure,
                    int(n_struct),
                    scratch_pb,
                    mode=str(args.posebusters_mode),
                    reference_ligand_sdf=ref_sdf,
                    initial_coords_for_reference_ligand=init_ref,
                    ligand_chain=args.posebusters_ligand_chain,
                )
                posebusters_context = {"evaluator": ev}
            else:
                bias_cv_work = bias_raw
                names_chk, raw_chk = expand_posebusters_all_to_cv_names(
                    structure, int(n_struct), scratch_pb / "verify", probe, **pb_kw
                )
                parts_chk = [p.strip() for p in bias_cv_work.split(",") if p.strip()]
                if parts_chk != names_chk:
                    print(
                        "[TPS-OPES] ERROR: --bias-cv posebusters__* list does not match "
                        "current --posebusters-mode / reference settings. "
                        "Use --bias-cv posebusters_all or run with --list-posebusters-cvs.",
                        file=sys.stderr,
                        flush=True,
                    )
                    sys.exit(1)
                ev = PoseBustersTrajEvaluator(
                    structure,
                    int(n_struct),
                    scratch_pb,
                    mode=str(args.posebusters_mode),
                    reference_ligand_sdf=ref_sdf,
                    initial_coords_for_reference_ligand=init_ref,
                    ligand_chain=args.posebusters_ligand_chain,
                )
                posebusters_context = {"evaluator": ev, "ordered_columns": raw_chk}

    # Parse --bias-cv (single name or comma-separated list for multi-D).
    try:
        bias_cv_names = _parse_bias_cv_list(bias_cv_work)
        validate_posebusters_bias_cv_names(bias_cv_names)
        validate_posebusters_gpu_bias_cv_names(bias_cv_names)
    except ValueError as exc:
        print(f"[TPS-OPES] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    ndim = len(bias_cv_names)

    # Auto-detect topo_npz if still unset (e.g. only openmm in bias).
    if topo_npz is None and _cv_names_require_topo_npz(bias_cv_names):
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
                "[TPS-OPES] ERROR: --bias-cv requires "
                "--topo-npz but no processed/structures/*.npz found.",
                file=sys.stderr, flush=True,
            )
            sys.exit(1)

    skrinjar_context: tuple | None = None
    if "training_sucos_pocket_qcov" in bias_cv_names:
        from genai_tps.io.boltz_npz_export import load_topo  # noqa: PLC0415
        from genai_tps.evaluation.skrinjar_similarity import IncrementalSkrinjarScorer  # noqa: PLC0415

        if topo_npz is None:
            print(
                "[TPS-OPES] ERROR: --bias-cv training_sucos_pocket_qcov requires --topo-npz.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)
        tdir = args.skrinjar_training_complexes_dir
        ldir = args.skrinjar_training_ligands_dir
        if tdir is None and ldir is None:
            print(
                "[TPS-OPES] ERROR: training_sucos_pocket_qcov needs at least one of "
                "--skrinjar-training-complexes-dir or --skrinjar-training-ligands-dir.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)
        structure, n_struct = load_topo(Path(topo_npz))
        sdfs: tuple[Path, ...] = ()
        if ldir is not None:
            lp = Path(ldir).expanduser().resolve()
            sdfs = tuple(sorted(p for p in lp.glob("*.sdf") if p.is_file()))
        fdb = Path(args.skrinjar_foldseek_db).resolve() if args.skrinjar_foldseek_db else None
        cdir = Path(tdir).expanduser().resolve() if tdir is not None else None
        scorer = IncrementalSkrinjarScorer(
            foldseek_bin=args.skrinjar_foldseek_bin,
            foldseek_db=fdb,
            training_complexes_dir=cdir,
            training_ligand_sdfs=sdfs,
            use_foldseek_gpu=not args.skrinjar_no_gpu,
            top_k=int(args.skrinjar_top_k),
            pocket_radius=float(args.skrinjar_pocket_radius),
            pocket_qcov_cutoff=float(args.skrinjar_pocket_qcov_cutoff),
            enable_coord_cache=not args.skrinjar_no_cache,
            scratch_dir=work_root / "skrinjar_cv_scratch",
        )
        skrinjar_context = (scorer, structure, n_struct)
        print(
            f"[TPS-OPES] Skrinjar scorer: complexes_dir={cdir!s} n_sdfs={len(sdfs)} "
            f"foldseek_db={fdb!s}",
            flush=True,
        )

    cv_function, ndim = _make_multi_cv_function(
        bias_cv_names,
        reference_coords=ref_coords,
        topo_npz=topo_npz,
        openmm_platform=args.openmm_platform,
        openmm_max_iter=args.openmm_max_iter,
        openmm_device_index=openmm_gpu_index,
        mol_dir=mol_dir,
        lddt_reference=lddt_reference,
        pocket_radius=args.pocket_radius,
        contact_r0=args.contact_r0,
        hbond_cutoff=args.hbond_cutoff,
        skrinjar_context=skrinjar_context,
        posebusters_context=posebusters_context,
    )
    print(
        f"[TPS-OPES] Bias CV(s): {bias_cv_names}  ndim={ndim}",
        flush=True,
    )

    if args.opes_restart is not None:
        print(f"[TPS-OPES] Restarting from {args.opes_restart}", flush=True)
        bias = OPESBias.load_state(args.opes_restart)
    else:
        biasfactor = float("inf") if args.opes_biasfactor <= 0 else args.opes_biasfactor

        # Parse --opes-fixed-sigma: may be scalar string or comma-separated list
        fixed_sigma: np.ndarray | None = None
        if args.opes_fixed_sigma is not None:
            fs_parts = [float(x.strip()) for x in str(args.opes_fixed_sigma).split(",") if x.strip()]
            if len(fs_parts) == 1:
                fixed_sigma = np.full(ndim, fs_parts[0])
            elif len(fs_parts) == ndim:
                fixed_sigma = np.array(fs_parts, dtype=np.float64)
            else:
                print(
                    f"[TPS-OPES] ERROR: --opes-fixed-sigma has {len(fs_parts)} value(s) "
                    f"but --bias-cv has {ndim} dimension(s).",
                    file=sys.stderr,
                )
                sys.exit(1)

        bias = OPESBias(
            ndim=ndim,
            kbt=args.opes_kbt,
            barrier=args.opes_barrier,
            biasfactor=biasfactor,
            epsilon=args.opes_epsilon,
            kernel_cutoff=None,
            compression_threshold=args.opes_compression_threshold,
            pace=args.opes_pace,
            sigma_min=args.opes_sigma_min,
            fixed_sigma=fixed_sigma,
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
<<<<<<< Updated upstream
        periodic_extra.append((trajectory_checkpoint_callback(work_root), save_traj_every))
=======
        periodic_extra.append(
            (trajectory_checkpoint_callback(work_root, bracket_tag="[TPS-OPES]"), save_traj_every)
        )
>>>>>>> Stashed changes

    save_opes_every = max(0, int(args.save_opes_state_every))
    if save_opes_every > 0:
        periodic_extra.append(
            (
                opes_state_checkpoint_callback(
                    bias, work_root, save_opes_every,
                    args.bias_cv, bias_cv_names,
                ),
                save_opes_every,
            )
        )

    if args.save_cv_json_every is not None:
        save_cv_every = max(1, int(args.save_cv_json_every))
    else:
        save_cv_every = save_opes_every if save_opes_every > 0 else 1

    cv_values_log: list[dict] = []
    cv_data_path = work_root / "cv_values.json"
    jsonl_path = work_root / "tps_steps.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    def _opes_step_postprocess(mc_step: int, entry: dict) -> None:
        cv_raw = entry.get("cv_value")
        log_entry: dict = {"mc_step": mc_step}
        if cv_raw is not None:
            # cv_raw may be a float (1-D) or a list (multi-D, already tolist()-ed)
            if isinstance(cv_raw, list):
                log_entry["cv"] = cv_raw  # list of floats for multi-D
            else:
                try:
                    log_entry["cv"] = float(cv_raw)
                except (TypeError, ValueError):
                    pass
            cv_values_log.append(log_entry)
        # Append any diagnostic CVs to the log entry
        for key, val in entry.items():
            if key.startswith("diag_") and val is not None:
                try:
                    log_entry[key] = float(val)
                except (TypeError, ValueError):
                    pass
        with jsonl_path.open("a", encoding="utf-8") as jf:
            jf.write(json.dumps(entry, default=str) + "\n")
            jf.flush()
        if save_cv_every > 0 and mc_step % save_cv_every == 0 and cv_values_log:
            atomic_write_json(
                cv_data_path,
                {
                    "cv_type": args.bias_cv,
                    "cv_names": bias_cv_names,
                    "ndim": ndim,
                    "cv_values": [x["cv"] for x in cv_values_log if "cv" in x],
                    "mc_steps": [x["mc_step"] for x in cv_values_log if "cv" in x],
                },
            )

    periodic_step: list[tuple[Callable[[int, dict], None], int]] = [
        (_opes_step_postprocess, 1),
    ]

    # WDSM data collection: paired callbacks that atomically save terminal
    # frame coords + CV + logw at the same MC step, guaranteeing consistency.
    save_wdsm_every = max(0, int(args.save_wdsm_data_every))
    _wdsm_n_saved = [0]
    if save_wdsm_every > 0:
        wdsm_dir = (args.wdsm_data_dir or (work_root / "wdsm_samples"))
        wdsm_dir = Path(wdsm_dir).expanduser().resolve()
        wdsm_dir.mkdir(parents=True, exist_ok=True)

        _wdsm_pending: dict = {"cv": None, "mc_step": None}

        def _wdsm_capture_cv(mc_step: int, entry: dict) -> None:
            cv_raw = entry.get("cv_value")
            if cv_raw is not None:
                _wdsm_pending["cv"] = cv_raw
                _wdsm_pending["mc_step"] = mc_step

        def _wdsm_save_frame(mc_step: int, traj: Trajectory) -> None:
            if _wdsm_pending["mc_step"] != mc_step or _wdsm_pending["cv"] is None:
                return
            snap = traj[-1]
            if not isinstance(snap, BoltzSnapshot) or snap.tensor_coords is None:
                return
            frame = snapshot_frame_numpy_copy(snap)
            if not np.any(np.isfinite(frame)):
                return
            cv_arr = np.atleast_1d(np.asarray(_wdsm_pending["cv"], dtype=np.float64))
            logw_val = float(bias.evaluate(cv_arr)) / bias.kbt
            out_npz = wdsm_dir / f"wdsm_step_{mc_step:08d}.npz"
            tmp_stem = wdsm_dir / f"_tmp_wdsm_{mc_step:08d}"
            np.savez_compressed(
                str(tmp_stem),
                coords=frame.astype(np.float32),
                cv=cv_arr,
                logw=np.float64(logw_val),
                mc_step=np.int64(mc_step),
            )
            os.replace(str(tmp_stem) + ".npz", out_npz)
            _wdsm_n_saved[0] += 1

        periodic_step.append((_wdsm_capture_cv, save_wdsm_every))
        periodic_extra.append((_wdsm_save_frame, save_wdsm_every))
        print(
            f"[TPS-OPES] WDSM data collection: saving every {save_wdsm_every} steps to {wdsm_dir}",
            file=sys.stderr, flush=True,
        )

    final_traj, tps_step_log = run_tps_path_sampling(
        engine, init_traj, args.shoot_rounds, shoot_log,
        progress_every=args.progress_every,
        periodic_callbacks=periodic_extra if periodic_extra else None,
        periodic_step_callbacks=periodic_step,
        log_path_prob_every=args.log_path_prob_every,
        forward_only=args.forward_only,
        reshuffle_probability=args.reshuffle_probability,
        enhanced_bias=bias,
        cv_function=cv_function,
        diagnostic_cv_functions=_build_diagnostic_cv_functions(args.diagnostic_cvs),
    )

    final_state_path = work_root / "opes_state_final.json"
    bias.save_state(
        final_state_path, bias_cv=args.bias_cv, bias_cv_names=bias_cv_names,
    )
    print(
        f"[TPS-OPES] Final OPES state: {bias.n_kernels} kernels, "
        f"{bias.counter} depositions, zed={bias.zed:.4g}",
        flush=True,
    )
    if save_wdsm_every > 0:
        print(
            f"[TPS-OPES] WDSM: saved {_wdsm_n_saved[0]} terminal-frame samples to {wdsm_dir}",
            flush=True,
        )

    cv_data = {
        "cv_type": args.bias_cv,
        "cv_names": bias_cv_names,
        "ndim": ndim,
        "cv_values": [entry["cv"] for entry in cv_values_log if "cv" in entry],
        "mc_steps": [entry["mc_step"] for entry in cv_values_log if "cv" in entry],
    }
    atomic_write_json(cv_data_path, cv_data)

    cv_stats = {}
    if ndim == 1 and isinstance(cv_function, (OpenMMLocalMinRMSD, OpenMMEnergy)):
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
        "bias_cv_names": bias_cv_names,
        "ndim": ndim,
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
