"""Factory helpers for building OPES / TPS collective-variable callables."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
import numpy as np
import torch
from openpathsampling.engines.trajectory import Trajectory

from genai_tps.backends.boltz.cv_geometric import (
    ca_contact_count,
    clash_count,
    contact_order,
    end_to_end_distance,
    lddt_to_reference,
    radius_of_gyration,
    ramachandran_outlier_fraction,
    rmsd_to_reference,
    shape_acylindricity,
    shape_kappa2,
)
from genai_tps.backends.boltz.cv_pose import (
    PoseCVIndexer,
    ligand_pose_rmsd,
    ligand_pocket_distance,
    protein_ligand_contacts,
    protein_ligand_hbond_count,
)
from genai_tps.evaluation.posebusters import (
    POSEBUSTERS_CV_PREFIX,
    POSEBUSTERS_GPU_CV_PREFIX,
    POSEBUSTERS_GPU_PASS_FRACTION,
)
from genai_tps.simulation.openmm_cv import OpenMMEnergy, OpenMMLocalMinRMSD

# All supported single-CV names for ``--bias-cv`` (OPES / TPS scripts).
SINGLE_CV_NAMES: list[str] = [
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


def bias_cv_string_needs_topo_early(bias_cv_raw: str, list_posebusters_cvs: bool) -> bool:
    """Whether resolving ``--topo-npz`` must happen before parsing ``--bias-cv``."""
    if list_posebusters_cvs:
        return True
    low = bias_cv_raw.lower()
    for key in ("openmm", "ligand_", "training_sucos", "posebusters"):
        if key in low:
            return True
    return False


def cv_names_require_topo_npz(cv_names: list[str]) -> bool:
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


def make_cv_function(
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
) -> Callable[[Trajectory], float]:
    """Build a scalar CV function (single-CV bias, returns ``float``)."""

    def _cv_rmsd(traj: Trajectory) -> float:
        assert reference_coords is not None
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
    if cv_type == "rg":
        return _cv_rg
    if cv_type == "end_to_end":
        return _cv_end_to_end
    if cv_type == "ca_contact_count":
        return _cv_ca_contact_count
    if cv_type == "shape_kappa2":
        return _cv_shape_kappa2
    if cv_type == "shape_acylindricity":
        return _cv_shape_acylindricity
    if cv_type == "openmm":
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
    if cv_type == "openmm_energy":
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
    if cv_type == "contact_order":
        return _cv_contact_order
    if cv_type == "clash_count":
        return _cv_clash_count
    if cv_type == "ramachandran_outlier":
        return _cv_ramachandran_outlier
    if cv_type == "lddt":
        return _cv_lddt
    if cv_type in ("ligand_rmsd", "ligand_pocket_dist", "ligand_contacts", "ligand_hbonds"):
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
    if cv_type == "training_sucos_pocket_qcov":
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
    if cv_type == "posebusters_pass_fraction":
        if posebusters_context is None or posebusters_context.get("evaluator") is None:
            raise ValueError(
                "--bias-cv posebusters_pass_fraction requires PoseBusters setup (internal error)."
            )
        from genai_tps.evaluation.posebusters import (  # noqa: PLC0415
            make_posebusters_pass_fraction_traj_fn,
        )

        return make_posebusters_pass_fraction_traj_fn(posebusters_context["evaluator"])
    if cv_type == POSEBUSTERS_GPU_PASS_FRACTION:
        if posebusters_context is None or posebusters_context.get("gpu_evaluator") is None:
            raise ValueError(
                "--bias-cv posebusters_gpu_pass_fraction requires GPU PoseBusters setup (internal error)."
            )
        from genai_tps.evaluation.posebusters import (  # noqa: PLC0415
            make_posebusters_gpu_pass_fraction_traj_fn,
        )

        return make_posebusters_gpu_pass_fraction_traj_fn(posebusters_context["gpu_evaluator"])
    raise ValueError(
        f"Unknown CV type: {cv_type!r}. "
        f"Available: {SINGLE_CV_NAMES}"
    )


def parse_bias_cv_list(bias_cv_str: str) -> list[str]:
    """Parse comma-separated ``--bias-cv`` string into a list of CV names."""
    names = [n.strip() for n in bias_cv_str.split(",") if n.strip()]
    for n in names:
        if (
            n in SINGLE_CV_NAMES
            or n.startswith(POSEBUSTERS_CV_PREFIX)
            or n.startswith(POSEBUSTERS_GPU_CV_PREFIX)
        ):
            continue
        raise ValueError(
            f"Unknown --bias-cv: {n!r}. "
            f"Supported base names: {SINGLE_CV_NAMES}.  "
            "PoseBusters per-check columns use the posebusters__ prefix (see "
            "--list-posebusters-cvs). GPU-native geometry checks use the "
            "posebusters_gpu__ prefix.  "
            "Separate multiple CVs with commas, e.g. "
            "--bias-cv contact_order,clash_count"
        )
    return names


def make_multi_cv_function(
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
) -> tuple[Callable[[Trajectory], np.ndarray], int] | tuple[Callable[[Trajectory], float], int]:
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
            make_cv_function(
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
        fn0 = scalar_fns[0]

        def _scalar_cv(traj: Trajectory) -> float:
            return fn0(traj)

        return _scalar_cv, 1  # type: ignore[return-value]

    def _vector_cv(traj: Trajectory) -> np.ndarray:
        return np.array([fn(traj) for fn in scalar_fns], dtype=np.float64)

    return _vector_cv, ndim


def build_diagnostic_cv_functions(
    names_csv: str | None,
) -> dict[str, Callable[[Trajectory], float]] | None:
    """Map comma-separated diagnostic CV names to callables on ``Trajectory``."""
    if not names_csv:
        return None
    from genai_tps.backends.boltz.cv_geometric import (  # noqa: PLC0415
        ca_contact_count as _ca_contact_count,
        clash_count as _clash_count,
        contact_order as _contact_order,
        end_to_end_distance as _end_to_end_distance,
        ramachandran_outlier_fraction as _ramachandran_outlier_fraction,
        radius_of_gyration as _radius_of_gyration,
        shape_acylindricity as _shape_acylindricity,
        shape_kappa2 as _shape_kappa2,
    )

    _registry: dict[str, Callable[[Trajectory], float]] = {
        "contact_order": lambda traj: _contact_order(traj[-1]),
        "clash_count": lambda traj: _clash_count(traj[-1]),
        "end_to_end": lambda traj: _end_to_end_distance(traj[-1]),
        "ca_contact_count": lambda traj: _ca_contact_count(traj[-1]),
        "shape_kappa2": lambda traj: _shape_kappa2(traj[-1]),
        "shape_acylindricity": lambda traj: _shape_acylindricity(traj[-1]),
        "ramachandran_outlier": lambda traj: _ramachandran_outlier_fraction(traj[-1]),
        "rg": lambda traj: _radius_of_gyration(traj[-1]),
    }
    result: dict[str, Callable[[Trajectory], float]] = {}
    for name in [n.strip() for n in names_csv.split(",") if n.strip()]:
        if name in _registry:
            result[name] = _registry[name]
        else:
            print(
                f"[TPS-OPES] WARNING: unknown diagnostic CV '{name}'; skipping. "
                f"Available: {sorted(_registry)}",
                file=sys.stderr,
                flush=True,
            )
    return result if result else None


__all__ = [
    "SINGLE_CV_NAMES",
    "bias_cv_string_needs_topo_early",
    "build_diagnostic_cv_functions",
    "cv_names_require_topo_npz",
    "make_cv_function",
    "make_multi_cv_function",
    "parse_bias_cv_list",
]
