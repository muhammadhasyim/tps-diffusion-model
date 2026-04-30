"""Shared Boltz-2 inference session setup (YAML preprocess, dataloader, model, core).

Used by training scripts and TPS drivers to avoid copy-pasting ~120 lines of
``process_inputs`` / ``Boltz2.load_from_checkpoint`` / ``BoltzSamplerCore`` wiring.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "DEFAULT_BOLTZ_CKPT_NAME",
    "DEFAULT_MSA_SERVER_URL",
    "boltz_prep_run_dir",
    "boltz_results_run_dir",
    "build_boltz_session",
    "write_ref_pdb_from_structure",
]

DEFAULT_MSA_SERVER_URL = "https://api.colabfold.com"
DEFAULT_BOLTZ_CKPT_NAME = "boltz2_conf.ckpt"


def boltz_prep_run_dir(work_root: Path, yaml_stem: str) -> Path:
    """Training / FES scripts: ``<work_root>/boltz_prep_<stem>``."""
    return work_root / f"boltz_prep_{yaml_stem}"


def boltz_results_run_dir(work_root: Path, yaml_stem: str) -> Path:
    """TPS drivers: ``<work_root>/boltz_results_<stem>``."""
    return work_root / f"boltz_results_{yaml_stem}"


def write_ref_pdb_from_structure(structure: Any, n_struct: int, out_pdb: Path) -> None:
    """Write a single-frame heavy-atom PDB from Boltz topology coordinates (OpenMM ref)."""
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


def build_boltz_session(
    *,
    yaml_path: Path,
    cache: Path,
    boltz_run_dir: Path,
    device: torch.device,
    diffusion_steps: int,
    recycling_steps: int,
    kernels: bool,
    use_msa_server: bool = False,
    model_eval_mode: bool = False,
    sampler_core_extra_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any, Path, Path | None, Path, dict[str, Any]]:
    """Preprocess YAML, load Boltz2, build :class:`BoltzSamplerCore`, return session bundle.

    Parameters
    ----------
    yaml_path
        Boltz input YAML.
    cache
        Boltz cache directory (CCD, checkpoint, mols).
    boltz_run_dir
        Preprocessing output directory (caller chooses ``boltz_prep_*`` vs ``boltz_results_*``).
    device
        Torch device for the model and batch.
    diffusion_steps, recycling_steps, kernels
        Passed through to predict args and trunk.
    use_msa_server
        Forwarded to ``process_inputs`` (TPS scripts may enable MSA server).
    model_eval_mode
        If True, call ``model.eval()`` after moving to device.
    sampler_core_extra_kwargs
        Optional extra kwargs for ``BoltzSamplerCore`` (e.g. ``compile_model``, ``inference_dtype``).

    Returns
    -------
    model, core, batch, processed_dir, topo_npz_or_none, boltz_run_dir, network_kwargs
        ``topo_npz_or_none`` is the first ``processed/structures/*.npz`` if any exist.
        ``network_kwargs`` is the conditioning dict from ``boltz2_trunk_to_network_kwargs``.
    """
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

    boltz_run_dir.mkdir(parents=True, exist_ok=True)

    data_list = check_inputs(yaml_path)
    process_inputs(
        data=data_list,
        out_dir=boltz_run_dir,
        ccd_path=cache / "ccd.pkl",
        mol_dir=mol_dir,
        msa_server_url=DEFAULT_MSA_SERVER_URL,
        msa_pairing_strategy="greedy",
        use_msa_server=use_msa_server,
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

    ckpt = cache / DEFAULT_BOLTZ_CKPT_NAME
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
    if model_eval_mode:
        model.eval()

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
    core_kw: dict[str, Any] = {"multiplicity": 1}
    if sampler_core_extra_kwargs:
        core_kw.update(sampler_core_extra_kwargs)
    core = BoltzSamplerCore(diffusion, atom_mask, network_kwargs, **core_kw)
    core.build_schedule(diffusion_steps)

    struct_candidates = sorted((processed_dir / "structures").glob("*.npz"))
    topo_npz = struct_candidates[0] if struct_candidates else None

    return model, core, batch, processed_dir, topo_npz, boltz_run_dir, network_kwargs
