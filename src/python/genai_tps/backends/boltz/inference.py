"""Shared Boltz-2 inference session construction for training and evaluation scripts.

Centralizes preprocessing, checkpoint loading, :class:`~genai_tps.backends.boltz.gpu_core.BoltzSamplerCore`
construction, and schedule building so ``train_weighted_dsm``, ``evaluate_wdsm_model``, and ``compare_models``
do not duplicate ~120 lines each.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class BoltzInferenceBundle:
    """Everything needed to run diffusion sampling or WDSM training on Boltz-2."""

    model: Any
    core: Any  # BoltzSamplerCore
    atom_mask: Any
    network_kwargs: dict[str, Any]
    batch: Any
    processed_dir: Path
    boltz_run_dir: Path
    topo_npz: Path | None


def quotient_space_sampling_for_checkpoint(checkpoint_path: Path | None) -> bool:
    """Whether Boltz sampling should use quotient-space ODE updates (horizontal projection).

    When a fine-tuned checkpoint was produced with true-quotient loss
    (``loss_type`` is ``true-quotient`` in ``training_summary.json``),
    inference must match training (see arXiv:2604.21809); otherwise use Boltz default
    sampling (no quotient projection in the Euler step).

    Reads ``training_summary.json`` beside ``checkpoint_path`` when present.
    If missing, falls back to directory-name heuristics (e.g. ``sft_true-quotient``).

    Parameters
    ----------
    checkpoint_path:
        Path to ``boltz2_wdsm_final.pt`` or similar, or ``None`` for baseline inference.

    Returns
    -------
    bool
        ``True`` only for true-quotient–trained checkpoints.
    """
    if checkpoint_path is None:
        return False
    ckpt = Path(checkpoint_path).expanduser().resolve()
    summary_path = ckpt.parent / "training_summary.json"
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as f:
            payload = json.load(f)
        cfg = payload.get("config") or {}
        if "quotient_space_sampling" in cfg:
            return bool(cfg["quotient_space_sampling"])
        lt = cfg.get("loss_type")
        if lt is not None:
            return str(lt) == "true-quotient"
    parent = ckpt.parent.name
    return "true-quotient" in parent


def predict_sample_atom_coords_numpy(bundle: BoltzInferenceBundle):
    """Run one Boltz-2 structure sample and return coordinates for all real atoms.

    Processed ``structures/*.npz`` topology bundles often store placeholder zeros
    in the ``coords`` arrays.  This helper runs the loaded Boltz-2 model once
    (using ``predict_args`` from the bundle) and returns a physical pose in
    Boltz atom order.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_atoms, 3)``, dtype ``float32``, coordinates in ångströms.
    """
    import numpy as np

    model = bundle.model
    batch = bundle.batch
    pa = model.predict_args
    model.eval()
    with torch.inference_mode():
        out = model(
            batch,
            recycling_steps=pa["recycling_steps"],
            num_sampling_steps=pa["sampling_steps"],
            diffusion_samples=pa["diffusion_samples"],
            max_parallel_samples=pa["max_parallel_samples"],
            run_confidence_sequentially=True,
        )
    coords_b = out["sample_atom_coords"]
    mask = batch["atom_pad_mask"][0].bool()
    xyz = coords_b[0, mask].float().cpu().numpy()
    return np.asarray(xyz, dtype=np.float32)


def build_boltz_inference_session(
    *,
    yaml_path: Path,
    cache: Path,
    boltz_prep_dir: Path,
    device: torch.device,
    diffusion_steps: int,
    recycling_steps: int = 3,
    kernels: bool = False,
    quotient_space_sampling: bool = False,
) -> BoltzInferenceBundle:
    """Load Boltz-2 from checkpoint, preprocess ``yaml_path``, and build a :class:`BoltzSamplerCore`.

    Mirrors the logic previously duplicated in ``scripts/train_weighted_dsm.py``,
    ``scripts/evaluate_wdsm_model.py``, and ``scripts/compare_models.py``.

    Default ``recycling_steps=3`` and the caller's ``diffusion_steps`` (passed as
    model ``sampling_steps``) match the official ``boltz predict`` CLI defaults
    documented in the upstream ``docs/prediction.md`` (``--recycling_steps 3``,
    ``--sampling_steps 200``).  ``step_scale`` and other diffusion hyperparameters
    follow ``Boltz2DiffusionParams`` from the Boltz checkpoint.

    Parameters
    ----------
    quotient_space_sampling:
        If True, reverse-diffusion steps project velocities through :math:`P_x`
        in the mask-centered shape frame. This removes rotational vertical
        motion while retaining Boltz's random translation as an auxiliary draw
        in the TPS extended state.
        If False, use Boltz default updates (same as pretrained inference).
    """
    from dataclasses import asdict

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
        template_dir=processed_dir / "templates"
        if (processed_dir / "templates").exists()
        else None,
        extra_mols_dir=processed_dir / "mols"
        if (processed_dir / "mols").exists()
        else None,
    )
    loader = dm.predict_dataloader()
    batch = next(iter(loader))
    batch = dm.transfer_batch_to_device(batch, device, dataloader_idx=0)

    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(
        subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True
    )
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
    core = BoltzSamplerCore(
        diffusion,
        atom_mask,
        network_kwargs,
        multiplicity=1,
        quotient_space_sampling=quotient_space_sampling,
    )
    core.build_schedule(diffusion_steps)

    structs = sorted((processed_dir / "structures").glob("*.npz"))
    topo_npz = structs[0] if structs else None

    return BoltzInferenceBundle(
        model=model,
        core=core,
        atom_mask=atom_mask,
        network_kwargs=network_kwargs,
        batch=batch,
        processed_dir=processed_dir,
        boltz_run_dir=boltz_run_dir,
        topo_npz=topo_npz,
    )

