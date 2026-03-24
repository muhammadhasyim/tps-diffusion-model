"""Run Boltz-2 trunk + diffusion conditioning for :class:`BoltzSamplerCore` (no ``sample`` call)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from genai_tps.backends.boltz.utils import build_network_condition_kwargs


def transfer_inference_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Match ``Boltz2InferenceDataModule.transfer_batch_to_device``."""
    skip = {
        "all_coords",
        "all_resolved_mask",
        "crop_to_all_atom_map",
        "chain_symmetries",
        "amino_acids_symmetries",
        "ligand_symmetries",
        "record",
        "affinity_mw",
    }
    out: dict[str, Any] = {}
    for key, val in batch.items():
        if key in skip:
            out[key] = val
        elif torch.is_tensor(val):
            out[key] = val.to(device)
        else:
            out[key] = val
    return out


@torch.no_grad()
def boltz2_trunk_to_network_kwargs(
    model: Any,
    feats: dict[str, Any],
    *,
    recycling_steps: int = 3,
) -> tuple[Tensor, dict[str, Any]]:
    """Pairformer + ``diffusion_conditioning`` only (mirrors ``Boltz2.forward`` before ``sample``).

    Returns
    -------
    atom_mask
        ``feats[\"atom_pad_mask\"].float()`` on the model device.
    network_kwargs
        Argument for :class:`~genai_tps.backends.boltz.gpu_core.BoltzSamplerCore` via
        :func:`~genai_tps.backends.boltz.utils.build_network_condition_kwargs`.
    """
    model.eval()
    device = next(model.parameters()).device
    feats = transfer_inference_batch_to_device(feats, device)

    s_inputs = model.input_embedder(feats)
    s_init = model.s_init(s_inputs)
    z_init = model.z_init_1(s_inputs)[:, :, None] + model.z_init_2(s_inputs)[:, None, :]
    relative_position_encoding = model.rel_pos(feats)
    z_init = z_init + relative_position_encoding
    z_init = z_init + model.token_bonds(feats["token_bonds"].float())
    if model.bond_type_feature:
        z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())
    z_init = z_init + model.contact_conditioning(feats)

    s = torch.zeros_like(s_init)
    z = torch.zeros_like(z_init)
    mask = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    if model.run_trunk_and_structure:
        for _ in range(recycling_steps + 1):
            s = s_init + model.s_recycle(model.s_norm(s))
            z = z_init + model.z_recycle(model.z_norm(z))
            if model.use_templates:
                tmpl = (
                    model.template_module._orig_mod
                    if model.is_template_compiled and not model.training
                    else model.template_module
                )
                z = z + tmpl(z, feats, pair_mask, use_kernels=model.use_kernels)
            msa_m = (
                model.msa_module._orig_mod
                if model.is_msa_compiled and not model.training
                else model.msa_module
            )
            z = z + msa_m(z, s_inputs, feats, use_kernels=model.use_kernels)
            pf = (
                model.pairformer_module._orig_mod
                if model.is_pairformer_compiled and not model.training
                else model.pairformer_module
            )
            s, z = pf(s, z, mask=mask, pair_mask=pair_mask, use_kernels=model.use_kernels)

    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = model.diffusion_conditioning(
        s_trunk=s,
        z_trunk=z,
        relative_position_encoding=relative_position_encoding,
        feats=feats,
    )
    diffusion_conditioning = {
        "q": q,
        "c": c,
        "to_keys": to_keys,
        "atom_enc_bias": atom_enc_bias,
        "atom_dec_bias": atom_dec_bias,
        "token_trans_bias": token_trans_bias,
    }
    bundle = {
        "s_trunk": s,
        "s_inputs": s_inputs,
        "feats": feats,
        "diffusion_conditioning": diffusion_conditioning,
    }
    kwargs = build_network_condition_kwargs(bundle)
    atom_mask = feats["atom_pad_mask"].float()
    return atom_mask, kwargs
