"""GPU-friendly collective variables; OPS CVs wrap scalar functions on snapshots."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from openpathsampling.collectivevariable import FunctionCV


def _coords_torch(snapshot) -> torch.Tensor:
    if getattr(snapshot, "_tensor_coords_gpu", None) is not None:
        return snapshot._tensor_coords_gpu
    c = torch.as_tensor(snapshot.coordinates, dtype=torch.float32)
    if c.dim() == 2:
        c = c.unsqueeze(0)
    return c


def radius_of_gyration(snapshot, atom_mask: torch.Tensor | None = None) -> float:
    """Mass-unweighted :math:`R_g` for the first batch element (Angstrom units)."""
    x = _coords_torch(snapshot)[0]
    if atom_mask is not None:
        m = atom_mask[0] if atom_mask.dim() > 1 else atom_mask
        sel = m.bool()
        pts = x[sel]
    else:
        pts = x
    c = pts.mean(dim=0)
    rg = torch.sqrt(((pts - c) ** 2).sum(dim=-1).mean() + 1e-12)
    return float(rg.detach().cpu())


def rmsd_to_reference(
    snapshot,
    reference: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> float:
    """RMSD (no alignment) between snapshot coords and ``reference`` (M, 3)."""
    x = _coords_torch(snapshot)[0]
    ref = reference.to(device=x.device, dtype=x.dtype)
    if atom_mask is not None:
        m = atom_mask[0] if atom_mask.dim() > 1 else atom_mask
        sel = m.bool()
        d = (x[sel] - ref[sel]).pow(2).sum(dim=-1).mean()
    else:
        d = (x - ref).pow(2).sum(dim=-1).mean()
    return float(torch.sqrt(d + 1e-12).detach().cpu())


def diffusion_step_index_cv(snapshot) -> float:
    return float(getattr(snapshot, "step_index", 0))


def make_rg_cv(name: str = "Rg", atom_mask: torch.Tensor | None = None) -> FunctionCV:
    return FunctionCV(name, _rg_callable, atom_mask=atom_mask)


def _rg_callable(snap, atom_mask=None):
    return radius_of_gyration(snap, atom_mask)


def make_rmsd_cv(
    name: str,
    reference: np.ndarray | torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> FunctionCV:
    ref = torch.as_tensor(reference, dtype=torch.float32)
    return FunctionCV(name, _rmsd_callable, reference=ref, atom_mask=atom_mask)


def _rmsd_callable(snap, reference=None, atom_mask=None):
    return rmsd_to_reference(snap, reference, atom_mask)


def make_sigma_cv(name: str = "sigma") -> FunctionCV:
    return FunctionCV(
        name,
        lambda snap: float(snap.sigma) if getattr(snap, "sigma", None) is not None else 0.0,
    )


def make_plddt_proxy_cv(
    name: str = "pLDDT_proxy",
    predictor: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> FunctionCV:
    """pLDDT from denoised coords; supply ``predictor`` for real Boltz confidence."""

    def _pred(snap):
        x = _coords_torch(snap)
        if predictor is None:
            return 50.0 + 0.0 * float(x.detach().cpu().mean())
        with torch.no_grad():
            out = predictor(x)
        return float(out.detach().cpu().reshape(-1)[0])

    return FunctionCV(name, _pred)
