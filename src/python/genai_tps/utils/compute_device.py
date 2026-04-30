"""PyTorch device strings and OpenMM ``DeviceIndex`` platform properties."""

from __future__ import annotations

import torch

__all__ = [
    "parse_torch_device",
    "cuda_device_index_for_openmm",
    "openmm_device_index_properties",
    "maybe_set_torch_cuda_current_device",
]


def parse_torch_device(s: str) -> torch.device:
    """Parse a device string (e.g. ``cpu``, ``cuda``, ``cuda:1``) into :class:`torch.device`."""
    t = s.strip()
    if not t:
        raise ValueError("device string must be non-empty")
    return torch.device(t)


def cuda_device_index_for_openmm(device: torch.device) -> int | None:
    """Return the CUDA ordinal for OpenMM, or ``None`` if not a CUDA device.

    Bare ``torch.device("cuda")`` has ``index is None``; treat that as ``0``.
    """
    if device.type != "cuda":
        return None
    idx = device.index
    return 0 if idx is None else int(idx)


def openmm_device_index_properties(
    platform_name: str, device_index: int | None
) -> dict[str, str]:
    """Build OpenMM ``platformProperties`` fragment for GPU selection.

    CUDA and OpenCL both use the ``DeviceIndex`` property (see OpenMM user guide,
    "Platform-Specific Properties"). *platform_name* may be the requested platform
    or the resolved name after fallback (both use the same property).
    """
    if device_index is None:
        return {}
    if platform_name not in ("CUDA", "OpenCL"):
        return {}
    return {"DeviceIndex": str(int(device_index))}


def maybe_set_torch_cuda_current_device(device: torch.device) -> None:
    """If *device* is CUDA and CUDA is available, set the current device ordinal.

    Call once at the start of CLIs so code that implicitly uses the default CUDA
    stream matches an explicit ``cuda:N`` choice.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        idx = 0 if device.index is None else int(device.index)
        torch.cuda.set_device(idx)
