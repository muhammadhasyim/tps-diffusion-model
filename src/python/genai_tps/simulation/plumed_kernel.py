"""Locate the active PLUMED kernel and inspect its compile-time ``config.txt``.

The conda-forge ``plumed`` package is typically built with optional modules
disabled.  In particular, ``OPES_METAD`` lives in the ``opes`` module; if that
line reads ``module opes off``, PLUMED will reject ``OPES_METAD`` with a
generic "I cannot understand line" parse error.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "plumed_kernel_path",
    "plumed_config_txt_path",
    "plumed_module_enabled",
    "assert_plumed_opes_metad_available",
]


def plumed_kernel_path() -> Path | None:
    """Return ``libplumedKernel.so`` (or dylib) if found, else ``None``."""
    env = os.environ.get("PLUMED_KERNEL")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path(sys.prefix) / "lib" / "libplumedKernel.so")
    if sys.platform == "darwin":
        candidates.append(Path(sys.prefix) / "lib" / "libplumedKernel.dylib")
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def plumed_config_txt_path(*, kernel: Path | None = None) -> Path | None:
    """Return ``.../lib/plumed/src/config/config.txt`` for the given kernel, if present."""
    k = kernel or plumed_kernel_path()
    if k is None:
        return None
    cfg = k.parent / "plumed" / "src" / "config" / "config.txt"
    if cfg.is_file():
        return cfg.resolve()
    return None


def plumed_module_enabled(module: str, *, config_path: Path | None = None) -> bool | None:
    """Parse ``module <name> on|off`` from PLUMED's ``config.txt``.

    Returns
    -------
    bool
        ``True`` if the module is enabled, ``False`` if disabled.
    None
        If *config_path* is missing or has no matching ``module`` line.
    """
    cfg = config_path or plumed_config_txt_path()
    if cfg is None or not cfg.is_file():
        return None
    prefix = f"module {module.strip()} "
    for raw in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line.startswith(prefix):
            continue
        parts = line.split()
        # e.g. "module opes on (default-off)" or "module opes off (default-off)"
        if len(parts) >= 3 and parts[2] in {"on", "off"}:
            return parts[2] == "on"
    return None


def assert_plumed_opes_metad_available() -> None:
    """Raise ``RuntimeError`` if the linked PLUMED cannot parse ``OPES_METAD``."""
    kernel = plumed_kernel_path()
    cfg = plumed_config_txt_path(kernel=kernel) if kernel else None
    enabled = plumed_module_enabled("opes", config_path=cfg)

    if enabled is True:
        return

    kernel_msg = str(kernel) if kernel else "(not found; set PLUMED_KERNEL)"
    if enabled is False:
        raise RuntimeError(
            "This environment's PLUMED was built without the 'opes' module, so "
            "the action OPES_METAD is unavailable (PLUMED reports "
            "'module opes off' in its config.txt). "
            "The standard conda-forge plumed package omits optional modules.\n\n"
            "Fix (pick one):\n"
            "  • Build PLUMED from source with ./configure --enable-modules=opes "
            "(see https://www.plumed.org/doc-master/user-doc/html/mymodules.html) "
            "and point PLUMED_KERNEL at that libplumedKernel.so.\n"
            "  • For development only, run with --opes-mode observer (Python-side "
            "OPESBias bookkeeping on unbiased MD — not equivalent to biased OPES-MD).\n\n"
            f"Resolved PLUMED kernel: {kernel_msg}\n"
            f"config.txt: {cfg if cfg else '(not found next to kernel)'}",
        )

    hint = ""
    cpx = os.environ.get("CONDA_PREFIX")
    if kernel is None and cpx:
        try:
            if Path(cpx).resolve() != Path(sys.prefix).resolve():
                hint = (
                    "\n\nLikely cause: `python` is not the interpreter inside CONDA_PREFIX. "
                    f"CONDA_PREFIX={cpx} but sys.prefix={sys.prefix}. "
                    "Put this env's bin ahead on PATH (e.g. export PATH=\"${CONDA_PREFIX}/bin:$PATH\") "
                    "or run scripts with `${CONDA_PREFIX}/bin/python` so PLUMED is resolved from the same prefix."
                )
        except OSError:
            pass

    raise RuntimeError(
        "Could not determine whether PLUMED includes the 'opes' module "
        f"(kernel={kernel_msg}, config.txt={cfg}). "
        "If OPES_METAD fails at runtime, build PLUMED with --enable-modules=opes "
        "or use --opes-mode observer."
        f"{hint}",
    )
