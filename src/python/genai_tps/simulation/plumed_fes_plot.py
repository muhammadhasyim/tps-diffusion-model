"""Plot PLUMED ``FES_from_Reweighting.py`` output (``.dat``) to PNG."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


def load_plumed_fes_dat(
    path: Path,
) -> tuple[Literal[1, 2], tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Parse PLUMED-style FES ``.dat`` with ``#! FIELDS`` header.

    Returns
    -------
    dim
        Number of CV dimensions (1 or 2) before ``file.free``.
    cv_names
        CV column names (length *dim*).
    x, y, fes
        For ``dim==1``, *y* is shape ``(0,)``; use *x* and *fes* only.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    fields_line = None
    for line in text.splitlines():
        if line.strip().startswith("#! FIELDS"):
            fields_line = line.strip()
            break
    if fields_line is None:
        raise ValueError(f"No #! FIELDS line in {path}")

    toks = fields_line.split()
    # #! FIELDS cv1 file.free  OR  #! FIELDS cv1 cv2 file.free
    after = toks[2:]
    if not after or after[-1] != "file.free":
        raise ValueError(f"Expected ... file.free in FIELDS line: {fields_line}")
    cv_names = tuple(after[:-1])
    dim: Literal[1, 2]
    if len(cv_names) == 1:
        dim = 1
    elif len(cv_names) == 2:
        dim = 2
    else:
        raise ValueError(f"Only 1D or 2D FES supported, got CVs {cv_names!r}")

    rows: list[list[float]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if dim == 1 and len(parts) >= 2:
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
        elif dim == 2 and len(parts) >= 3:
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric data rows in {path}")

    arr = np.asarray(rows, dtype=np.float64)
    if dim == 1:
        return dim, cv_names, arr[:, 0], np.empty(0, dtype=np.float64), arr[:, 1]
    return dim, cv_names, arr[:, 0], arr[:, 1], arr[:, 2]


def plot_fes_dat_to_png(
    fes_path: Path,
    out_png: Path,
    *,
    dpi: int = 150,
    vmax_percentile: float = 98.0,
) -> None:
    """Write a matplotlib PNG for 1D (line) or 2D (filled contour) FES."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dim, cv_names, x, y, fes = load_plumed_fes_dat(fes_path)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if dim == 1:
        ax.plot(x, fes, color="#1565C0", linewidth=2.0)
        ax.set_xlabel(cv_names[0])
        ax.set_ylabel(r"Free energy (kJ mol$^{-1}$)")
        ax.set_title(f"{cv_names[0]} (FES)")
    else:
        fin = fes[np.isfinite(fes)]
        if fin.size == 0:
            raise ValueError("No finite FES values to plot")
        cap = float(np.percentile(fin, vmax_percentile))
        tcf = ax.tricontourf(
            x, y, fes, levels=48, cmap="viridis_r", vmin=float(np.nanmin(fes)), vmax=cap
        )
        fig.colorbar(tcf, ax=ax, label=r"Free energy (kJ mol$^{-1}$)")
        ax.set_xlabel(cv_names[0])
        ax.set_ylabel(cv_names[1])
        ax.set_title(f"{cv_names[0]} vs {cv_names[1]}")
        ax.set_aspect("auto")

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
