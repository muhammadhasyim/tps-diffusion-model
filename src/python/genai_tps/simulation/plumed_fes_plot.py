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


def _plumed_fields_tokens_from_line(fields_line: str) -> list[str]:
    toks = fields_line.split()
    if len(toks) < 3 or toks[0] != "#!" or toks[1] != "FIELDS":
        raise ValueError(f"Not a PLUMED FIELDS line: {fields_line!r}")
    return toks[2:]


def load_plumed_colvar_two_cvs(
    colvar_path: Path,
    cv_a: str,
    cv_b: str,
    *,
    skiprows: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load two CV columns from a PLUMED ``COLVAR`` (first ``#! FIELDS`` line).

    All non-comment lines are parsed as whitespace-separated floats; column
    order matches the FIELDS list (excluding the ``#! FIELDS`` prefix).

    Parameters
    ----------
    colvar_path
        Path to ``COLVAR`` (or ``bck.*.COLVAR``) with a ``#! FIELDS`` header.
    cv_a, cv_b
        Names exactly as in the FIELDS line (e.g. ``lig_rmsd``, ``lig_dist``).
    skiprows
        Number of **data** rows to drop after the header (burn-in).
    """
    text = colvar_path.read_text(encoding="utf-8", errors="replace")
    fields: list[str] | None = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#! FIELDS"):
            fields = _plumed_fields_tokens_from_line(s)
            break
    if fields is None:
        raise ValueError(f"No #! FIELDS line in {colvar_path}")
    try:
        ia = fields.index(cv_a)
        ib = fields.index(cv_b)
    except ValueError as exc:
        raise ValueError(
            f"CV {cv_a!r} or {cv_b!r} not in FIELDS {fields!r} ({colvar_path})"
        ) from exc

    rows: list[tuple[float, float]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) <= max(ia, ib):
            continue
        try:
            xa = float(parts[ia])
            xb = float(parts[ib])
        except ValueError:
            continue
        rows.append((xa, xb))

    if not rows:
        raise ValueError(f"No numeric data rows in {colvar_path}")
    if skiprows > 0:
        rows = rows[int(skiprows) :]
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def load_plumed_kernels_2d(
    kernels_path: Path,
) -> tuple[tuple[str, str], np.ndarray, np.ndarray, np.ndarray]:
    """Parse OPES ``KERNELS`` (2 CVs) into centers, Gaussian widths, and heights.

    Expected PLUMED layout (``#! FIELDS`` when present)::

        time cv1 cv2 sigma_cv1 sigma_cv2 height logweight

    If the file has no ``#! FIELDS`` line, the same 8-token numeric layout is
    assumed and CVs are labeled ``cv1``, ``cv2``.

    Returns
    -------
    cv_names
        Names of the two collective variables.
    centers
        Shape ``(N, 2)`` — kernel centers in CV space.
    sigmas
        Shape ``(N, 2)`` — Gaussian standard deviations per CV.
    height
        Shape ``(N,)`` — kernel height prefactor (PLUMED ``height`` field).
    """
    text = kernels_path.read_text(encoding="utf-8", errors="replace")
    fields: list[str] | None = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#! FIELDS"):
            fields = _plumed_fields_tokens_from_line(s)
            break

    if fields is not None:
        n_cv = (len(fields) - 3) // 2
        if n_cv != 2:
            raise ValueError(
                f"Only 2D kernels supported; FIELDS has {len(fields)} tokens "
                f"({fields!r}) in {kernels_path}"
            )
        expected = 1 + 2 * n_cv + 2
        if len(fields) != expected:
            raise ValueError(f"Unexpected FIELDS width in {kernels_path}: {fields!r}")
        cv_names = (fields[1], fields[2])
    else:
        cv_names = ("cv1", "cv2")

    rows: list[list[float]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 7:
            continue
        try:
            vals = [float(parts[i]) for i in range(7)]
        except ValueError:
            continue
        rows.append(vals)

    if not rows:
        raise ValueError(f"No kernel data rows in {kernels_path}")

    arr = np.asarray(rows, dtype=np.float64)
    centers = arr[:, 1:3]
    sigmas = arr[:, 3:5]
    heights = arr[:, 5]
    _ = arr[:, 6]  # logweight (PLUMED); unused for plotting
    return cv_names, centers, sigmas, heights


def _kernel_density_grid(
    centers: np.ndarray,
    sigmas: np.ndarray,
    heights: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
) -> np.ndarray:
    """Sum of 2D Gaussians on a Cartesian grid (*gx*, *gy* are 1-D)."""
    xg, yg = np.meshgrid(gx, gy, indexing="ij")
    acc = np.zeros_like(xg, dtype=np.float64)
    tiny = 1e-30
    for k in range(centers.shape[0]):
        cx, cy = centers[k, 0], centers[k, 1]
        sx = max(sigmas[k, 0], tiny)
        sy = max(sigmas[k, 1], tiny)
        h = heights[k]
        acc += h * np.exp(
            -0.5 * ((xg - cx) / sx) ** 2 - 0.5 * ((yg - cy) / sy) ** 2
        )
    return acc


def plot_opes_2d_fes_triptych(
    fes_dat: Path,
    kernels_path: Path,
    colvar_path: Path,
    out_png: Path,
    *,
    colvar_cv_names: tuple[str, str] | None = None,
    dpi: int = 150,
    grid_bins: int = 100,
    hist_bins: int = 90,
    skiprows: int = 0,
    vmax_percentile: float = 98.0,
) -> None:
    """Three-panel figure: kernel density + centers, reweighted FES, COLVAR histogram.

    Left: unnormalized sum of OPES Gaussian kernels on a regular grid (log
    color scale) with kernel centers overlaid.  Middle: same 2D FES as
    :func:`plot_fes_dat_to_png` (filled contours).  Right: 2D histogram of
    COLVAR samples (log counts) with linearly interpolated FES contours overlaid.

    Parameters
    ----------
    fes_dat
        PLUMED ``fes_reweighted_2d.dat`` (or any 2D ``file.free`` grid).
    kernels_path
        OPES ``KERNELS`` file from the same run.
    colvar_path
        ``COLVAR`` used for the histogram (and typically for reweighting).
    colvar_cv_names
        Two CV names matching ``COLVAR`` FIELDS.  If ``None``, names are taken
        from *fes_dat* (must match ``COLVAR`` column labels).
    grid_bins
        Resolution for the left kernel-density panel (square grid).
    hist_bins
        Bin count per axis for ``numpy.histogram2d`` (square grid).
    skiprows
        Data rows to skip in *colvar_path* after the header (burn-in).
    vmax_percentile
        Upper cap for the middle-panel FES color scale (finite values).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from scipy.interpolate import griddata

    dim, cv_names_fes, xf, yf, fes = load_plumed_fes_dat(fes_dat)
    if dim != 2:
        raise ValueError(f"Triptych requires 2D FES, got dim={dim} from {fes_dat}")
    cva, cvb = (
        colvar_cv_names if colvar_cv_names is not None else (cv_names_fes[0], cv_names_fes[1])
    )
    xc, yc = load_plumed_colvar_two_cvs(colvar_path, cva, cvb, skiprows=skiprows)

    _, k_centers, k_sigmas, k_heights = load_plumed_kernels_2d(kernels_path)

    xmin = float(min(xf.min(), xc.min(), k_centers[:, 0].min()))
    xmax = float(max(xf.max(), xc.max(), k_centers[:, 0].max()))
    ymin = float(min(yf.min(), yc.min(), k_centers[:, 1].min()))
    ymax = float(max(yf.max(), yc.max(), k_centers[:, 1].max()))
    pad_x = 0.02 * (xmax - xmin + 1e-12)
    pad_y = 0.02 * (ymax - ymin + 1e-12)
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    nxg = int(grid_bins)
    gx_edges = np.linspace(xmin, xmax, nxg + 1)
    gy_edges = np.linspace(ymin, ymax, nxg + 1)
    gx_c = 0.5 * (gx_edges[:-1] + gx_edges[1:])
    gy_c = 0.5 * (gy_edges[:-1] + gy_edges[1:])
    rho_k = _kernel_density_grid(k_centers, k_sigmas, k_heights, gx_c, gy_c)
    rho_pos = np.maximum(rho_k, 1e-300)
    rho_vmin = float(np.percentile(rho_pos[rho_pos > 0], 5)) if np.any(rho_pos > 0) else 1e-30
    rho_vmin = max(rho_vmin, float(rho_pos.max()) * 1e-6)

    hrange = [[xmin, xmax], [ymin, ymax]]
    bins_spec = (int(hist_bins), int(hist_bins))
    H, xe, ye = np.histogram2d(xc, yc, bins=bins_spec, range=hrange)
    xc_bin = 0.5 * (xe[:-1] + xe[1:])
    yc_bin = 0.5 * (ye[:-1] + ye[1:])
    xg_h, yg_h = np.meshgrid(xc_bin, yc_bin, indexing="ij")
    fes_on_hist = griddata(
        (xf, yf),
        fes,
        (xg_h, yg_h),
        method="linear",
        fill_value=np.nan,
    )

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), constrained_layout=True)

    ax0, ax1, ax2 = axes
    fin = fes[np.isfinite(fes)]
    if fin.size == 0:
        raise ValueError("No finite FES values to plot")
    cap = float(np.percentile(fin, vmax_percentile))

    im0 = ax0.pcolormesh(
        gx_edges,
        gy_edges,
        rho_pos,
        shading="flat",
        cmap="viridis",
        norm=LogNorm(vmin=rho_vmin, vmax=float(rho_pos.max())),
    )
    fig.colorbar(im0, ax=ax0, label=r"Kernel sum (arb.)")
    ax0.scatter(
        k_centers[:, 0],
        k_centers[:, 1],
        s=4,
        c="white",
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    ax0.set_xlabel(cva)
    ax0.set_ylabel(cvb)
    ax0.set_title("Kernel density + centers")
    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(ymin, ymax)
    ax0.set_aspect("auto")

    tcf = ax1.tricontourf(
        xf,
        yf,
        fes,
        levels=48,
        cmap="viridis_r",
        vmin=float(np.nanmin(fes)),
        vmax=cap,
    )
    fig.colorbar(tcf, ax=ax1, label=r"Free energy (kJ mol$^{-1}$)")
    ax1.set_xlabel(cv_names_fes[0])
    ax1.set_ylabel(cv_names_fes[1])
    ax1.set_title("Reweighted FES")
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_aspect("auto")

    Hplot = np.maximum(H, 0.5)
    im2 = ax2.pcolormesh(
        xe,
        ye,
        Hplot,
        shading="flat",
        cmap="magma",
        norm=LogNorm(vmin=0.5, vmax=float(np.max(Hplot))),
    )
    fig.colorbar(im2, ax=ax2, label="COLVAR count (+0.5)")
    cs_ok = np.isfinite(fes_on_hist)
    if np.any(cs_ok):
        zgood = fes_on_hist[cs_ok]
        z_lo = float(np.nanpercentile(zgood, 5.0))
        z_hi = float(np.nanpercentile(zgood, vmax_percentile))
        if z_hi > z_lo + 1e-9:
            ax2.contour(
                xg_h,
                yg_h,
                fes_on_hist,
                levels=np.linspace(z_lo, z_hi, 9),
                colors="0.85",
                linewidths=0.6,
            )
    ax2.set_xlabel(cva)
    ax2.set_ylabel(cvb)
    ax2.set_title("COLVAR histogram + FES contours")
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_aspect("auto")

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
