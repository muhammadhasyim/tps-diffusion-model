"""Plot PLUMED ``FES_from_Reweighting.py`` output (``.dat``) to PNG."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


def _parse_genai_grid_bins_3d(text: str) -> tuple[int, int, int] | None:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#! SET genai_grid_bins"):
            parts = s.split()
            if len(parts) >= 6:
                return int(parts[3]), int(parts[4]), int(parts[5])
    return None


def load_plumed_fes_dat(
    path: Path,
    text: str | None = None,
) -> tuple[
    Literal[1, 2, 3],
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Parse PLUMED-style FES ``.dat`` with ``#! FIELDS`` header.

    Returns
    -------
    dim
        Number of CV dimensions (1, 2, or 3) before ``file.free``.
    cv_names
        CV column names (length *dim*).
    x, y, z, fes
        Flattened grid coordinates and free energies.  For ``dim < 3``, *z*
        is an empty array.  For ``dim==1``, *y* is also empty.
    """
    text = (
        path.read_text(encoding="utf-8", errors="replace")
        if text is None
        else text
    )
    fields_line = None
    for line in text.splitlines():
        if line.strip().startswith("#! FIELDS"):
            fields_line = line.strip()
            break
    if fields_line is None:
        raise ValueError(f"No #! FIELDS line in {path}")

    toks = fields_line.split()
    # #! FIELDS cv1 file.free  OR  #! FIELDS cv1 cv2 file.free  OR  + cv3
    after = toks[2:]
    if not after or after[-1] != "file.free":
        raise ValueError(f"Expected ... file.free in FIELDS line: {fields_line}")
    cv_names = tuple(after[:-1])
    dim: Literal[1, 2, 3]
    if len(cv_names) == 1:
        dim = 1
    elif len(cv_names) == 2:
        dim = 2
    elif len(cv_names) == 3:
        dim = 3
    else:
        raise ValueError(
            f"Only 1D, 2D, or 3D FES supported (before file.free), got CVs {cv_names!r}"
        )

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
        elif dim == 3 and len(parts) >= 4:
            try:
                rows.append(
                    [
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    ]
                )
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric data rows in {path}")

    arr = np.asarray(rows, dtype=np.float64)
    if dim == 1:
        return (
            dim,
            cv_names,
            arr[:, 0],
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            arr[:, 1],
        )
    if dim == 2:
        return dim, cv_names, arr[:, 0], arr[:, 1], np.empty(0, dtype=np.float64), arr[:, 2]
    return dim, cv_names, arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def _mask_marginal_fes_far_from_colvar(
    fm: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    sigma_a: float,
    sigma_b: float,
    nsigma: float,
) -> np.ndarray:
    """Set marginal FES to NaN where no COLVAR sample lies within *nsigma* (σ-scaled distance).

    Distance for masking uses the same per-CV scaling as OPES kernels:
    :math:`\\sqrt{((x-x')/\\sigma_a)^2 + ((y-y')/\\sigma_b)^2}` minimized over
    trajectory samples.  Values above *nsigma* are treated as visually
    unsampled (Ramachandran-style blank regions).
    """
    if samples_a.size == 0 or samples_b.size == 0:
        return fm
    tiny = 1e-12
    sa = max(float(sigma_a), tiny)
    sb = max(float(sigma_b), tiny)
    gx_m, gy_m = np.meshgrid(gx, gy, indexing="ij")
    # (nx, ny, N)
    da = (gx_m[:, :, np.newaxis] - samples_a[np.newaxis, np.newaxis, :]) / sa
    db = (gy_m[:, :, np.newaxis] - samples_b[np.newaxis, np.newaxis, :]) / sb
    d_min = np.sqrt(np.min(da * da + db * db, axis=2))
    out = np.array(fm, dtype=np.float64, copy=True)
    out[d_min > float(nsigma)] = np.nan
    return out


def plot_fes_dat_to_png(
    fes_path: Path,
    out_png: Path,
    *,
    dpi: int = 150,
    vmax_percentile: float = 98.0,
    temperature_k: float = 300.0,
    marginal_colvar_mask_path: Path | None = None,
    marginal_colvar_skiprows: int = 0,
    marginal_mask_nsigma: float = 4.0,
    marginal_mask_sigmas: tuple[float, float, float] | None = None,
    marginal_axis_pad_fraction: float = 0.0,
) -> None:
    """Write a matplotlib PNG for 1D (line), 2D (filled contour), or 3D FES.

    For **3D** inputs (three CVs before ``file.free``), the figure has three
    panels: thermodynamic 2D marginals obtained with
    :math:`F_{ab}=-k_BT\\ln\\sum_c \\exp(-F_{abc}/k_BT)` on the internal grid
    (requires ``#! SET genai_grid_bins nx ny nz`` from
    :func:`genai_tps.simulation.reweighted_fes_kde.write_reweighted_fes_3d`).

    When *marginal_colvar_mask_path* is set, each 2D marginal is masked to
    ``NaN`` away from the COLVAR cloud (σ-scaled distance
    :math:`>\\,` *marginal_mask_nsigma*), and the colormap maps bad pixels to
    white so unexplored regions resemble sparse Ramachandran-style figures.
    Use a **wider FES grid** (regenerate ``.dat`` with broad ``grid_min`` /
    ``grid_max``) together with *marginal_axis_pad_fraction* to frame sampled
    ``islands`` on a larger canvas.
    """
    import copy

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    text = Path(fes_path).read_text(encoding="utf-8", errors="replace")
    dim, cv_names, x, y, z, fes = load_plumed_fes_dat(fes_path, text=text)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    kbt_kjmol = 0.0083144621 * float(temperature_k)

    if dim == 1:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.plot(x, fes, color="#1565C0", linewidth=2.0)
        ax.set_xlabel(cv_names[0])
        ax.set_ylabel(r"Free energy (kJ mol$^{-1}$)")
        ax.set_title(f"{cv_names[0]} (FES)")
    elif dim == 2:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
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
    else:
        from scipy.special import logsumexp

        bins = _parse_genai_grid_bins_3d(text)
        if bins is None:
            raise ValueError(
                "3D FES plotting needs '#! SET genai_grid_bins nx ny nz' in the .dat "
                f"header (genai_tps 3D KDE output). File: {fes_path}"
            )
        nx, ny, nz = bins
        if x.size != nx * ny * nz:
            raise ValueError(
                f"3D FES grid mismatch: genai_grid_bins {nx} {ny} {nz} but {x.size} data rows"
            )
        vol = fes.reshape((nx, ny, nz), order="C")
        gx = np.unique(x)
        gy = np.unique(y)
        gz = np.unique(z)
        if len(gx) != nx or len(gy) != ny or len(gz) != nz:
            raise ValueError("3D FES: unique axis lengths do not match genai_grid_bins")

        def _marg(axis: int) -> np.ndarray:
            logp = -vol / kbt_kjmol
            fm = -kbt_kjmol * logsumexp(logp, axis=axis)
            fm -= float(np.nanmin(fm))
            return fm

        f_xy = _marg(2)
        f_xz = _marg(1)
        f_yz = _marg(0)

        sig_triple = marginal_mask_sigmas or (0.3, 0.5, 1.0)
        name_to_sigma = {
            cv_names[0]: float(sig_triple[0]),
            cv_names[1]: float(sig_triple[1]),
            cv_names[2]: float(sig_triple[2]),
        }

        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
        pairs = [
            (f_xy, gx, gy, cv_names[0], cv_names[1], cv_names[2], (0, 1)),
            (f_xz, gx, gz, cv_names[0], cv_names[2], cv_names[1], (0, 2)),
            (f_yz, gy, gz, cv_names[1], cv_names[2], cv_names[0], (1, 2)),
        ]
        for ax, (fm0, xa, xb, na, nb, nsum, ij) in zip(axes, pairs):
            fm = np.array(fm0, dtype=np.float64, copy=True)
            if marginal_colvar_mask_path is not None:
                cva, cvb = load_plumed_colvar_two_cvs(
                    marginal_colvar_mask_path,
                    na,
                    nb,
                    skiprows=int(marginal_colvar_skiprows),
                )
                ia, ib = ij
                fm = _mask_marginal_fes_far_from_colvar(
                    fm,
                    xa,
                    xb,
                    cva,
                    cvb,
                    name_to_sigma[cv_names[ia]],
                    name_to_sigma[cv_names[ib]],
                    marginal_mask_nsigma,
                )
                fm -= np.nanmin(fm)
            fin = fm[np.isfinite(fm)]
            if fin.size == 0:
                raise ValueError("No finite values in 3D marginal (mask too tight?)")
            cap = float(np.percentile(fin, vmax_percentile))
            xg, yg = np.meshgrid(xa, xb, indexing="ij")
            try:
                cmap = plt.colormaps["viridis_r"].copy()
            except (AttributeError, KeyError):
                cmap = copy.copy(plt.cm.get_cmap("viridis_r"))
            cmap.set_bad("white", 1.0)
            tcf = ax.pcolormesh(
                xg,
                yg,
                fm,
                shading="auto",
                cmap=cmap,
                vmin=float(np.nanmin(fm)),
                vmax=cap,
            )
            fig.colorbar(tcf, ax=ax, label=r"Free energy (kJ mol$^{-1}$)")
            ax.set_xlabel(na)
            ax.set_ylabel(nb)
            ax.set_title(f"{na}–{nb} (marg. {nsum})")
            ax.set_aspect("auto")
            if float(marginal_axis_pad_fraction) > 0.0:
                pad = float(marginal_axis_pad_fraction)
                rx = float(np.nanmax(xa) - np.nanmin(xa) + 1e-12)
                ry = float(np.nanmax(xb) - np.nanmin(xb) + 1e-12)
                ax.set_xlim(
                    float(np.nanmin(xa)) - pad * rx,
                    float(np.nanmax(xa)) + pad * rx,
                )
                ax.set_ylim(
                    float(np.nanmin(xb)) - pad * ry,
                    float(np.nanmax(xb)) + pad * ry,
                )

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


def load_plumed_colvar_three_cvs(
    colvar_path: Path,
    cv_a: str,
    cv_b: str,
    cv_c: str,
    *,
    skiprows: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load three CV columns from a PLUMED ``COLVAR`` (first ``#! FIELDS`` line)."""
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
        ic = fields.index(cv_c)
    except ValueError as exc:
        raise ValueError(
            f"CV {cv_a!r}, {cv_b!r}, or {cv_c!r} not in FIELDS {fields!r} ({colvar_path})"
        ) from exc

    rows: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) <= max(ia, ib, ic):
            continue
        try:
            xa = float(parts[ia])
            xb = float(parts[ib])
            xc = float(parts[ic])
        except ValueError:
            continue
        rows.append((xa, xb, xc))

    if not rows:
        raise ValueError(f"No numeric data rows in {colvar_path}")
    if skiprows > 0:
        rows = rows[int(skiprows) :]
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _cell_edges_from_centers(c: np.ndarray) -> np.ndarray:
    """Piecewise-linear cell boundaries so ``len(edges) == len(centers) + 1``."""
    c = np.asarray(c, dtype=np.float64)
    if c.size == 0:
        raise ValueError("empty centers")
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5], dtype=np.float64)
    dif = np.diff(c)
    left = c[0] - 0.5 * dif[0]
    right = c[-1] + 0.5 * dif[-1]
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[left], mid, [right]])


def _kernel_mixture_slice(
    centers: np.ndarray,
    sigmas: np.ndarray,
    heights: np.ndarray,
    *,
    plane_axes: tuple[int, int],
    fixed_axis: int,
    fixed_value: float,
    g0: np.ndarray,
    g1: np.ndarray,
) -> np.ndarray:
    """Sum of 3D truncated Gaussians evaluated on a 2D plane (log-weights ignored).

    Each kernel contributes ``height * exp(-0.5 Q)`` with ``Q`` the full 3-term
    Mahalanobis sum along (axis0, axis1, fixed); *g0*, *g1* are 1-D grids for the
    two free axes (meshgrid ``ij``).
    """
    ia, ib = int(plane_axes[0]), int(plane_axes[1])
    ifixed = int(fixed_axis)
    u, v = np.meshgrid(g0, g1, indexing="ij")
    acc = np.zeros_like(u, dtype=np.float64)
    tiny = 1e-30
    for k in range(centers.shape[0]):
        sa = max(float(sigmas[k, ia]), tiny)
        sb = max(float(sigmas[k, ib]), tiny)
        sf = max(float(sigmas[k, ifixed]), tiny)
        h = float(heights[k]) * np.exp(
            -0.5 * ((fixed_value - centers[k, ifixed]) / sf) ** 2
        )
        acc += h * np.exp(
            -0.5 * ((u - centers[k, ia]) / sa) ** 2
            - 0.5 * ((v - centers[k, ib]) / sb) ** 2
        )
    return acc


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


def load_plumed_kernels_3d(
    kernels_path: Path,
    cv_names: tuple[str, str, str] | None = None,
) -> tuple[tuple[str, str, str], np.ndarray, np.ndarray, np.ndarray]:
    """Parse OPES ``KERNELS`` with three biased CVs into centers, widths, heights.

    PLUMED layout (``#! FIELDS`` when present)::

        time cv1 cv2 cv3 sigma_cv1 sigma_cv2 sigma_cv3 height logweight

    If there is no ``#! FIELDS`` line, each non-comment line must contain nine
    floats in that order; *cv_names* must then be supplied (or defaults
    ``lig_rmsd``, ``lig_dist``, ``lig_contacts`` are used).

    Returns
    -------
    cv_names
        Three collective variable names.
    centers
        Shape ``(N, 3)`` — kernel centers in CV space.
    sigmas
        Shape ``(N, 3)`` — Gaussian standard deviations per CV.
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

    names_resolved: tuple[str, str, str]
    if fields is not None:
        n_cv = (len(fields) - 3) // 2
        if n_cv != 3:
            raise ValueError(
                f"Only 3D kernels supported here; FIELDS has {len(fields)} tokens "
                f"({fields!r}) in {kernels_path}"
            )
        expected = 1 + 2 * n_cv + 2
        if len(fields) != expected:
            raise ValueError(f"Unexpected FIELDS width in {kernels_path}: {fields!r}")
        names_resolved = (fields[1], fields[2], fields[3])
    else:
        names_resolved = (
            cv_names
            if cv_names is not None
            else ("lig_rmsd", "lig_dist", "lig_contacts")
        )

    rows: list[list[float]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 9:
            continue
        try:
            vals = [float(parts[i]) for i in range(9)]
        except ValueError:
            continue
        rows.append(vals)

    if not rows:
        raise ValueError(f"No 3D kernel data rows in {kernels_path}")

    arr = np.asarray(rows, dtype=np.float64)
    centers = arr[:, 1:4]
    sigmas = arr[:, 4:7]
    heights = arr[:, 7]
    _ = arr[:, 8]
    return names_resolved, centers, sigmas, heights


def _integrate_trapezoid_2d(Z: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """∬ Z(x,y) dx dy with *Z* shape ``(len(y), len(x))`` (``meshgrid(..., indexing='xy')``)."""
    zx = Z.astype(np.float64, copy=False)
    if hasattr(np, "trapezoid"):
        inner = np.trapezoid(zx, x, axis=1)
        return float(np.trapezoid(inner, y))
    inner = np.trapz(zx, x, axis=1)
    return float(np.trapz(inner, y))


def _pdf_on_mesh_2d(Z: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Divide *Z* by ∬ Z dx dy on the tensor-product grid so the integral is ~1."""
    zz = np.asarray(Z, dtype=np.float64)
    integ = _integrate_trapezoid_2d(zz, x, y)
    if integ <= 0.0 or not np.isfinite(integ):
        return np.zeros_like(zz)
    return zz / integ


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

    dim, cv_names_fes, xf, yf, zf, fes = load_plumed_fes_dat(fes_dat)
    if dim != 2 or zf.size != 0:
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


def plot_opes_2d_fes_triptych_opes_style(
    fes_dat: Path,
    kernels_path: Path,
    colvar_path: Path,
    out_png: Path,
    *,
    colvar_cv_names: tuple[str, str] | None = None,
    dpi: int = 150,
    grid_bins: int = 96,
    hexbin_gridsize: int = 48,
    skiprows: int = 0,
    ln_rho_n_levels: int = 42,
    ln_rho_cap_percentile: float = 92.0,
    fes_contour_n_levels: int = 10,
    temperature_k: float = 300.0,
) -> None:
    """1×3 OPES-style figure matching ``plot_opes_fes_2d`` (TPS diagnostic layout).

    **Left:** :math:`\\hat{P}_{\\mathrm{OPES}}(s_1,s_2)` from the PLUMED kernel sum on a
    uniform grid, **normalized** so :math:`\\iint \\hat{P}\\,\\mathrm{d}s_1\\mathrm{d}s_2 \\approx 1`
    (same trapezoid rule as ``scripts/plot_opes_fes.py``).  Kernel centers as orange-edge
    scatter markers.

    **Middle:** :math:`-\\ln(\\hat{\\rho}/\\hat{\\rho}_{\\mathrm{max}})` from the reweighted
    FES grid: Boltzmann weights :math:`\\propto \\exp(-F/k_{\\mathrm{B}}T)` on the same mesh,
    then the relative ``-ln ρ`` map (filled contours + light black line contours).

    **Right:** ``hexbin`` of COLVAR samples with ``bins='log'``, plus **blue** contours of the
    interpolated reweighted *F* (kJ mol⁻¹) with inline labels — same information design as the
    internal OPES-TPS 2D figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    dim, cv_names_fes, xf, yf, zf, fes = load_plumed_fes_dat(fes_dat)
    if dim != 2 or zf.size != 0:
        raise ValueError(f"OPES-style triptych requires 2D FES, got dim={dim} from {fes_dat}")
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

    n_axis = int(max(24, min(280, int(grid_bins))))
    gx = np.linspace(xmin, xmax, n_axis, dtype=np.float64)
    gy = np.linspace(ymin, ymax, n_axis, dtype=np.float64)
    X, Y = np.meshgrid(gx, gy, indexing="xy")

    rho_ij = _kernel_density_grid(k_centers, k_sigmas, k_heights, gx, gy)
    p_kernel = rho_ij.T.astype(np.float64)
    p_hat = _pdf_on_mesh_2d(p_kernel, gx, gy)

    kbt_kjmol = 0.0083144621 * float(temperature_k)
    F_grid = griddata(
        (xf, yf),
        fes,
        (X, Y),
        method="linear",
        fill_value=np.nan,
    )
    fin_f = F_grid[np.isfinite(F_grid)]
    if fin_f.size == 0:
        raise ValueError("No finite interpolated FES on the plot mesh.")
    f_min = float(np.nanmin(fin_f))
    rho_f = np.exp(-(F_grid - f_min) / kbt_kjmol)
    rho_f[~np.isfinite(F_grid)] = np.nan
    z_ln = np.full_like(F_grid, np.nan, dtype=np.float64)
    vm = np.nanmax(rho_f)
    if np.isfinite(vm) and vm > 0:
        m = np.isfinite(rho_f) & (rho_f > 0)
        z_ln[m] = -np.log(rho_f[m] / vm)

    fin_ln = z_ln[np.isfinite(z_ln)]
    if fin_ln.size == 0:
        raise ValueError("Could not build -ln(ρ/ρ_max) map from FES.")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.0), layout="constrained")
    except TypeError:
        fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.0), constrained_layout=True)

    kw = dict(levels=18, cmap="viridis", extend="both")

    ax0 = axes[0]
    cf0 = ax0.contourf(X, Y, p_hat, **kw)
    ax0.scatter(
        k_centers[:, 0],
        k_centers[:, 1],
        c="none",
        edgecolors="#FF9800",
        s=28,
        linewidths=1.0,
        label=r"$\mathrm{kernel\ } c_k$",
        zorder=5,
    )
    ax0.set_xlabel(cva)
    ax0.set_ylabel(cvb)
    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(ymin, ymax)
    try:
        ax0.set_box_aspect(1.0)
    except Exception:
        ax0.set_aspect("equal", adjustable="box")
    cbar0 = fig.colorbar(cf0, ax=ax0, fraction=0.046, pad=0.02)
    cbar0.set_label(
        r"$\hat{P}_{\mathrm{OPES}}(s_1,s_2)$ "
        r"($\iint \hat{P}\,\mathrm{d}s_1\mathrm{d}s_2 \approx 1$)"
    )
    ax0.legend(loc="best")

    ax1 = axes[1]
    vmin_ln = float(np.nanmin(z_ln))
    vmax_full = float(np.nanmax(z_ln))
    cap_p = float(np.clip(ln_rho_cap_percentile, 50.0, 99.999))
    vmax_cap = float(np.percentile(fin_ln, cap_p)) if fin_ln.size else vmax_full
    vmax_plot = min(max(vmax_cap, vmin_ln + 1e-12), vmax_full)
    if vmax_plot <= vmin_ln + 1e-12:
        vmax_plot = vmax_full
    nlev = max(8, int(ln_rho_n_levels))
    levels_f = np.linspace(vmin_ln, vmax_plot, nlev)
    extend_ln = "max" if vmax_full > vmax_plot + 1e-6 * max(1.0, abs(vmax_full)) else "neither"
    cf1 = ax1.contourf(
        X,
        Y,
        z_ln,
        levels=levels_f,
        cmap="viridis",
        extend=extend_ln,
    )
    n_line = max(14, min(36, nlev))
    levels_line = np.linspace(vmin_ln, vmax_plot, n_line)
    ax1.contour(
        X,
        Y,
        z_ln,
        levels=levels_line,
        colors="k",
        linewidths=0.45,
        alpha=0.38,
    )
    ax1.set_xlabel(cv_names_fes[0])
    ax1.set_ylabel(cv_names_fes[1])
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    try:
        ax1.set_box_aspect(1.0)
    except Exception:
        ax1.set_aspect("equal", adjustable="box")
    cbar1 = fig.colorbar(cf1, ax=ax1, fraction=0.046, pad=0.02)
    cbar1.set_label(r"$-\ln\left(\hat{\rho}/\hat{\rho}_{\mathrm{max}}\right)$")

    ax2 = axes[2]
    gsz = int(max(12, min(90, hexbin_gridsize)))
    hb = ax2.hexbin(
        xc,
        yc,
        gridsize=gsz,
        mincnt=1,
        cmap="Purples",
        bins="log",
        extent=(xmin, xmax, ymin, ymax),
    )
    f_cont = np.array(F_grid, dtype=np.float64, copy=True)
    f_good = f_cont[np.isfinite(f_cont)]
    if f_good.size > 1:
        z_lo = float(np.nanpercentile(f_good, 5.0))
        z_hi = float(np.nanpercentile(f_good, 95.0))
        if z_hi > z_lo + 1e-9:
            ncv = max(5, int(fes_contour_n_levels))
            clev = np.linspace(z_lo, z_hi, ncv)
            cs = ax2.contour(
                X,
                Y,
                f_cont,
                levels=clev,
                colors="#1565C0",
                linewidths=1.0,
                alpha=0.9,
            )
            ax2.clabel(cs, inline=True, fontsize=9, fmt="%.3g")
    ax2.set_xlabel(cva)
    ax2.set_ylabel(cvb)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    try:
        ax2.set_box_aspect(1.0)
    except Exception:
        ax2.set_aspect("equal", adjustable="box")
    cbar2 = fig.colorbar(hb, ax=ax2, fraction=0.046, pad=0.02)
    cbar2.set_label(r"$\mathrm{counts\ (log\ scale)}$")

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_opes_3d_fes_triptych(
    fes_dat: Path,
    kernels_path: Path,
    colvar_path: Path,
    out_png: Path,
    *,
    colvar_cv_names: tuple[str, str, str] | None = None,
    dpi: int = 150,
    grid_bins: int = 80,
    hist_bins: int = 60,
    skiprows: int = 0,
    vmax_percentile: float = 98.0,
    temperature_k: float = 300.0,
) -> None:
    """Nine-panel figure (3×3): for each CV pair, kernel density, marginal FES, COLVAR hist.

    Mirrors :func:`plot_opes_2d_fes_triptych` for the three canonical 2D projections of a 3D
    reweighted FES (``lig_rmsd``–``lig_dist``, ``lig_rmsd``–``lig_contacts``,
    ``lig_dist``–``lig_contacts``).  OPES kernels are projected onto each plane by summing
    the same bivariate Gaussians used in the 2D helper (third CV ignored in the panel).

    Parameters
    ----------
    fes_dat
        3D ``#! FIELDS cv1 cv2 cv3 file.free`` grid with ``#! SET genai_grid_bins``.
    kernels_path
        OPES ``KERNELS`` from the same run (nine numeric columns per row, with or without
        ``#! FIELDS``).
    colvar_path
        ``COLVAR`` containing the three CV columns (and bias).
    colvar_cv_names
        Names matching ``COLVAR`` / ``fes_dat``.  If ``None``, taken from *fes_dat* FIELDS.
    grid_bins
        Square grid resolution for each kernel-density panel.
    hist_bins
        Bin count per axis for each ``numpy.histogram2d``.
    skiprows
        COLVAR data rows to skip after the header (burn-in).
    vmax_percentile
        Upper cap for each middle-panel marginal FES color scale.
    temperature_k
        Kelvin, for Boltzmann marginalization along the collapsed CV axis.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from scipy.interpolate import griddata
    from scipy.special import logsumexp

    text = Path(fes_dat).read_text(encoding="utf-8", errors="replace")
    dim, cv_names_fes, xf, yf, zf, fes_flat = load_plumed_fes_dat(fes_dat, text=text)
    if dim != 3:
        raise ValueError(f"3D triptych requires 3D FES, got dim={dim} from {fes_dat}")
    names = (
        colvar_cv_names
        if colvar_cv_names is not None
        else (cv_names_fes[0], cv_names_fes[1], cv_names_fes[2])
    )

    bins = _parse_genai_grid_bins_3d(text)
    if bins is None:
        raise ValueError(
            "3D triptych needs '#! SET genai_grid_bins nx ny nz' in the FES .dat header."
        )
    nx, ny, nz = bins
    if xf.size != nx * ny * nz:
        raise ValueError(
            f"3D FES grid mismatch: genai_grid_bins {nx} {ny} {nz} vs {xf.size} rows"
        )
    vol = fes_flat.reshape((nx, ny, nz), order="C")
    gx = np.unique(xf)
    gy = np.unique(yf)
    gz = np.unique(zf)
    if len(gx) != nx or len(gy) != ny or len(gz) != nz:
        raise ValueError("3D FES: unique axis lengths do not match genai_grid_bins")

    kbt_kjmol = 0.0083144621 * float(temperature_k)

    def _marg(axis: int) -> np.ndarray:
        logp = -vol / kbt_kjmol
        fm = -kbt_kjmol * logsumexp(logp, axis=axis)
        fm -= float(np.nanmin(fm))
        return fm

    f_xy = _marg(2)
    f_xz = _marg(1)
    f_yz = _marg(0)

    _, k_centers, k_sigmas, k_heights = load_plumed_kernels_3d(
        kernels_path, cv_names=names
    )

    row_specs: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, str, str, tuple[int, int]]
    ] = [
        (f_xy, gx, gy, names[0], names[1], (0, 1)),
        (f_xz, gx, gz, names[0], names[2], (0, 2)),
        (f_yz, gy, gz, names[1], names[2], (1, 2)),
    ]

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(14.0, 12.0), constrained_layout=True)

    for row, (fm, xa, xb, na, nb, ij) in enumerate(row_specs):
        ia, ib = ij
        xc, yc = load_plumed_colvar_two_cvs(colvar_path, na, nb, skiprows=skiprows)
        k2 = k_centers[:, [ia, ib]]
        s2 = k_sigmas[:, [ia, ib]]

        xf_m, yf_m = np.meshgrid(xa, xb, indexing="ij")
        fes_scatter_x = xf_m.ravel()
        fes_scatter_y = yf_m.ravel()
        fes_scatter_z = fm.ravel()

        xmin = float(
            min(fes_scatter_x.min(), xc.min(), k2[:, 0].min())
        )
        xmax = float(
            max(fes_scatter_x.max(), xc.max(), k2[:, 0].max())
        )
        ymin = float(
            min(fes_scatter_y.min(), yc.min(), k2[:, 1].min())
        )
        ymax = float(
            max(fes_scatter_y.max(), yc.max(), k2[:, 1].max())
        )
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
        rho_k = _kernel_density_grid(k2, s2, k_heights, gx_c, gy_c)
        rho_pos = np.maximum(rho_k, 1e-300)
        rho_vmin = (
            float(np.percentile(rho_pos[rho_pos > 0], 5))
            if np.any(rho_pos > 0)
            else 1e-30
        )
        rho_vmin = max(rho_vmin, float(rho_pos.max()) * 1e-6)

        hrange = [[xmin, xmax], [ymin, ymax]]
        bins_spec = (int(hist_bins), int(hist_bins))
        H, xe, ye = np.histogram2d(xc, yc, bins=bins_spec, range=hrange)
        xc_bin = 0.5 * (xe[:-1] + xe[1:])
        yc_bin = 0.5 * (ye[:-1] + ye[1:])
        xg_h, yg_h = np.meshgrid(xc_bin, yc_bin, indexing="ij")
        fes_on_hist = griddata(
            (fes_scatter_x, fes_scatter_y),
            fes_scatter_z,
            (xg_h, yg_h),
            method="linear",
            fill_value=np.nan,
        )

        ax0, ax1, ax2 = axes[row, 0], axes[row, 1], axes[row, 2]
        fin = fm[np.isfinite(fm)]
        if fin.size == 0:
            raise ValueError(f"No finite FES values for panel {na}–{nb}")
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
            k2[:, 0],
            k2[:, 1],
            s=3,
            c="white",
            alpha=0.3,
            linewidths=0,
            rasterized=True,
        )
        ax0.set_xlabel(na)
        ax0.set_ylabel(nb)
        ax0.set_title(f"{na}–{nb}: kernel density + centers")
        ax0.set_xlim(xmin, xmax)
        ax0.set_ylim(ymin, ymax)
        ax0.set_aspect("auto")

        tcf = ax1.pcolormesh(
            xf_m,
            yf_m,
            fm,
            shading="auto",
            cmap="viridis_r",
            vmin=float(np.nanmin(fm)),
            vmax=cap,
        )
        fig.colorbar(tcf, ax=ax1, label=r"Free energy (kJ mol$^{-1}$)")
        ax1.set_xlabel(na)
        ax1.set_ylabel(nb)
        ax1.set_title(f"{na}–{nb}: marginal FES")
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
        ax2.set_xlabel(na)
        ax2.set_ylabel(nb)
        ax2.set_title(f"{na}–{nb}: COLVAR + FES contours")
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)
        ax2.set_aspect("auto")

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_opes_3d_fes_triptych_opes_style(
    fes_dat: Path,
    kernels_path: Path,
    colvar_path: Path,
    out_png: Path,
    *,
    colvar_cv_names: tuple[str, str, str] | None = None,
    dpi: int = 150,
    hexbin_gridsize: int = 36,
    hexbin_linewidths: float | None = 0.55,
    skiprows: int = 0,
    ln_rho_n_levels: int = 42,
    ln_rho_cap_percentile: float = 92.0,
    fes_contour_n_levels: int = 10,
    temperature_k: float = 300.0,
    kernel_center_alpha: float = 0.22,
) -> None:
    """3×3 figure: three CV-plane rows, each with OPES-TPS-style columns.

    For each canonical slice (``lig_rmsd``–``lig_dist`` at mid ``lig_contacts``, etc.):

    1. **Normalized** 3D kernel mixture on that plane (:func:`_kernel_mixture_slice`)
       with trapezoid-normalized :math:`\\hat{P}` and orange-edge kernel markers
       (opacity ``kernel_center_alpha`` so dense deposits remain readable).
    2. :math:`-\\ln(\\hat{\\rho}/\\hat{\\rho}_{\\mathrm{max}})` from the **same** 3D FES
       **volume slice** on the native grid (not a Boltzmann marginal).
    3. ``hexbin`` of COLVAR (log counts) with blue contours of *F* (kJ mol⁻¹) on the slice.
       Use a **moderate** ``hexbin_gridsize`` (e.g. 28–40): large values yield many small
       hex cells. ``hexbin_linewidths`` draws cell edges so filled hexes read more solidly.

    This is the 3D analogue of :func:`plot_opes_2d_fes_triptych_opes_style` /
    ``plot_opes_fes_2d`` in ``scripts/plot_opes_fes.py``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    text = Path(fes_dat).read_text(encoding="utf-8", errors="replace")
    dim, cv_names_fes, xf, yf, zf, fes_flat = load_plumed_fes_dat(fes_dat, text=text)
    if dim != 3:
        raise ValueError(f"3D OPES-style triptych requires 3D FES, got dim={dim} from {fes_dat}")
    names = (
        colvar_cv_names
        if colvar_cv_names is not None
        else (cv_names_fes[0], cv_names_fes[1], cv_names_fes[2])
    )

    bins = _parse_genai_grid_bins_3d(text)
    if bins is None:
        raise ValueError(
            "3D OPES-style triptych needs '#! SET genai_grid_bins nx ny nz' in the FES .dat header."
        )
    nx, ny, nz = bins
    if xf.size != nx * ny * nz:
        raise ValueError(
            f"3D FES grid mismatch: genai_grid_bins {nx} {ny} {nz} vs {xf.size} rows"
        )
    vol = fes_flat.reshape((nx, ny, nz), order="C")
    gx = np.unique(xf)
    gy = np.unique(yf)
    gz = np.unique(zf)
    if len(gx) != nx or len(gy) != ny or len(gz) != nz:
        raise ValueError("3D FES: unique axis lengths do not match genai_grid_bins")

    _, k_centers, k_sigmas, k_heights = load_plumed_kernels_3d(
        kernels_path, cv_names=names
    )

    ix = int(nx // 2)
    iy = int(ny // 2)
    iz = int(nz // 2)
    z0 = float(gz[iz])
    y0 = float(gy[iy])
    x0 = float(gx[ix])

    kbt_kjmol = 0.0083144621 * float(temperature_k)
    k_alpha = float(np.clip(kernel_center_alpha, 0.03, 1.0))

    row_defs: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, str, str, tuple[int, int], int, float]
    ] = [
        (vol[:, :, iz], gx, gy, names[0], names[1], (0, 1), 2, z0),
        (vol[:, iy, :], gx, gz, names[0], names[2], (0, 2), 1, y0),
        (vol[ix, :, :], gy, gz, names[1], names[2], (1, 2), 0, x0),
    ]

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(17.5, 17.0), constrained_layout=True)
    kw = dict(levels=18, cmap="viridis", extend="both")

    for row, (f_sl, ga, gb, na, nb, plane_axes, i_fix, u_slice) in enumerate(row_defs):
        ia, ib = plane_axes
        xc, yc = load_plumed_colvar_two_cvs(colvar_path, na, nb, skiprows=skiprows)

        xmin = float(
            min(np.nanmin(ga), float(xc.min()), float(k_centers[:, ia].min()))
        )
        xmax = float(
            max(np.nanmax(ga), float(xc.max()), float(k_centers[:, ia].max()))
        )
        ymin = float(
            min(np.nanmin(gb), float(yc.min()), float(k_centers[:, ib].min()))
        )
        ymax = float(
            max(np.nanmax(gb), float(yc.max()), float(k_centers[:, ib].max()))
        )
        pad_x = 0.02 * (xmax - xmin + 1e-12)
        pad_y = 0.02 * (ymax - ymin + 1e-12)
        xmin -= pad_x
        xmax += pad_x
        ymin -= pad_y
        ymax += pad_y

        rho = _kernel_mixture_slice(
            k_centers,
            k_sigmas,
            k_heights,
            plane_axes=plane_axes,
            fixed_axis=i_fix,
            fixed_value=u_slice,
            g0=ga,
            g1=gb,
        )
        X, Y = np.meshgrid(ga, gb, indexing="xy")
        rho_plot = rho.T.astype(np.float64)
        p_hat = _pdf_on_mesh_2d(rho_plot, ga, gb)

        F_plot = np.asarray(f_sl.T, dtype=np.float64)
        fin_f = F_plot[np.isfinite(F_plot)]
        if fin_f.size == 0:
            raise ValueError(f"No finite FES on slice row {row} ({na}–{nb}).")
        f_min = float(np.nanmin(fin_f))
        rho_f = np.exp(-(F_plot - f_min) / kbt_kjmol)
        rho_f[~np.isfinite(F_plot)] = np.nan
        z_ln = np.full_like(F_plot, np.nan, dtype=np.float64)
        vm = np.nanmax(rho_f)
        if np.isfinite(vm) and vm > 0:
            m = np.isfinite(rho_f) & (rho_f > 0)
            z_ln[m] = -np.log(rho_f[m] / vm)

        fin_ln = z_ln[np.isfinite(z_ln)]
        if fin_ln.size == 0:
            raise ValueError(f"Could not build -ln(ρ/ρ_max) for row {row}.")

        vmin_ln = float(np.nanmin(z_ln))
        vmax_full = float(np.nanmax(z_ln))
        cap_p = float(np.clip(ln_rho_cap_percentile, 50.0, 99.999))
        vmax_cap = float(np.percentile(fin_ln, cap_p)) if fin_ln.size else vmax_full
        vmax_plot = min(max(vmax_cap, vmin_ln + 1e-12), vmax_full)
        if vmax_plot <= vmin_ln + 1e-12:
            vmax_plot = vmax_full
        nlev = max(8, int(ln_rho_n_levels))
        levels_f = np.linspace(vmin_ln, vmax_plot, nlev)
        extend_ln = "max" if vmax_full > vmax_plot + 1e-6 * max(1.0, abs(vmax_full)) else "neither"

        ax0, ax1, ax2 = axes[row, 0], axes[row, 1], axes[row, 2]

        cf0 = ax0.contourf(X, Y, p_hat, **kw)
        ax0.scatter(
            k_centers[:, ia],
            k_centers[:, ib],
            c="none",
            edgecolors="#FF9800",
            s=22,
            linewidths=0.9,
            alpha=k_alpha,
            label=r"$\mathrm{kernel\ } c_k$",
            zorder=5,
        )
        ax0.set_xlim(xmin, xmax)
        ax0.set_ylim(ymin, ymax)
        ax0.set_xlabel(na)
        ax0.set_ylabel(nb)
        slice_txt = f"{names[i_fix]} = {u_slice:.4g}"
        ax0.set_title(
            f"{na}–{nb} @ {slice_txt}: "
            r"$\hat{P}_{\mathrm{OPES}}$ (normalized on slice, $\iint \hat{P}\,\mathrm{d}s_1\mathrm{d}s_2 \approx 1$)"
        )
        try:
            ax0.set_box_aspect(1.0)
        except Exception:
            ax0.set_aspect("equal", adjustable="box")
        cbar0 = fig.colorbar(cf0, ax=ax0, fraction=0.046, pad=0.02)
        cbar0.set_label(r"$\hat{P}_{\mathrm{OPES}}$ (normalized on slice)")
        if row == 0:
            ax0.legend(loc="best", fontsize=8)

        cf1 = ax1.contourf(
            X,
            Y,
            z_ln,
            levels=levels_f,
            cmap="viridis",
            extend=extend_ln,
        )
        n_line = max(14, min(36, nlev))
        levels_line = np.linspace(vmin_ln, vmax_plot, n_line)
        ax1.contour(
            X,
            Y,
            z_ln,
            levels=levels_line,
            colors="k",
            linewidths=0.4,
            alpha=0.35,
        )
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        ax1.set_xlabel(na)
        ax1.set_ylabel(nb)
        ax1.set_title(
            f"{na}–{nb}: "
            r"$-\ln(\hat{\rho}/\hat{\rho}_{\mathrm{max}})$ "
            f"(slice {slice_txt})"
        )
        try:
            ax1.set_box_aspect(1.0)
        except Exception:
            ax1.set_aspect("equal", adjustable="box")
        cbar1 = fig.colorbar(cf1, ax=ax1, fraction=0.046, pad=0.02)
        cbar1.set_label(r"$-\ln\left(\hat{\rho}/\hat{\rho}_{\mathrm{max}}\right)$")

        gsz = int(max(10, min(80, int(hexbin_gridsize))))
        hb_kw: dict = dict(
            x=xc,
            y=yc,
            gridsize=gsz,
            mincnt=1,
            cmap="Purples",
            bins="log",
            extent=(xmin, xmax, ymin, ymax),
        )
        if hexbin_linewidths is not None and float(hexbin_linewidths) > 0:
            hb_kw["linewidths"] = float(hexbin_linewidths)
            hb_kw["edgecolors"] = "face"
        hb = ax2.hexbin(**hb_kw)
        f_cont = np.array(F_plot, dtype=np.float64, copy=True)
        f_good = f_cont[np.isfinite(f_cont)]
        if f_good.size > 1:
            z_lo = float(np.nanpercentile(f_good, 5.0))
            z_hi = float(np.nanpercentile(f_good, 95.0))
            if z_hi > z_lo + 1e-9:
                ncv = max(5, int(fes_contour_n_levels))
                clev = np.linspace(z_lo, z_hi, ncv)
                cs = ax2.contour(
                    X,
                    Y,
                    f_cont,
                    levels=clev,
                    colors="#1565C0",
                    linewidths=0.9,
                    alpha=0.9,
                )
                ax2.clabel(cs, inline=True, fontsize=7, fmt="%.3g")
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)
        ax2.set_xlabel(na)
        ax2.set_ylabel(nb)
        ax2.set_title(f"{na}–{nb}: COLVAR hexbin + " + r"$F$ contours (kJ mol$^{-1}$)")
        try:
            ax2.set_box_aspect(1.0)
        except Exception:
            ax2.set_aspect("equal", adjustable="box")
        cbar2 = fig.colorbar(hb, ax=ax2, fraction=0.046, pad=0.02)
        cbar2.set_label(r"$\mathrm{counts\ (log\ scale)}$")

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_opes_3d_fes_slices_kde_deposits(
    fes_dat: Path,
    kernels_path: Path,
    colvar_path: Path,
    out_png: Path,
    *,
    colvar_cv_names: tuple[str, str, str] | None = None,
    dpi: int = 150,
    skiprows: int = 0,
    vmax_percentile: float = 98.0,
) -> None:
    """3×3 dashboard: true FES volume slices, OPES kernel mixture slices, deposit histograms.

    **Columns (same for each row):**

    1. **Reweighted 3D FES** — planar slice through the PLUMED KDE grid (not a
       Boltzmann marginal): *xy* at mid-*z*, *xz* at mid-*y*, *yz* at mid-*x*.
    2. **OPES kernel mixture** on that plane: sum of 3D Gaussians with the
       out-of-plane factor :math:`\\exp(-(u-u_k)^2/(2\\sigma^2))` at the slice
       coordinate (same construction the model uses along biased CVs).
    3. **Deposited kernels** — ``numpy.histogram2d`` counts of OPES **kernel
       center** positions projected onto the plane (one count per PLUMED
       ``KERNELS`` row / deposit event).

    Optional faint scatter: COLVAR samples on the same two CVs (alpha) on column
    0 so trajectory coverage is visible against the FES slice.

    Parameters
    ----------
    fes_dat
        3D ``#! FIELDS cv1 cv2 cv3 file.free`` with ``#! SET genai_grid_bins``.
    kernels_path
        OPES ``KERNELS`` (3 CVs + three sigmas + height + logweight).
    colvar_path
        ``COLVAR`` with the three CV columns (for optional trajectory scatter).
    colvar_cv_names
        If ``None``, names are taken from *fes_dat* ``#! FIELDS``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    text = Path(fes_dat).read_text(encoding="utf-8", errors="replace")
    dim, cv_names_fes, xf, yf, zf, fes_flat = load_plumed_fes_dat(fes_dat, text=text)
    if dim != 3:
        raise ValueError(f"3D dashboard requires 3D FES, got dim={dim} from {fes_dat}")
    names = (
        colvar_cv_names
        if colvar_cv_names is not None
        else (cv_names_fes[0], cv_names_fes[1], cv_names_fes[2])
    )

    bins = _parse_genai_grid_bins_3d(text)
    if bins is None:
        raise ValueError(
            "3D slice dashboard needs '#! SET genai_grid_bins nx ny nz' in the FES .dat."
        )
    nx, ny, nz = bins
    if xf.size != nx * ny * nz:
        raise ValueError(
            f"3D FES grid mismatch: genai_grid_bins {nx} {ny} {nz} vs {xf.size} rows"
        )
    vol = fes_flat.reshape((nx, ny, nz), order="C")
    gx = np.unique(xf)
    gy = np.unique(yf)
    gz = np.unique(zf)
    if len(gx) != nx or len(gy) != ny or len(gz) != nz:
        raise ValueError("3D FES: unique axis lengths do not match genai_grid_bins")

    names, k_centers, k_sigmas, k_heights = load_plumed_kernels_3d(
        kernels_path, cv_names=names
    )

    xc, yc, zc = load_plumed_colvar_three_cvs(
        colvar_path, names[0], names[1], names[2], skiprows=skiprows
    )

    ix = int(nx // 2)
    iy = int(ny // 2)
    iz = int(nz // 2)
    z0 = float(gz[iz])
    y0 = float(gy[iy])
    x0 = float(gx[ix])

    ex = _cell_edges_from_centers(gx)
    ey = _cell_edges_from_centers(gy)
    ez = _cell_edges_from_centers(gz)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(15.0, 13.0), constrained_layout=True)

    row_specs: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, str, str, tuple[int, int], int, float, tuple[np.ndarray, np.ndarray]]
    ] = [
        (vol[:, :, iz], gx, gy, names[0], names[1], (0, 1), 2, z0, (ex, ey)),
        (vol[:, iy, :], gx, gz, names[0], names[2], (0, 2), 1, y0, (ex, ez)),
        (vol[ix, :, :], gy, gz, names[1], names[2], (1, 2), 0, x0, (ey, ez)),
    ]

    for row, spec in enumerate(row_specs):
        f_sl, ga, gb, na, nb, plane_axes, i_fix, u_fix, (e0, e1) = spec
        ia, ib = plane_axes

        rho = _kernel_mixture_slice(
            k_centers,
            k_sigmas,
            k_heights,
            plane_axes=plane_axes,
            fixed_axis=i_fix,
            fixed_value=u_fix,
            g0=ga,
            g1=gb,
        )
        rho_pos = np.maximum(rho, 1e-300)
        rho_vmin = (
            float(np.percentile(rho_pos[rho_pos > 0], 5))
            if np.any(rho_pos > 0)
            else 1e-30
        )
        rho_vmin = max(rho_vmin, float(rho_pos.max()) * 1e-6)
        rho_vmax = float(np.max(rho_pos))
        if not np.isfinite(rho_vmax) or rho_vmax <= 0.0:
            rho_vmax = 1.0
        rho_vmin = float(np.clip(rho_vmin, 1e-12, rho_vmax * 0.999))
        if rho_vmax <= rho_vmin * 1.000001:
            rho_vmax = rho_vmin * 100.0
        if rho_vmax / rho_vmin < 1.05:
            mix_norm: Normalize | LogNorm = Normalize(
                vmin=float(np.min(rho_pos)), vmax=float(np.max(rho_pos))
            )
        else:
            mix_norm = LogNorm(vmin=rho_vmin, vmax=rho_vmax)

        H_dep, xe_dep, ye_dep = np.histogram2d(
            k_centers[:, ia],
            k_centers[:, ib],
            bins=[e0, e1],
        )
        Hplot = np.maximum(H_dep, 0.5)
        hmax = float(np.max(Hplot))
        if hmax <= 0.500001:
            hmax = 2.0
        if hmax / 0.5 < 1.05:
            dep_norm: Normalize | LogNorm = Normalize(vmin=0.0, vmax=hmax)
        else:
            dep_norm = LogNorm(vmin=0.5, vmax=hmax)

        fin = f_sl[np.isfinite(f_sl)]
        if fin.size == 0:
            raise ValueError(f"No finite FES values for slice row {row}")
        cap = float(np.percentile(fin, vmax_percentile))

        g0m, g1m = np.meshgrid(ga, gb, indexing="ij")
        ax0, ax1, ax2 = axes[row, 0], axes[row, 1], axes[row, 2]

        tcf0 = ax0.pcolormesh(
            g0m,
            g1m,
            f_sl,
            shading="auto",
            cmap="viridis_r",
            vmin=float(np.nanmin(f_sl)),
            vmax=cap,
        )
        fig.colorbar(tcf0, ax=ax0, label=r"FES slice (kJ mol$^{-1}$)")
        ax0.set_xlabel(na)
        ax0.set_ylabel(nb)
        ax0.set_title(
            f"{na}–{nb} slice @ {names[i_fix]}={u_fix:.4g} (reweighted KDE grid)"
        )
        ax0.set_aspect("auto")
        cv_a = (xc, yc, zc)[ia]
        cv_b = (xc, yc, zc)[ib]
        ax0.scatter(
            cv_a,
            cv_b,
            s=1,
            c="white",
            alpha=0.08,
            linewidths=0,
            rasterized=True,
        )

        im1 = ax1.pcolormesh(
            g0m,
            g1m,
            rho_pos,
            shading="auto",
            cmap="viridis",
            norm=mix_norm,
        )
        fig.colorbar(im1, ax=ax1, label=r"OPES kernel mixture (slice)")
        ax1.scatter(
            k_centers[:, ia],
            k_centers[:, ib],
            s=4,
            c="white",
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        ax1.set_xlabel(na)
        ax1.set_ylabel(nb)
        ax1.set_title(f"{na}–{nb}: Gaussian mixture @ same slice")
        ax1.set_aspect("auto")

        # ``histogram2d`` first dim is *x* bins; ``pcolormesh(..., shading='flat')``
        # expects ``C.shape == (len(Y)-1, len(X)-1)``.
        im2 = ax2.pcolormesh(
            xe_dep,
            ye_dep,
            Hplot.T,
            shading="flat",
            cmap="magma",
            norm=dep_norm,
        )
        fig.colorbar(im2, ax=ax2, label="Kernel deposit count (+0.5)")
        ax2.set_xlabel(na)
        ax2.set_ylabel(nb)
        ax2.set_title(f"{na}–{nb}: histogram of kernel centers (deposits)")
        ax2.set_aspect("auto")

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
