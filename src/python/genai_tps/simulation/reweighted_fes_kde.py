"""3D reweighted KDE FES on a regular grid (COLVAR + OPES bias).

The PLUMED tutorial ``FES_from_Reweighting.py`` supports only 1D and 2D.  For
three bias CVs we evaluate the same log-weighted kernel sum on a 3D Cartesian
grid and write a PLUMED-style ``#! FIELDS cv1 cv2 cv3 file.free`` table.

This module is used by :func:`genai_tps.simulation.plumed_colvar_fes.run_fes_from_reweighting_script`
when three comma-separated CV names are requested.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# PLUMED / OPES tutorial convention (kJ/mol per Kelvin)
KBT_KJ_MOL_PER_K = 0.0083144621


def _colvar_field_names(colvar_path: Path) -> list[str]:
    with colvar_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("#! FIELDS"):
                toks = s.split()
                return toks[2:]
            if s and not s.startswith("#"):
                break
    raise ValueError(f"No #! FIELDS line found in {colvar_path}")


def _field_column_index(field_names: list[str], name: str) -> int:
    try:
        return field_names.index(name)
    except ValueError as exc:
        raise ValueError(f"CV {name!r} not in COLVAR FIELDS {field_names!r}") from exc


def _count_header_lines(colvar_path: Path) -> int:
    n = 0
    with colvar_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip().startswith("#"):
                n += 1
            else:
                break
    return max(n, 1)


def _read_colvar_columns(
    colvar_path: Path,
    col_indices: list[int],
    *,
    skiprows_after_header: int,
) -> np.ndarray:
    """Return float array shape ``(n_rows, len(col_indices))`` in *col_indices* order."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "3D FES reweighting requires pandas (same as PLUMED FES script)."
        ) from exc

    header_lines = _count_header_lines(colvar_path)
    skip = header_lines + int(skiprows_after_header)
    # PLUMED COLVAR: whitespace-separated floats; comment lines skipped by pandas
    data = pd.read_table(
        colvar_path,
        dtype=float,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=sorted(col_indices),
        skiprows=skip,
    )
    if data.isnull().values.any():
        raise ValueError(f"COLVAR contains NaNs (check last line): {colvar_path}")
    # Reorder columns to match col_indices order (not sorted)
    order = [sorted(col_indices).index(c) for c in col_indices]
    arr = np.asarray(data.iloc[:, order], dtype=np.float64)
    return arr


def write_reweighted_fes_3d(
    colvar_path: Path,
    outfile: Path,
    *,
    temperature_k: float,
    sigma: tuple[float, float, float],
    cv_names: tuple[str, str, str],
    bias_name: str = "opes.bias",
    grid_bin: tuple[int, int, int] = (40, 40, 40),
    grid_min: tuple[float, float, float] | None = None,
    grid_max: tuple[float, float, float] | None = None,
    skiprows: int = 0,
    fmt: str = "% 12.6f",
) -> None:
    """Reweighted KDE estimate of F(s1,s2,s3) on a regular 3D grid.

    For each grid point *g*, with samples indexed by *t* and bias
    :math:`V(t)/k_BT` stored in *bias* (already dimensionless if passed as
    ``opes.bias / kbt`` internally):

    .. math::

        F(g) = -k_B T \\ln \\sum_t \\exp\\Big(
            V(t) - \\sum_{d=1}^3 \\frac{(g_d - s_d(t))^2}{2\\sigma_d^2}
        \\Big)

    The global minimum is shifted to zero (PLUMED ``mintozero`` convention).

    Parameters
    ----------
    colvar_path, outfile
        Same layout as :func:`~genai_tps.simulation.plumed_colvar_fes.run_fes_from_reweighting_script`.
    temperature_k
        Simulation temperature (Kelvin).
    sigma
        Gaussian bandwidths :math:`\\sigma_1,\\sigma_2,\\sigma_3` (same units as CVs).
    cv_names
        Three names matching the COLVAR ``#! FIELDS`` line.
    bias_name
        Bias column name(s), comma-separated; multiple columns are summed like
        the PLUMED tutorial script (e.g. ``opes.bias``).
    grid_bin
        Number of **interior** bins per axis; grid edges use ``bins+1`` linspace
        endpoints like the 2D PLUMED script (then endpoints may be dropped if
        periodic — not used here).
    grid_min, grid_max
        Optional axis bounds; default: min/max of each CV in the COLVAR sample.
    skiprows
        Data rows to skip after the PLUMED header (burn-in).
    """
    colvar_path = Path(colvar_path).expanduser().resolve()
    outfile = Path(outfile).expanduser().resolve()
    if colvar_path.parent.resolve() != outfile.parent.resolve():
        raise ValueError(
            f"outfile parent must match COLVAR directory ({colvar_path.parent}); "
            f"got {outfile.parent}"
        )

    field_names = _colvar_field_names(colvar_path)
    ix = _field_column_index(field_names, cv_names[0])
    iy = _field_column_index(field_names, cv_names[1])
    iz = _field_column_index(field_names, cv_names[2])
    bias_labels = [b.strip() for b in bias_name.split(",") if b.strip()]
    ibs = [_field_column_index(field_names, bn) for bn in bias_labels]
    col_order = [ix, iy, iz] + ibs
    arr = _read_colvar_columns(colvar_path, col_order, skiprows_after_header=skiprows)
    cv_x = arr[:, 0]
    cv_y = arr[:, 1]
    cv_z = arr[:, 2]
    bias_raw = np.sum(arr[:, 3:], axis=1)
    bias_dimless = bias_raw / (float(temperature_k) * KBT_KJ_MOL_PER_K)

    sx, sy, sz = (max(float(s), 1e-12) for s in sigma)
    kbt = float(temperature_k) * KBT_KJ_MOL_PER_K

    def _axis_grid(
        v: np.ndarray, nbin: int, lo: float | None, hi: float | None
    ) -> tuple[np.ndarray, int]:
        nbin = int(nbin)
        if nbin < 2:
            raise ValueError("grid_bin components must be >= 2")
        lo_v = float(v.min()) if lo is None else float(lo)
        hi_v = float(v.max()) if hi is None else float(hi)
        edges = np.linspace(lo_v, hi_v, nbin + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, nbin

    gx, nx = _axis_grid(cv_x, grid_bin[0], grid_min[0] if grid_min else None, grid_max[0] if grid_max else None)
    gy, ny = _axis_grid(cv_y, grid_bin[1], grid_min[1] if grid_min else None, grid_max[1] if grid_max else None)
    gz, nz = _axis_grid(cv_z, grid_bin[2], grid_min[2] if grid_min else None, grid_max[2] if grid_max else None)

    fes = np.empty((nx, ny, nz), dtype=np.float64)
    for i, px in enumerate(gx):
        dx = (px - cv_x) / sx
        base_xy = bias_dimless - 0.5 * dx * dx
        for j, py in enumerate(gy):
            dy = (py - cv_y) / sy
            base = base_xy - 0.5 * dy * dy
            for k, pz in enumerate(gz):
                dz = (pz - cv_z) / sz
                arg = base - 0.5 * dz * dz
                fes[i, j, k] = -kbt * np.logaddexp.reduce(arg)

    fes -= float(np.min(fes))

    outfile.parent.mkdir(parents=True, exist_ok=True)
    xg, yg, zg = np.meshgrid(gx, gy, gz, indexing="ij")
    with outfile.open("w", encoding="utf-8") as fh:
        fields = (
            f"#! FIELDS {cv_names[0]} {cv_names[1]} {cv_names[2]} file.free\n"
        )
        fh.write(fields)
        fh.write(f"#! SET genai_grid_bins {nx} {ny} {nz}\n")
        fh.write(f"#! SET sample_size {cv_x.shape[0]}\n")
        weights = np.exp(bias_dimless - np.max(bias_dimless))
        effsize = float(np.sum(weights) ** 2 / np.sum(weights**2))
        fh.write(f"#! SET effective_sample_size {effsize:g}\n")
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    line = (fmt + " " + fmt + " " + fmt + "  " + fmt + "\n") % (
                        float(xg[i, j, k]),
                        float(yg[i, j, k]),
                        float(zg[i, j, k]),
                        float(fes[i, j, k]),
                    )
                    fh.write(line)
                fh.write("\n")
