"""Parse PLUMED FES ``.dat``, ``COLVAR``, and OPES ``KERNELS`` files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

__all__ = [
    "load_plumed_fes_dat",
    "load_plumed_colvar_two_cvs",
    "load_plumed_colvar_three_cvs",
    "load_plumed_kernels_2d",
    "load_plumed_kernels_3d",
    "parse_genai_grid_bins_3d",
]


def parse_genai_grid_bins_3d(text: str) -> tuple[int, int, int] | None:
    """Read ``#! SET genai_grid_bins nx ny nz`` from a FES ``.dat`` header."""
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
