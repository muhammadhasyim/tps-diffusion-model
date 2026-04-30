"""Grid math, KDE helpers, and marginal masking for PLUMED FES visualization."""

from __future__ import annotations

import numpy as np

__all__ = [
    "integrate_trapezoid_1d",
    "integrate_trapezoid_2d",
    "pdf_on_grid_1d",
    "pdf_on_mesh_2d",
    "kernel_density_grid_2d",
    "kernel_mixture_slice_3d",
    "cell_edges_from_centers",
    "mask_marginal_fes_far_from_colvar",
]


def mask_marginal_fes_far_from_colvar(
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


def cell_edges_from_centers(c: np.ndarray) -> np.ndarray:
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


def kernel_mixture_slice_3d(
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


def integrate_trapezoid_1d(y: np.ndarray, x: np.ndarray) -> float:
    """∫ y(x) dx with NumPy 1.x/2.x compatible API."""
    ya = np.asarray(y, dtype=np.float64)
    xa = np.asarray(x, dtype=np.float64)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(ya, xa))
    return float(np.trapz(ya, xa))


def pdf_on_grid_1d(grid: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Divide *density* by ∫ density dx on *grid* so the result integrates to ~1."""
    y = np.asarray(density, dtype=np.float64)
    g = np.asarray(grid, dtype=np.float64)
    integ = integrate_trapezoid_1d(y, g)
    if integ <= 0.0 or not np.isfinite(integ):
        return np.zeros_like(y)
    return y / integ


def integrate_trapezoid_2d(Z: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """∬ Z(x,y) dx dy with *Z* shape ``(len(y), len(x))`` (``meshgrid(..., indexing='xy')``)."""
    zx = Z.astype(np.float64, copy=False)
    if hasattr(np, "trapezoid"):
        inner = np.trapezoid(zx, x, axis=1)
        return float(np.trapezoid(inner, y))
    inner = np.trapz(zx, x, axis=1)
    return float(np.trapz(inner, y))


def pdf_on_mesh_2d(Z: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Divide *Z* by ∬ Z dx dy on the tensor-product grid so the integral is ~1."""
    zz = np.asarray(Z, dtype=np.float64)
    integ = integrate_trapezoid_2d(zz, x, y)
    if integ <= 0.0 or not np.isfinite(integ):
        return np.zeros_like(zz)
    return zz / integ


def kernel_density_grid_2d(
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
