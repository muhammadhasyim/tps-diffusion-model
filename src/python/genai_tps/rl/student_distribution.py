"""Running KDE over Boltz collective-variable samples for FES-guided RL."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = ["BoltzStudentKDE"]


class BoltzStudentKDE:
    """Sliding-window isotropic Gaussian KDE for :math:`\\log \\hat p_{\\mathrm{Boltz}}(\\mathbf{s})`.

    Maintains the last *window* CV vectors (shape ``(D,)`` each).  The log-density
    estimate is::

        \\log \\hat p(\\mathbf{x}) = \\log \\sum_i \\mathcal{N}(\\mathbf{x}; \\mathbf{x}_i, h^2 I)
        - \\log N - \\text{const}(h, D)

    The additive normalisation constant depends only on *h* and *D*; it cancels
    when comparing two queries at fixed *h* and buffer length, but is included
    for a more interpretable scale.

    Parameters
    ----------
    ndim:
        CV dimension *D*.
    window:
        Maximum number of recent samples to retain.
    bandwidth:
        Isotropic Gaussian standard deviation *h* in CV units.  If ``None``,
        defaults to ``0.5`` (reasonable Å-scale CVs).
    log_floor:
        Returned when the buffer is empty (before any ``update``).
    """

    def __init__(
        self,
        ndim: int,
        *,
        window: int = 200,
        bandwidth: float | None = None,
        log_floor: float = -700.0,
    ) -> None:
        if ndim < 1:
            raise ValueError("ndim must be >= 1.")
        if window < 1:
            raise ValueError("window must be >= 1.")
        self.ndim = int(ndim)
        self.window = int(window)
        self.bandwidth = float(bandwidth) if bandwidth is not None else 0.5
        if self.bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive.")
        self.log_floor = float(log_floor)
        self._buf: list[np.ndarray] = []

    def clear(self) -> None:
        """Drop all stored samples."""
        self._buf.clear()

    def update(self, cv: np.ndarray | Sequence[float]) -> None:
        """Append a CV sample (shape ``(D,)``) and trim to *window*."""
        v = np.asarray(cv, dtype=np.float64).ravel()
        if v.shape != (self.ndim,):
            raise ValueError(
                f"CV has shape {v.shape}; expected ({self.ndim},) for this KDE."
            )
        if not np.all(np.isfinite(v)):
            return
        self._buf.append(v.copy())
        while len(self._buf) > self.window:
            self._buf.pop(0)

    def log_density(self, cv: np.ndarray | Sequence[float]) -> float:
        """Isotropic Gaussian KDE log-density at *cv* (unnormalised kernel sum, log domain)."""
        x = np.asarray(cv, dtype=np.float64).ravel()
        if x.shape != (self.ndim,):
            raise ValueError(
                f"CV has shape {x.shape}; expected ({self.ndim},) for this KDE."
            )
        if not np.all(np.isfinite(x)):
            return self.log_floor
        if not self._buf:
            return self.log_floor

        X = np.stack(self._buf, axis=0)  # (N, D)
        h = self.bandwidth
        d = self.ndim
        # Squared Mahalanobis distance with I/h^2
        diff = X - x
        r2 = np.sum(diff * diff, axis=1)
        log_kernel = -0.5 * r2 / (h * h)
        m = float(np.max(log_kernel))
        log_sum_w = m + math.log(float(np.sum(np.exp(log_kernel - m))) + 1e-300)
        # Normalisation: (2π h^2)^{-D/2} / N
        log_norm = -0.5 * d * math.log(2.0 * math.pi * h * h) - math.log(len(self._buf))
        return log_sum_w + log_norm
