"""Hyperparameters for Boltz RL fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BoltzRLConfig", "FESTeacherConfig"]


@dataclass(frozen=True)
class FESTeacherConfig:
    """FES-guided RL: OpenMM+OPES teacher bursts and student KDE / advantage."""

    md_steps_per_burst: int = 2000
    md_deposit_pace: int = 10
    boltz_rollouts_per_iter: int = 8
    n_iters: int = 1000
    opes_barrier: float = 5.0
    opes_biasfactor: float = 10.0
    opes_kbt: float = 2.494
    student_kde_window: int = 200
    student_kde_bandwidth: float | None = None
    advantage_clip: float = 5.0
    teacher_minimize_steps: int = 0
    pocket_radius: float = 6.0

    # Bidirectional loop: generative deposits into shared OPES
    generative_deposit_weight: float = 0.2
    # Bidirectional loop: warm-start MD from Boltz structures
    disagreement_warmstart: bool = False
    warmstart_minimize_steps: int = 200
    warmstart_fraction: float = 0.5


@dataclass(frozen=True)
class BoltzRLConfig:
    """Configuration mirroring key RLDiff / DDPO-IS knobs (see RLDiff ``train_utils``)."""

    learning_rate: float = 1e-5
    clip_range: float = 0.2
    max_grad_norm: float = 1.0
    # Early diffusion steps: cap surrogate magnitude (RLDiff-style).
    no_early_step_guidance: bool = True
    early_step_cap: float = 0.03
    early_step_end: int = 10
    alpha_step: int | None = None
    # Surrogate Gaussian variance for denoiser-velocity IS ratio (see boltz_likelihood).
    velocity_log_prob_tau_sq: float = 1e-2
    log_importance_clip: float = 20.0
