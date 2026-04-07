"""RL fine-tuning utilities for Boltz-2 diffusion (RLDiff-style DDPO-IS surrogate).

This subpackage implements an offline clipped surrogate and Boltz-specific
importance weights for trajectory replay. It does **not** depend on the
third-party ``posebusters`` package; rewards use :mod:`genai_tps.analysis.posebusters_gpu`.
"""

from genai_tps.rl.config import BoltzRLConfig, FESTeacherConfig
from genai_tps.rl.ppo_surrogate import compute_ppo_loss, normalize_rewards_per_trajectory
from genai_tps.rl.rollout import BoltzRolloutStep, rollout_forward_trajectory

__all__ = [
    "BoltzRLConfig",
    "BoltzRolloutStep",
    "FESTeacherConfig",
    "compute_ppo_loss",
    "normalize_rewards_per_trajectory",
    "rollout_forward_trajectory",
]
