"""RL fine-tuning utilities for Boltz-2 diffusion (RLDiff-style DDPO-IS surrogate).

This subpackage implements an offline clipped surrogate and Boltz-specific
importance weights for trajectory replay. It does **not** depend on the
third-party ``posebusters`` package; rewards use :mod:`genai_tps.analysis.posebusters_gpu`.
"""

from genai_tps.rl.boltz_likelihood import (
    denoiser_velocity_importance_weight,
    forward_step_meta_tensor,
)
from genai_tps.rl.config import BoltzRLConfig, FESTeacherConfig
from genai_tps.rl.fes_teacher import OpenMMTeacher, boltz_terminal_pose_cv_numpy
from genai_tps.rl.ppo_surrogate import compute_ppo_loss, normalize_rewards_per_trajectory
from genai_tps.rl.reward_gpu import gpu_pass_fraction_reward_from_coords
from genai_tps.rl.rollout import BoltzRolloutStep, rollout_forward_trajectory
from genai_tps.rl.student_distribution import BoltzStudentKDE
from genai_tps.rl.training import fes_guided_trajectory_loss, replay_trajectory_loss

__all__ = [
    "BoltzRLConfig",
    "BoltzRolloutStep",
    "BoltzStudentKDE",
    "FESTeacherConfig",
    "OpenMMTeacher",
    "boltz_terminal_pose_cv_numpy",
    "compute_ppo_loss",
    "denoiser_velocity_importance_weight",
    "forward_step_meta_tensor",
    "fes_guided_trajectory_loss",
    "gpu_pass_fraction_reward_from_coords",
    "normalize_rewards_per_trajectory",
    "replay_trajectory_loss",
    "rollout_forward_trajectory",
]
