"""Configuration for weighted denoising score matching fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeightedDSMConfig:
    """Hyperparameters for weighted DSM fine-tuning of a diffusion model.

    Parameters
    ----------
    learning_rate:
        Optimizer learning rate.
    beta:
        Regularization strength toward the frozen pretrained model.
        0.0 disables regularization entirely.
    gamma:
        Weight tempering exponent in (0, 1]. ``gamma=1`` uses raw weights;
        ``gamma < 1`` flattens the weight distribution toward uniform.
    max_grad_norm:
        Gradient clipping norm.
    n_eff_min:
        Minimum effective sample size (as a fraction of batch size) below
        which a warning is emitted. Purely diagnostic — does not alter training.
    max_log_weight_ratio:
        Cap on ``max(logw) - mean(logw)`` before softmax normalization.
        Prevents a single sample from dominating the batch. Set to ``None``
        or ``inf`` to disable.
    epochs:
        Number of training epochs.
    checkpoint_every:
        Save a checkpoint every N epochs. 0 disables periodic checkpoints.
    """

    learning_rate: float = 3e-6
    beta: float = 0.01
    gamma: float = 1.0
    max_grad_norm: float = 1.0
    n_eff_min: float = 0.1
    max_log_weight_ratio: float = 10.0
    epochs: int = 10
    checkpoint_every: int = 2

    def __post_init__(self) -> None:
        if self.beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.max_grad_norm <= 0.0:
            raise ValueError(f"max_grad_norm must be > 0, got {self.max_grad_norm}")
