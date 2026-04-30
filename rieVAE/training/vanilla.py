"""VanillaTrainingPlan: ELBO with no iso term.

Two-term objective:

    L = L_rec / s_rec + beta(t) * L_KL / s_kl

The L_iso term is dropped entirely. Used as the baseline against
which the iso architecture's certified isometry is benchmarked.
This plan replaces the deleted ``rieVAE/train/vanilla_trainer.py``.
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn

from rieVAE.training._base import (
    Term, TrainingPlanBase, constant, beta_linear_decay,
)
from rieVAE.training._terms import likelihood_recon_term, manifold_kl_term


class VanillaTrainingPlan(TrainingPlanBase):
    """Standard ELBO (likelihood + KL) with no isometry regulariser."""

    def __init__(
        self,
        model: nn.Module,
        beta: float = 0.01,
        beta_min: float = 0.0,
        beta_linear_decay_on: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        decoder_lr_scale: float = 1.0,
        grad_clip_norm: float = 0.0,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 1e-6,
        lr_warmup_steps: int = 0,
        max_steps: int = 50_000,
        use_initial_scale_norm: bool = True,
        scale_eps: float = 1e-6,
    ) -> None:
        if beta_linear_decay_on:
            kl_schedule = beta_linear_decay(beta, beta_min)
        else:
            kl_schedule = constant(beta)

        terms = [
            Term("rec", likelihood_recon_term, schedule=constant(1.0)),
            Term("kl",  manifold_kl_term,      schedule=kl_schedule),
        ]
        super().__init__(
            model=model,
            terms=terms,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            decoder_lr_scale=decoder_lr_scale,
            grad_clip_norm=grad_clip_norm,
            lr_scheduler=lr_scheduler,
            lr_min=lr_min,
            lr_warmup_steps=lr_warmup_steps,
            max_steps=max_steps,
            use_initial_scale_norm=use_initial_scale_norm,
            scale_eps=scale_eps,
        )
