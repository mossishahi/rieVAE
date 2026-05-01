"""IsoTrainingPlan: the iso-architecture default (op47C C.3.3).

Three-term objective:

    L = L_rec/s_rec + beta(t) * L_KL/s_kl + gamma(t) * L_iso/s_iso

with the manifold-aware reconstruction (``-likelihood.log_prob``),
manifold-aware KL (``manifold.kl_to_prior``), and manifold-aware iso
loss (squared difference of ``manifold.distance(mu_i, mu_j)`` and the
spectral target).

This is the post-Phase-2 manifold-VAE template applied to the iso
architecture. Setting ``LatentManifold = Euclidean(d, prior='partial')``
and ``Likelihood = Gaussian`` recovers the Phase-1 / Phase-2
``IsoVAELoss`` byte-for-byte (regression-tested in the Phase-3 smoke).

Defaults match the paper's Algorithm (method_overview.tex, Sec. 4-5):
  - decoder_lr_scale = 0.1  (two-timescale: decoder 10x slower than encoder)
  - grad_clip_norm   = 1.0  (global L2-norm clip before every step)
  - weight_decay     = 1e-4 (AdamW decoupled weight decay)
The two-timescale design prevents the decoder from converging on
reconstruction before the encoder has organized the latent geometry;
see method_overview.tex Sec. 4 for the rationale.
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn

from rieVAE.training._base import (
    Term, TrainingPlanBase,
    constant, sigmoid, beta_linear_decay,
)
from rieVAE.training._terms import (
    likelihood_recon_term, manifold_kl_term, iso_term,
)


class IsoTrainingPlan(TrainingPlanBase):
    """Iso-architecture training plan (manifold-aware ELBO + iso term)."""

    def __init__(
        self,
        model: nn.Module,
        beta: float = 0.01,
        beta_min: float = 0.0,
        beta_linear_decay_on: bool = False,
        gamma_max: float = 1.0,
        gamma_sigmoid_k: float = 8.0,
        gamma_sigmoid_center: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        decoder_lr_scale: float = 0.1,
        grad_clip_norm: float = 1.0,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 1e-6,
        lr_warmup_steps: int = 0,
        max_steps: int = 50_000,
        use_initial_scale_norm: bool = True,
        scale_eps: float = 1e-6,
    ) -> None:
        # KL schedule: constant beta or linear-decay from beta -> beta_min.
        if beta_linear_decay_on:
            kl_schedule = beta_linear_decay(beta, beta_min)
        else:
            kl_schedule = constant(beta)

        # Iso schedule: sigmoid ramp 0 -> gamma_max with paper's k=8, c=0.2.
        # (Activates early because tilde_w is decoder-independent and a
        # valid target from step zero; cf. main.tex Algorithm
        # alg:training, gamma(t) discussion.)
        iso_schedule = sigmoid(
            target=gamma_max,
            k=gamma_sigmoid_k,
            center=gamma_sigmoid_center,
        )

        terms = [
            Term("rec", likelihood_recon_term, schedule=constant(1.0)),
            Term("kl",  manifold_kl_term,      schedule=kl_schedule),
            Term("iso", iso_term,              schedule=iso_schedule),
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
