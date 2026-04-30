"""IsoPlusGlobalOrderTrainingPlan: iso plan + RankNet global ordinal loss.

Adds a fourth term L_rank that enforces ordinal isometry on randomly
sampled global pairs (3sum.md Section 3.1). The rank loss uses the
global-ordinal oracle ``psi_full`` (precomputed by
:class:`SpectralPreprocessor`) rather than the encoder's PE features,
since the oracle has Spearman ~0.99 with d^M; driving L_rank -> 0
implies ordinal isometry globally.

Activated only after rho >= ``global_order_warmup_frac`` so L_iso has
established the local scale before ranking is enforced.
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn

from rieVAE.training._base import Term, warmup_then_constant
from rieVAE.training._terms import global_ordinal_term_factory
from rieVAE.training.iso import IsoTrainingPlan


class IsoPlusGlobalOrderTrainingPlan(IsoTrainingPlan):
    """IsoTrainingPlan + RankNet global-ordinal loss term."""

    def __init__(
        self,
        model: nn.Module,
        global_order_eta: float = 0.1,
        global_order_k_near: int = 5,
        global_order_k_far: int = 16,
        global_order_batch: int = 128,
        global_order_warmup_frac: float = 0.2,
        **iso_kwargs,
    ) -> None:
        super().__init__(model=model, **iso_kwargs)
        rank_term = global_ordinal_term_factory(
            k_near=int(global_order_k_near),
            k_far=int(global_order_k_far),
            batch_size=int(global_order_batch),
        )
        self.add_term(
            Term(
                name="rank",
                fn=rank_term,
                schedule=warmup_then_constant(
                    target=global_order_eta,
                    warmup_frac=global_order_warmup_frac,
                ),
            )
        )
