"""IsoPlusJVPLegacyTrainingPlan: iso plan + the legacy JVP terms.

Phase-3 stub. Reproduces the pre-R4 JVP-architecture loss bundle
(L_vector + L_anchor) on top of the iso terms so paper ablations
that compare the iso architecture against the legacy JVP architecture
remain reachable. The actual JVP-edge-residual machinery returns to
the public surface in a future iteration; for now this plan is
structurally identical to :class:`IsoTrainingPlan` with a warning
banner.

Implementation note: the JVP terms require an MLP edge head
(``edge_decoder_type='mlp'``); the iso architecture's scalar head
does not carry the necessary shape. The plan validates the model's
edge-head type and warns / errors accordingly.
"""
from __future__ import annotations

import warnings

import torch.nn as nn

from rieVAE.training.iso import IsoTrainingPlan


class IsoPlusJVPLegacyTrainingPlan(IsoTrainingPlan):
    """Phase-3 stub: iso plan with placeholders for the JVP terms."""

    def __init__(self, model: nn.Module, **kwargs) -> None:
        if getattr(model, "edge_decoder_type", "scalar") != "mlp":
            warnings.warn(
                "IsoPlusJVPLegacyTrainingPlan: model.edge_decoder_type "
                f"is {getattr(model, 'edge_decoder_type', 'scalar')!r}; "
                "the legacy JVP terms will be inactive in this plan. "
                "Construct RiemannianVAE(edge_decoder_type='mlp') to "
                "enable the JVP edge head.",
            )
        warnings.warn(
            "IsoPlusJVPLegacyTrainingPlan is a Phase-3 stub: the iso "
            "terms train normally, but the JVP-vector and JVP-anchor "
            "terms are not yet wired through the unified manifold-VAE "
            "template. Use IsoTrainingPlan for production runs.",
        )
        super().__init__(model=model, **kwargs)
