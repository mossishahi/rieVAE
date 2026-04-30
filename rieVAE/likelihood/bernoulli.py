"""Bernoulli observation likelihood for binary features."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bernoulli(nn.Module):
    """Bernoulli (logistic) likelihood for binary features in {0, 1}.

    p(x_i | params) = sigmoid(logits_i) ** x_i * (1 - sigmoid(logits_i)) ** (1 - x_i).

    Parameters
    ----------
    n_features : int
    """

    name = "bernoulli"
    n_decoder_outputs_per_feature = 1
    requires_scale_factor = False

    def __init__(self, n_features: int) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError(
                f"Bernoulli(n_features) requires n_features >= 1, got {n_features}"
            )
        self.n_features = int(n_features)

    def parse(
        self,
        decoder_out: torch.Tensor,
        *,
        scale_factor: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        return {"logits": decoder_out}

    def log_prob(
        self,
        x: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = params["logits"]
        # PyTorch's BCEWithLogits is numerically robust; use it
        # element-wise (without reduction).
        return -F.binary_cross_entropy_with_logits(
            logits, x.float(), reduction="none",
        )

    def expected_value(
        self,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return torch.sigmoid(params["logits"])

    def sample(
        self,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        probs = torch.sigmoid(params["logits"])
        return torch.bernoulli(probs)

    def __repr__(self) -> str:
        return f"Bernoulli(n_features={self.n_features})"
