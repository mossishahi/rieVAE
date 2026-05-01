"""Poisson observation likelihood for non-negative integer features."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class Poisson(nn.Module):
    """Generic Poisson likelihood for count data.

    The decoder emits per-feature log-rates; the per-sample rate is

        rate_i = scale_factor * softmax(decoder_out)_i

    where ``scale_factor`` is the GLM-standard offset / exposure
    (defaults to ``n_features`` when not supplied, so the rate is on
    the same scale as the input even without explicit normalisation).

    Parameters
    ----------
    n_features : int
    eps : float
        Small constant added to ``rate`` for numerical stability.
    """

    name = "poisson"
    n_decoder_outputs_per_feature = 1
    requires_scale_factor = True

    def __init__(self, n_features: int, eps: float = 1e-8) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError(
                f"Poisson(n_features) requires n_features >= 1, got {n_features}"
            )
        self.n_features = int(n_features)
        self.eps = float(eps)

    def parse(
        self,
        decoder_out: torch.Tensor,
        *,
        scale_factor: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        rho = torch.softmax(decoder_out, dim=-1)
        if scale_factor is None:
            scale = decoder_out.new_ones(decoder_out.shape[:-1] + (1,)) * float(self.n_features)
        else:
            scale = scale_factor.to(decoder_out.device, decoder_out.dtype)
            if scale.dim() == decoder_out.dim() - 1:
                scale = scale.unsqueeze(-1)
        rate = (rho * scale).clamp(min=self.eps)
        return {"rate": rate, "rho": rho, "scale": scale}

    def log_prob(
        self,
        x: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        rate = params["rate"]
        dist = torch.distributions.Poisson(rate=rate)
        return dist.log_prob(x.float())

    def expected_value(
        self,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return params["rate"]

    def sample(
        self,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        rate = params["rate"]
        return torch.distributions.Poisson(rate=rate).sample()

    def __repr__(self) -> str:
        return f"Poisson(n_features={self.n_features})"
