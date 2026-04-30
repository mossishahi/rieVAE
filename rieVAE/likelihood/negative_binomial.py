"""Negative-binomial observation likelihood for non-negative integer
features with overdispersion.

A generic count-data plug-in. The decoder emits a per-sample
unnormalised mean rho (passed through softmax over features so it
sums to 1); the per-sample mean is then ``mean = scale_factor * rho``
where ``scale_factor`` is the GLM-standard offset / exposure
(typically the per-sample row total in count tables; we treat it as
1 when not supplied).

The NB(mean, theta) parameterisation used here is the one supported
by ``torch.distributions.NegativeBinomial`` after converting from
mean / dispersion to total_count / probs:

    total_count = theta
    probs       = mean / (mean + theta)        (i.e., 1 - p in some texts)
    p(x | mean, theta) = NB(x ; total_count, probs)

The dispersion theta is learnable. Three modes:

  - 'feature' (default): one theta per feature, shared across samples
  - 'sample-feature'  : per-(sample, feature) theta predicted by the
                        decoder (requires n_decoder_outputs_per_feature = 2)
  - 'constant'        : a single global theta scalar
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NegativeBinomial(nn.Module):
    """Generic negative-binomial likelihood for overdispersed counts.

    Parameters
    ----------
    n_features : int
        Number of features G.
    dispersion : str
        'feature' (default), 'sample-feature', or 'constant'.
    init_theta : float
        Initial dispersion (used for 'feature' and 'constant'; ignored
        for 'sample-feature' where theta is decoder-predicted).
    eps : float
        Small constant added inside log() and divisions for numerical
        stability.
    """

    name = "negative_binomial"
    requires_scale_factor = True

    def __init__(
        self,
        n_features: int,
        dispersion: str = "feature",
        init_theta: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError(
                f"NegativeBinomial(n_features) requires n_features >= 1, "
                f"got {n_features}"
            )
        if dispersion not in ("feature", "sample-feature", "constant"):
            raise ValueError(
                f"NegativeBinomial.dispersion must be one of "
                f"'feature' / 'sample-feature' / 'constant'; "
                f"got {dispersion!r}"
            )
        if init_theta <= 0.0:
            raise ValueError(
                f"NegativeBinomial.init_theta requires init_theta > 0, "
                f"got {init_theta}"
            )
        self.n_features = int(n_features)
        self.dispersion = str(dispersion)
        self.eps = float(eps)
        # n_decoder_outputs_per_feature: 1 for 'feature' / 'constant'
        # (only mean is decoder-predicted), 2 for 'sample-feature'
        # (mean + log_theta per (sample, feature)).
        if self.dispersion == "sample-feature":
            self.n_decoder_outputs_per_feature = 2
            self.log_theta = None  # decoder-predicted
        elif self.dispersion == "feature":
            self.n_decoder_outputs_per_feature = 1
            self.log_theta = nn.Parameter(
                torch.full((int(n_features),), float(torch.log(torch.tensor(init_theta))))
            )
        else:  # constant
            self.n_decoder_outputs_per_feature = 1
            self.log_theta = nn.Parameter(
                torch.tensor(float(torch.log(torch.tensor(init_theta))))
            )

    def parse(
        self,
        decoder_out: torch.Tensor,
        *,
        scale_factor: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        # decoder_out shape:
        #   (B, n_features) for 'feature' / 'constant'
        #   (B, 2 * n_features) for 'sample-feature'
        if self.dispersion == "sample-feature":
            mean_logits = decoder_out[..., : self.n_features]
            log_theta_per = decoder_out[..., self.n_features:]
            theta = log_theta_per.exp().clamp(min=self.eps)
        else:
            mean_logits = decoder_out
            theta = self.log_theta.exp().clamp(min=self.eps)
        # Normalise the predicted means via softmax over features so
        # that the un-scaled mean is a probability vector summing to 1.
        rho = torch.softmax(mean_logits, dim=-1)
        # Apply the GLM offset; default to a uniform scale of
        # n_features when not supplied (so the mean is on the same
        # scale as the input even without explicit normalisation --
        # callers with proper count totals should supply scale_factor).
        if scale_factor is None:
            scale = mean_logits.new_ones(mean_logits.shape[:-1] + (1,)) * float(self.n_features)
        else:
            scale = scale_factor.to(mean_logits.device, mean_logits.dtype)
            if scale.dim() == mean_logits.dim() - 1:
                scale = scale.unsqueeze(-1)
        mean = rho * scale
        return {"mean": mean, "theta": theta, "rho": rho, "scale": scale}

    def log_prob(
        self,
        x: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        mean = params["mean"]
        theta = params["theta"]
        # Use torch.distributions for numerical robustness; convert
        # (mean, theta) -> (total_count, probs).
        eps = self.eps
        total_count = theta
        probs = (mean + eps) / (mean + theta + 2 * eps)
        dist = torch.distributions.NegativeBinomial(
            total_count=total_count.expand_as(mean),
            probs=probs,
        )
        return dist.log_prob(x.float())

    def expected_value(
        self,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return params["mean"]

    def sample(
        self,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        mean = params["mean"]
        theta = params["theta"]
        total_count = theta.expand_as(mean)
        probs = (mean + self.eps) / (mean + theta + 2 * self.eps)
        dist = torch.distributions.NegativeBinomial(total_count=total_count, probs=probs)
        return dist.sample()

    def __repr__(self) -> str:
        return (
            f"NegativeBinomial(n_features={self.n_features}, "
            f"dispersion={self.dispersion!r})"
        )
