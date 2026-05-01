"""Gaussian observation likelihood (the iso-architecture default)."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class Gaussian(nn.Module):
    """Gaussian observation likelihood with isotropic variance.

    p(x_i | params) = N(x_i ; mean_i, sigma^2 I)

    With the default ``sigma = 1`` and ``learn_sigma = False``, the
    reconstruction term reduces to MSE up to a constant (recovering
    the Phase-1 iso-architecture loss bit-for-bit).

    Parameters
    ----------
    n_features : int
        Number of features G. Required so that ``learn_sigma=True``
        can size its parameter; ignored when ``learn_sigma=False``.
    sigma : float
        Initial / fixed observation standard deviation per feature.
        Default 1.
    learn_sigma : bool
        If True, sigma becomes a learnable per-feature parameter
        (initialised at the supplied ``sigma``).
    """

    name = "gaussian"
    n_decoder_outputs_per_feature = 1
    requires_scale_factor = False

    def __init__(
        self,
        n_features: int,
        sigma: float = 1.0,
        learn_sigma: bool = False,
    ) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError(
                f"Gaussian(n_features) requires n_features >= 1, got {n_features}"
            )
        if sigma <= 0.0:
            raise ValueError(
                f"Gaussian(sigma) requires sigma > 0, got {sigma}"
            )
        self.n_features = int(n_features)
        if learn_sigma:
            self.log_sigma = nn.Parameter(
                torch.full((int(n_features),), math.log(sigma))
            )
        else:
            self.register_buffer(
                "log_sigma_buf",
                torch.full((int(n_features),), math.log(sigma)),
            )

    @property
    def _log_sigma(self) -> torch.Tensor:
        if hasattr(self, "log_sigma"):
            return self.log_sigma
        return self.log_sigma_buf

    def parse(
        self,
        decoder_out: torch.Tensor,
        *,
        scale_factor: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        # decoder_out has shape (B, n_features * 1) = (B, n_features).
        return {"mean": decoder_out}

    def log_prob(
        self,
        x: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        mean = params["mean"]
        log_sigma = self._log_sigma.to(mean.device, mean.dtype)
        sigma2 = (2.0 * log_sigma).exp()
        # log N(x; mean, sigma^2) = -0.5 (x - mean)^2 / sigma^2
        #                          - 0.5 log(2 pi sigma^2)
        sq_err = (x - mean).pow(2)
        log_p = -0.5 * (sq_err / sigma2 + math.log(2.0 * math.pi) + 2.0 * log_sigma)
        return log_p

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
        log_sigma = self._log_sigma.to(mean.device, mean.dtype)
        sigma = log_sigma.exp()
        eps = torch.randn_like(mean)
        return mean + eps * sigma

    def __repr__(self) -> str:
        learnable = isinstance(getattr(self, "log_sigma", None), nn.Parameter)
        return f"Gaussian(n_features={self.n_features}, learn_sigma={learnable})"
