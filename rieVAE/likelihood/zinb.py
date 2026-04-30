"""Zero-inflated negative-binomial likelihood for non-negative integer
features with both overdispersion and excess zeros.

Three decoder output channels per feature: the NB mean logits, the
NB log-dispersion logits (when ``dispersion='sample-feature'``;
otherwise this slot is unused), and the dropout (zero-inflation)
logits which are passed through a sigmoid.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rieVAE.likelihood.negative_binomial import NegativeBinomial


class ZeroInflatedNegativeBinomial(nn.Module):
    """Zero-inflated NB likelihood.

    Density:
        p(x | mean, theta, pi) = pi * delta_0(x)
                                 + (1 - pi) * NB(x; mean, theta).

    Parameters
    ----------
    n_features : int
    dispersion : str  -- 'feature', 'sample-feature', or 'constant'
                         (forwarded to the inner NB).
    init_theta : float
    eps : float
    """

    name = "zinb"
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
                f"ZINB(n_features) requires n_features >= 1, got {n_features}"
            )
        self.n_features = int(n_features)
        # The inner NB owns dispersion bookkeeping; we add one extra
        # decoder output per feature for the dropout logit.
        self.nb = NegativeBinomial(
            n_features=int(n_features),
            dispersion=dispersion,
            init_theta=init_theta,
            eps=eps,
        )
        self.n_decoder_outputs_per_feature = (
            self.nb.n_decoder_outputs_per_feature + 1
        )
        self.eps = float(eps)
        self.dispersion = str(dispersion)

    def parse(
        self,
        decoder_out: torch.Tensor,
        *,
        scale_factor: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        # decoder_out has shape
        #   (B, n_features * (n_dec_per_feat_NB + 1))
        # Split into NB-side and dropout-side.
        n_nb = self.nb.n_decoder_outputs_per_feature * self.n_features
        nb_out = decoder_out[..., :n_nb]
        dropout_logits = decoder_out[..., n_nb : n_nb + self.n_features]
        nb_params = self.nb.parse(nb_out, scale_factor=scale_factor)
        pi = torch.sigmoid(dropout_logits).clamp(min=self.eps, max=1.0 - self.eps)
        return {**nb_params, "pi": pi, "dropout_logits": dropout_logits}

    def log_prob(
        self,
        x: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # log p_zinb(x) = log pi    + log delta_0(x)               if x == 0
        #                = log (1 - pi) + log p_nb(x)               if x >  0
        #
        # Equivalently:
        #   x == 0 :  logsumexp( log_pi, log(1-pi) + log_p_nb(0) )
        #   x  > 0 :  log(1 - pi) + log_p_nb(x)
        pi = params["pi"]
        log_pi = (pi + self.eps).log()
        log_1m_pi = ((1.0 - pi) + self.eps).log()

        log_p_nb = self.nb.log_prob(x, params)

        # NB log-prob at x = 0 (vectorised; we just call nb.log_prob with
        # zeros and pick out the per-feature value -- shape-compatible).
        zero_x = torch.zeros_like(x)
        log_p_nb_at_0 = self.nb.log_prob(zero_x, params)

        is_zero = (x < 0.5)
        log_p_zero_branch = torch.logsumexp(
            torch.stack([log_pi, log_1m_pi + log_p_nb_at_0], dim=-1),
            dim=-1,
        )
        log_p_pos_branch = log_1m_pi + log_p_nb
        return torch.where(is_zero, log_p_zero_branch, log_p_pos_branch)

    def expected_value(
        self,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # E[X] = (1 - pi) * mean_NB
        mean_nb = self.nb.expected_value(params)
        pi = params["pi"]
        return (1.0 - pi) * mean_nb

    def sample(
        self,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        nb_sample = self.nb.sample(params)
        pi = params["pi"]
        # Bernoulli mask: 1 = drop to zero with prob pi.
        mask = torch.bernoulli(pi).to(nb_sample.dtype)
        return nb_sample * (1.0 - mask)

    def __repr__(self) -> str:
        return (
            f"ZeroInflatedNegativeBinomial(n_features={self.n_features}, "
            f"dispersion={self.dispersion!r})"
        )
