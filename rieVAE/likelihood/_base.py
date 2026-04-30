"""Observation-likelihood protocol for the Certified Riemannian VAE.

Phase-2 unification (op47C C.2): the reconstruction term in the
manifold-VAE template

    L_rec = -E_{q_phi(z|x)}[log p_theta(x | z)]

is supplied by a ``Likelihood`` plug-in. The plug-in:

  - declares how many decoder output channels per feature it needs
    (``n_decoder_outputs_per_feature``);
  - parses the decoder's raw output into named distribution
    parameters (``parse``);
  - evaluates the per-sample log-probability of the observed x
    (``log_prob``);
  - exposes the expected value E[X | params] used by the C2
    reconstruction residual delta_rec = sup_i ||E[X | mu_i] - x_i||
    (``expected_value``);
  - samples from the distribution (``sample``).

The likelihood is decoupled from the manifold: the same Likelihood
class works with any LatentManifold. Concrete classes shipping in
this package are:

  - ``Gaussian``                         (default; recovers MSE reconstruction)
  - ``NegativeBinomial``                 (overdispersed counts)
  - ``ZeroInflatedNegativeBinomial``     (counts with extra zeros)
  - ``Poisson``                          (counts)
  - ``Bernoulli``                        (binary)

The names are general (no scRNA-domain identifiers); ``scale_factor``
is the GLM-standard offset / exposure that count likelihoods accept
when present, and is used for any features whose totals vary per
sample (count tabular data, image patch normalisation, document
term frequencies, etc.).
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import torch


@runtime_checkable
class Likelihood(Protocol):
    """Observation-likelihood protocol."""

    name: str
    """Lower-case identifier, used by the registry."""

    n_decoder_outputs_per_feature: int
    """Number of decoder output channels per feature. The decoder's
    last layer outputs ``n_features * n_decoder_outputs_per_feature``
    raw values; ``parse`` then groups them into named distribution
    parameters."""

    requires_scale_factor: bool
    """If True, ``parse`` requires a per-sample ``scale_factor`` (the
    GLM offset / exposure) to be supplied; if None is supplied, the
    likelihood falls back to a uniform scale of 1. False for
    likelihoods where scale is implicit in the parameters
    (e.g., Gaussian)."""

    def parse(
        self,
        decoder_out: torch.Tensor,
        *,
        scale_factor: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Parse the decoder's raw output into distribution parameters.

        Parameters
        ----------
        decoder_out : (B, n_features * n_decoder_outputs_per_feature)
            Raw decoder output.
        scale_factor : (B,) or None
            Per-sample scale (e.g., total counts for an NB / Poisson
            count vector). Ignored when ``requires_scale_factor`` is
            False. When None, defaults to a uniform scale of 1.

        Returns
        -------
        dict mapping parameter names to tensors of shape (B, n_features)
        (or whatever the distribution requires).
        """
        ...

    def log_prob(
        self,
        x: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Per-sample log-probability log p(x | params).

        Parameters
        ----------
        x : (B, n_features)
        params : output of ``parse``

        Returns
        -------
        log_p : (B, n_features) or (B,)
            Per-feature or per-sample log-probabilities. Higher is
            better; the reconstruction loss is the negation reduced
            over features and averaged over the batch.
        """
        ...

    def expected_value(
        self,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """E[X | params]. Used to compute the C2 reconstruction residual
        ``delta_rec = sup_i ||E[X | mu_i] - x_i||``."""
        ...

    def sample(
        self,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Sample from the distribution at the given parameters."""
        ...
