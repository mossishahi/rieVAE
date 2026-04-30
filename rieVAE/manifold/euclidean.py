"""Euclidean latent manifold M_z = R^d (the iso-architecture default)."""
from __future__ import annotations

import torch
import torch.nn as nn


class Euclidean(nn.Module):
    """Euclidean latent space R^d.

    This is the contractible-latent default of Theorem
    thm:encoder_isometry, recovering the pre-Phase-2 iso-architecture
    behaviour bit-for-bit when paired with ``Likelihood = Gaussian``
    and ``kl_mode = 'partial'``.

    Parameters
    ----------
    dim : int
        Latent dimension d.
    default_kl_mode : str
        One of 'standard' (full Gaussian KL with the mu^2 attractor),
        'partial' (mu^2 dropped; the iso default), 'flat'
        (entropy-only; pre-R4 alias of 'partial' that is
        deprecated in favour of 'partial'). The choice determines
        the implicit prior:

          - 'standard' : p(z) = N(0, I), full KL.
          - 'partial'  : translation-invariant prior; KL regularises
                         sigma only, leaves mu unconstrained (this
                         is the iso default; see main.tex sec:method).
          - 'flat'     : entropy-only KL = -0.5 * (1 + log var). The
                         manuscript reserves "flat" for flat-curvature
                         manifolds (FlatTorus); we keep this string for
                         backward compatibility but prefer 'partial'.
    """

    name = "euclidean"

    def __init__(self, dim: int, default_kl_mode: str = "partial") -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"Euclidean(dim) requires dim >= 1, got {dim}")
        if default_kl_mode not in ("standard", "partial", "flat"):
            raise ValueError(
                f"Euclidean.default_kl_mode must be 'standard', "
                f"'partial', or 'flat'; got {default_kl_mode!r}"
            )
        self.dim = int(dim)
        self.chart_dim = int(dim)
        self.decoder_input_dim = int(dim)
        self.default_kl_mode = str(default_kl_mode)

    def kl_to_prior(
        self,
        mu: torch.Tensor,
        var: torch.Tensor,
        *,
        kl_mode: str | None = None,
        free_bits: float = 0.0,
    ) -> torch.Tensor:
        mode = kl_mode if kl_mode is not None else self.default_kl_mode
        if mode == "standard":
            kl_per_dim = 0.5 * (mu.pow(2) + var - 1.0 - var.log())
            if free_bits > 0.0:
                kl_per_dim = kl_per_dim.clamp(min=float(free_bits))
        elif mode == "partial":
            kl_per_dim = 0.5 * (var - 1.0 - var.log())
        elif mode == "flat":
            kl_per_dim = -0.5 * (1.0 + var.log())
        else:
            raise ValueError(
                f"Euclidean: unknown kl_mode {mode!r}; expected "
                "'standard', 'partial', or 'flat'."
            )
        return kl_per_dim.sum(dim=-1).mean()

    def reparameterize(
        self,
        mu: torch.Tensor,
        var: torch.Tensor,
    ) -> torch.Tensor:
        if not mu.requires_grad and not var.requires_grad:
            return mu
        eps = torch.randn_like(mu)
        return mu + eps * var.sqrt()

    def distance(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> torch.Tensor:
        return (z_b - z_a).norm(dim=-1)

    def embed_for_decoder(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def __repr__(self) -> str:
        return (
            f"Euclidean(dim={self.dim}, "
            f"default_kl_mode={self.default_kl_mode!r})"
        )
