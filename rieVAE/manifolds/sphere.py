"""Round sphere latent manifold M_z = S^d."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class Sphere(nn.Module):
    """Round sphere M_z = S^d, parameterised in chart R^d as a tangent
    coordinate at the north pole p0 = (0, ..., 0, 1) in R^{d+1}, with
    the exponential map applied at decoder-input time.

    Concretely, given chart coordinates z in R^d:

        v = (z[0], ..., z[d-1], 0) in R^{d+1}      (tangent at p0)
        Sphere(v) = cos(||z||) p0 + sin(||z||) v / ||z||
                                                       (exp_p0(v) in S^d)

    This is the standard "tangent at the pole" chart used by the
    Hyperspherical VAE family (Davidson et al. 2018) without the
    vMF parameterisation; under op47C option (ii) we keep the
    posterior Gaussian on the chart and let ``embed_for_decoder``
    handle the exp-map. The geodesic distance is the great-circle
    angle between the two exp-mapped points.

    KL is taken against the uniform prior on S^d (translation
    invariant under the rotation group), giving the entropy-only KL
    -0.5 * (1 + log var) per chart dim (the same convention as
    FlatTorus; see ``kl_to_prior``).

    Parameters
    ----------
    dim : int
        Intrinsic dimension d. The sphere lives in R^{d+1}.
    """

    name = "sphere"
    default_kl_mode = "entropy_only"

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"Sphere(dim) requires dim >= 1, got {dim}")
        self.dim = int(dim)
        self.chart_dim = int(dim)
        self.decoder_input_dim = int(dim) + 1

    def kl_to_prior(
        self,
        mu: torch.Tensor,
        var: torch.Tensor,
        *,
        kl_mode: str | None = None,
        free_bits: float = 0.0,
    ) -> torch.Tensor:
        if kl_mode not in (None, "entropy_only", "partial"):
            raise ValueError(
                f"Sphere.kl_to_prior: unsupported kl_mode {kl_mode!r}; "
                "expected 'entropy_only' or 'partial' (an alias)."
            )
        kl_per_dim = -0.5 * (1.0 + var.log())
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

    @staticmethod
    def _exp_at_north_pole(v: torch.Tensor) -> torch.Tensor:
        """Exponential map at the north pole p0 = (0, ..., 0, 1) in
        R^{d+1}, applied to a tangent vector v in R^d (interpreted as
        the first d coordinates of a tangent in R^{d+1}).

        Returns the resulting point on S^d in R^{d+1}.
        """
        # ||v|| with a small clamp for the v=0 case (taylor: exp(0)=p0).
        norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sin_part = torch.sin(norm) / norm * v          # (..., d)
        cos_part = torch.cos(norm)                       # (..., 1)
        return torch.cat([sin_part, cos_part], dim=-1)   # (..., d+1)

    def distance(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> torch.Tensor:
        # Geodesic distance on S^d is the great-circle angle between
        # exp(z_a) and exp(z_b). We use the numerically robust
        # acos-of-clipped-inner-product formulation; the alternative
        # 2 * asin(||p - q||/2) is equally good on the chart.
        p = self._exp_at_north_pole(z_a)
        q = self._exp_at_north_pole(z_b)
        dot = (p * q).sum(dim=-1).clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        return torch.acos(dot)

    def embed_for_decoder(self, z: torch.Tensor) -> torch.Tensor:
        return self._exp_at_north_pole(z)

    def __repr__(self) -> str:
        return f"Sphere(dim={self.dim})"
