"""Stereographic product latent manifold.

A product of constant-curvature factors composed via stereographic
projection (Skopek, Ganea, Becigneul 2020, "Mixed-Curvature
Variational Autoencoders"). Each factor is one of the four
single-curvature manifolds we ship: Euclidean, FlatTorus, Sphere,
Hyperbolic. The product structure factorises through the chart, the
KL, the geodesic distance, and the decoder embedding.

Under op47C option (ii), the ``StereographicProduct`` is a thin
adapter that delegates each operation to its constituent factors:
chart coordinates are concatenated factor-by-factor, the KL is the
sum of factor KLs, the geodesic distance is the sqrt-sum-of-squares
of factor distances (Pythagoras on the product Riemannian metric),
and the decoder embedding is the concatenation of factor embeddings.
"""
from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn

from rieVAE.manifold._base import LatentManifold


class StereographicProduct(nn.Module):
    """Product of constant-curvature factors.

    Parameters
    ----------
    factors : iterable of LatentManifold
        Each factor is one of the four base manifolds (Euclidean,
        FlatTorus, Sphere, Hyperbolic). The intrinsic dimension and
        chart dimension of the product are the sums of the factors';
        likewise the decoder input dim and the KL.

    Examples
    --------
    A 6-dimensional latent split into a 2-D Euclidean plane, a
    1-D flat torus, and a 3-D round sphere:

        StereographicProduct([
            Euclidean(2),
            FlatTorus(1),
            Sphere(3),
        ])

    The chart is R^6 = R^2 x R^1 x R^3; the decoder consumes a
    tensor of dim 2 + 2 + 4 = 8.
    """

    name = "stereographic_product"

    def __init__(self, factors: Iterable[LatentManifold]) -> None:
        super().__init__()
        factor_list: List[LatentManifold] = list(factors)
        if len(factor_list) < 2:
            raise ValueError(
                "StereographicProduct requires at least two factors; "
                f"got {len(factor_list)}."
            )
        self.factors = nn.ModuleList(factor_list)
        self.dim = sum(int(f.dim) for f in factor_list)
        self.chart_dim = sum(int(f.chart_dim) for f in factor_list)
        self.decoder_input_dim = sum(int(f.decoder_input_dim) for f in factor_list)
        # Use 'partial' as the umbrella default; concrete behaviour is
        # determined by each factor's default_kl_mode at call time.
        self.default_kl_mode = "partial"
        # Cache slice boundaries on the chart axis for forward/backward
        # passes.
        self._chart_slices: list[slice] = []
        offset = 0
        for f in factor_list:
            self._chart_slices.append(slice(offset, offset + int(f.chart_dim)))
            offset += int(f.chart_dim)
        self._decoder_slices: list[slice] = []
        offset = 0
        for f in factor_list:
            self._decoder_slices.append(slice(offset, offset + int(f.decoder_input_dim)))
            offset += int(f.decoder_input_dim)

    def kl_to_prior(
        self,
        mu: torch.Tensor,
        var: torch.Tensor,
        *,
        kl_mode: str | None = None,
        free_bits: float = 0.0,
    ) -> torch.Tensor:
        total = mu.new_zeros(())
        for f, sl in zip(self.factors, self._chart_slices):
            total = total + f.kl_to_prior(
                mu[..., sl], var[..., sl],
                kl_mode=kl_mode, free_bits=free_bits,
            )
        return total

    def reparameterize(
        self,
        mu: torch.Tensor,
        var: torch.Tensor,
    ) -> torch.Tensor:
        # Vanilla Gaussian on the chart. We delegate to each factor
        # so subclasses with a non-default reparameterisation are
        # honoured; under op47C option (ii) all factors use the
        # vanilla form.
        parts = []
        for f, sl in zip(self.factors, self._chart_slices):
            parts.append(f.reparameterize(mu[..., sl], var[..., sl]))
        return torch.cat(parts, dim=-1)

    def distance(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> torch.Tensor:
        # Pythagoras on the product Riemannian metric:
        #   d_prod(p, q)^2 = sum_k d_k(p_k, q_k)^2.
        sq_sum = z_a.new_zeros(z_a.shape[:-1])
        for f, sl in zip(self.factors, self._chart_slices):
            d_k = f.distance(z_a[..., sl], z_b[..., sl])
            sq_sum = sq_sum + d_k.pow(2)
        return sq_sum.clamp(min=0.0).sqrt()

    def embed_for_decoder(self, z: torch.Tensor) -> torch.Tensor:
        parts = []
        for f, sl in zip(self.factors, self._chart_slices):
            parts.append(f.embed_for_decoder(z[..., sl]))
        return torch.cat(parts, dim=-1)

    def __repr__(self) -> str:
        factor_repr = ", ".join(repr(f) for f in self.factors)
        return f"StereographicProduct([{factor_repr}])"
