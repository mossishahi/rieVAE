"""Hyperbolic latent manifold M_z = H^d (Lorentz hyperboloid model).

Chart coordinates (the encoder's mu output) live in R^d, interpreted
as a tangent vector at the origin o = (0, ..., 0, 1) in R^{d+1} of
the Lorentz hyperboloid

    H^d = { x in R^{d+1} : -x_0^2 + x_1^2 + ... + x_d^2 = -1, x_0 > 0 }
    (with the (-, +, ..., +) Lorentzian inner product).

The decoder consumes the exp-map of the chart coordinates onto H^d
(dimension d + 1). The geodesic distance is the standard Lorentzian
arccosh formula evaluated on the exp-mapped points.

KL is taken against the standard wrapped-normal at the origin
(Nagano et al. 2019): a translation-invariant prior on the tangent
space at o, giving the same entropy-only KL form as Sphere /
FlatTorus.

Phase-1 / Phase-2 scope note: under op47C option (ii) we take the
posterior on the chart to be a vanilla Gaussian and rely on
``embed_for_decoder`` to map it onto H^d at decoder time; we do NOT
implement the parallel-transport-based wrapped-normal posterior
sampler of Nagano et al. since the simpler Gaussian-on-chart
behaviour is consistent with the "general VAE" interpretation the
user selected.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class Hyperbolic(nn.Module):
    """Hyperbolic latent space H^d (Lorentz / hyperboloid model).

    Parameters
    ----------
    dim : int
        Intrinsic dimension d.
    curvature : float
        Sectional curvature K. Must be negative; default -1. The
        hyperboloid is rescaled so its sectional curvature is K
        (we multiply tangent vectors and distances by 1/sqrt(-K)
        relative to the unit hyperboloid).
    """

    name = "hyperbolic"
    default_kl_mode = "entropy_only"

    def __init__(self, dim: int, curvature: float = -1.0) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"Hyperbolic(dim) requires dim >= 1, got {dim}")
        if curvature >= 0.0:
            raise ValueError(
                f"Hyperbolic.curvature must be negative, got {curvature}"
            )
        self.dim = int(dim)
        self.chart_dim = int(dim)
        self.decoder_input_dim = int(dim) + 1
        # Length scale: distances on the K-curvature hyperboloid are
        # 1/sqrt(-K) times distances on the unit hyperboloid. We
        # store sqrt(-K) and scale chart vectors at exp-map time.
        self._sqrt_neg_K = float(math.sqrt(-curvature))
        self.curvature = float(curvature)

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
                f"Hyperbolic.kl_to_prior: unsupported kl_mode {kl_mode!r}; "
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

    def _exp_at_origin(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map at the origin o = (1, 0, ..., 0) in
        R^{d+1} (Lorentzian convention: the timelike component is
        index 0). v is in R^d, interpreted as the spacelike
        components of a tangent at o.

        Returns the corresponding point on H^d in R^{d+1}.
        """
        # Scale by 1/sqrt(-K) so distances are correctly proportional.
        v_scaled = v * (1.0 / self._sqrt_neg_K)
        norm = v_scaled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        timelike = torch.cosh(norm)                                 # (..., 1)
        spacelike = torch.sinh(norm) / norm * v_scaled               # (..., d)
        return torch.cat([timelike, spacelike], dim=-1)              # (..., d+1)

    def distance(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> torch.Tensor:
        # Geodesic distance on H^d:
        #   d(p, q) = (1 / sqrt(-K)) * arccosh( -<p, q>_L )
        # where <p, q>_L = -p_0 q_0 + sum_{i=1}^d p_i q_i.
        p = self._exp_at_origin(z_a)
        q = self._exp_at_origin(z_b)
        # Lorentzian inner product: flip the sign on the timelike axis.
        timelike = -p[..., 0] * q[..., 0]
        spacelike = (p[..., 1:] * q[..., 1:]).sum(dim=-1)
        lor = timelike + spacelike
        # -<p, q>_L is >= 1 with equality iff p == q; clamp for stability.
        arg = (-lor).clamp(min=1.0 + 1e-7)
        d_unit = torch.acosh(arg)
        return d_unit * (1.0 / self._sqrt_neg_K)

    def embed_for_decoder(self, z: torch.Tensor) -> torch.Tensor:
        return self._exp_at_origin(z)

    def __repr__(self) -> str:
        return f"Hyperbolic(dim={self.dim}, curvature={self.curvature})"
