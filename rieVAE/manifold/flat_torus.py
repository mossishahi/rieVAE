"""Flat torus latent manifold M_z = R/(2 pi Z)^d (Clifford torus).

This is the topology-matched regime of Cor.~cor:topo_matched at
``p = 2`` for data manifolds with fundamental group Z^d (e.g., the
Clifford torus T^2 = S^1 x S^1 with radii (R, r)).

Chart coordinates (the encoder's mu output) live in R^d; the wrap
function ``atan2(sin x, cos x)`` projects them to (-pi, pi]^d when
required by the geodesic distance. The decoder consumes the
component-wise (cos, sin) embedding of dimension 2 d (so the
periodicity is built into the decoder by construction).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class FlatTorus(nn.Module):
    """Flat torus M_z = T^d = R/(2 pi Z)^d, optionally with anisotropic
    radii (R_1, ..., R_d) so the geodesic distance becomes the
    Riemannian length on the corresponding Clifford torus
    S^1(R_1) x ... x S^1(R_d).

    KL is taken against the uniform prior on T^d (translation
    invariant). The closed form is the entropy-only KL
    ``-0.5 * (1 + log var)`` per circle dim: the prior is constant in
    mu, so the KL never penalises the posterior mean; it only
    penalises the posterior variance via the entropy term.

    Reparameterisation is the vanilla z = mu + eps * sqrt(var) per
    op47C option (ii); the wrap is applied only when chart coordinates
    are consumed (in ``distance`` and ``embed_for_decoder``), not at
    sample time.

    Parameters
    ----------
    dim : int
        Number of circle factors d. The intrinsic dimension is d.
    radii : tuple[float, ...] or None
        Per-circle radii (R_1, ..., R_d). When None, all radii are 1.
        Length must equal ``dim``.
    """

    name = "flat_torus"
    default_kl_mode = "entropy_only"

    def __init__(
        self,
        dim: int,
        radii: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"FlatTorus(dim) requires dim >= 1, got {dim}")
        if radii is None:
            radii_t = torch.ones(int(dim), dtype=torch.float32)
        else:
            radii_t = torch.tensor(list(radii), dtype=torch.float32)
            if radii_t.numel() != int(dim):
                raise ValueError(
                    f"FlatTorus(dim={dim}, radii=...) requires "
                    f"len(radii) == dim; got len(radii)={radii_t.numel()}."
                )
        self.dim = int(dim)
        self.chart_dim = int(dim)
        self.decoder_input_dim = 2 * int(dim)
        self.register_buffer("radii", radii_t.clone())

    def kl_to_prior(
        self,
        mu: torch.Tensor,
        var: torch.Tensor,
        *,
        kl_mode: str | None = None,
        free_bits: float = 0.0,
    ) -> torch.Tensor:
        # Uniform prior on T^d is translation invariant; the KL is
        # the entropy of the wrapped-Gaussian posterior up to a
        # constant. We use the entropy-only form -0.5 * (1 + log var)
        # which is the standard manifold-VAE convention (cf. Davidson
        # et al. 2018 and Mathieu et al. 2019 for the analogous
        # spherical / hyperbolic forms). free_bits is ignored on
        # T^d -- there is no mu-attractor to short-circuit.
        if kl_mode not in (None, "entropy_only"):
            # Allow the iso-architecture's 'partial' alias: the
            # algebraic form 0.5 * (var - 1 - log var) coincides with
            # entropy-only up to an additive constant in var, both
            # of which have grad zero on mu and pull sigma -> 1.
            # We honor the alias for ergonomics.
            if kl_mode != "partial":
                raise ValueError(
                    f"FlatTorus.kl_to_prior: unsupported kl_mode "
                    f"{kl_mode!r}; expected 'entropy_only' or 'partial'."
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

    def distance(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> torch.Tensor:
        # Wrap the difference to (-pi, pi] component-wise; multiply
        # by per-circle radii so the result is the Riemannian length
        # on the Clifford torus S^1(R_1) x ... x S^1(R_d).
        diff = z_b - z_a
        wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
        radii = self.radii.to(wrapped.device, wrapped.dtype)
        return (wrapped * radii).norm(dim=-1)

    def embed_for_decoder(self, z: torch.Tensor) -> torch.Tensor:
        # (cos, sin) per circle: enforces 2 pi periodicity on the
        # decoder by construction (the decoder's first layer takes
        # only periodic inputs in [-1, 1]^{2d}).
        return torch.cat([torch.cos(z), torch.sin(z)], dim=-1)

    def __repr__(self) -> str:
        radii_list = self.radii.detach().cpu().tolist()
        return f"FlatTorus(dim={self.dim}, radii={tuple(radii_list)})"
