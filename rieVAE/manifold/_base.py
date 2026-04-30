"""Latent-manifold protocol for the Certified Riemannian VAE.

Phase-2 unification (op47C C.2): the latent space M_z of the
manifold-VAE template (eq:loss in main.tex) is supplied by a
``LatentManifold`` plug-in. The encoder emits raw chart coordinates
in R^k (the "general VAE" convention of op47C option (ii)); the
manifold object is responsible for:

  - the closed-form KL divergence to the manifold's prior;
  - the chart-coordinate reparameterisation;
  - the geodesic distance d_{M_z}: R^k x R^k -> R+, evaluated by
    projecting/wrapping/exp-mapping the chart coordinates onto M_z
    before measuring (so the iso loss and the certificate's
    delta_iso measure the same intrinsic object);
  - the input embedding for the decoder (the chart coordinates are
    mapped to a representation the decoder is built against, e.g.
    (cos, sin) for the flat torus, ambient embedding for the sphere,
    exp-map for hyperbolic).

The protocol also exposes the dimensions a caller needs to size the
encoder/decoder: ``chart_dim`` is the dimension of the encoder's mu
output, ``decoder_input_dim`` is the dimension the decoder consumes
after ``embed_for_decoder``.

Theoretical implications of option (ii) (cf. main.tex sec:isometry,
post-Phase-2 remark):

  - Theorem thm:encoder_isometry: ``mu_phi(x_i)`` is now a chart
    coordinate in R^k; ``d_{M_z}`` projects/wraps internally and is
    well-defined and continuous on R^k. The bound's form is
    unchanged.
  - Theorem thm:isometry_main: the decoder pullback metric is
    ``J_f^T J_f`` where f is the *composed* map
    ``embed_for_decoder o decoder``; the JVP machinery of
    Lemma lem:dist_approx still applies.
  - Theorem thm:topo_floor: "Euclidean latent" means
    ``LatentManifold = Euclidean(d)``. With a topology-matched
    manifold (FlatTorus, Sphere, Hyperbolic, ...) the architectural
    fold of thm:topo_floor does not occur; the relevant rate is
    Cor.~cor:topo_matched at p = 2.
  - Cor. cor:pe_euclidean: still silent on non-Euclidean latents
    (the Omega(inj M) floor it tightens is itself Euclidean).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class LatentManifold(Protocol):
    """Latent-manifold protocol for the Certified Riemannian VAE.

    Concrete subclasses live in this package: ``Euclidean``,
    ``FlatTorus``, ``Sphere``, ``Hyperbolic``, ``StereographicProduct``.
    """

    name: str
    """Lower-case identifier, used by the registry."""

    dim: int
    """Intrinsic dimension d of M_z."""

    chart_dim: int
    """Dimension of the encoder's mu / var output (the chart in
    which posterior parameters live). For the manifolds we ship this
    equals ``dim``; the chart used per manifold is documented on the
    concrete class."""

    decoder_input_dim: int
    """Dimension of the tensor the decoder consumes, after
    ``embed_for_decoder``. Equals ``dim`` for Euclidean and
    Hyperbolic (tangent-at-origin chart), ``2 * dim`` for FlatTorus
    (cos, sin embedding per circle), ``dim + 1`` for Sphere (ambient
    R^{d+1} embedding), and the sum of factor decoder dims for
    StereographicProduct."""

    default_kl_mode: str
    """Default KL form for ``kl_to_prior``. 'standard' for Euclidean
    with the standard Gaussian prior; 'partial' for Euclidean with
    the mu^2-dropped prior (the iso default); 'entropy_only' for all
    translation-invariant priors on compact / origin-centred
    manifolds (FlatTorus uniform, Hyperbolic standard wrapped-normal
    at origin, etc.)."""

    def kl_to_prior(
        self,
        mu: torch.Tensor,
        var: torch.Tensor,
        *,
        kl_mode: str | None = None,
        free_bits: float = 0.0,
    ) -> torch.Tensor:
        """KL divergence q_phi(z|x) || p(z), reduced over the latent
        and averaged over the batch.

        Parameters
        ----------
        mu : (B, chart_dim)
        var : (B, chart_dim)
        kl_mode : str or None
            Optional override of ``default_kl_mode``. Only meaningful
            for Euclidean (where 'standard' / 'partial' / 'flat' all
            make sense); ignored otherwise.
        free_bits : float
            Per-dimension free-bits threshold (Kingma et al. 2016).
            Only meaningful when kl_mode == 'standard'; clamps each
            kl_per_dim from below at this value.

        Returns
        -------
        Scalar loss tensor.
        """
        ...

    def reparameterize(
        self,
        mu: torch.Tensor,
        var: torch.Tensor,
    ) -> torch.Tensor:
        """Sample z ~ q_phi(z|x). Under op47C option (ii), this is the
        vanilla Gaussian reparameterisation z = mu + eps * sqrt(var)
        on the chart for ALL manifolds (the manifold structure is
        applied when ``embed_for_decoder`` and ``distance`` consume
        z; we do not project z back onto M_z at sample time)."""
        ...

    def distance(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> torch.Tensor:
        """Geodesic distance d_{M_z}(z_a, z_b), batched over leading
        dims. The implementation projects/wraps/exp-maps chart
        coordinates onto M_z before measuring, so the result is a
        true intrinsic geodesic distance regardless of where the
        chart coordinates land in R^k.

        Parameters
        ----------
        z_a, z_b : (..., chart_dim)

        Returns
        -------
        d : (...,) non-negative
        """
        ...

    def embed_for_decoder(self, z: torch.Tensor) -> torch.Tensor:
        """Map chart coordinates z (..., chart_dim) to the decoder's
        input (..., decoder_input_dim).

        Concrete behaviour:
          - Euclidean: identity.
          - FlatTorus: per-coordinate (cos, sin) embedding.
          - Sphere: normalise (or exp-map at north pole) onto the
            ambient (d+1)-dimensional sphere.
          - Hyperbolic: exp-map at origin onto the (d+1)-dimensional
            hyperboloid (or Poincare disc; documented on the class).
          - StereographicProduct: factor-wise concatenation.
        """
        ...
