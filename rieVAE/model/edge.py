"""Edge head for the certified Riemannian VAE.

Single learnable module beyond the encoder and decoder:

* ``JointEdgeDecoder`` (``F_phi``). A single MLP on the concatenated
  pair ``[z_i ; z_j]`` that predicts the log map in ambient space,
  approximating ``J_f(z_i)(z_j - z_i)`` at every base point.

The deformation module ``Def_psi`` of earlier drafts is removed: the
training graph is now the static biharmonic spectral edge set of
:mod:`rieVAE.geometry.spectral_premetric`, computed once from the data
Laplacian and not dependent on any learnable parameter. See Section
2.1 and Section 5 of the main paper, Lemmas lem:spec_premetric and
lem:lap_convergence.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rieVAE.model._activations import make_activation


class JointEdgeDecoder(nn.Module):
    """F_phi : R^d x R^d -> R^G.

    Takes the pair (z_i, z_j) as CONCATENATED input (dim 2d) and outputs
    a predicted log map in R^G:

        \\hat ell_{ij} = F_phi([z_i ; z_j]) ~= J_f(z_i) (z_j - z_i)

    Having both z_i and z_j separately is critical: a function of
    z_j - z_i alone can at best learn the mean Jacobian
    E_{z_i}[J_f(z_i)], which vanishes on symmetric manifolds. With
    (z_i, z_j) input, the network represents J_f(z_i)(z_j - z_i) at
    every specific base point.

    Parameters
    ----------
    dim_latent : int
        Latent dimension d.
    dim_out : int
        Ambient dimension G (log map lives in R^G).
    hidden_dims : tuple[int, ...]
        Hidden widths of the MLP.
    dropout : float
        Dropout rate (0 recommended for a stable JVP target).
    """

    def __init__(
        self,
        dim_latent: int,
        dim_out: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.dim_latent = dim_latent
        self.dim_out = dim_out
        self.activation = activation

        dims = (2 * dim_latent,) + hidden_dims
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(d_in, d_out), make_activation(activation)])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], dim_out))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Predict log maps for a batch of directed edges.

        Parameters
        ----------
        z_src : (E, d) latent codes at the source node z_i.
        z_dst : (E, d) latent codes at the destination node z_j.

        Returns
        -------
        l_hat : (E, G) predicted Riemannian log maps in R^G.
        """
        zij = torch.cat([z_src, z_dst], dim=-1)
        return self.net(zij)


class ScalarEdgeDecoder(nn.Module):
    """Scalar distance edge head F_phi : R^d x R^d -> R_{>=0}.

    Prediction:

        F_phi(z_i, z_j) = softplus(w_raw) * ||z_i - z_j||

    A single learnable scalar w_raw, passed through softplus to enforce
    positivity. By construction the head is symmetric in (z_i, z_j) and
    non-negative. It is the minimal non-trivial inference-time distance
    proxy in the iso-architecture: a global learnable scale factor that
    maps Euclidean latent distances to biharmonic targets.

    A scalar linear head cannot absorb encoder non-isometry beyond a
    uniform global scale; this is the specific reason it is safe to
    couple the edge head to the encoder through a shared training
    signal. A non-linear MLP head would absorb arbitrary non-linear
    encoder distortion and reintroduce the moving-target failure mode
    of the JVP-based architecture; that is why a single scalar parameter
    is used here instead of an MLP.

    The scalar w_raw is calibrated post hoc by ordinary least squares
    against the spectral target on E* (see
    :func:`rieVAE.train.loss.calibrate_edge_decoder_scale` and the
    trainer's ``_calibrate_edge_scale_posthoc``); it receives no
    gradient from the iso loss. The encoder is constrained directly
    by L_iso, which uses ``latent_distance_fn`` to measure the same
    object the certificate's ``delta_iso`` measures.

    Parameters
    ----------
    w_init : float
        Initial value of the raw scalar parameter (before softplus).
        With w_init=0.0 the initial scale is softplus(0)=ln(2)~0.6931.
        Use :meth:`set_scale_from_value` after computing an empirical
        ratio between d^bih and ||mu_i - mu_j|| at Phase 2 entry to
        warm-start the scale.
    """

    def __init__(
        self,
        w_init: float = 0.0,
        latent_distance_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.w_raw = nn.Parameter(torch.tensor(float(w_init)))
        # Latent distance d_{M_z}(z_src, z_dst). When None, falls back to
        # the Euclidean norm (the contractible-latent default of
        # Theorem thm:encoder_isometry with M_z = R^d). For compact
        # latents (e.g. Clifford torus, sphere) the trainer constructs
        # a wrapped/intrinsic distance closure and passes it here so
        # that the certified C1' residual delta_iso (= |softplus(w*) *
        # d_{M_z}(mu_i, mu_j) - tilde_w_ij|) measures the same object
        # the iso loss optimises (Phase-0 bug B.3 of op47C).
        self.latent_distance_fn = latent_distance_fn

    @property
    def scale(self) -> torch.Tensor:
        """The current positive scale factor softplus(w_raw)."""
        return F.softplus(self.w_raw)

    def set_latent_distance_fn(
        self,
        fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> None:
        """Install or replace the latent distance closure post-construction.

        Used by the trainer when latent_topology is set on TrainingConfig
        rather than on model construction; the trainer builds the closure
        from manifold parameters (e.g. Clifford radii) and passes it in
        before the first forward pass.
        """
        self.latent_distance_fn = fn

    def forward(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Predict scalar distance per directed edge.

        Parameters
        ----------
        z_src : (E, d) latent codes at the source.
        z_dst : (E, d) latent codes at the destination.

        Returns
        -------
        d_hat : (E,) non-negative predicted distances. Equals
            softplus(w_raw) * d_{M_z}(z_src, z_dst) where d_{M_z} is
            the latent_distance_fn supplied at construction (or the
            Euclidean norm if none was supplied).
        """
        if self.latent_distance_fn is None:
            d = (z_dst - z_src).norm(dim=-1)
        else:
            d = self.latent_distance_fn(z_src, z_dst)
        return self.scale * d

    @torch.no_grad()
    def set_scale_from_value(self, value: float) -> None:
        """Set w_raw so softplus(w_raw) = value.

        Used to warm-start the scale at Phase 2 entry from the
        empirical ratio
            value = mean(d^bih) / mean(||mu_i - mu_j||)
        on the static edge set, so that L_edge starts close to zero
        and the optimizer only refines the scale rather than building
        it from scratch.
        """
        if value <= 0.0:
            return
        if value > 20.0:
            self.w_raw.fill_(float(value))
            return
        self.w_raw.fill_(float(math.log(math.expm1(value))))
