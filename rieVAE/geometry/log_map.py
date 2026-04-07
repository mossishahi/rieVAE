"""Riemannian log map computation via Jacobian-vector products.

The Riemannian log map at f_theta(z_i) in the direction of z_j is approximated
by the Jacobian-vector product:

    l_ij = J_{f_theta}(z_i) @ (z_j - z_i)  in R^G

This is the first-order Taylor approximation of f_theta(z_j) - f_theta(z_i),
exact when f_theta is linear (Corollary 1 of the theory paper). The approximation
error is O(||Gamma(z_i)|| * r^2) + O(K_max * r^3) in general (Lemma 1), where
Gamma(z_i) are the Christoffel symbols of the pullback metric in Euclidean latent
coordinates. The error reduces to O(K_max * r^3 / 6) when G(z_i) = I (e.g., at
the isometric fixed point in near-normal coordinates).

All computations use STOP-GRADIENT on f_theta: the decoder does not receive
gradient from the Riemannian loss. Gradient flows only from L_node_recon and
L_node_KL to f_theta.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.func import jvp, vmap  # type: ignore[attr-defined]


def riemannian_log_map_single(
    decoder: nn.Module,
    z_src: torch.Tensor,
    delta_z: torch.Tensor,
) -> torch.Tensor:
    """Compute the Riemannian log map for a single edge.

    l = J_{f}(z_src) @ delta_z  in R^G

    Parameters
    ----------
    decoder : nn.Module
        Node decoder f_theta: R^d -> R^G. Must be in eval() mode and C^1 smooth.
    z_src : (d,)
        Source latent point.
    delta_z : (d,)
        Latent difference z_dst - z_src.

    Returns
    -------
    log_map : (G,)
        Riemannian log map vector. Detached from decoder parameters.
    """
    def f_single(z: torch.Tensor) -> torch.Tensor:
        return decoder(z.unsqueeze(0)).squeeze(0)

    with torch.no_grad():
        _, tangent = jvp(f_single, (z_src,), (delta_z,))

    return tangent


def riemannian_log_maps_batched(
    decoder: nn.Module,
    z_src: torch.Tensor,
    delta_z: torch.Tensor,
) -> torch.Tensor:
    """Compute Riemannian log maps for all edges in a batch.

    Uses vmap over edges for efficiency. The decoder must be in eval() mode
    so that dropout is disabled during JVP computation.

    Parameters
    ----------
    decoder : nn.Module
        Node decoder f_theta: must support functional API (stateless).
    z_src : (E, d)
        Source latent points for each directed edge.
    delta_z : (E, d)
        Latent differences z_dst - z_src for each directed edge.

    Returns
    -------
    log_maps : (E, G)
        Riemannian log map vectors. Detached from the decoder's parameters.
    """
    was_training = decoder.training
    decoder.eval()

    def f_single(z: torch.Tensor) -> torch.Tensor:
        return decoder(z.unsqueeze(0)).squeeze(0)

    def jvp_single(z: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, tangent = jvp(f_single, (z,), (dz,))
        return tangent

    log_maps = vmap(jvp_single)(z_src, delta_z)

    if was_training:
        decoder.train()

    return log_maps.detach()


def riemannian_distances(log_maps: torch.Tensor) -> torch.Tensor:
    """Compute Riemannian distances w_ij = ||l_ij||_2 from log maps.

    These distances are used as edge weights in the Riemannian KNN graph.
    By Lemma 1 of the theory paper:
        |w_ij - d_R(z_i, z_j)| <= C_Gamma(z_i) * r^2 + K_max * r^3 / 6
    where C_Gamma = ||Gamma^{Eucl}(z_i)|| is the Christoffel-symbol norm at z_i.
    The O(K r^3/6) bound holds exactly when G(z_i) = I (near-normal coordinates).

    Parameters
    ----------
    log_maps : (E, G)

    Returns
    -------
    distances : (E,)
    """
    return log_maps.norm(dim=-1)
