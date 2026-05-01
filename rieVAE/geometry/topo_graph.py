"""Wrapped-angular distance utilities for the flat-torus latent.

When the latent space is Z = S^1 x S^1 with the flat metric, all
Christoffel symbols vanish, geodesics are straight lines in angular
coordinates, and the Riemannian log map reduces to the component-wise
wrapped difference. These two helpers are used by:

  - the ``ProximalSCRVAETrainer``'s (deleted in Phase 3;
    ``_build_latent_distance_fn`` (so the iso loss measures the wrapped
    Clifford-torus distance when ``latent_topology='torus'``);
  - :class:`rieVAE.evaluate.isometry` evaluation paths that need a
    wrapped geodesic distance.

Phase-1 dead-code carve (op47C C.1.3): the legacy graph builders
``torus_euclidean_knn_graph``, ``torus_riemannian_knn_graph``, and the
internal ``_deduplicate_edges`` helper have been removed. They were
G-step / pullback-graph rebuilders for the pre-R4 self-consistent
iteration; the static-graph iso architecture does not call them. The
unified manifold abstraction of Phase 2 (``LatentManifold``) absorbs
the surviving distance helper behind an interface.

Angular coordinates are in radians. The wrap function is the standard
atan2(sin(x), cos(x)) which maps any angle to (-pi, pi].
"""
from __future__ import annotations

import torch


def torus_latent_delta(z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
    """Compute the shortest angular displacement on S^1 x S^1.

    Geodesics on the flat torus are straight lines in angular coordinates.
    The Riemannian log map at z_src pointing toward z_dst is exactly the
    component-wise wrapped difference (Corollary cor:topo_matched, part b).

    Parameters
    ----------
    z_src : (..., 2)  -- source angular coordinates (theta, phi) in radians.
    z_dst : (..., 2)  -- destination angular coordinates.

    Returns
    -------
    delta : (..., 2)  -- wrapped displacement, each component in (-pi, pi].
    """
    diff = z_dst - z_src
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def torus_geodesic_distance(z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
    """Exact geodesic distance on S^1(1) x S^1(1).

    d(z_i, z_j) = ||wrap(z_j - z_i)||_2

    For a general flat torus S^1(R_z) x S^1(r_z), multiply by the radii
    before calling this function (or incorporate them into the
    parameterisation).

    Parameters
    ----------
    z_src : (..., 2)
    z_dst : (..., 2)

    Returns
    -------
    dist : (...,)  non-negative
    """
    delta = torus_latent_delta(z_src, z_dst)
    return delta.norm(dim=-1)
