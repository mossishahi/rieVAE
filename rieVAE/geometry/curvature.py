"""Curvature analysis from learned Riemannian log maps.

The ambient closure vector c_ijk = l_{i->j} + l_{j->k} + l_{k->i} in R^G
measures the second fundamental form of the decoder manifold:
    c_ijk ≈ -(h(u,u) + h(v,v)) + h(u,v) + O(eps^3)
where h is the second fundamental form and u = Delta z_ij, v = Delta z_ik.

Key properties (proved in Proposition 3 of the theory paper):
    - c_ijk = 0 iff f_theta is locally affine at z_i (flat manifold locally).
    - c_ijk lies in the NORMAL space of M_theta to leading order.
    - ||c_ijk|| / Area(i,j,k) is a curvature proxy (ambient curvature proxy).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rieVAE.geometry.log_map import riemannian_log_maps_batched


def find_triangles(
    edge_index: torch.Tensor,
    max_triangles: int = 10_000,
) -> torch.Tensor:
    """Find all triangles (i, j, k) in the graph.

    Parameters
    ----------
    edge_index : (2, E)
    max_triangles : int
        Maximum number of triangles to return (randomly subsampled if exceeded).

    Returns
    -------
    triangles : (T, 3) -- node indices of each triangle, with i < j < k.
    """
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    adj: dict[int, set[int]] = {}
    for s, d in zip(src, dst):
        if s != d:
            adj.setdefault(s, set()).add(d)

    triangles = []
    nodes = sorted(adj.keys())
    for i in nodes:
        for j in sorted(adj.get(i, set())):
            if j <= i:
                continue
            common = adj.get(i, set()) & adj.get(j, set())
            for k in sorted(common):
                if k > j:
                    triangles.append((i, j, k))

    if len(triangles) == 0:
        return torch.zeros((0, 3), dtype=torch.long, device=edge_index.device)

    if len(triangles) > max_triangles:
        idx = torch.randperm(len(triangles))[:max_triangles]
        triangles = [triangles[i] for i in idx.tolist()]

    return torch.tensor(triangles, dtype=torch.long, device=edge_index.device)


def ambient_closure_vectors(
    decoder: nn.Module,
    z_mu: torch.Tensor,
    triangles: torch.Tensor,
) -> torch.Tensor:
    """Compute the ambient closure vector c_ijk for each triangle.

    c_ijk = l_{i->j} + l_{j->k} + l_{k->i}  in R^G

    where each l_{a->b} = J_f(z_a) * (z_b - z_a) is the Riemannian log map
    computed via JVP with stop-gradient on f_theta.

    Parameters
    ----------
    decoder : nn.Module
        Node decoder (eval mode during JVP).
    z_mu : (N, d)
        Posterior means.
    triangles : (T, 3)
        Triangle node indices (i, j, k).

    Returns
    -------
    c : (T, G)
        Ambient closure vectors.
    """
    if triangles.shape[0] == 0:
        G = decoder(z_mu[:1]).shape[-1]
        return torch.zeros((0, G), dtype=z_mu.dtype, device=z_mu.device)

    i_idx, j_idx, k_idx = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    z_i = z_mu[i_idx]
    z_j = z_mu[j_idx]
    z_k = z_mu[k_idx]

    src_all = torch.cat([z_i, z_j, z_k], dim=0)
    dz_all = torch.cat([
        z_j - z_i,
        z_k - z_j,
        z_i - z_k,
    ], dim=0)

    log_maps_all = riemannian_log_maps_batched(decoder, src_all, dz_all)

    T = triangles.shape[0]
    l_ij = log_maps_all[:T]
    l_jk = log_maps_all[T:2*T]
    l_ki = log_maps_all[2*T:]

    return l_ij + l_jk + l_ki


def triangle_areas(
    decoder: nn.Module,
    z_mu: torch.Tensor,
    triangles: torch.Tensor,
) -> torch.Tensor:
    """Compute triangle areas in ambient space.

    Area = 0.5 * ||l_{i->j} x l_{i->k}|| for 3D ambient space,
    or more generally 0.5 * sqrt(||l_ij||^2 ||l_ik||^2 - (l_ij . l_ik)^2)
    (parallelogram formula, valid in any dimension G).

    NOTE: This function is NOT used by curvature_proxy, which normalizes by
    ||l_{i->j}|| * ||l_{i->k}|| (product of edge lengths, Definition 4) rather
    than the triangle area. The two normalizations differ by a factor of
    1/(2 sin theta) where theta is the angle between legs. This function is
    retained for potential future use (e.g., area-normalized curvature estimates
    or surface area computations).

    Parameters
    ----------
    decoder : nn.Module
    z_mu : (N, d)
    triangles : (T, 3)

    Returns
    -------
    areas : (T,)
    """
    if triangles.shape[0] == 0:
        return torch.zeros(0, dtype=z_mu.dtype, device=z_mu.device)

    i_idx, j_idx, k_idx = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    z_i, z_j, z_k = z_mu[i_idx], z_mu[j_idx], z_mu[k_idx]

    l_ij = riemannian_log_maps_batched(decoder, z_i, z_j - z_i)
    l_ik = riemannian_log_maps_batched(decoder, z_i, z_k - z_i)

    dot = (l_ij * l_ik).sum(dim=-1)
    norm_ij_sq = l_ij.pow(2).sum(dim=-1)
    norm_ik_sq = l_ik.pow(2).sum(dim=-1)

    cross_norm_sq = (norm_ij_sq * norm_ik_sq - dot.pow(2)).clamp(min=0.0)
    return 0.5 * cross_norm_sq.sqrt()


def curvature_proxy(
    decoder: nn.Module,
    z_mu: torch.Tensor,
    triangles: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scale-invariant curvature proxy kappa_hat for each triangle.

    kappa_hat(i,j,k)  =  ||c_ijk|| / (||l_{i->j}|| * ||l_{i->k}||)

    where c_ijk = l_{i->j} + l_{j->k} + l_{k->i} is the ambient closure
    vector and l_{a->b} = J_f(z_a)(z_b - z_a) are the Riemannian log maps.

    DENOMINATOR: uses ||l_{i->k}|| (base z_i, direction z_k - z_i), matching
    Definition 4 of the theory paper exactly. Both denominator terms use the
    Jacobian at the SAME base point z_i, which is theoretically correct and
    avoids systematic bias from mixing Jacobian scales at z_i and z_k.

    NOTE: An earlier implementation reused l_{k->i} from the closure to save
    one JVP call. That was rejected because ||l_{k->i}|| != ||l_{i->k}|| when
    the Jacobian norm varies across the manifold (always true for non-isometric
    decoders), introducing a systematic bias proportional to ||nabla J_f|| * r_n.
    Since curvature_proxy is only called at evaluation time (not in the training
    loop), the extra JVP call is negligible.

    THEORETICAL MEANING (Proposition 4 of the theory paper):
    kappa_hat measures MEAN CURVATURE |H|, NOT Gaussian curvature K.
    - For codimension-1 surfaces with legs along principal directions:
          kappa_hat -> 2|H| = |kappa_1 + kappa_2|
    - For umbilic surfaces (sphere, kappa_1 = kappa_2 = 1/R):
          kappa_hat -> 2/R = 2*sqrt(K)  [special case only]
    - K = 0 does NOT imply kappa_hat = 0 (cylinder: K=0, H=1/(2R) > 0).
    - kappa_hat = 0 iff the decoder is locally affine (h = 0 everywhere).

    Reconstruction-dominated regime (MSE-trained decoders, both manifolds):
    - Sphere  (R=1,   G=50, A ~ N(0,1/sqrt(3))): kappa_hat* ~ 0.489
    - Clifford flat torus (R=2, r=1, G=50):       kappa_hat* ~ 0.316
    - Scale-invariant with respect to latent coordinate rescaling.

    Parameters
    ----------
    decoder : nn.Module
    z_mu : (N, d)
    triangles : (T, 3)
    eps : float

    Returns
    -------
    kappa : (T,)
    """
    if triangles.shape[0] == 0:
        return torch.zeros(0, dtype=z_mu.dtype, device=z_mu.device)

    i_idx, j_idx, k_idx = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    z_i = z_mu[i_idx]
    z_j = z_mu[j_idx]
    z_k = z_mu[k_idx]

    # Four JVP calls: three for the closure triangle, one for the denominator.
    # l_ij, l_jk, l_ki: the three directed edges of the closure c = l_ij + l_jk + l_ki
    # l_ik:             the denominator's second edge, base z_i (matches Definition 4)
    l_ij = riemannian_log_maps_batched(decoder, z_i, z_j - z_i)
    l_jk = riemannian_log_maps_batched(decoder, z_j, z_k - z_j)
    l_ki = riemannian_log_maps_batched(decoder, z_k, z_i - z_k)
    l_ik = riemannian_log_maps_batched(decoder, z_i, z_k - z_i)  # denominator, base z_i

    c = l_ij + l_jk + l_ki
    closure_norm = c.norm(dim=-1)

    norm_ij = l_ij.norm(dim=-1)
    norm_ik = l_ik.norm(dim=-1)  # ||l_{i->k}||, exact match to Definition 4
    denom = norm_ij * norm_ik

    return closure_norm / (denom + eps)
