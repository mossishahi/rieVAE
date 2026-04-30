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
from rieVAE.geometry.topo_graph import torus_latent_delta


# ---------------------------------------------------------------------------
# Latent-space delta dispatch
# ---------------------------------------------------------------------------

def _latent_delta(
    z_from: torch.Tensor,
    z_to: torch.Tensor,
    latent_space: str = "euclidean",
) -> torch.Tensor:
    """Compute z_to - z_from respecting the latent geometry.

    latent_space = "euclidean": plain difference (R^d).
    latent_space = "torus":     wrapped angular difference (atan2(sin, cos)),
                                 correct for the FlatTorus latent of
                                 ``RiemannianVAE(latent_manifold='torus')``.
    """
    if latent_space == "torus":
        return torus_latent_delta(z_from, z_to)
    return z_to - z_from


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
    latent_space: str = "euclidean",
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
        _latent_delta(z_i, z_j, latent_space),
        _latent_delta(z_j, z_k, latent_space),
        _latent_delta(z_k, z_i, latent_space),
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
    latent_space: str = "euclidean",
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

    l_ij = riemannian_log_maps_batched(decoder, z_i, _latent_delta(z_i, z_j, latent_space))
    l_ik = riemannian_log_maps_batched(decoder, z_i, _latent_delta(z_i, z_k, latent_space))

    dot = (l_ij * l_ik).sum(dim=-1)
    norm_ij_sq = l_ij.pow(2).sum(dim=-1)
    norm_ik_sq = l_ik.pow(2).sum(dim=-1)

    cross_norm_sq = (norm_ij_sq * norm_ik_sq - dot.pow(2)).clamp(min=0.0)
    return 0.5 * cross_norm_sq.sqrt()


def closure_proxy_per_node(
    z_mu: torch.Tensor,
    triangles: torch.Tensor,
    c_ijk: torch.Tensor,
    eps: float = 1e-8,
    latent_space: str = "euclidean",
) -> torch.Tensor:
    """Per-node unnormalized closure proxy Ĉ_Γ(z_i).

    Ĉ_Γ(z_i) = max_{triangles (i,j,k) incident to i} ||c_ijk|| / ||Δz_ij||²

    This is the computable proxy from Lemma lem:proxy_bound(a): it upper-bounds
    ||H_f(z_i)||_op up to a factor of 3 and O(r_n) correction (from the tangential
    Hessian H_f^{tan} = O(r_n) at the self-consistent fixed point, rem:PT_closure).
    When multiplied by kappa_cond(z_i) = sqrt(Lambda_max/lambda_0), it bounds
    C_Gamma(z_i)/sqrt(lambda_0) from above (Lemma lem:proxy_bound(c)).

    Parameters
    ----------
    z_mu      : (N, d) latent codes
    triangles : (T, 3) node indices (i, j, k) from find_triangles
    c_ijk     : (T, G) ambient closure vectors from ambient_closure_vectors
    eps       : float  numerical floor for ||Δz_ij||²

    Returns
    -------
    proxy : (N,) per-node max; 0.0 at nodes with no incident triangles.
    """
    N = z_mu.shape[0]
    device, dtype = z_mu.device, z_mu.dtype

    proxy = torch.zeros(N, dtype=dtype, device=device)
    if triangles.shape[0] == 0:
        return proxy

    i_idx, j_idx, k_idx = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    closure_norms = c_ijk.norm(dim=-1)                                   # (T,)
    dz_ij = _latent_delta(z_mu[i_idx], z_mu[j_idx], latent_space)       # (T, d)
    dz_ij_sq = dz_ij.pow(2).sum(dim=-1)                                  # (T,)
    ratios = closure_norms / (dz_ij_sq + eps)                           # (T,)

    # Max-aggregate over all three vertices of each triangle.
    # A node contributes to the max both when it is the base vertex i
    # and when it is vertex j or k (asymmetric but conservative: the Lemma
    # bound holds for all vertices since the Hessian bound is at z_i only;
    # using all three gives a tighter per-node estimate in practice).
    for vertex_col in [i_idx, j_idx, k_idx]:
        proxy.scatter_reduce_(0, vertex_col, ratios, reduce="amax", include_self=True)

    return proxy


def adaptive_knn_radii(
    z_mu: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    triangles: torch.Tensor,
    c_ijk: torch.Tensor,
    r_n: float,
    eps_accuracy: float = 0.1,
    C_prime: float = 3.0,
    r_min_fraction: float = 0.1,
    eps_denom: float = 1e-8,
    latent_space: str = "euclidean",
) -> torch.Tensor:
    """Per-node adaptive KNN radius from Theorem thm:adaptive_radius.

    For each node i sets:
        r_i = min(r_n, sqrt(eps_accuracy / (C_prime * kappa_cond(i) * C_hat_Gamma(i))))

    where:
        C_hat_Gamma(i) = max over incident triangles of ||c_ijk|| / ||Δz_ij||²
                         (Lemma lem:proxy_bound(a))
        kappa_cond(i)  = max_j(w_ij/||Δz_ij||) / min_j(w_ij/||Δz_ij||)
                         ≈ sqrt(Lambda_max(z_i) / lambda_0(z_i))
                         (local metric condition number, Lemma lem:proxy_bound(b))

    The condition C_prime * kappa_cond(i) * C_hat_Gamma(i) * r_i² ≤ eps_accuracy
    guarantees C_Gamma(z_i)/sqrt(lambda_0) * r_i² ≤ eps_accuracy (Lemma
    lem:proxy_bound(c)), so the per-edge linearization error |d_R - w_ij| ≤
    eps_accuracy * r_i / 2 (Theorem thm:adaptive_radius(a)).

    Nodes with no incident triangles (C_hat_Gamma = 0) are assigned r_i = r_n
    (no restriction: the linear approximation is exact for locally flat regions).

    Parameters
    ----------
    z_mu         : (N, d) latent codes
    edge_index   : (2, E) current graph
    edge_weight  : (E,)  Riemannian distances w_ij = ||l_ij||
    triangles    : (T, 3) triangles in the graph (from find_triangles)
    c_ijk        : (T, G) ambient closure vectors (from ambient_closure_vectors)
    r_n          : float  global KNN radius (hard upper bound on r_i)
    eps_accuracy : float  target per-edge accuracy ε (Theorem thm:adaptive_radius)
    C_prime      : float  constant from Lemma lem:proxy_bound(c); default 3.0
                          (from the 3-term closure bound ||c|| ≤ 3||H_f||r²)
    r_min_fraction : float  clamp r_i ≥ r_min_fraction * r_n (prevent collapse)
    eps_denom    : float  numerical floor for divisions

    Returns
    -------
    node_radii : (N,) tensor in [r_min_fraction * r_n, r_n]
    """
    N = z_mu.shape[0]
    device, dtype = z_mu.device, z_mu.dtype

    # --- Step 1: Closure proxy per node ---
    c_hat = closure_proxy_per_node(
        z_mu, triangles, c_ijk, eps=eps_denom, latent_space=latent_space,
    )  # (N,)

    # --- Step 2: Local metric condition number per node ---
    # kappa_cond(i) = max_j(w_ij/||Δz_ij||) / min_j(w_ij/||Δz_ij||)
    # The ratio w_ij/||Δz_ij|| = ||J_f(z_i) * (Δz_ij/||Δz_ij||)|| estimates
    # sqrt(lambda_max) in direction Δz_ij. Taking max/min over neighbors
    # estimates sqrt(Lambda_max(z_i) / lambda_0(z_i)).
    src, dst = edge_index[0], edge_index[1]
    dz_norm = _latent_delta(z_mu[src], z_mu[dst], latent_space).norm(dim=-1).clamp(min=eps_denom)  # (E,)
    stretch = edge_weight / dz_norm                                        # (E,)

    stretch_max = torch.zeros(N, dtype=dtype, device=device)
    stretch_min = torch.full((N,), float("inf"), dtype=dtype, device=device)

    stretch_max.scatter_reduce_(0, src, stretch, reduce="amax", include_self=True)
    stretch_min.scatter_reduce_(0, src, stretch, reduce="amin", include_self=True)

    # Nodes with no outgoing edges: condition number = 1 (isotropic, no restriction)
    no_edges = stretch_max < eps_denom
    stretch_min = torch.where(no_edges, torch.ones_like(stretch_min), stretch_min)
    stretch_max = torch.where(no_edges, torch.ones_like(stretch_max), stretch_max)

    kappa_cond = (stretch_max / stretch_min.clamp(min=eps_denom)).clamp(min=1.0)  # (N,)

    # --- Step 3: Adaptive radius ---
    # r_i = min(r_n, sqrt(eps_accuracy / (C_prime * kappa_cond * c_hat)))
    # When c_hat = 0 (locally flat), r_i = r_n (no restriction).
    denom = C_prime * kappa_cond * c_hat                                   # (N,)

    r_adaptive = torch.where(
        denom > eps_denom,
        (eps_accuracy / denom).clamp(min=0.0).sqrt(),
        torch.full((N,), r_n, dtype=dtype, device=device),
    )

    r_floor = r_min_fraction * r_n
    node_radii = r_adaptive.clamp(min=r_floor, max=r_n)
    return node_radii


def curvature_proxy(
    decoder: nn.Module,
    z_mu: torch.Tensor,
    triangles: torch.Tensor,
    eps: float = 1e-8,
    latent_space: str = "euclidean",
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
    l_ij = riemannian_log_maps_batched(decoder, z_i, _latent_delta(z_i, z_j, latent_space))
    l_jk = riemannian_log_maps_batched(decoder, z_j, _latent_delta(z_j, z_k, latent_space))
    l_ki = riemannian_log_maps_batched(decoder, z_k, _latent_delta(z_k, z_i, latent_space))
    l_ik = riemannian_log_maps_batched(decoder, z_i, _latent_delta(z_i, z_k, latent_space))  # denominator, base z_i

    c = l_ij + l_jk + l_ki
    closure_norm = c.norm(dim=-1)

    norm_ij = l_ij.norm(dim=-1)
    norm_ik = l_ik.norm(dim=-1)  # ||l_{i->k}||, exact match to Definition 4
    denom = norm_ij * norm_ik

    # Filter degenerate triangles: when either log-map has near-zero norm the
    # denominator is ~1e-16 and swamps the eps=1e-8 guard, producing kappa ≫ 1.
    # We use a threshold of 1e-4 on denom (≈ edge lengths of 1e-2 each), which
    # is well below any physically meaningful triangle in a trained latent space.
    min_denom = 1e-4
    valid = denom > min_denom
    kappa = torch.zeros_like(closure_norm)
    kappa[valid] = closure_norm[valid] / denom[valid]
    return kappa
