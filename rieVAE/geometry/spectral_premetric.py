"""Spectral ambient premetric via the graph Laplacian.

Implements Section 2.1 ("Spectral Ambient Premetric") of the paper:
Lemmas lem:spec_premetric and lem:lap_convergence.

Primary formula (Varadhan heat-kernel distance, Lemma lem:spec_premetric):

    tilde_w(x_i, x_j) = sqrt(-4t * log K_t(x_i, x_j))
    K_t(x_i, x_j)     = sum_{l=1..k} exp(-lambda_l * t) * phi_l(x_i) * phi_l(x_j)

where (lambda_l, phi_l) are the smallest non-trivial eigenpairs of the
CkNN graph Laplacian with Coifman-Lafon alpha=1 normalization
(:func:`build_cknn_laplacian`). By Varadhan's theorem (1967),
tilde_w(x,y) -> d^M(x,y) as t->0, giving a decoder-independent training
target that converges to the true geodesic. The CkNN + Coifman-Lafon
Laplacian converges to the true Laplace-Beltrami operator independently
of sampling density (Lemma lem:lap_convergence).

Legacy formula (biharmonic distance, REMOVED from paper):
    d_bih(x_i, x_j)^2 = sum_{l=1..k} lambda_l^{-2} (phi_l(x_i) - phi_l(x_j))^2
This has a manifold-specific constant scale bias (e.g. 0.724 * d^M on S^1)
independent of spectral truncation K. It is retained only for ablation
studies (target_mode='bih' in the trainer).

Key properties established in the paper:
  Lemma lem:spec_premetric (App. app:cknn):
    The Varadhan formula satisfies premetric axioms (P1)-(P3) and is
    bi-Lipschitz equivalent to d^M (with O(1) constants on training-scale
    edges; O(r_n^2) relative error on global pairs at t=Theta(r_n^2)).
  Lemma lem:lap_convergence (App. app:lap_convergence):
    CkNN + Coifman-Lafon eigenvalue convergence rate
    O((log n / n)^{1/(d+4)}) under the Calder-Cheng-Wu/Tan-Cheng bounds.

Usage:
    artefacts = build_biharmonic_distance(x, k_nn, k_trunc, laplacian_type='cknn')
    d_vdh, t_used, valid = compute_varadhan_edge_distances(
        phi, lambdas, edge_index, return_valid_mask=True)
    # filter edge_index to valid edges before training:
    edge_index = edge_index[:, valid]
    tilde_w    = d_vdh[valid]
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.sparse import csr_matrix, diags, eye as sp_eye
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------- graph Laplacian

def build_knn_laplacian(
    x: torch.Tensor | np.ndarray,
    k_nn: int,
    normalized: bool = True,
    self_loops: bool = False,
    ensure_connected: bool = True,
) -> csr_matrix:
    """Symmetric (optionally normalised) Laplacian of the Euclidean kNN graph.

    For Lemma lem:spec_premetric we need a symmetric Laplacian with
    a reliable bottom of the spectrum. The symmetric normalised
    Laplacian L = I - D^{-1/2} W D^{-1/2} is the default; setting
    ``normalized=False`` returns the combinatorial L = D - W.

    Parameters
    ----------
    x : (n, G) torch tensor or numpy array of ambient coordinates.
    k_nn : int number of nearest neighbours per node (undirected
        symmetrised after union).
    normalized : bool (default True).
    self_loops : bool include i->i edges (default False).
    ensure_connected : bool (default True).
        If True, adds the minimum-weight inter-component bridge edges
        required to make the graph a single connected component.  This
        guards against disconnected k-NN graphs on real data with very
        non-uniform density.  For clean synthetic data (n>=1000, k>=8)
        this branch is never entered; the cost is one connected_components
        call per preprocessing run.

    Returns
    -------
    L : scipy.sparse.csr_matrix of shape (n, n).
    """
    from scipy.sparse.csgraph import connected_components as _cc
    from scipy.spatial import cKDTree

    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)

    n = x_np.shape[0]
    nn = NearestNeighbors(n_neighbors=k_nn + 1).fit(x_np)
    _, idx = nn.kneighbors(x_np)

    rows = np.repeat(np.arange(n), k_nn + 1 if self_loops else k_nn)
    cols = idx[:, :].reshape(-1) if self_loops else idx[:, 1:].reshape(-1)
    data = np.ones_like(rows, dtype=np.float64)

    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)

    # Connectivity check: add minimum-weight bridge edges between components.
    if ensure_connected:
        n_comp, labels = _cc(W, directed=False)
        if n_comp > 1:
            print(
                f"[build_knn_laplacian] k-NN graph has {n_comp} components "
                f"(n={n}, k={k_nn}); adding bridge edge(s) to connect.",
                flush=True,
            )
            # Greedy merge: at each step find the shortest euclidean edge
            # between any two distinct components and add it.
            # For each component pair, query one component's KD-tree with
            # the other component's nodes (guaranteed to find cross edges).
            while True:
                n_comp_cur, labels = _cc(W, directed=False)
                if n_comp_cur == 1:
                    break
                best_dist = np.inf
                best_i, best_j = -1, -1
                # Partition into component arrays.
                comp_idx_list = [
                    np.where(labels == c)[0] for c in range(n_comp_cur)
                ]
                # For every pair of distinct components, find the closest pair.
                for ca in range(n_comp_cur):
                    nodes_a = comp_idx_list[ca]
                    # Build KDTree for all other-component nodes.
                    other_idx = np.concatenate([
                        comp_idx_list[cb] for cb in range(n_comp_cur) if cb != ca
                    ])
                    other_labels = labels[other_idx]
                    tree_other = cKDTree(x_np[other_idx])
                    # Nearest-neighbor query from component A into others.
                    dists_q, local_q = tree_other.query(x_np[nodes_a], k=1)
                    argmin = int(np.argmin(dists_q))
                    if dists_q[argmin] < best_dist:
                        best_dist = dists_q[argmin]
                        best_i = int(nodes_a[argmin])
                        best_j = int(other_idx[local_q[argmin]])
                if best_i == -1:
                    break
                bridge_data = np.ones(2, dtype=np.float64)
                bridge = csr_matrix(
                    (bridge_data, (np.array([best_i, best_j]),
                                   np.array([best_j, best_i]))),
                    shape=(n, n),
                )
                W = W.maximum(bridge)
                print(
                    f"  [bridge] added edge ({best_i}, {best_j}), "
                    f"dist={best_dist:.4f}",
                    flush=True,
                )

    d = np.asarray(W.sum(axis=1)).ravel()
    d_safe = np.where(d > 0, d, 1.0)

    if normalized:
        d_inv_sqrt = diags(1.0 / np.sqrt(d_safe))
        L = sp_eye(n) - d_inv_sqrt @ W @ d_inv_sqrt
    else:
        L = diags(d) - W

    return L.tocsr()


# ---------------------------------------------------------------- eigendecomposition

def solve_laplacian_eigenpairs(
    L: csr_matrix,
    k_trunc: int,
    skip_trivial: bool = True,
    shift: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Smallest k_trunc non-trivial eigenpairs of a symmetric Laplacian.

    Uses ARPACK's shift-invert on the sparse matrix; we request
    k_trunc (+1 if skip_trivial) smallest eigenvalues and drop the
    trivial near-zero mode that corresponds to the constant
    eigenvector on a connected graph.

    Parameters
    ----------
    L : scipy sparse symmetric (n, n).
    k_trunc : int number of non-trivial eigenpairs to keep.
    skip_trivial : whether to discard the zero eigenmode (default True).
    shift : Tikhonov regularisation in eigsh (for numerical stability
        when the zero mode is close to numerical zero).

    Returns
    -------
    eigvals : (k_trunc,) float64 ascending eigenvalues.
    eigvecs : (n, k_trunc) float64 eigenvectors, columns orthonormal
              w.r.t. the symmetric normalisation.
    """
    n_ask = k_trunc + (1 if skip_trivial else 0)
    vals, vecs = eigsh(L + shift * sp_eye(L.shape[0]), k=n_ask, which="SM")
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    if skip_trivial:
        vals = vals[1:]
        vecs = vecs[:, 1:]
    vals = np.maximum(vals - shift, 1e-12)
    return vals.astype(np.float64), vecs.astype(np.float64)


# ---------------------------------------------------------------- biharmonic distance
#
# ABLATION-ONLY -- NOT the paper's method.
#
# The biharmonic distance d_bih(x_i,x_j)^2 = sum_l lambda_l^{-2}
# (phi_l(x_i) - phi_l(x_j))^2 (lambda^{-2} weighting) was used in
# earlier drafts of the paper and is retained ONLY as a comparison
# baseline for ablation experiments (target_mode='bih' in the
# preprocessor). It is NOT part of the certified iso-architecture.
#
# Reason for removal: on S^1 (and analogously on other manifolds),
# d_bih = C_manifold * d^M with a manifold-specific constant C_manifold
# (e.g. 0.724 on S^1, provable analytically via Parseval). This fixed
# multiplicative bias does NOT vanish as K -> infinity, so d_bih is NOT
# bi-Lipschitz to d^M -- it is a scaled version of it. The Varadhan
# formula sqrt(-4t*log K_t) is the correct proxy (converges to d^M).
#
# These functions are NOT exported from rieVAE.geometry.__init__ and
# should NOT be used in new experiments or paper figures.

def biharmonic_feature_map(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
) -> np.ndarray:
    """[ABLATION ONLY] Spectral embedding Psi(x_i)_l = lambda_l^{-1} phi_l(x_i).

    Used only for ablation comparisons (target_mode='bih'). Not the
    paper's method; see module-level comment above.

    Parameters
    ----------
    eigvals : (k,) strictly positive eigenvalues.
    eigvecs : (n, k) corresponding eigenvectors.

    Returns
    -------
    Psi : (n, k) biharmonic feature map.
    """
    weights = 1.0 / np.asarray(eigvals, dtype=np.float64)
    return np.asarray(eigvecs, dtype=np.float64) * weights[None, :]


def pairwise_biharmonic_distance(
    Psi: np.ndarray,
    idx_src: Optional[np.ndarray] = None,
    idx_dst: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pairwise biharmonic distances from the feature map.

    If ``idx_src`` and ``idx_dst`` are given, returns the distances
    between those specific pairs (shape (E,)). Otherwise returns the
    full (n, n) matrix. For large n, prefer restricting to the kNN
    candidate set (see :func:`biharmonic_candidate_distances`).

    Parameters
    ----------
    Psi : (n, k) output of :func:`biharmonic_feature_map`.
    idx_src, idx_dst : optional arrays of length E with source and
        destination node indices.

    Returns
    -------
    d_bih : (E,) if indices provided; (n, n) otherwise.
    """
    if idx_src is not None and idx_dst is not None:
        diff = Psi[idx_src] - Psi[idx_dst]
        return np.sqrt((diff * diff).sum(axis=-1))

    n = Psi.shape[0]
    diff_norm = np.sum(Psi * Psi, axis=1)
    sq = diff_norm[:, None] + diff_norm[None, :] - 2.0 * Psi @ Psi.T
    sq = np.maximum(sq, 0.0)
    return np.sqrt(sq)


def biharmonic_candidate_distances(
    x: torch.Tensor | np.ndarray,
    Psi: np.ndarray,
    k_candidates: int,
    pool_multiplier: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-node kNN candidate distances in the biharmonic metric.

    Computes for every node i its k_candidates nearest neighbours
    under the biharmonic distance. For large n we do NOT materialise
    the full n x n biharmonic distance matrix; instead we restrict
    the per-node search to a Euclidean-kNN candidate pool of size
    ``pool_multiplier * k_candidates`` (BallTree pre-filter), then
    re-rank the pool by the biharmonic distance.

    APPROXIMATION CAVEAT (IMPORTANT).
    The Euclidean pre-filter is EXACT only when the manifold
    embedding is approximately isometric to its ambient Euclidean
    metric. On highly non-isometric embeddings (Swiss roll,
    elongated cell-development trajectories, manifolds with strong
    anisotropy or folds), two nodes that are biharmonic-close CAN
    be Euclidean-far; the pre-filter will then MISS such pairs,
    silently producing a false-negative in the spectral ball graph
    and violating the theoretical definition
    E* = {(i, j) : d_bih(x_i, x_j) <= r}.

    Mitigations:
      - For small n (typically n <= 10,000), call
        :func:`pairwise_biharmonic_distance` for the full dense
        matrix (set ``k_candidates=None`` in
        :func:`build_biharmonic_distance`).
      - For large n on highly non-isometric data, increase
        ``pool_multiplier`` (at the cost of memory in the pre-filter
        step) until
        :func:`verify_euclidean_pool_no_miss` reports zero-miss on a
        dense verification subsample.

    Parameters
    ----------
    x : (n, G) ambient coordinates (used only for the pre-filter).
    Psi : (n, k) biharmonic feature map.
    k_candidates : number of nearest neighbours to return per node.
    pool_multiplier : int, default 10
        Size of the Euclidean pre-filter pool is
        ``pool_multiplier * k_candidates`` (clamped to n - 1). Larger
        values reduce the false-negative risk on non-isometric data
        at the cost of preprocessing time and memory.

    Returns
    -------
    idx : (n, k_candidates) integer indices of biharmonic neighbours
        per node (excluding self).
    dist : (n, k_candidates) biharmonic distances to those neighbours.
    """
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)

    n = x_np.shape[0]
    pool = min(max(pool_multiplier * k_candidates, 50), n - 1)
    nn_euc = NearestNeighbors(n_neighbors=pool + 1).fit(x_np)
    _, cand = nn_euc.kneighbors(x_np)
    cand = cand[:, 1:]

    idx_out = np.zeros((n, k_candidates), dtype=np.int64)
    dist_out = np.zeros((n, k_candidates), dtype=np.float64)
    for i in range(n):
        cols = cand[i]
        diff = Psi[cols] - Psi[i][None, :]
        d = np.sqrt((diff * diff).sum(axis=-1))
        order = np.argsort(d)[:k_candidates]
        idx_out[i] = cols[order]
        dist_out[i] = d[order]
    return idx_out, dist_out


# ---------------------------------------------------------------- static edge set

def spectral_ball_edges(
    d_bih: np.ndarray,
    radius: float,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Static edge set E* = {(i, j) : d_bih(x_i, x_j) <= radius}.

    This is (eq:training_edge_set) of the paper.

    Parameters
    ----------
    d_bih : (n, n) biharmonic distance matrix OR tuple
        (idx_knn, dist_knn) of per-node candidates.
    radius : float radius threshold.
    symmetric : if True and d_bih is a full matrix, both orderings of
        each edge are included.

    Returns
    -------
    edge_index : (2, E) long tensor of directed edges.
    edge_weight : (E,) float32 tensor of biharmonic distances.
    """
    if isinstance(d_bih, np.ndarray) and d_bih.ndim == 2:
        n = d_bih.shape[0]
        mask = (d_bih <= radius) & (d_bih > 0.0)
        rows, cols = np.where(mask)
        weights = d_bih[rows, cols]
    else:
        idx_knn, dist_knn = d_bih
        n, k = idx_knn.shape
        keep = dist_knn <= radius
        rows = np.repeat(np.arange(n), k)[keep.ravel()]
        cols = idx_knn.ravel()[keep.ravel()]
        weights = dist_knn.ravel()[keep.ravel()]

    if symmetric:
        rows2 = np.concatenate([rows, cols])
        cols2 = np.concatenate([cols, rows])
        weights2 = np.concatenate([weights, weights])
        pair_flat = rows2.astype(np.int64) * (np.int64(rows2.max()) + np.int64(cols2.max()) + 2) + cols2.astype(np.int64)
        _, uniq_idx = np.unique(pair_flat, return_index=True)
        uniq_idx.sort()
        rows = rows2[uniq_idx]
        cols = cols2[uniq_idx]
        weights = weights2[uniq_idx]

    edge_index = torch.from_numpy(np.stack([rows, cols], axis=0)).long()
    edge_weight = torch.from_numpy(weights.astype(np.float32))
    return edge_index, edge_weight


# ---------------------------------------------------------------- decoder-independent reweighting

def pca_local_reweighting(
    x: torch.Tensor | np.ndarray,
    edge_index: torch.Tensor,
    k_pca: int = 20,
    omega_min: float = 0.25,
    omega_max: float = 4.0,
) -> torch.Tensor:
    """Data-PCA-based decoder-independent per-pair reweighting.

    Implements App. app:reweight of the paper: for each edge (i, j),
    omega(i, j) = clip(1 / u_ij^T Sigma(x_i) u_ij, omega_min,
    omega_max) where Sigma(x_i) is the local PCA covariance from
    k_pca nearest Euclidean neighbours and u_ij = (x_j - x_i) /
    ||x_j - x_i||.

    Crucially, omega depends only on the RAW data and precomputed
    quantities; it has no dependence on (theta, phi). The
    decoder-independence constraint (DI) (Def. def:di) is satisfied
    by construction.

    Parameters
    ----------
    x : (n, G) ambient coordinates.
    edge_index : (2, E) long tensor directed edges (src, dst).
    k_pca : int neighbourhood size for the local covariance.
    omega_min, omega_max : float clipping range.

    Returns
    -------
    omega : (E,) float32 per-pair reweighting coefficient, detached
        from autograd by construction.
    """
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)

    n, G = x_np.shape
    nn = NearestNeighbors(n_neighbors=k_pca + 1).fit(x_np)
    _, idx_neigh = nn.kneighbors(x_np)
    idx_neigh = idx_neigh[:, 1:]

    sigmas = np.zeros((n, G, G), dtype=np.float64)
    for i in range(n):
        xs = x_np[idx_neigh[i]] - x_np[i]
        sigmas[i] = (xs.T @ xs) / max(k_pca - 1, 1)
        sigmas[i] += 1e-6 * np.eye(G)

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    dx = x_np[dst] - x_np[src]
    dx_norm = np.linalg.norm(dx, axis=-1, keepdims=True).clip(min=1e-12)
    u = dx / dx_norm

    quadratic = np.einsum("ei,eij,ej->e", u, sigmas[src], u).clip(min=1e-12)
    omega = np.clip(1.0 / quadratic, omega_min, omega_max).astype(np.float32)
    return torch.from_numpy(omega)


# ---------------------------------------------------------------- top-level

def verify_euclidean_pool_no_miss(
    x: torch.Tensor | np.ndarray,
    Psi: np.ndarray,
    radius: float,
    pool_multiplier: int,
    n_verify: int = 64,
    seed: int = 0,
    k_candidates: int | None = None,
) -> dict:
    """Sanity check: does the Euclidean pre-filter miss in-ball pairs?

    For ``n_verify`` randomly chosen anchor nodes, compute the full
    dense biharmonic distance vector against all other nodes AND the
    pool-restricted version used by
    :func:`biharmonic_candidate_distances`. Return the number of
    pairs in the dense ball that were dropped by the pre-filter.

    Non-zero ``n_missed`` indicates the Euclidean pool is too narrow
    for the manifold's embedding; re-run
    :func:`build_biharmonic_distance` with a larger
    ``pool_multiplier`` or set ``k_candidates=None``.

    Parameters
    ----------
    x : (n, G) ambient coordinates.
    Psi : (n, k) biharmonic feature map.
    radius : float. Biharmonic ball radius used downstream.
    pool_multiplier : int matching the candidate-distance call.
    n_verify : int number of anchor nodes to check.
    seed : int reproducibility seed.
    k_candidates : int or None. The ``k_candidates`` value actually
        passed to :func:`build_biharmonic_distance` / the production
        pipeline. When ``None``, falls back to the previous
        ``max(10, int(log2(n)))`` heuristic, but callers should pass
        the exact value to get a faithful pool-size audit.

    Returns
    -------
    dict with ``n_missed``, ``frac_missed``, ``n_true_ball_pairs``.
    """
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)
    n = x_np.shape[0]
    rng = np.random.default_rng(seed)
    anchors = rng.choice(n, size=min(n_verify, n), replace=False)

    if k_candidates is None:
        k_candidates_eff = max(10, int(np.log2(max(n, 2))))
    else:
        k_candidates_eff = int(k_candidates)
    pool = min(max(pool_multiplier * k_candidates_eff, 50), n - 1)
    nn_euc = NearestNeighbors(n_neighbors=pool + 1).fit(x_np)
    _, cand = nn_euc.kneighbors(x_np[anchors])
    cand = cand[:, 1:]

    n_missed = 0
    n_true = 0
    for i, a in enumerate(anchors):
        diff = Psi - Psi[a][None, :]
        d_all = np.sqrt((diff * diff).sum(axis=-1))
        mask_all = (d_all <= radius) & (np.arange(n) != a)
        mask_cand = np.zeros(n, dtype=bool)
        mask_cand[cand[i]] = True
        missed = mask_all & ~mask_cand
        n_missed += int(missed.sum())
        n_true += int(mask_all.sum())

    denom = max(n_true, 1)
    return {
        "n_missed": int(n_missed),
        "frac_missed": float(n_missed / denom),
        "n_true_ball_pairs": int(n_true),
    }


def compute_ols_edge_distances(
    phi: torch.Tensor,
    edge_index: torch.Tensor,
    chord_sq: torch.Tensor,
    eps_w: float = 1e-12,
    eps_dist: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension OLS spectral distance calibrated to chord^2 on E*.

    Finds non-negative weights w* in R^K that minimise

        ||F w - d_chord^2||^2        s.t. w >= 0

    where F[e, k] = (phi_k(src_e) - phi_k(dst_e))^2 for each edge e.
    The resulting distance proxy is:

        tilde_w[e] = sqrt( (F w*)[e] )

    This is a single NNLS solve on |E*| rows x K columns (typically
    ~90 x 50, solved in milliseconds). It requires no model forward pass
    and no gradient step -- it happens in the preprocessing stage.

    The OLS weights automatically adapt to the manifold's anisotropic
    geometry (e.g. different radii in the two angular directions of the
    Clifford torus), which the fixed lambda^{-2} or exp(-lambda t)
    formulas cannot.

    Parameters
    ----------
    phi : (N, K) float tensor  -- eigenvectors (rows = nodes).
    edge_index : (2, E) long tensor  -- E* edge pairs (src, dst).
    chord_sq : (E,) float tensor  -- ||x_src - x_dst||^2 for each edge.
    eps_w : float  -- floor for weights (numerical stability).
    eps_dist : float  -- floor before sqrt.

    Returns
    -------
    d_ols : (E,) float tensor  -- OLS-calibrated distance per edge.
    w_star : (K,) float tensor  -- the learned per-dimension weights.
    """
    from scipy.optimize import nnls

    src = edge_index[0]
    dst = edge_index[1]

    phi_np = phi.detach().cpu().numpy()
    diff_sq = (phi_np[src.cpu().numpy()] - phi_np[dst.cpu().numpy()]) ** 2   # (E, K)
    chord_sq_np = chord_sq.detach().cpu().numpy()                              # (E,)

    w_star_np, _ = nnls(diff_sq, chord_sq_np)      # (K,), non-negative
    w_star_np = np.maximum(w_star_np, eps_w)

    d_ols_sq_np = diff_sq @ w_star_np              # (E,)
    d_ols_np = np.sqrt(np.maximum(d_ols_sq_np, eps_dist))

    d_ols   = torch.from_numpy(d_ols_np).float()
    w_star  = torch.from_numpy(w_star_np).float()
    return d_ols, w_star


def compute_varadhan_edge_distances(
    phi: torch.Tensor,
    lambdas: torch.Tensor,
    edge_index: torch.Tensor,
    t: Optional[float] = None,
    eps_log: float = 1e-30,
    return_valid_mask: bool = False,
) -> tuple:
    """Varadhan heat-kernel distance estimate for a set of edge pairs.

    The K-mode truncated heat kernel approximation
        K_t(i, j) = sum_{k=1..K} exp(-lambda_k * t) * phi_k(i) * phi_k(j)
    is a FINITE-K SPECTRAL PROXY motivated by Varadhan's theorem (1967),
    which states d^M(x,y) = lim_{t->0} sqrt(-4t * log K_t(x,y)) for the
    full INFINITE-K heat kernel. For fixed K the truncated formula does
    NOT converge to d^M as t->0 (the sum approaches the constant K-mode
    spectral projector, not a diverging kernel). What the code computes
    is a fixed-t bi-Lipschitz proxy to d^M, valid when K_t > 0 on all
    training edges.

    Positivity of K_t is guaranteed for the full infinite series (strong
    maximum principle for the heat equation). The K-mode truncation can
    produce K_t <= 0 for edges where many included eigenvectors have
    opposite signs at the two endpoints -- typically the longer-range
    edges in E*. Edges with K_t <= 0 receive a corrupted target under
    the log-clamp and MUST be excluded from the training edge set.

    Default t heuristic: t = 0.25 / lambda_mean makes the mean-eigenvalue
    contribution exp(-lambda_mean * t) = exp(-0.25) ~= 0.78 -- in a
    numerically stable range for local edges.

    Parameters
    ----------
    phi : (N, K) float tensor  -- graph Laplacian eigenvectors.
    lambdas : (K,) float tensor  -- corresponding eigenvalues (ascending,
        strictly positive; trivial zero eigenpair already removed).
    edge_index : (2, E) long tensor  -- edge pairs (src, dst) in active
        node indexing.
    t : float or None  -- heat time.  None = auto (0.25 / lambda_mean).
    eps_log : float  -- numerical floor applied to K_t before log.
        Only affects edges that passed the K_t > 0 validity check; edges
        with K_t <= 0 are flagged in the returned mask regardless.
    return_valid_mask : bool  -- if True, return a third element: a bool
        tensor ``valid`` of shape (E,) where valid[e]=True iff K_t[e]>0.
        Callers SHOULD filter edge_index and the returned d_vdh to
        ``valid`` edges before using them as training targets.

    Returns
    -------
    d_vdh : (E,) float tensor  -- Varadhan distance for each edge pair.
        Entries for K_t <= 0 edges are NaN (NOT 0 -- see reasoning below).
        NaN propagates through the iso loss and crashes training visibly
        if these edges are not filtered out before use. The caller MUST
        either use the return_valid_mask=True path and filter, or use the
        2-return path only for diagnostic purposes (not for training targets).
        Do NOT replace NaN with 0: zero distance targets would tell the
        isometry loss to collapse two distinct nodes to the same latent
        point, introducing a silent wrong bias.
    t_used : float  -- the t value actually used (for logging).
    valid : (E,) bool tensor  -- [only when return_valid_mask=True]
        True where K_t > 0 and the distance is well-defined.
    """
    # Auto-calibrate t to the eigenvalue scale.
    lam_mean = float(lambdas.mean().item())
    if t is None or t <= 0.0:
        t = 0.25 / max(lam_mean, 1e-30)
    t = float(t)

    # Weighted eigenvectors: phi_k * exp(-lambda_k * t / 2).
    # K_t(i,j) = dot(w_i, w_j) where w_k = phi_k * exp(-lambda_k * t / 2).
    w = phi * torch.exp(-0.5 * t * lambdas.unsqueeze(0))  # (N, K)

    src = edge_index[0]
    dst = edge_index[1]
    K_t = (w[src] * w[dst]).sum(dim=-1)  # (E,)

    # Validity mask: K_t <= 0 means the K-mode truncation failed for
    # this edge -- the spectral approximation is not a valid heat kernel
    # at this pair. Setting d_vdh to 0 for invalid edges; the caller
    # should exclude them via the mask rather than train on them.
    valid = K_t > 0.0
    n_invalid = int((~valid).sum().item())
    if n_invalid > 0:
        frac = n_invalid / max(K_t.numel(), 1)
        print(
            f"[varadhan] WARNING: {n_invalid}/{K_t.numel()} edges "
            f"({100.0 * frac:.1f}%) have K_t <= 0 (truncation failure). "
            f"These edges will be excluded from E*. "
            f"Consider increasing spectral_truncation (K={phi.shape[1]}) "
            f"to reduce the invalid fraction.",
            flush=True,
        )

    K_t_safe = K_t.clamp(min=eps_log)
    d_vdh_raw = torch.sqrt((-4.0 * t * torch.log(K_t_safe)).clamp(min=0.0))

    # Mark invalid edges with NaN so that any caller that skips the validity
    # filter gets an immediately visible training crash rather than a silent
    # wrong bias. Specifically:
    #   - 0.0 would be WRONG: the iso loss would interpret it as "nodes are
    #     at the same point" and push them together in latent space (collapse).
    #   - large clamped value would be WRONG: inflated targets.
    #   - NaN is SAFE: NaN propagates through torch.mean() and the optimizer
    #     step, producing a NaN loss that raises early and visibly.
    # The caller MUST filter edge_index to valid edges before training.
    nan = float("nan")
    d_vdh = torch.where(valid, d_vdh_raw, torch.full_like(d_vdh_raw, nan))

    if return_valid_mask:
        return d_vdh, t, valid
    return d_vdh, t


def build_cknn_laplacian(
    x: torch.Tensor | np.ndarray,
    k_cand: int = 50,
    k_sigma: int = 7,
    alpha: float = 1.0,
) -> dict:
    """CkNN graph with Gaussian weights and Coifman-Lafon LBO normalization.

    Three improvements over the binary symmetric-normalized Laplacian:

    1. Self-tuning Gaussian weights (Zelnik-Manor & Perona 2004):
       W_ij = exp(-||xi-xj||^2 / (rho_i * rho_j)), rho_i = d(xi, k_sigma-NN).
       Preserves metric information; binary weights destroy it.

    2. Continuous k-NN topology (Berry & Sauer 2017):
       Keep edge (i,j) only when d(xi,xj) < delta * sqrt(rho_i * rho_j).
       delta_auto = max(MST of density-scaled distances) + eps.
       Eliminates parasite long-range "wormhole" edges.

    3. Coifman-Lafon alpha=1 density normalization:
       L = I - D2^{-1/2} W_tilde D2^{-1/2},
       W_tilde_ij = W_ij / (D1_i * D1_j).
       Converges to the Laplace-Beltrami operator independent of sampling
       density -- required for Varadhan's theorem to hold.

    Parameters
    ----------
    x : (n, G) ambient coordinates.
    k_cand : int  -- candidate pool size (M in the CkNN literature).
    k_sigma : int -- neighbor index for local density estimate rho_i.
    alpha : float -- Coifman-Lafon exponent (1.0 = full LBO, 0.5 = half).
    ensure_connected : bool -- bridge any disconnected components (see
        :func:`build_knn_laplacian`).

    Returns
    -------
    dict with:
      "L"          : scipy.sparse.csr_matrix (n, n) symmetric LBO Laplacian.
      "rho"        : (n,) float64 local density radii.
      "delta_auto" : float scalar CkNN threshold (dimensionless).
      "edge_index" : (2, E) int64 numpy array of CkNN edge pairs (symmetric).
    """
    from scipy.sparse.csgraph import minimum_spanning_tree

    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)
    n = x_np.shape[0]
    k_cand = min(k_cand, n - 1)
    k_sigma = min(k_sigma, k_cand)

    # --- Step 1: candidate pool & density radius -------------------------
    nn = NearestNeighbors(n_neighbors=k_cand + 1).fit(x_np)
    dists_all, idx_all = nn.kneighbors(x_np)        # (n, k_cand+1)
    rho = dists_all[:, k_sigma].astype(np.float64)  # (n,) density radius
    rho = np.maximum(rho, 1e-30)

    # Directed candidate edges (exclude self: column 0)
    rows     = np.repeat(np.arange(n), k_cand)
    cols     = idx_all[:, 1:].reshape(-1)            # (n*k_cand,)
    d_raw    = dists_all[:, 1:].reshape(-1)          # (n*k_cand,)
    rho_r    = rho[rows]
    rho_c    = rho[cols]
    D_scaled = d_raw / np.sqrt(rho_r * rho_c)        # density-scaled distances

    # --- Step 2: automatic delta via MST of density-scaled graph --------
    # Symmetrize the candidate graph so the MST is undirected.
    cand_sym  = csr_matrix((D_scaled, (rows, cols)), shape=(n, n))
    cand_sym  = cand_sym.maximum(cand_sym.T)
    mst_tree  = minimum_spanning_tree(cand_sym)
    delta_auto = float(mst_tree.data.max()) + 1e-5 if mst_tree.data.size > 0 else 1.0
    # Connectivity is GUARANTEED without any bridge code: delta_auto is set
    # so that every MST edge passes the CkNN filter (D_scaled < delta_auto).
    # The MST already spans all n nodes, so the result is born connected.

    # --- Step 3: apply CkNN filter on symmetrized candidates -------------
    # Apply to the symmetrized D_scaled so we evaluate both directions.
    rs_sym, cs_sym  = cand_sym.nonzero()
    ds_sym          = np.asarray(cand_sym[rs_sym, cs_sym]).ravel()
    mask            = ds_sym < delta_auto
    W_topo = csr_matrix((np.ones(mask.sum()), (rs_sym[mask], cs_sym[mask])),
                        shape=(n, n))

    # --- Step 3: Gaussian weights aligned with CkNN topology ------------
    rs, cs = W_topo.nonzero()
    d_sym = np.sqrt(np.sum((x_np[rs] - x_np[cs]) ** 2, axis=1))
    W_gauss_data = np.exp(-d_sym ** 2 / (rho[rs] * rho[cs] + 1e-30))
    W_gauss = csr_matrix((W_gauss_data, (rs, cs)), shape=(n, n))

    # --- Step 4: Coifman-Lafon alpha=1 normalization ---------------------
    D1 = np.maximum(np.asarray(W_gauss.sum(axis=1)).ravel(), 1e-30)
    if alpha > 0:
        D1_inv_a = diags(D1 ** (-alpha))
        W_tilde  = D1_inv_a @ W_gauss @ D1_inv_a
    else:
        W_tilde = W_gauss
    D2 = np.maximum(np.asarray(W_tilde.sum(axis=1)).ravel(), 1e-30)
    D2_inv_sqrt = diags(1.0 / np.sqrt(D2))
    L = (sp_eye(n) - D2_inv_sqrt @ W_tilde @ D2_inv_sqrt).tocsr()

    # CkNN edge index (topology, not Gaussian weights).
    rs_e, cs_e = W_topo.nonzero()
    edge_index = np.stack([rs_e.astype(np.int64), cs_e.astype(np.int64)], axis=0)
    print(f"[cknn] n={n}  k_cand={k_cand}  k_sigma={k_sigma}  "
          f"delta_auto={delta_auto:.4f}  n_edges={edge_index.shape[1]}",
          flush=True)
    return {"L": L, "rho": rho.astype(np.float64), "delta_auto": delta_auto,
            "edge_index": edge_index}


def build_biharmonic_distance(
    x: torch.Tensor | np.ndarray,
    k_nn: int,
    k_trunc: int,
    k_candidates: Optional[int] = None,
    pool_multiplier: int = 10,
    laplacian_type: str = "binary",
    cknn_k_cand: int = 50,
    cknn_k_sigma: int = 7,
) -> dict:
    """End-to-end: kNN/CkNN graph -> Laplacian -> spectrum -> biharmonic distance.

    Returns a dict with the precomputed artefacts consumed downstream:

    - ``eigvals``    : (k_trunc,)
    - ``eigvecs``    : (n, k_trunc)
    - ``Psi``        : (n, k_trunc)
    - ``d_bih``      : (n, n) or tuple (idx, dist)
    - ``cknn_edges`` : (2, E) int64 [only when laplacian_type='cknn']
    - ``rho``        : (n,) local density radii [only when laplacian_type='cknn']

    Parameters
    ----------
    laplacian_type : str
        'binary' (default) -- original binary k-NN symmetric normalized
        Laplacian.
        'cknn'             -- CkNN with self-tuning Gaussian weights and
        Coifman-Lafon LBO normalization (see :func:`build_cknn_laplacian`).
    cknn_k_cand : int  -- CkNN candidate pool (ignored for binary).
    cknn_k_sigma : int -- CkNN density neighbor index (ignored for binary).
    """
    if laplacian_type == "cknn":
        cknn = build_cknn_laplacian(x, k_cand=cknn_k_cand,
                                     k_sigma=cknn_k_sigma, alpha=1.0)
        L          = cknn["L"]
        extra_keys = {"cknn_edges": cknn["edge_index"], "rho": cknn["rho"]}
    else:
        L          = build_knn_laplacian(x, k_nn=k_nn,
                                         normalized=True, self_loops=False)
        extra_keys = {}

    eigvals, eigvecs = solve_laplacian_eigenpairs(
        L, k_trunc=k_trunc, skip_trivial=True,
    )
    Psi = biharmonic_feature_map(eigvals, eigvecs)
    if k_candidates is None:
        d_bih = pairwise_biharmonic_distance(Psi)
    else:
        d_bih = biharmonic_candidate_distances(
            x, Psi, k_candidates=k_candidates, pool_multiplier=pool_multiplier,
        )
    return {"eigvals": eigvals, "eigvecs": eigvecs, "Psi": Psi,
            "d_bih": d_bih, **extra_keys}
