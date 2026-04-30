"""Graph construction for the Certified Riemannian VAE.

The training graph is static: it is the biharmonic spectral edge set
E* = {(i, j) : d_bih(x_i, x_j) <= r} computed once from the data's
graph Laplacian eigenfunctions in
:mod:`rieVAE.geometry.spectral_premetric`. The helper functions below
support the preprocessing stage that picks the radius r via the MST
connectivity threshold and constructs the initial Euclidean kNN
candidate set.

The graph is represented as a pair:
  edge_index : (2, E) long tensor  -- directed edges (src, dst)
  edge_weight : (E,)  float tensor -- biharmonic distance (default) or
                                       Euclidean distance (initialiser).

Convention: edge (i -> j) has src=i, dst=j.
For each undirected edge {i,j}, both (i->j) and (j->i) are stored.

The previous deformation-driven graph-update machinery (G-step,
deformed ambient ball, temperature-annealed Gibbs weights,
soft-to-hard gap stability) has been removed; see Section 5 of the
main paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors

from rieVAE.geometry.log_map import riemannian_log_maps_batched, riemannian_distances


def euclidean_ball_graph(
    x: torch.Tensor,
    radius: float,
    max_neighbors: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a symmetric ambient-ball graph of fixed Euclidean radius.

    For each node i, all other nodes j with ||x_j - x_i|| <= radius are
    its neighbors. Used as the INITIAL graph (Theorem thm:prox_fp:
    Euclidean ambient ball corresponds to p = 1).

    Parameters
    ----------
    x : (N, G) data points in ambient space.
    radius : float
        Euclidean radius r in R^G. Points x_j with ||x_j - x_i|| <= r
        are accepted as neighbors of x_i.
    max_neighbors : int or None
        Optional cap on the per-node degree (truncates to the closest
        ``max_neighbors`` candidates). ``None`` means no cap. A hard cap
        is important for memory in the high-density regime since a ball
        can contain O(n) points.

    Returns
    -------
    edge_index : (2, E)
    edge_weight : (E,) -- Euclidean distances (used as edge weights)
    """
    x_np = x.detach().cpu().numpy()
    if max_neighbors is None or max_neighbors >= len(x_np):
        query_k = len(x_np)
    else:
        # Over-sample a bit to ensure we keep all points within radius.
        query_k = max(max_neighbors * 2, 64)
    query_k = min(query_k, len(x_np))

    nn_model = NearestNeighbors(
        n_neighbors=query_k, metric="euclidean", n_jobs=-1,
    )
    nn_model.fit(x_np)
    distances, indices = nn_model.kneighbors(x_np)

    src_list, dst_list, dist_list = [], [], []
    for i in range(len(x_np)):
        keep = 0
        for rank in range(1, query_k):
            j = int(indices[i, rank])
            d = float(distances[i, rank])
            if d > radius:
                continue
            src_list.extend([i, j])
            dst_list.extend([j, i])
            dist_list.extend([d, d])
            keep += 1
            if max_neighbors is not None and keep >= max_neighbors:
                break

    if not src_list:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=x.device)
        edge_weight = torch.zeros(0, dtype=x.dtype, device=x.device)
        return edge_index, edge_weight

    edge_index = torch.tensor(
        [src_list, dst_list], dtype=torch.long, device=x.device,
    )
    edge_weight = torch.tensor(dist_list, dtype=x.dtype, device=x.device)
    edge_index, edge_weight = _deduplicate_edges(edge_index, edge_weight)
    return edge_index, edge_weight


def median_neighbor_radius(
    x: torch.Tensor,
    k_probe: int = 16,
) -> float:
    """Return the median k-th-Euclidean-nearest-neighbor distance.

    A robust data-driven default for the ambient-ball radius r that
    makes the expected ball degree roughly ``k_probe``. The theoretical
    scaling is r = Theta(r_n) = Theta((log n / n)^{1/d}), which matches
    ``median_k_nn(k_probe)`` at small intrinsic dimension up to a
    constant factor. Kept as a fallback when ``mst_connectivity_radius``
    is disabled.
    """
    x_np = x.detach().cpu().numpy()
    k = min(k_probe + 1, len(x_np))
    nn_model = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn_model.fit(x_np)
    distances, _ = nn_model.kneighbors(x_np)
    return float(np.median(distances[:, -1]))


def mst_connectivity_radius(
    x: torch.Tensor,
    quantile: float = 0.995,
    k_safe: int | None = None,
    return_diagnostics: bool = False,
):
    """Connectivity-threshold ambient-ball radius (Option B initializer).

    Computes the smallest Euclidean radius r such that the (1 - q)-quantile
    of the data is contained in a connected ambient-ball graph, where
    q = 1 - quantile. The radius is read off the Euclidean minimum
    spanning tree (MST) of a sparse k-NN graph (``k_safe = ceil(2 log_2 N)``
    by default, doubled until the kNN graph is connected). Theoretical
    scaling: for i.i.d. samples from a density bounded below on a compact
    d-manifold, this radius is Theta(r_n) = Theta((log n / n)^{1/d})
    (Penrose-type connectivity results).

    Outliers (points whose MST edge to the rest of the graph exceeds r)
    are flagged in ``outlier_mask`` so the trainer can drop them. For
    clean synthetic data with quantile=0.995 this is typically 0 to a
    few points; for noisy real data it is at most 0.5%.

    Parameters
    ----------
    x : (N, G) ambient coordinates.
    quantile : float in (0, 1]
        Fraction of MST edges to retain when picking the radius. The
        radius is the ``quantile`` quantile of the MST edge weights.
        ``quantile = 1.0`` recovers the strict max-MST connectivity
        threshold (no outlier rejection); 0.995 drops the top 0.5%.
    k_safe : int or None
        kNN-graph degree used to build the sparse MST candidate set.
        ``None`` selects ``max(8, ceil(2 * log_2 N))`` and doubles it
        until the kNN graph is connected (rare).
    return_diagnostics : bool
        If True, also return a dict with ``n_components_at_r``,
        ``n_components_kNN``, ``mst_edge_weights``, and ``k_safe_used``.

    Returns
    -------
    radius : float
        Connectivity radius r (Euclidean units).
    outlier_mask : (N,) bool tensor
        True where the node is an outlier (its MST edge exceeds r).
    diagnostics : dict (only if return_diagnostics=True)
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import (
        connected_components,
        minimum_spanning_tree,
    )

    x_np = x.detach().cpu().numpy()
    N = x_np.shape[0]
    if N <= 1:
        radius = 0.0
        mask = torch.zeros(N, dtype=torch.bool, device=x.device)
        if return_diagnostics:
            return radius, mask, {"n_components_kNN": N, "k_safe_used": 0,
                                  "n_components_at_r": N, "mst_edge_weights": np.array([])}
        return radius, mask

    if k_safe is None:
        k_safe = max(8, int(np.ceil(2.0 * np.log2(max(N, 2)))))
    k_safe = min(k_safe, N - 1)

    # Build the sparse kNN candidate set, doubling k_safe if the
    # resulting graph is disconnected (very rare in practice).
    n_components_knn = N + 1
    k_used = k_safe
    while True:
        nn_model = NearestNeighbors(
            n_neighbors=k_used + 1, metric="euclidean", n_jobs=-1,
        )
        nn_model.fit(x_np)
        distances, indices = nn_model.kneighbors(x_np)

        rows = np.repeat(np.arange(N), k_used)
        cols = indices[:, 1:].reshape(-1)
        dists = distances[:, 1:].reshape(-1)
        adj = csr_matrix((dists, (rows, cols)), shape=(N, N))
        # Symmetrize so MST sees both endpoints of every edge.
        adj = adj.maximum(adj.T)

        n_components_knn, _ = connected_components(adj, directed=False)
        if n_components_knn == 1 or k_used >= N - 1:
            break
        k_used = min(k_used * 2, N - 1)

    mst = minimum_spanning_tree(adj).tocoo()
    mst_weights = np.asarray(mst.data, dtype=np.float64)
    if mst_weights.size == 0:
        radius = 0.0
        mask = torch.zeros(N, dtype=torch.bool, device=x.device)
        if return_diagnostics:
            return radius, mask, {
                "n_components_kNN": int(n_components_knn),
                "k_safe_used": int(k_used),
                "n_components_at_r": int(n_components_knn),
                "mst_edge_weights": mst_weights,
            }
        return radius, mask

    radius = float(np.quantile(mst_weights, quantile))

    # A node is flagged as an outlier if all of its MST-edges exceed r.
    # In a tree, every non-root node has exactly one parent-edge; we
    # mark the node whose only MST edge exceeds r (or any leaf node
    # whose unique edge exceeds r).
    rows_mst = np.asarray(mst.row, dtype=np.int64)
    cols_mst = np.asarray(mst.col, dtype=np.int64)

    # For each node, the smallest incident MST edge weight.
    min_edge = np.full(N, np.inf, dtype=np.float64)
    for r_, c_, w_ in zip(rows_mst, cols_mst, mst_weights):
        if w_ < min_edge[r_]:
            min_edge[r_] = w_
        if w_ < min_edge[c_]:
            min_edge[c_] = w_
    outlier_np = min_edge > radius
    outlier_mask = torch.from_numpy(outlier_np).to(device=x.device)

    n_components_at_r = int((mst_weights > radius).sum() + 1)

    if return_diagnostics:
        diagnostics = {
            "n_components_kNN": int(n_components_knn),
            "k_safe_used": int(k_used),
            "n_components_at_r": n_components_at_r,
            "mst_edge_weights": mst_weights,
        }
        return radius, outlier_mask, diagnostics
    return radius, outlier_mask


def spectral_ball_graph(
    d_bih: np.ndarray | tuple[np.ndarray, np.ndarray],
    radius: float,
    device: torch.device | str = "cpu",
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the static biharmonic spectral edge set E*.

    Thin wrapper around
    :func:`rieVAE.geometry.spectral_premetric.spectral_ball_edges`
    that returns torch tensors on the requested device.

    Parameters
    ----------
    d_bih : (n, n) biharmonic-distance matrix OR tuple (idx, dist)
        of per-node kNN candidates in the biharmonic metric, as
        returned by
        :func:`rieVAE.geometry.spectral_premetric.build_biharmonic_distance`.
    radius : float
        Biharmonic radius threshold. Pairs with d_bih(x_i, x_j)
        <= radius are kept as edges.
    device : torch device to place the output tensors on.
    symmetric : bool whether to return an undirected graph (both
        (i, j) and (j, i) are edges).

    Returns
    -------
    edge_index : (2, E) long tensor.
    edge_weight : (E,) float32 tensor of biharmonic distances.
    """
    from rieVAE.geometry.spectral_premetric import spectral_ball_edges

    edge_index, edge_weight = spectral_ball_edges(
        d_bih=d_bih, radius=radius, symmetric=symmetric,
    )
    return edge_index.to(device), edge_weight.to(device)


def _full_pair_candidates(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all directed pairs (i, j) with i != j on the given device.

    O(n^2). For large n use an external candidate set instead.
    """
    ii, jj = torch.meshgrid(
        torch.arange(n, device=device),
        torch.arange(n, device=device),
        indexing="ij",
    )
    mask = ii != jj
    src = ii[mask]
    dst = jj[mask]
    return src, dst


def _cap_degree_per_node(
    src: torch.Tensor,
    dst: torch.Tensor,
    d: torch.Tensor,
    n_nodes: int,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep at most ``max_neighbors`` smallest-distance edges per source."""
    src_np = src.cpu().numpy()
    dst_np = dst.cpu().numpy()
    d_np = d.detach().cpu().numpy()

    keep_idx: list[int] = []
    bucket: dict[int, list[tuple[float, int]]] = {}
    for e, s in enumerate(src_np):
        bucket.setdefault(int(s), []).append((float(d_np[e]), e))
    for s, entries in bucket.items():
        entries.sort(key=lambda t: t[0])
        for _, e in entries[:max_neighbors]:
            keep_idx.append(e)
    keep_idx_tensor = torch.tensor(sorted(keep_idx), dtype=torch.long, device=src.device)
    return src[keep_idx_tensor], dst[keep_idx_tensor], d[keep_idx_tensor]


def euclidean_knn_graph(
    x: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a symmetric k-NN graph using Euclidean distances.

    Legacy initializer kept as the hard-kNN-limit corollary path.

    Parameters
    ----------
    x : (N, G) data points in ambient space.
    k : number of neighbors per node.

    Returns
    -------
    edge_index : (2, E) with E <= 2*N*k
    edge_weight : (E,) Euclidean distances (used as initial weights).
    """
    x_np = x.detach().cpu().numpy()
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn_model.fit(x_np)
    distances, indices = nn_model.kneighbors(x_np)

    src_list, dst_list, dist_list = [], [], []
    for i in range(len(x_np)):
        for rank in range(1, k + 1):
            j = int(indices[i, rank])
            d = float(distances[i, rank])
            src_list.extend([i, j])
            dst_list.extend([j, i])
            dist_list.extend([d, d])

    edge_index = torch.tensor(
        [src_list, dst_list], dtype=torch.long, device=x.device
    )
    edge_weight = torch.tensor(dist_list, dtype=x.dtype, device=x.device)

    edge_index, edge_weight = _deduplicate_edges(edge_index, edge_weight)
    return edge_index, edge_weight


def graph_changed(
    old_index: torch.Tensor,
    new_index: torch.Tensor,
) -> bool:
    """Check whether the graph topology has changed between iterations."""
    return graph_change_fraction(old_index, new_index) > 0.0


def graph_change_fraction(
    old_index: torch.Tensor,
    new_index: torch.Tensor,
) -> float:
    """Fraction of edges that changed between two graph iterations.

    Computes |E_old △ E_new| / max(|E_old|, |E_new|) where △ is symmetric
    difference on canonical (undirected) edges.

    Returns
    -------
    float in [0, 1]:  0 = identical graphs, 1 = completely different
    """
    def canonical_edges(idx: torch.Tensor) -> set[tuple[int, int]]:
        src, dst = idx[0].tolist(), idx[1].tolist()
        return {(min(s, d), max(s, d)) for s, d in zip(src, dst)}

    old_set = canonical_edges(old_index)
    new_set = canonical_edges(new_index)
    symmetric_diff = len(old_set.symmetric_difference(new_set))
    total = max(len(old_set), len(new_set), 1)
    return float(symmetric_diff / total)


def merge_candidate_graphs(
    edge_index_a: torch.Tensor,
    edge_weight_a: torch.Tensor,
    edge_index_b: torch.Tensor,
    edge_weight_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two directed edge sets, keeping the minimum-weight copy of each.

    Used by the density-based graph collapse guard in SCRVAETrainer: when a
    single G-step drops more than ``max_edge_loss_fraction`` of edges, the
    new collapsed graph is merged with the previous graph so that all
    previously-known neighbors remain candidates in the next iteration.

    The minimum-weight copy of each directed edge is retained, consistent
    with the KNN selection criterion (prefer the shortest Riemannian distance
    seen across any computation).

    Parameters
    ----------
    edge_index_a : (2, E_a)  -- first edge set (typically the new graph)
    edge_weight_a : (E_a,)   -- Riemannian distances for set A
    edge_index_b : (2, E_b)  -- second edge set (typically the previous graph)
    edge_weight_b : (E_b,)   -- Riemannian distances for set B

    Returns
    -------
    merged_edge_index : (2, E_merged)   -- union, deduplicated
    merged_edge_weight : (E_merged,)    -- minimum weight per unique edge
    """
    merged_index = torch.cat([edge_index_a, edge_index_b], dim=1)
    merged_weight = torch.cat([edge_weight_a, edge_weight_b])
    return _deduplicate_edges(merged_index, merged_weight)


def _deduplicate_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove duplicate directed edges, keeping the minimum-weight copy.

    Fully vectorized -- no Python loops over edges.
    """
    src, dst = edge_index[0], edge_index[1]
    no_self = src != dst
    src, dst = src[no_self], dst[no_self]
    edge_weight = edge_weight[no_self]

    if src.numel() == 0:
        return edge_index[:, :0], edge_weight[:0]

    stride = max(src.max().item(), dst.max().item()) + 1
    keys = src.long() * stride + dst.long()

    weight_order = torch.argsort(edge_weight)
    keys_by_weight = keys[weight_order]

    key_order = torch.argsort(keys_by_weight, stable=True)
    final_order = weight_order[key_order]

    keys_sorted = keys[final_order]
    first_mask = torch.cat([
        keys_sorted.new_ones(1, dtype=torch.bool),
        keys_sorted[1:] != keys_sorted[:-1],
    ])

    keep = final_order[first_mask]

    out_index = torch.stack([src[keep], dst[keep]], dim=0)
    return out_index, edge_weight[keep]
