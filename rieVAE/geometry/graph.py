"""Riemannian KNN graph construction and update.

The self-consistent iteration alternates between:
  1. M-step: train the VAE on the current graph.
  2. E-step: rebuild the graph using the decoder's current Riemannian distances.

The graph is represented as a pair:
  edge_index : (2, E) long tensor  -- directed edges (src, dst)
  edge_weight : (E,)  float tensor -- Riemannian distances w_ij = ||l_ij||

Convention: edge (i -> j) has src=i, dst=j.
For each undirected edge {i,j}, both (i->j) and (j->i) are stored.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors

from rieVAE.geometry.log_map import riemannian_log_maps_batched, riemannian_distances


def euclidean_knn_graph(
    x: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a symmetric k-NN graph using Euclidean distances.

    Used as the INITIAL graph before any Riemannian training.

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


def riemannian_knn_graph(
    decoder: nn.Module,
    z_mu: torch.Tensor,
    k: int,
    current_edge_index: torch.Tensor,
    n_extra_candidates: int = 0,
    distance_clip_factor: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild the graph using the decoder's Riemannian metric.

    For each node i, finds the k nearest neighbors measured by
    the linearized Riemannian distance w_ij = ||J_f(z_i)(z_j - z_i)||.

    The candidate set includes all current edges plus a random sample of
    additional pairs. This allows the graph to discover new true neighbors
    that were missed in previous iterations.

    Parameters
    ----------
    decoder : nn.Module
        Node decoder (eval mode during JVP computation).
    z_mu : (N, d)
        Posterior means of latent codes.
    k : int
        Number of neighbors.
    current_edge_index : (2, E_current)
        Current graph edges (candidates for the new graph).
    n_extra_candidates : int
        Number of random node pairs to add to the candidate set.
        Set to 0 to only reconsider current edges.
    distance_clip_factor : float
        Clip Riemannian distances at ``clip_factor * median(dists)`` before
        KNN ranking. The linearized formula ||J_f(z_i)Dz|| is only valid for
        small Dz; far-away candidate pairs produce pathologically large values
        that cause the graph to oscillate (2-cycle). Clipping at a multiple of
        the median focuses the KNN on the local regime where the linearization
        is accurate (Lemma 1: error O(K r^3)). Set to 0 to disable clipping.

    Returns
    -------
    new_edge_index : (2, E_new)
    new_edge_weight : (E_new,) -- Riemannian distances (clipped)
    """
    N, d = z_mu.shape
    device = z_mu.device

    src = current_edge_index[0]
    dst = current_edge_index[1]

    if n_extra_candidates > 0:
        extra_src = torch.randint(0, N, (n_extra_candidates,), device=device)
        extra_dst = torch.randint(0, N, (n_extra_candidates,), device=device)
        valid = extra_src != extra_dst
        extra_src = extra_src[valid]
        extra_dst = extra_dst[valid]
        src = torch.cat([src, extra_src, extra_dst])
        dst = torch.cat([dst, extra_dst, extra_src])

    delta_z_fwd = z_mu[dst] - z_mu[src]
    delta_z_bwd = z_mu[src] - z_mu[dst]

    log_fwd = riemannian_log_maps_batched(decoder, z_mu[src], delta_z_fwd)
    log_bwd = riemannian_log_maps_batched(decoder, z_mu[dst], delta_z_bwd)

    # Symmetrized Riemannian distance: average of forward and backward.
    # Both ||J_f(z_i) dz_ij|| and ||J_f(z_j) dz_ji|| approximate d_R(z_i,z_j)
    # with error O(K r^3) by Lemma 1. The symmetric average is more stable:
    # small perturbations to f_theta shift both terms in the same direction,
    # preserving neighbor rank ordering better than either term alone.
    dists = (riemannian_distances(log_fwd) + riemannian_distances(log_bwd)) / 2.0

    src_np = src.cpu().numpy()
    dst_np = dst.cpu().numpy()
    dists_np = dists.detach().cpu().numpy()

    # Clip outlier distances before KNN ranking.
    # The linearized Riemannian distance is valid only for small Dz (Lemma 1);
    # large Dz produces linearization artifacts that can be orders of magnitude
    # larger than the true geodesic distance, causing the graph to flip neighbors
    # between iterations (2-cycle). Clipping at clip_factor * median removes
    # these artifacts while keeping all pairs that fall in the valid local regime.
    if distance_clip_factor > 0.0:
        positive_dists = dists_np[dists_np > 0]
        if len(positive_dists) > 0:
            median_d = float(np.median(positive_dists))
            clip_cap = distance_clip_factor * median_d
            n_clipped = int((dists_np > clip_cap).sum())
            if n_clipped > 0:
                dists_np = np.clip(dists_np, 0.0, clip_cap)

    dist_matrix: dict[int, list[tuple[float, int]]] = {i: [] for i in range(N)}
    for idx in range(len(src_np)):
        i, j = int(src_np[idx]), int(dst_np[idx])
        if i != j:
            dist_matrix[i].append((float(dists_np[idx]), j))

    new_src_list, new_dst_list, new_dist_list = [], [], []
    for i in range(N):
        neighbors = sorted(dist_matrix[i], key=lambda x: x[0])[:k]
        for dist, j in neighbors:
            new_src_list.extend([i, j])
            new_dst_list.extend([j, i])
            new_dist_list.extend([dist, dist])

    new_edge_index = torch.tensor(
        [new_src_list, new_dst_list], dtype=torch.long, device=device
    )
    new_edge_weight = torch.tensor(new_dist_list, dtype=z_mu.dtype, device=device)

    new_edge_index, new_edge_weight = _deduplicate_edges(new_edge_index, new_edge_weight)
    return new_edge_index, new_edge_weight


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


def _deduplicate_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove duplicate directed edges, keeping the minimum-weight copy."""
    src, dst = edge_index[0], edge_index[1]
    # Hash: key = src * stride + dst.  stride = src_max + dst_max + 2 ensures
    # injectivity (key(s,d) != key(s',d') whenever (s,d) != (s',d')).
    # For N=3000: max_key ~ 3000 * 6002 ~ 18M, well within int64.
    # For N > ~3M the product could approach int64 limits; use torch.int64 below.
    keys = src.long() * (src.max().item() + dst.max().item() + 2) + dst.long()
    unique_keys, inverse = torch.unique(keys, return_inverse=True)

    n_unique = int(unique_keys.shape[0])
    best_weight = torch.full((n_unique,), float("inf"), dtype=edge_weight.dtype,
                             device=edge_weight.device)
    best_weight.scatter_reduce_(0, inverse, edge_weight, reduce="amin",
                                include_self=True)

    mask = (best_weight[inverse] == edge_weight) & (src != dst)
    first_occurrence = torch.zeros(n_unique, dtype=torch.bool, device=edge_index.device)
    for idx in torch.where(mask)[0]:
        k = int(inverse[idx].item())
        if not first_occurrence[k]:
            first_occurrence[k] = True

    keep = torch.zeros(len(src), dtype=torch.bool, device=edge_index.device)
    seen = torch.zeros(n_unique, dtype=torch.bool, device=edge_index.device)
    for idx in range(len(src)):
        if not mask[idx]:
            continue
        k = int(inverse[idx].item())
        if not seen[k]:
            seen[k] = True
            keep[idx] = True

    return edge_index[:, keep], edge_weight[keep]
