"""User-facing latent-distance helpers for trained RiemannianVAE models.

Four point-to-point distance objects coexist in the latent space (see
Section 5 of the paper); this module exposes them through a uniform
interface so a downstream user does not have to assemble the JVP /
edge-decoder / path-accumulation machinery themselves:

  * 'euclidean'      : || z_j - z_i ||_2 in the flat latent.
                       Good only at kNN scale.
  * 'edge_head'      : || F_phi(z_i, z_j) ||_2 from the trained edge
                       decoder (Section 3 of the paper). Single MLP
                       forward; matches the symmetric JVP edge weight
                       at the certified fixed point.
  * 'jvp'            : || J_f(z_i) (z_j - z_i) ||_2 from the decoder
                       Jacobian-vector product. One JVP per pair.
  * 'jvp_symmetric'  : average of the forward and backward JVPs.
                       This is the exact object L_Riem trains against.

For arbitrary-range queries on a trained model, the recommended
operator is 'path' (Bernstein path accumulation):
:func:`latent_distance_path` computes shortest-path distance under
the chosen edge-weight estimator on the kNN graph; this is what
inherits the certificate's O(r_n) bound globally.

The dense pairwise matrix entry point :func:`compute_pairwise_distances`
covers both 'edge_head' / 'jvp' (O(N^2)) and 'path'
(O(N k log N + N k)) families through one signature.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from rieVAE.geometry.log_map import riemannian_log_maps_batched


_VALID_PAIR_MODES = ("euclidean", "edge_head", "jvp", "jvp_symmetric")
_VALID_PAIRWISE_MODES = _VALID_PAIR_MODES + ("path",)


@torch.no_grad()
def latent_distance(
    model,
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    mode: str = "jvp_symmetric",
) -> torch.Tensor:
    """Latent-distance proxy between corresponding pairs of latent codes.

    Parameters
    ----------
    model : trained RiemannianVAE
        Must expose ``model.node_decoder`` and (for ``mode='edge_head'``)
        ``model.edge_decoder``.
    z_i, z_j : (..., d) latent code tensors of identical shape.
    mode : str
        One of {'euclidean', 'edge_head', 'jvp', 'jvp_symmetric'}.

    Returns
    -------
    distance : (...,) tensor of pairwise distances in the chosen mode.
    """
    if mode not in _VALID_PAIR_MODES:
        raise ValueError(
            f"unknown mode {mode!r}; valid options are {_VALID_PAIR_MODES}"
        )
    if z_i.shape != z_j.shape:
        raise ValueError(
            f"shape mismatch: z_i.shape={tuple(z_i.shape)} vs "
            f"z_j.shape={tuple(z_j.shape)}"
        )
    if mode == "euclidean":
        return (z_j - z_i).norm(dim=-1)
    if mode == "edge_head":
        return model.edge_decoder(z_i, z_j).norm(dim=-1)
    fwd = riemannian_log_maps_batched(
        model.node_decoder, z_i, z_j - z_i,
    ).norm(dim=-1)
    if mode == "jvp":
        return fwd
    bwd = riemannian_log_maps_batched(
        model.node_decoder, z_j, z_i - z_j,
    ).norm(dim=-1)
    return 0.5 * (fwd + bwd)


@torch.no_grad()
def compute_pairwise_distances(
    model,
    z: torch.Tensor,
    mode: str = "edge_head",
    k: int = 16,
    chunk_size: int = 4096,
    return_numpy: bool = False,
) -> torch.Tensor | np.ndarray:
    """Compute the (N, N) pairwise distance matrix.

    Parameters
    ----------
    model : trained RiemannianVAE.
    z : (N, d) latent codes.
    mode : str in {'euclidean', 'edge_head', 'jvp', 'jvp_symmetric', 'path'}
        - 'euclidean': O(N^2) plain Euclidean (dense, very fast).
        - 'edge_head': O(N^2) edge-decoder forward; vectorised in
          ``chunk_size``-sized rows.
        - 'jvp' / 'jvp_symmetric': O(N^2) JVP through the decoder.
        - 'path': O(N k log N) Euclidean kNN graph + Dijkstra
          (Bernstein path accumulation, the recommended global proxy
          for d^M).
    k : int
        Neighbour count for 'path'. Ignored for the other modes.
    chunk_size : int
        Row-block size for the dense modes; trades GPU memory for
        speed.
    return_numpy : bool
        If True, returns a NumPy array (CPU-only).

    Returns
    -------
    distances : (N, N) tensor (or NumPy array if ``return_numpy=True``).
    """
    if mode not in _VALID_PAIRWISE_MODES:
        raise ValueError(
            f"unknown mode {mode!r}; valid options are {_VALID_PAIRWISE_MODES}"
        )
    n, d = z.shape
    device = z.device

    if mode == "euclidean":
        D = torch.cdist(z, z, p=2.0)
    elif mode == "path":
        D = latent_distance_path(
            model, z, sources=None, edge_weight_mode="edge_head", k=k,
        )
        if not return_numpy:
            D = torch.as_tensor(D, dtype=z.dtype, device=device)
    else:
        D = torch.empty(n, n, dtype=z.dtype, device=device)
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            block = z[start:stop]                       # (B, d)
            B = block.shape[0]
            # Compute distances from `block` to every node j.
            zi = block.unsqueeze(1).expand(B, n, d).reshape(B * n, d)
            zj = z.unsqueeze(0).expand(B, n, d).reshape(B * n, d)
            row = latent_distance(model, zi, zj, mode=mode)
            D[start:stop] = row.view(B, n)
    if return_numpy and isinstance(D, torch.Tensor):
        D = D.detach().cpu().numpy()
    return D


@torch.no_grad()
def latent_distance_path(
    model,
    z: torch.Tensor,
    sources: Optional[torch.Tensor] = None,
    edge_weight_mode: str = "edge_head",
    k: int = 16,
) -> np.ndarray:
    """Bernstein path accumulation: shortest paths on the kNN graph.

    For arbitrary pairs of nodes this is the practitioner's workhorse.
    Build the Euclidean kNN graph on the latent codes, weight each
    edge by the chosen estimator (``edge_head`` or
    ``jvp_symmetric``), then Dijkstra from each source.

    Parameters
    ----------
    model : trained RiemannianVAE.
    z : (N, d) latent codes.
    sources : (S,) long tensor of source node indices, or None for
        all-pairs.
    edge_weight_mode : str
        One of {'edge_head', 'jvp', 'jvp_symmetric'}; the per-edge
        distance proxy used to weight the Dijkstra graph.
    k : int
        kNN degree.

    Returns
    -------
    distances : NumPy array of shape (S, N) (or (N, N) if
        sources is None) of shortest-path distances; +inf for
        disconnected pairs.
    """
    if edge_weight_mode not in ("edge_head", "jvp", "jvp_symmetric"):
        raise ValueError(
            f"edge_weight_mode {edge_weight_mode!r} not allowed for "
            f"path accumulation; choose 'edge_head', 'jvp', or "
            f"'jvp_symmetric'."
        )
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    n = z.shape[0]
    z_np = z.detach().cpu().numpy()
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(z_np)
    _, idx = nn.kneighbors(z_np)
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:k + 1].reshape(-1)

    z_src = torch.as_tensor(z_np[src], dtype=z.dtype, device=z.device)
    z_dst = torch.as_tensor(z_np[dst], dtype=z.dtype, device=z.device)
    w = latent_distance(model, z_src, z_dst, mode=edge_weight_mode)
    w_np = w.detach().cpu().numpy()

    adj = csr_matrix((w_np, (src, dst)), shape=(n, n))
    adj = adj.maximum(adj.T)  # symmetrise for undirected Dijkstra

    if sources is not None:
        src_idx = sources.detach().cpu().numpy().tolist()
    else:
        src_idx = None
    return dijkstra(adj, directed=False, indices=src_idx)
