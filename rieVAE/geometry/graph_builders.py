"""Pluggable kNN-graph builders for the spectral preprocessor.

Phase 4 of op47C (C.4.3): the preprocessor's kNN step is the
hot bottleneck for n > 1e4. We expose a small protocol with three
backends so the user can pick the right tradeoff without touching
the preprocessor code:

  * :class:`SklearnGraphBuilder` -- exact kNN via
    ``sklearn.neighbors.NearestNeighbors``. Default. O(n log n) for
    small n; degrades to O(n^2) for high G.
  * :class:`PyNNDescentGraphBuilder` -- approximate kNN via the
    NNDescent algorithm (Dong et al. 2011). Optional dependency
    (``pip install pynndescent``).
  * :class:`FaissGraphBuilder` -- approximate kNN via FAISS
    (Johnson, Douze, Jegou 2017). Optional dependency
    (``pip install faiss-cpu`` or ``faiss-gpu``).

Currently the spectral preprocessor calls sklearn directly via
``build_cknn_laplacian`` / ``build_knn_laplacian``. The Phase-5
work is to thread a ``GraphBuilder`` protocol through those
constructors so the user can pass ``preprocess_kwargs={'graph_builder':
'faiss'}``. For Phase 4 we ship the protocol + the three backend
classes so callers can already build an alternative graph manually
and pass it in via ``SpectralArtefacts.edge_index``.

Usage
-----

    from rieVAE.geometry.graph_builders import resolve_graph_builder
    builder = resolve_graph_builder("faiss")
    edge_index, edge_weight = builder.knn_graph(x, k=15)

The builders all return ``(edge_index, edge_weight)`` of shape
``(2, E)`` and ``(E,)`` with ``E = n * k`` directed edges.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class GraphBuilder(Protocol):
    """Pluggable kNN-graph builder protocol.

    Implementations must return ``(edge_index, edge_weight)`` where:
      - ``edge_index`` is a ``(2, E)`` long tensor of directed edges
        (each (source, destination) pair appears once);
      - ``edge_weight`` is a ``(E,)`` float tensor with the Euclidean
        distance for each edge.

    The protocol does NOT subsume the CkNN topology / Coifman-Lafon
    LBO (those live on the preprocessor); a graph builder is a
    drop-in for the kNN step alone.
    """

    name: str

    def knn_graph(
        self,
        x: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...


# ---------------------------------------------------------------------------
# Backend 1: sklearn (default; exact kNN).
# ---------------------------------------------------------------------------

class SklearnGraphBuilder:
    """Exact kNN via ``sklearn.neighbors.NearestNeighbors``."""

    name = "sklearn"

    def __init__(self, leaf_size: int = 30, algorithm: str = "auto") -> None:
        self.leaf_size = int(leaf_size)
        self.algorithm = str(algorithm)

    def knn_graph(
        self, x: torch.Tensor, k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from sklearn.neighbors import NearestNeighbors
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        n = x_np.shape[0]
        nn = NearestNeighbors(
            n_neighbors=int(k) + 1,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
        ).fit(x_np)
        dists, idx = nn.kneighbors(x_np)
        # Drop self-loop (column 0).
        dists = dists[:, 1:]
        idx = idx[:, 1:]
        rows = np.repeat(np.arange(n), int(k))
        cols = idx.reshape(-1)
        edge_index = torch.tensor(
            np.stack([rows, cols], axis=0).astype(np.int64),
            dtype=torch.long,
        )
        edge_weight = torch.tensor(
            dists.reshape(-1).astype(np.float32),
            dtype=torch.float32,
        )
        return edge_index, edge_weight


# ---------------------------------------------------------------------------
# Backend 2: PyNNDescent (approximate kNN; optional dep).
# ---------------------------------------------------------------------------

class PyNNDescentGraphBuilder:
    """Approximate kNN via the NNDescent algorithm.

    Requires the optional ``pynndescent`` package; raises
    ``ImportError`` at call time when unavailable.
    """

    name = "pynndescent"

    def __init__(
        self,
        n_trees: Optional[int] = None,
        leaf_size: int = 30,
        n_iters: int = 5,
        random_state: int = 0,
    ) -> None:
        self.n_trees = n_trees
        self.leaf_size = int(leaf_size)
        self.n_iters = int(n_iters)
        self.random_state = int(random_state)

    def knn_graph(
        self, x: torch.Tensor, k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            from pynndescent import NNDescent
        except ImportError as e:
            raise ImportError(
                "PyNNDescentGraphBuilder requires `pynndescent`. "
                "Install it with `pip install pynndescent`."
            ) from e
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        idx = NNDescent(
            x_np,
            n_neighbors=int(k) + 1,
            n_trees=self.n_trees,
            leaf_size=self.leaf_size,
            n_iters=self.n_iters,
            random_state=self.random_state,
        )
        neighbours, distances = idx.neighbor_graph
        # Drop self-loop.
        neighbours = neighbours[:, 1:]
        distances = distances[:, 1:]
        n = x_np.shape[0]
        rows = np.repeat(np.arange(n), int(k))
        cols = neighbours.reshape(-1)
        edge_index = torch.tensor(
            np.stack([rows, cols], axis=0).astype(np.int64),
            dtype=torch.long,
        )
        edge_weight = torch.tensor(
            distances.reshape(-1).astype(np.float32),
            dtype=torch.float32,
        )
        return edge_index, edge_weight


# ---------------------------------------------------------------------------
# Backend 3: FAISS (approximate kNN; optional dep).
# ---------------------------------------------------------------------------

class FaissGraphBuilder:
    """Approximate kNN via FAISS (Johnson, Douze, Jegou 2017).

    Requires the optional ``faiss-cpu`` or ``faiss-gpu`` package.
    Default uses an L2 IndexFlat (exact for small n; the user can
    swap to ``index='ivfflat'`` for very large n).
    """

    name = "faiss"

    def __init__(
        self,
        index: str = "flatl2",
        nlist: int = 100,
        nprobe: int = 10,
    ) -> None:
        self.index = str(index).lower()
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)

    def knn_graph(
        self, x: torch.Tensor, k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "FaissGraphBuilder requires `faiss`. Install via "
                "`pip install faiss-cpu` (CPU) or `faiss-gpu` (GPU)."
            ) from e
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy().astype("float32", copy=False)
        else:
            x_np = np.asarray(x, dtype="float32")
        n, G = x_np.shape
        if self.index == "flatl2":
            idx = faiss.IndexFlatL2(G)
        elif self.index == "ivfflat":
            quantiser = faiss.IndexFlatL2(G)
            idx = faiss.IndexIVFFlat(quantiser, G, self.nlist)
            idx.train(x_np)
            idx.nprobe = self.nprobe
        else:
            raise ValueError(
                f"Unknown FaissGraphBuilder.index {self.index!r}; "
                "expected 'flatl2' or 'ivfflat'."
            )
        idx.add(x_np)
        D, I = idx.search(x_np, int(k) + 1)
        # Drop self-loop. faiss uses squared L2; take sqrt.
        D = np.sqrt(np.maximum(D[:, 1:], 0.0))
        I = I[:, 1:]
        rows = np.repeat(np.arange(n), int(k))
        cols = I.reshape(-1)
        edge_index = torch.tensor(
            np.stack([rows, cols], axis=0).astype(np.int64),
            dtype=torch.long,
        )
        edge_weight = torch.tensor(
            D.reshape(-1).astype(np.float32),
            dtype=torch.float32,
        )
        return edge_index, edge_weight


# ---------------------------------------------------------------------------
# Registry / resolver.
# ---------------------------------------------------------------------------

_GRAPH_BUILDERS = {
    "sklearn":     SklearnGraphBuilder,
    "pynndescent": PyNNDescentGraphBuilder,
    "faiss":       FaissGraphBuilder,
}


def resolve_graph_builder(
    spec, **kwargs,
) -> GraphBuilder:
    """Resolve a graph-builder spec to a concrete instance.

    Parameters
    ----------
    spec : str or GraphBuilder
        ``"sklearn"`` (default), ``"pynndescent"``, ``"faiss"``, or a
        GraphBuilder instance (returned unchanged).
    **kwargs : forwarded to the constructor.
    """
    if isinstance(spec, GraphBuilder):
        return spec
    if not isinstance(spec, str):
        raise TypeError(
            f"resolve_graph_builder(spec): expected str or GraphBuilder, "
            f"got {type(spec).__name__}."
        )
    key = spec.lower().strip()
    if key not in _GRAPH_BUILDERS:
        raise ValueError(
            f"Unknown graph_builder spec {spec!r}; expected one of "
            f"{sorted(_GRAPH_BUILDERS)}."
        )
    return _GRAPH_BUILDERS[key](**kwargs)


__all__ = [
    "GraphBuilder",
    "SklearnGraphBuilder",
    "PyNNDescentGraphBuilder",
    "FaissGraphBuilder",
    "resolve_graph_builder",
]
