"""Anchor samplers for the closed-1-hop subgraph training loop.

The trainer's anchor batch determines two things:
  * which encoder/decoder forwards we pay for this step (cost), and
  * which (anchor, candidate) pairs feed L_def, L_Riem, and the BallTree
    expansion check (gradient signal coverage).

Four sampling strategies are provided; the convergence guarantees of
Theorem thm:prox_fp (proximal alternating minimisation; see
Bolte-Sabach-Teboulle 2014 and Attouch-Bolte 2009 in the paper's
App. app:prox_fp) hold under any of them, but the per-batch variance
and the rate constants differ:

  * 'uniform'              -- i.i.d. uniform with replacement (legacy).
                              No coverage guarantee per epoch.
  * 'without_replacement'  -- shuffle once per epoch, partition into
                              ceil(N / batch_size) batches.
                              Each node is an anchor exactly once per
                              epoch. Recht-Re (2012) shows strictly
                              better constants than uniform; the rate
                              is unchanged. This is the recommended
                              default.
  * 'fps'                  -- farthest-point sampling per batch on
                              the ambient X. Spreads anchors across
                              the manifold each batch, reducing
                              redundant 1-hop overlap between
                              consecutive batches at a small per-batch
                              CPU cost (O(N * batch_size)).
  * 'stratified'           -- one-time MiniBatchKMeans clustering on
                              X with K = batch_size cells; each batch
                              picks one anchor per cell, without
                              replacement within each cell per epoch.
                              Guarantees uniform spatial coverage
                              every batch; clusters are not refreshed
                              during training.

All strategies maintain a per-node visit counter so the trainer can
log graph/anchor_visit_count_{max,min,std} per epoch.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


_VALID_MODES = ("uniform", "without_replacement", "fps", "stratified")


class EpochAnchorSampler:
    """Anchor index sampler for the anchor-batched proximal trainer.

    Parameters
    ----------
    n_nodes : int
        Number of training nodes (after MST outlier drop, if any).
    batch_size : int
        Target number of anchors per batch. The strategies are allowed
        to return slightly fewer (for example, if ``stratified`` ends
        up with one or two empty cells after K-means).
    device : torch.device or str
        Device for the returned anchor tensors.
    mode : str
        One of ``{'uniform', 'without_replacement', 'fps', 'stratified'}``.
        Default is ``'without_replacement'``.
    x_full : torch.Tensor or None
        Required for ``mode in {'fps', 'stratified'}``. Ambient
        coordinates of the training nodes; lives on any device.
    seed : int or None
        Optional integer seed for reproducibility.
    """

    def __init__(
        self,
        n_nodes: int,
        batch_size: int,
        device,
        mode: str = "without_replacement",
        x_full: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"unknown anchor_sampling mode {mode!r}; "
                f"valid options are {_VALID_MODES}"
            )
        if mode in ("fps", "stratified") and x_full is None:
            raise ValueError(
                f"mode={mode!r} requires x_full to be passed to "
                f"EpochAnchorSampler"
            )

        self.n_nodes = int(n_nodes)
        self.batch_size = max(1, min(int(batch_size), self.n_nodes))
        self.device = torch.device(device)
        self.mode = mode

        # Per-node visit counter (host-side; cheap to update).
        self._visit_counts = torch.zeros(self.n_nodes, dtype=torch.long)
        # Outer-loop epoch counter, incremented every time the
        # without-replacement perm is reshuffled.
        self._epoch = 0
        self._step = 0

        # Mode-specific state.
        self._np_rng = np.random.default_rng(seed)
        self._torch_gen = (
            torch.Generator(device="cpu").manual_seed(seed)
            if seed is not None else None
        )
        self._perm: Optional[torch.Tensor] = None  # without_replacement
        self._cursor: int = 0                      # without_replacement
        self._x_cpu: Optional[np.ndarray] = None   # fps, stratified
        self._cell_indices: list = []              # stratified
        self._cell_perm: list = []                 # stratified
        self._cell_cursor: list = []               # stratified

        if x_full is not None:
            self._x_cpu = x_full.detach().cpu().numpy().astype(np.float32)

        if mode == "without_replacement":
            self._reshuffle()
        elif mode == "stratified":
            self._build_strata()

    # ------------------------------------------------------------------ public

    def next_anchors(self) -> torch.Tensor:
        """Return the next batch of anchor indices on ``self.device``."""
        self._step += 1
        if self.mode == "uniform":
            anchors = self._next_uniform()
        elif self.mode == "without_replacement":
            anchors = self._next_without_replacement()
        elif self.mode == "fps":
            anchors = self._next_fps()
        elif self.mode == "stratified":
            anchors = self._next_stratified()
        else:  # pragma: no cover - guarded by __init__
            raise RuntimeError(f"unhandled mode {self.mode}")

        if anchors.numel() > 0:
            anchors_cpu = anchors.detach().cpu()
            self._visit_counts.index_add_(
                0, anchors_cpu, torch.ones_like(anchors_cpu, dtype=torch.long),
            )
        return anchors.to(self.device)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def step(self) -> int:
        return self._step

    @property
    def visit_counts(self) -> torch.Tensor:
        return self._visit_counts

    def visit_count_summary(self) -> dict:
        """Return a metric dict with the visit-count quantiles."""
        c = self._visit_counts
        return {
            "graph/anchor_visit_count_max": float(c.max().item()),
            "graph/anchor_visit_count_min": float(c.min().item()),
            "graph/anchor_visit_count_std": float(c.float().std().item()),
            "graph/anchor_sampler_epoch": float(self._epoch),
        }

    # --------------------------------------------------------------- uniform

    def _next_uniform(self) -> torch.Tensor:
        if self._torch_gen is not None:
            anchors = torch.randint(
                0, self.n_nodes, (self.batch_size,),
                generator=self._torch_gen, device="cpu",
            )
        else:
            anchors = torch.randint(0, self.n_nodes, (self.batch_size,))
        return torch.unique(anchors)

    # ---------------------------------------------------- without_replacement

    def _reshuffle(self) -> None:
        if self._torch_gen is not None:
            perm = torch.randperm(
                self.n_nodes, generator=self._torch_gen, device="cpu",
            )
        else:
            perm = torch.randperm(self.n_nodes)
        self._perm = perm
        self._cursor = 0
        self._epoch += 1

    def _next_without_replacement(self) -> torch.Tensor:
        # First call ever: make sure a permutation exists.
        if self._perm is None:
            self._reshuffle()

        # Emit the remainder of the current permutation first, even if it
        # contains fewer than self.batch_size entries. This guarantees that
        # every node is visited exactly once per epoch (no partial-tail
        # drop).  The next call will detect cursor == n_nodes and reshuffle.
        if self._cursor >= self.n_nodes:
            self._reshuffle()

        end = min(self._cursor + self.batch_size, self.n_nodes)
        out = self._perm[self._cursor:end]
        self._cursor = end
        return out.clone()

    # ---------------------------------------------------------------- fps

    def _next_fps(self) -> torch.Tensor:
        """Visit-count-balanced farthest-point sampling.

        Pure FPS would pick the same spread of "extremal" nodes every
        batch, providing per-batch isotropy but no cross-batch coverage.
        We combine the FPS objective with a visit-count weight
        ``w[j] = 1 / (1 + visit_counts[j])`` so that nodes visited many
        times in earlier batches are down-weighted. The first anchor is
        sampled with probability proportional to ``w``; subsequent
        anchors maximise ``min_dist[j] * w[j]`` greedily.

        O(N * batch_size) per batch on CPU. Cross-batch coverage
        approaches without_replacement after a few epochs while keeping
        each batch maximally spread.
        """
        x = self._x_cpu  # type: ignore[union-attr]
        n = self.n_nodes
        bs = self.batch_size
        visit_w = 1.0 / (
            self._visit_counts.numpy().astype(np.float32) + 1.0
        )
        # First anchor: weighted by inverse visit count (prefer unvisited).
        probs = visit_w / visit_w.sum()
        first = int(self._np_rng.choice(n, p=probs))
        picked = [first]
        diff = x - x[first]
        min_dist = np.einsum("ij,ij->i", diff, diff)  # squared distance
        min_dist[first] = -1.0
        for _ in range(bs - 1):
            score = min_dist * visit_w
            nxt = int(np.argmax(score))
            if min_dist[nxt] <= 0:
                break  # all candidates already picked this batch
            picked.append(nxt)
            diff = x - x[nxt]
            new_d = np.einsum("ij,ij->i", diff, diff)
            np.minimum(min_dist, new_d, out=min_dist)
            min_dist[nxt] = -1.0
        # Track epoch as ceil(step * batch_size / n).
        if self._step * self.batch_size >= self.n_nodes * (self._epoch + 1):
            self._epoch += 1
        return torch.tensor(picked, dtype=torch.long)

    # ---------------------------------------------------------- stratified

    def _build_strata(self) -> None:
        """One-time K-means clustering on the ambient coordinates."""
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "anchor_sampling='stratified' requires scikit-learn"
            ) from exc

        n_cells = min(self.batch_size, self.n_nodes)
        km = MiniBatchKMeans(
            n_clusters=n_cells,
            batch_size=min(1024, self.n_nodes),
            random_state=int(self._np_rng.integers(0, 2**31 - 1)),
            n_init=3,
            max_iter=100,
        )
        labels = km.fit_predict(self._x_cpu)
        cells = []
        for c in range(n_cells):
            members = np.where(labels == c)[0]
            if members.size > 0:
                cells.append(members)
        # Drop empty cells; the per-batch size becomes the surviving count.
        self._cell_indices = cells
        self._cell_perm = [self._np_rng.permutation(c) for c in cells]
        self._cell_cursor = [0] * len(cells)

    def _next_stratified(self) -> torch.Tensor:
        indices = []
        any_reshuffled = False
        for c, cell in enumerate(self._cell_indices):
            cur = self._cell_cursor[c]
            if cur >= len(self._cell_perm[c]):
                self._cell_perm[c] = self._np_rng.permutation(cell)
                cur = 0
                any_reshuffled = True
            indices.append(int(self._cell_perm[c][cur]))
            self._cell_cursor[c] = cur + 1
        if any_reshuffled:
            self._epoch += 1
        return torch.tensor(indices, dtype=torch.long)
