"""Lightning ``DataModule`` for the Certified Riemannian VAE.

Wraps a tensor of ambient features + the artefacts produced by
:class:`rieVAE.data.SpectralPreprocessor` into a Lightning data module
that yields anchor-sampled subgraph batches matching the pre-Phase-3
trainer's ``_sample_subgraph`` contract.

Each batch dict contains:
  - ``x``           : (B', G)  ambient features for the batch nodes
  - ``edge_index``  : (2, E')  edges entirely within the batch nodes
  - ``tilde_w``     : (E',)    spectral targets for those edges
  - ``omega``       : (E',) or None
  - ``pe_feat``     : (B', K)  or None (encoder PE features)
  - ``psi_full_batch`` : (B', K) or None (global-ordinal oracle)
  - ``nodes``       : (B',)    indices into the active set
  - ``edge_keep_idx`` : indices into the static edge weight tensor

where B' is the number of nodes touched by the batch (anchors + their
1-hop neighbours).
"""
from __future__ import annotations

from typing import Optional

import math
import numpy as np
import torch

try:
    import pytorch_lightning as pl
    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False
    pl = None  # type: ignore[assignment]

from rieVAE.data.preprocessor import SpectralArtefacts


class _SubgraphIterableDataset(torch.utils.data.IterableDataset):
    """Yields anchor-sampled subgraph dicts on demand.

    Length ``n_steps`` is the number of training steps per epoch
    (i.e. the dataloader produces ``n_steps`` batches per epoch).
    Each batch dict mirrors the pre-Phase-3 trainer's
    ``_sample_subgraph`` output.
    """

    def __init__(
        self,
        artefacts: SpectralArtefacts,
        anchor_batch_size: int,
        n_steps: int,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.artefacts = artefacts
        self.anchor_batch_size = int(anchor_batch_size)
        self.n_steps = int(n_steps)
        self.seed = int(seed)
        # Build adjacency once so subgraph sampling is O(B) per batch.
        src = artefacts.edge_index[0].cpu().numpy()
        dst = artefacts.edge_index[1].cpu().numpy()
        n = artefacts.n_active
        self._adjacency: list[np.ndarray] = [[] for _ in range(n)]
        for s, d in zip(src, dst):
            self._adjacency[int(s)].append(int(d))
        self._adjacency = [
            np.array(a, dtype=np.int64) for a in self._adjacency
        ]
        self._src_np = src
        self._dst_np = dst

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_steps):
            yield self._sample_subgraph(rng)
        # IterableDataset can be re-iterated; the seed is bumped so
        # successive epochs see different draws.
        self.seed += 1

    def _sample_subgraph(self, rng: np.random.Generator) -> dict:
        n = self.artefacts.n_active
        batch_size = min(self.anchor_batch_size, n)
        anchor_idx = rng.choice(n, size=batch_size, replace=False)
        hop1: set[int] = set(int(a) for a in anchor_idx)
        for a in anchor_idx:
            hop1.update(int(x) for x in self._adjacency[int(a)])
        nodes = np.array(sorted(hop1), dtype=np.int64)
        node_to_local = {int(n_): i for i, n_ in enumerate(nodes)}

        keep = np.array(
            [
                (int(s) in hop1) and (int(d) in hop1)
                for s, d in zip(self._src_np, self._dst_np)
            ],
            dtype=bool,
        )
        sub_src = np.array(
            [node_to_local[int(s)] for s in self._src_np[keep]], dtype=np.int64
        )
        sub_dst = np.array(
            [node_to_local[int(d)] for d in self._dst_np[keep]], dtype=np.int64
        )
        sub_edge_index = torch.tensor(
            np.stack([sub_src, sub_dst], axis=0), dtype=torch.long,
        )
        keep_idx = torch.from_numpy(np.where(keep)[0]).long()

        a = self.artefacts
        nodes_t = torch.from_numpy(nodes).long()
        x_batch = a.x_active[nodes_t]
        sub_tilde_w = (
            a.edge_weight[keep_idx]
            if (a.edge_weight.numel() > 0 and keep_idx.numel() > 0)
            else None
        )
        sub_omega = a.omega[keep_idx] if a.omega is not None else None
        pe_batch = a.pe_feat[nodes_t] if a.pe_feat is not None else None
        psi_batch = a.psi_full[nodes_t] if a.psi_full is not None else None
        return {
            "x":                 x_batch,
            "edge_index":        sub_edge_index,
            "tilde_w":           sub_tilde_w,
            "omega":             sub_omega,
            "pe_feat":           pe_batch,
            "psi_full_batch":    psi_batch,
            "nodes":             nodes_t,
            "edge_keep_idx":     keep_idx,
        }


def _identity_collate(batch):
    """Each item from ``_SubgraphIterableDataset.__iter__`` is already
    a fully assembled batch dict; we just return the first element."""
    return batch[0]


if _PL_AVAILABLE:

    class TensorDataModule(pl.LightningDataModule):
        """Lightning data module for an in-memory tensor + spectral artefacts.

        Parameters
        ----------
        x : (N, G) torch.Tensor
            Ambient features. Used by the preprocessor before the
            data module is constructed; the data module itself
            consumes ``artefacts.x_active`` (the outlier-filtered view)
            for batching.
        artefacts : SpectralArtefacts
            Output of :class:`rieVAE.data.SpectralPreprocessor`.
        anchor_batch_size : int
            Anchor sample size per training step. The actual batch size
            after 1-hop expansion is typically larger.
        n_steps_per_epoch : int
            Number of subgraph draws per epoch; setting
            ``n_epochs * n_steps_per_epoch`` to the desired total
            number of optimisation steps gives a clean separation
            between Lightning's epoch counter and the actual training
            budget.
        seed : int
            Seed for the per-step subgraph sampler.
        """

        def __init__(
            self,
            x: torch.Tensor,
            artefacts: SpectralArtefacts,
            anchor_batch_size: int = 512,
            n_steps_per_epoch: int = 1000,
            seed: int = 0,
        ) -> None:
            super().__init__()
            self.x = x
            self.artefacts = artefacts
            self.anchor_batch_size = int(anchor_batch_size)
            self.n_steps_per_epoch = int(n_steps_per_epoch)
            self.seed = int(seed)

        def train_dataloader(self):
            ds = _SubgraphIterableDataset(
                artefacts=self.artefacts,
                anchor_batch_size=self.anchor_batch_size,
                n_steps=self.n_steps_per_epoch,
                seed=self.seed,
            )
            return torch.utils.data.DataLoader(
                ds,
                batch_size=1,
                num_workers=0,
                collate_fn=_identity_collate,
            )

        # Phase-3 leaves val/test hooks empty; users add their own
        # validation metrics via Lightning Callbacks (e.g.,
        # CertificateObserver).

else:  # pragma: no cover -- pytorch_lightning unavailable

    class TensorDataModule:  # type: ignore[no-redef]
        """Stub shown when ``pytorch_lightning`` is not installed.

        Falls back to a plain ``torch.utils.data.DataLoader`` that the
        user can drive directly. The full Lightning lifecycle is gated
        behind the optional dependency; install ``pytorch-lightning``
        to enable it.
        """

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "rieVAE.data.TensorDataModule requires "
                "pytorch_lightning; install it via "
                "`pip install pytorch-lightning`."
            )
