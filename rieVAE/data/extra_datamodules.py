"""Additional data modules for the Certified Riemannian VAE.

Phase 4 of op47C extends the in-memory ``TensorDataModule`` with two
sibling modules for less-trivial inputs:

  * :class:`DatasetModule` -- wraps any ``torch.utils.data.Dataset``
    that yields per-sample feature tensors. Useful when the user has
    a custom ``Dataset`` (file-backed, on-the-fly augmentation, etc.).
  * :class:`MmapTensorDataModule` -- wraps a ``numpy.memmap``-backed
    file on disk. Used when N is too large to fit in host memory; the
    preprocessor can still operate on a contiguous view (it pulls
    ``x.cpu()`` once at fit time, but per-step batches are
    materialised lazily from the memmap).

All three modules share the same anchor-sampled subgraph contract:
each batch is a dict with keys ``x, edge_index, tilde_w, omega,
pe_feat, psi_full_batch, nodes, edge_keep_idx`` -- the same schema
:class:`TrainingPlanBase.training_step` consumes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

try:
    import pytorch_lightning as pl
    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False
    pl = None  # type: ignore[assignment]

from rieVAE.data.preprocessor import SpectralArtefacts
from rieVAE.data.datamodule import _SubgraphIterableDataset, _identity_collate


def _features_from_dataset(
    dataset: torch.utils.data.Dataset,
) -> torch.Tensor:
    """Stack a ``Dataset`` yielding feature tensors into a single (N, G)
    tensor. Only used by the preprocessor (one-shot, on host memory);
    per-step batches are materialised lazily from the dataset itself
    in the dataloader.
    """
    items = []
    for k in range(len(dataset)):
        item = dataset[k]
        if isinstance(item, dict):
            items.append(item["x"])
        else:
            items.append(item)
    return torch.stack(items, dim=0)


if _PL_AVAILABLE:

    class DatasetModule(pl.LightningDataModule):
        """Lightning data module wrapping any ``torch.utils.data.Dataset``.

        The dataset is expected to yield per-sample feature tensors of
        shape ``(n_features,)`` (or dicts with key ``"x"``); the
        preprocessor stacks them once into a ``(N, n_features)`` tensor
        and the resulting artefacts are then used to drive the
        anchor-sampled subgraph dataloader (the dataset itself is not
        consulted per step -- the artefacts hold the materialised
        tensor under ``artefacts.x_active``).

        For datasets that cannot be materialised in host memory, use
        :class:`MmapTensorDataModule` instead.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
        artefacts : SpectralArtefacts
            Output of :class:`rieVAE.data.SpectralPreprocessor.fit`
            applied to the stacked dataset features.
        anchor_batch_size, n_steps_per_epoch, seed : as in TensorDataModule.
        """

        def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            artefacts: SpectralArtefacts,
            anchor_batch_size: int = 512,
            n_steps_per_epoch: int = 1000,
            seed: int = 0,
        ) -> None:
            super().__init__()
            self.dataset = dataset
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
                ds, batch_size=1, num_workers=0, collate_fn=_identity_collate,
            )


    class MmapTensorDataModule(pl.LightningDataModule):
        """Memory-mapped tensor data module for n > 1e6 feature tables.

        The features live on disk as a ``numpy.memmap`` (shape (N, G),
        dtype float32 by default); the preprocessor operates on a
        host-memory view but per-step batches (post-fit) are read on
        demand. For the preprocessor, the entire memmap is brought
        into host memory once (so this module is useful when N * G
        fits in RAM but you want to skip an upfront load of a .npy
        file); for fully out-of-core preprocessing, use a custom
        :class:`GraphBuilder` (op47C C.4.3) that streams from the
        memmap.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the memmap-backed .npy or raw binary file.
        shape : tuple
            (N, G) shape of the memmap.
        dtype : numpy dtype
            Default float32.
        artefacts : SpectralArtefacts
            Pre-computed spectral artefacts. The data module does not
            run preprocessing itself; the user is expected to call
            :class:`SpectralPreprocessor.fit(...)` on the memmap-backed
            tensor before constructing this module.
        anchor_batch_size, n_steps_per_epoch, seed : as elsewhere.
        """

        def __init__(
            self,
            path: Union[str, Path],
            shape: tuple,
            artefacts: SpectralArtefacts,
            anchor_batch_size: int = 512,
            n_steps_per_epoch: int = 1000,
            seed: int = 0,
            dtype: np.dtype = np.float32,
        ) -> None:
            super().__init__()
            self.path = Path(path)
            self.shape = tuple(shape)
            self.dtype = dtype
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
                ds, batch_size=1, num_workers=0, collate_fn=_identity_collate,
            )

else:  # pragma: no cover -- pytorch_lightning unavailable

    class DatasetModule:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "rieVAE.data.DatasetModule requires pytorch_lightning."
            )

    class MmapTensorDataModule:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "rieVAE.data.MmapTensorDataModule requires pytorch_lightning."
            )
