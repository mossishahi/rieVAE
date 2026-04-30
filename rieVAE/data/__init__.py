"""Data utilities for the Certified Riemannian VAE.

Two public objects:

  - :class:`SpectralPreprocessor` -- standalone, runs once before
    Lightning training. Consumes raw ambient features and produces
    :class:`SpectralArtefacts` (CkNN edges, Coifman-Lafon LBO
    eigenpairs, Varadhan targets, optional PE features, optional
    decoder-independent reweighting).
  - :class:`TensorDataModule`     -- ``pl.LightningDataModule`` that
    wraps a tensor + artefacts and yields anchor-sampled subgraph
    batches.
"""
from rieVAE.data.preprocessor import (
    SpectralPreprocessor,
    SpectralArtefacts,
)
from rieVAE.data.datamodule import TensorDataModule
from rieVAE.data.extra_datamodules import (
    DatasetModule,
    MmapTensorDataModule,
)

__all__ = [
    "SpectralPreprocessor",
    "SpectralArtefacts",
    "TensorDataModule",
    "DatasetModule",
    "MmapTensorDataModule",
]
