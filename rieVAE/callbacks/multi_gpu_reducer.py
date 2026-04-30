"""MultiGPUCertificateReducer.

DDP utility: at certificate evaluation time the per-rank cert dict
holds the local supremum scalars (``delta_rec``, ``delta_iso``, ...);
this callback all-reduces them across ranks so the reported cert is
the global supremum, not the rank-0 local supremum.

Activates only when ``trainer.world_size > 1``; on single-process runs
it is a no-op so users can leave it on by default. The reduction
operates on values already produced by
:class:`CertificateObserverCallback`'s history.
"""
from __future__ import annotations

from typing import Iterable

import torch

try:
    import pytorch_lightning as pl
    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False
    pl = None  # type: ignore[assignment]


_REDUCE_MAX_KEYS = (
    "delta_rec",
    "delta_iso",
    "delta_edge_scalar",
    "delta_pe_aux_sup",
    "L_phi_observed",
)
_REDUCE_MIN_KEYS = (
    "mu_hat_1",
)


if _PL_AVAILABLE:

    class MultiGPUCertificateReducer(pl.Callback):
        """All-reduce certificate sup-norm scalars across DDP ranks."""

        def __init__(
            self,
            cert_callback: pl.Callback,
        ) -> None:
            super().__init__()
            self.cert_callback = cert_callback

        def _maybe_reduce(self, trainer, cert: dict) -> None:
            if not trainer.world_size or trainer.world_size <= 1:
                return
            if not torch.distributed.is_available():
                return
            if not torch.distributed.is_initialized():
                return
            for k in _REDUCE_MAX_KEYS:
                v = cert.get(k)
                if v is None or not isinstance(v, (int, float)):
                    continue
                t = torch.tensor(float(v), device=trainer.strategy.root_device)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
                cert[k] = float(t.item())
            for k in _REDUCE_MIN_KEYS:
                v = cert.get(k)
                if v is None or not isinstance(v, (int, float)):
                    continue
                t = torch.tensor(float(v), device=trainer.strategy.root_device)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
                cert[k] = float(t.item())

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            history: list[dict] = getattr(self.cert_callback, "history", [])
            if not history:
                return
            cert = history[-1]
            self._maybe_reduce(trainer, cert)

        def on_train_end(self, trainer, pl_module):
            history: list[dict] = getattr(self.cert_callback, "history", [])
            if not history:
                return
            cert = history[-1]
            self._maybe_reduce(trainer, cert)

else:  # pragma: no cover

    class MultiGPUCertificateReducer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "rieVAE.callbacks.MultiGPUCertificateReducer requires "
                "pytorch_lightning."
            )
