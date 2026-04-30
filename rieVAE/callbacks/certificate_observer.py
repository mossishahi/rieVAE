"""CertificateObserverCallback.

Computes the runtime certificate at the chosen interval during
training and at the end of training, and logs the result through
Lightning's logger registry. Replaces the pre-Phase-3 trainer's
``_update_certificate``.
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    import pytorch_lightning as pl
    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False
    pl = None  # type: ignore[assignment]

from rieVAE.callbacks._certificate_compute import compute_global_certificate


if _PL_AVAILABLE:

    class CertificateObserverCallback(pl.Callback):
        """Run the certificate every ``every_n_steps`` and at end of run.

        Parameters
        ----------
        every_n_steps : int
            Evaluation cadence; default 500.
        cert_subsample : int
            Per-cert subsample size for mid-training evaluations.
            Set to a value >= n_active to always evaluate globally
            (slower but unbiased). Default 2048.
        cert_pullback_nodes : int
            Number of points at which to estimate the pullback
            spectrum (drives the cost of the SC / Lambda_max
            estimate). Default 32.
        force_global_at_end : bool
            When True, the final post-training certificate runs over
            the full active set. Default True.
        chart_regime : 'general' (p = 1) or 'flat' (p = 2 = topology-matched).
        activation : str
            Activation name; passes through to
            ``encoder_lipschitz_bound`` for the L_phi_observed scalar.
        """

        def __init__(
            self,
            every_n_steps: int = 500,
            cert_subsample: int = 2048,
            cert_pullback_nodes: int = 32,
            force_global_at_end: bool = True,
            chart_regime: str = "general",
            activation: str = "silu",
        ) -> None:
            super().__init__()
            self.every_n_steps = int(every_n_steps)
            self.cert_subsample = int(cert_subsample)
            self.cert_pullback_nodes = int(cert_pullback_nodes)
            self.force_global_at_end = bool(force_global_at_end)
            self.chart_regime = str(chart_regime)
            self.activation = str(activation)
            self.history: list[dict] = []

        def _datamodule_artefacts(self, trainer):
            dm = getattr(trainer, "datamodule", None)
            if dm is None:
                raise RuntimeError(
                    "CertificateObserverCallback requires a "
                    "rieVAE.data.TensorDataModule (or any datamodule "
                    "exposing ``.artefacts``) on the trainer."
                )
            return dm.artefacts

        def _gamma_t_now(self, pl_module) -> float:
            # Find the iso term's current schedule weight, if present.
            for term in getattr(pl_module, "terms", []):
                if term.name == "iso":
                    try:
                        return float(term.schedule(
                            int(pl_module.global_step),
                            int(getattr(pl_module, "max_steps", 1)),
                        ))
                    except Exception:
                        return 0.0
            return 0.0

        def _alpha_pe_now(self, pl_module) -> float:
            # Phase-3 has no PE warm-up schedule on the plan; PE is
            # passed through at full strength when present.
            if getattr(pl_module.model, "use_pe", False):
                return 1.0
            return 0.0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = int(pl_module.global_step)
            if self.every_n_steps <= 0 or step % self.every_n_steps != 0:
                return
            cert = compute_global_certificate(
                model=pl_module.model,
                artefacts=self._datamodule_artefacts(trainer),
                pe_feat=None,
                alpha_pe=self._alpha_pe_now(pl_module),
                cert_subsample=self.cert_subsample,
                cert_pullback_nodes=self.cert_pullback_nodes,
                rng_seed=step,
                force_global=False,
                gamma_t=self._gamma_t_now(pl_module),
                chart_regime=self.chart_regime,
                activation=self.activation,
            )
            cert["step"] = step
            cert["kind"] = "cert_intermediate"
            self.history.append(cert)
            pl_module.log_dict(
                {f"cert/{k}": float(v) for k, v in cert.items()
                 if isinstance(v, (int, float)) and v is not None},
                on_step=True,
            )

        def on_train_end(self, trainer, pl_module):
            if not self.force_global_at_end:
                return
            cert = compute_global_certificate(
                model=pl_module.model,
                artefacts=self._datamodule_artefacts(trainer),
                pe_feat=None,
                alpha_pe=self._alpha_pe_now(pl_module),
                cert_subsample=None,
                cert_pullback_nodes=self.cert_pullback_nodes,
                rng_seed=0,
                force_global=True,
                gamma_t=self._gamma_t_now(pl_module),
                chart_regime=self.chart_regime,
                activation=self.activation,
            )
            cert["kind"] = "cert_final"
            cert["step"] = int(pl_module.global_step)
            self.history.append(cert)

else:  # pragma: no cover

    class CertificateObserverCallback:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "rieVAE.callbacks.CertificateObserverCallback requires "
                "pytorch_lightning."
            )
