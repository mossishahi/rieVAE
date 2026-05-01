"""PostHocCalibrationCallback.

Runs the OLS calibration of the scalar edge head's softplus(w*) on
the converged encoder, exactly once at ``on_train_end``. Replaces
the pre-Phase-3 trainer's ``_calibrate_edge_scale_posthoc``.
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

from rieVAE.training.loss import calibrate_edge_decoder_scale


if _PL_AVAILABLE:

    class PostHocCalibrationCallback(pl.Callback):
        """OLS calibration of the scalar edge head, once after training."""

        def __init__(self) -> None:
            super().__init__()
            self.calibrated_scale: Optional[float] = None

        def on_train_end(self, trainer, pl_module):
            model = pl_module.model
            if not hasattr(model, "edge_decoder") or not hasattr(
                model.edge_decoder, "set_scale_from_value"
            ):
                return
            dm = getattr(trainer, "datamodule", None)
            if dm is None or getattr(dm, "artefacts", None) is None:
                return
            artefacts = dm.artefacts
            edge_index = artefacts.edge_index
            edge_weight = artefacts.edge_weight
            if edge_index.numel() == 0 or edge_weight.numel() == 0:
                return
            device = next(model.parameters()).device
            x_active = artefacts.x_active.to(device)
            edge_index_d = edge_index.to(device)
            edge_weight_d = edge_weight.to(device)
            pe_feat = artefacts.pe_feat.to(device, x_active.dtype) if artefacts.pe_feat is not None else None

            model.eval()
            with torch.no_grad():
                if pe_feat is not None and getattr(model, "use_pe", False):
                    mu_full, _ = model.encode_nodes(
                        x_active, pe_feat=pe_feat, alpha_pe=1.0,
                    )
                else:
                    mu_full, _ = model.encode_nodes(x_active)
                # The edge head's latent_distance_fn was installed at
                # construction time (RiemannianVAE wires it directly to
                # ``manifold.distance``); calibrate_edge_decoder_scale
                # also takes the function explicitly.
                latent_distance_fn = (
                    model.manifold.distance
                    if model.manifold.name != "euclidean" else None
                )
                scale = calibrate_edge_decoder_scale(
                    edge_decoder=model.edge_decoder,
                    mu=mu_full,
                    edge_index=edge_index_d,
                    tilde_w=edge_weight_d,
                    latent_distance_fn=latent_distance_fn,
                )
                if scale != scale:  # NaN
                    return
                self.calibrated_scale = float(scale)
                # Store on the model for downstream certificates.
                model.edge_decoder.set_scale_from_value(self.calibrated_scale)
            model.train()
            print(
                f"[iso] post-hoc edge scale calibration: "
                f"softplus(w*) = {self.calibrated_scale:.4f}",
                flush=True,
            )

else:  # pragma: no cover

    class PostHocCalibrationCallback:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "rieVAE.callbacks.PostHocCalibrationCallback requires "
                "pytorch_lightning."
            )
