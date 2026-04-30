"""Loss-helper module retained from Phase 1 / 2.

After Phase 3 of op47C the imperative ``ProximalSCRVAETrainer`` is
removed; training is driven by ``pytorch_lightning.Trainer`` plus a
``rieVAE.training.TrainingPlanBase`` subclass. The remaining public
surface of this package is the loss helpers:

  - ``IsoVAELoss``                     -- pre-Phase-3 loss bundle, kept
                                          as a back-compat helper for
                                          callers that wire the loss
                                          directly without a plan.
  - ``calibrate_edge_decoder_scale``    -- OLS calibration helper used
                                          by ``PostHocCalibrationCallback``.
  - ``compute_delta_iso``               -- C1' residual.
  - ``compute_delta_edge_scalar``       -- scalar edge-head residual.
  - ``node_reconstruction_loss``        -- MSE (Gaussian likelihood
                                          short-cut).
  - ``node_kl_loss``                    -- standalone KL.
  - ``iso_loss``                        -- standalone iso term.

For new code, prefer ``rieVAE.training`` (the Lightning training-plan
module).
"""
from rieVAE.train.loss import (
    IsoVAELoss,
    node_reconstruction_loss,
    node_kl_loss,
    iso_loss,
    calibrate_edge_decoder_scale,
    compute_delta_iso,
    compute_delta_edge_scalar,
)

__all__ = [
    "IsoVAELoss",
    "node_reconstruction_loss",
    "node_kl_loss",
    "iso_loss",
    "calibrate_edge_decoder_scale",
    "compute_delta_iso",
    "compute_delta_edge_scalar",
]
