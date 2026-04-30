"""Unified experiment logging for rieVAE.

Provides a single ExperimentLogger that simultaneously writes to:
  1. wandb (when available and enabled)
  2. CSV files (always, for publication figures and debugging)
  3. JSON summary (always, for programmatic result access)

All metric names follow a strict convention:
  train/{metric}      — per-step training metrics
  eval/{metric}       — evaluation metrics (computed periodically)
  sc/{metric}         — strong-convexity diagnostics
  sgd/{metric}        — SGD noise floor diagnostics
  enc/{metric}        — encoder regularity diagnostics
  proximal/{metric}   — proximal-specific diagnostics
  minimax/{metric}    — minimax bound metrics
  graph/{metric}      — graph structure metrics
  ablation/{metric}   — ablation-specific metrics

Step counter is always `step` (gradient step for proximal, global_epoch for
full-batch), never `epoch` or `iter` — this eliminates the iter/epoch ambiguity.
"""

from rieVAE.logging.experiment_logger import ExperimentLogger, METRIC_REGISTRY

__all__ = ["ExperimentLogger", "METRIC_REGISTRY"]
