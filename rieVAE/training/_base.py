"""Lightning-based training plan core for the Certified Riemannian VAE.

Phase 3 of op47C (C.3.2): a ``TrainingPlanBase`` that owns

  * the term registry (a list of ``Term(name, fn, schedule)`` triples);
  * the optimiser configuration;
  * the LR scheduler;
  * the per-step training loop -- a single backward through the sum
    of weighted terms;

and concrete subclasses register specific bundles (``IsoTrainingPlan``,
``IsoPlusGlobalOrderTrainingPlan``, ``IsoPlusJVPLegacyTrainingPlan``,
``VanillaTrainingPlan``).

A ``Term``'s schedule is a callable ``(step, max_steps) -> float`` that
gives the term's weight at the current step. The default schedules
defined here cover the common cases: constant, linear warm-up,
sigmoid, beta-decay, and "warm-up then constant" (the global-order
recipe).

Adding a new loss is a 3-line patch: subclass ``TrainingPlanBase``
(or call its constructor with extra terms) and supply a
``term_fn(model, outputs, batch) -> Tensor`` plus a schedule.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

try:
    import pytorch_lightning as pl
    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False
    pl = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

ScheduleFn = Callable[[int, int], float]


def constant(value: float) -> ScheduleFn:
    """Return a schedule that always returns ``value``."""
    val = float(value)

    def _schedule(step: int, max_steps: int) -> float:
        return val

    return _schedule


def linear_warmup(
    target: float,
    warmup_steps: int,
) -> ScheduleFn:
    """Linear ramp from 0 -> ``target`` over the first ``warmup_steps``,
    then constant ``target``."""
    target = float(target)
    warmup_steps = max(1, int(warmup_steps))

    def _schedule(step: int, max_steps: int) -> float:
        if step >= warmup_steps:
            return target
        return target * (step / warmup_steps)

    return _schedule


def sigmoid(
    target: float,
    k: float = 8.0,
    center: float = 0.5,
    k_max: float = 15.0,
) -> ScheduleFn:
    """Sigmoid ramp ``target * sigmoid(k * (rho - center))`` where
    ``rho = step / max_steps``. The ``k_max`` clamp prevents the
    sigmoid from degenerating into a step function (k > 15 is
    essentially a Heaviside)."""
    target = float(target)
    k_safe = float(min(k, k_max))
    center_f = float(center)

    def _schedule(step: int, max_steps: int) -> float:
        denom = max(int(max_steps) - 1, 1)
        rho = float(step) / float(denom)
        rho = min(max(rho, 0.0), 1.0)
        return target * (1.0 / (1.0 + math.exp(-k_safe * (rho - center_f))))

    return _schedule


def beta_linear_decay(
    beta_start: float,
    beta_end: float,
) -> ScheduleFn:
    """Linear decay from ``beta_start`` to ``beta_end`` over the run."""
    b0 = float(beta_start)
    b1 = float(beta_end)

    def _schedule(step: int, max_steps: int) -> float:
        denom = max(int(max_steps) - 1, 1)
        rho = float(step) / float(denom)
        rho = min(max(rho, 0.0), 1.0)
        return b1 + (b0 - b1) * (1.0 - rho)

    return _schedule


def warmup_then_constant(
    target: float,
    warmup_frac: float,
) -> ScheduleFn:
    """Zero until ``rho >= warmup_frac``, then constant ``target``.

    Used by the global-ordinal loss: the rank loss is disabled until
    ``rho >= global_order_warmup_frac`` so that L_iso has established
    the local scale before ranking is enforced.
    """
    target = float(target)
    warmup_frac = float(warmup_frac)

    def _schedule(step: int, max_steps: int) -> float:
        denom = max(int(max_steps) - 1, 1)
        rho = float(step) / float(denom)
        if rho < warmup_frac:
            return 0.0
        return target

    return _schedule


# ---------------------------------------------------------------------------
# Term registry
# ---------------------------------------------------------------------------

TermFn = Callable[[nn.Module, dict, dict], torch.Tensor]


@dataclasses.dataclass
class Term:
    """A single weighted loss term in a training plan.

    Attributes
    ----------
    name : str
        Logged as ``L_{name}_raw``, ``L_{name}_eff``, ``w_{name}``.
    fn : Callable[[model, outputs, batch], Tensor]
        Computes the raw scalar value of the term given the model,
        the model's forward outputs (with all the standard keys
        ``mu``, ``var``, ``z``, ``x_hat``, ``likelihood_params``),
        and the batch dict produced by the data module.
    schedule : Callable[[step, max_steps], float]
        Schedule callable; defaults to a constant 1.0.
    """

    name: str
    fn: TermFn
    schedule: ScheduleFn = dataclasses.field(default_factory=lambda: constant(1.0))


# ---------------------------------------------------------------------------
# TrainingPlanBase
# ---------------------------------------------------------------------------

if _PL_AVAILABLE:

    class TrainingPlanBase(pl.LightningModule):
        """LightningModule that drives the term-registry training step.

        Parameters
        ----------
        model : nn.Module
            The :class:`rieVAE.RiemannianVAE` (or any nn.Module
            exposing the same forward signature).
        terms : list[Term]
            The training-objective terms.
        learning_rate : float
        weight_decay : float
            Decoupled weight decay for AdamW (paper default 1e-4).
        decoder_lr_scale : float
            Multiplier on the decoder's learning rate. The paper's
            two-timescale default is 0.1 (decoder 10x slower than
            encoder). With ``decoder_lr_scale=1.0`` all parameters
            share a single learning rate.
        grad_clip_norm : float
            Global L2-norm gradient clip before every step (paper
            default 1.0). Set to 0 to disable.
        lr_scheduler : str or None
            'cosine', 'linear', 'constant', or None.
        lr_min : float
            Floor learning rate for the scheduler.
        lr_warmup_steps : int
        max_steps : int
            Total budget; the schedules consume this directly.
        use_initial_scale_norm : bool
            One-time scale normalisation: at step 0 the raw values of
            each term are recorded and used as the per-term denominators
            for the rest of training (mirroring the pre-Phase-3
            ``_record_phase2_initial_scales``). Default True.
        scale_eps : float
        """

        def __init__(
            self,
            model: nn.Module,
            terms: list[Term],
            learning_rate: float = 1e-3,
            weight_decay: float = 0.0,
            decoder_lr_scale: float = 1.0,
            grad_clip_norm: float = 0.0,
            lr_scheduler: Optional[str] = None,
            lr_min: float = 1e-6,
            lr_warmup_steps: int = 0,
            max_steps: int = 50_000,
            use_initial_scale_norm: bool = True,
            scale_eps: float = 1e-6,
        ) -> None:
            super().__init__()
            self.model = model
            self.terms: list[Term] = list(terms)
            self.learning_rate = float(learning_rate)
            self.weight_decay = float(weight_decay)
            self.decoder_lr_scale = float(decoder_lr_scale)
            self.grad_clip_norm = float(grad_clip_norm)
            self.lr_scheduler_kind = lr_scheduler
            self.lr_min = float(lr_min)
            self.lr_warmup_steps = int(lr_warmup_steps)
            self.max_steps = int(max_steps)
            self.use_initial_scale_norm = bool(use_initial_scale_norm)
            self.scale_eps = float(scale_eps)
            # Initial-scale normalisation: each term's raw value at
            # step 0 is stored as the per-term denominator; the
            # effective term contribution becomes
            # ``schedule(t) * raw / denom``.
            self.register_buffer(
                "_term_scales",
                torch.ones(len(self.terms), dtype=torch.float32),
            )
            self.register_buffer(
                "_term_scales_initialised", torch.tensor(0.0),
            )

        # --- term registry helpers ---

        def add_term(self, term: Term) -> None:
            """Append a new term after construction. Useful for callers
            who want to compose plans on the fly. Note: for use_initial_
            scale_norm to remain consistent, append before fit()."""
            self.terms.append(term)
            new_scales = torch.ones(len(self.terms), dtype=torch.float32)
            if self._term_scales.numel() > 0:
                new_scales[: self._term_scales.numel()] = self._term_scales
            # We can't re-register an existing buffer, so swap the data.
            self._term_scales = new_scales.to(self._term_scales.device)

        # --- Lightning hooks ---

        def forward(self, *args, **kwargs):  # noqa: D401
            """Delegate to the wrapped model so plan(x) == model(x)."""
            return self.model(*args, **kwargs)

        def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
            outputs = self.model(
                batch["x"],
                edge_index=batch.get("edge_index"),
                pe_feat=batch.get("pe_feat"),
                alpha_pe=batch.get("alpha_pe", 1.0),
                scale_factor=batch.get("scale_factor"),
            )

            step = int(self.global_step)
            max_steps = int(self.max_steps)

            # Mo1 fix: record per-term initial scales BEFORE computing the
            # normalised loss so that step 0's backward uses correct denoms.
            # We do a cheap no-grad pre-pass on the already-computed outputs
            # (no extra model forward) to measure raw term magnitudes, then
            # store them for the main loop below.
            if (
                self.use_initial_scale_norm
                and float(self._term_scales_initialised.item()) < 0.5
            ):
                with torch.no_grad():
                    pre_vals = [
                        float(term.fn(self.model, outputs, batch).item())
                        for term in self.terms
                    ]
                eps = self.scale_eps
                new_scales = torch.tensor(
                    [max(abs(v), eps) for v in pre_vals],
                    dtype=self._term_scales.dtype,
                    device=self._term_scales.device,
                )
                self._term_scales.copy_(new_scales)
                self._term_scales_initialised.fill_(1.0)

            total = outputs["mu"].new_zeros(())
            diag: dict[str, Any] = {}
            for k, term in enumerate(self.terms):
                w = float(term.schedule(step, max_steps))
                val = term.fn(self.model, outputs, batch)
                eps = self.scale_eps
                denom = (
                    self._term_scales[k].item()
                    if self.use_initial_scale_norm else 1.0
                )
                denom = max(abs(denom), eps)
                eff = w * val / denom
                total = total + eff
                diag[f"L_{term.name}_raw"] = float(val.detach().item())
                diag[f"L_{term.name}_eff"] = float(eff.detach().item())
                diag[f"w_{term.name}"]     = w

            diag["L_total"] = float(total.detach().item())
            self.log_dict({f"train/{k}": v for k, v in diag.items()}, on_step=True)
            return total

        def configure_optimizers(self):
            base_lr = self.learning_rate
            params = []
            decoder_params: list[nn.Parameter] = []
            other_params: list[nn.Parameter] = []
            try:
                pg = self.model.parameter_groups()
                decoder_params = list(pg.get("decoder", []))
                for k, v in pg.items():
                    if k != "decoder":
                        other_params.extend(v)
            except AttributeError:
                # If the model does not split parameters by role, fall
                # back to a single group.
                other_params = list(self.model.parameters())
            if abs(self.decoder_lr_scale - 1.0) < 1e-9 or not decoder_params:
                params = [{"params": other_params + decoder_params, "lr": base_lr}]
            else:
                params = [
                    {"params": other_params,   "lr": base_lr,                          "name": "rest"},
                    {"params": decoder_params, "lr": base_lr * self.decoder_lr_scale,   "name": "decoder"},
                ]
            optimizer = torch.optim.AdamW(params, weight_decay=self.weight_decay)
            if not self.lr_scheduler_kind:
                return optimizer
            sched = self._make_lr_scheduler(optimizer)
            return [optimizer], [sched]

        def _make_lr_scheduler(self, optimizer: torch.optim.Optimizer):
            n_total  = max(int(self.max_steps), 1)
            n_warmup = max(int(self.lr_warmup_steps), 0)
            kind     = str(self.lr_scheduler_kind).lower()
            lr_max   = float(self.learning_rate)
            lr_min   = float(self.lr_min)
            lr_min_ratio = lr_min / lr_max if lr_max > 0.0 else 0.0

            def lr_lambda(step: int) -> float:
                step = int(step)
                if n_warmup > 0 and step < n_warmup:
                    return max(step, 1) / max(n_warmup, 1)
                denom = max(n_total - n_warmup, 1)
                progress = min(max((step - n_warmup) / denom, 0.0), 1.0)
                if kind == "cosine":
                    cos_v = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return lr_min_ratio + (1.0 - lr_min_ratio) * cos_v
                if kind == "linear":
                    return lr_min_ratio + (1.0 - lr_min_ratio) * (1.0 - progress)
                if kind == "constant":
                    return 1.0
                raise ValueError(f"unknown lr_scheduler {kind!r}")

            return {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lr_lambda,
                ),
                "interval": "step",
            }

        def configure_gradient_clipping(
            self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None,
        ):
            # Plan-level gradient clipping. Lightning forwards user-set
            # values through this hook; we honour both Lightning's value
            # and our own ``grad_clip_norm`` (whichever is non-zero).
            clip_val = gradient_clip_val if gradient_clip_val is not None else self.grad_clip_norm
            if clip_val and clip_val > 0.0:
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=clip_val,
                    gradient_clip_algorithm=(
                        gradient_clip_algorithm or "norm"
                    ),
                )

else:  # pragma: no cover -- pytorch_lightning unavailable

    class TrainingPlanBase:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "rieVAE.training.TrainingPlanBase requires "
                "pytorch_lightning; install it via "
                "`pip install pytorch-lightning`."
            )
