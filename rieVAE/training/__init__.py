"""Training plans for the Certified Riemannian VAE (op47C C.3.3).

Four concrete plans + the base class:

  - :class:`TrainingPlanBase` ............... abstract; owns the
    term-registry training step, optimiser configuration, LR
    scheduler.
  - :class:`IsoTrainingPlan` ................ iso-architecture default
    (L_rec + beta * L_KL + gamma(t) * L_iso).
  - :class:`IsoPlusGlobalOrderTrainingPlan` . iso + RankNet rank loss.
  - :class:`IsoPlusJVPLegacyTrainingPlan` ... iso + (Phase-3 stub of)
    the legacy JVP vector/anchor terms.
  - :class:`VanillaTrainingPlan` ............ baseline ELBO with no
    isometry term (replaces the deleted vanilla_trainer.py).

All plans inherit from :class:`TrainingPlanBase`, which means a new
training plan is a 5-line subclass that registers extra ``Term``s in
its ``__init__``.

Schedules are exposed at the package level for direct composition:

    from rieVAE.training import constant, sigmoid, beta_linear_decay,
                                 linear_warmup, warmup_then_constant
"""
from rieVAE.training._base import (
    TrainingPlanBase,
    Term,
    constant,
    linear_warmup,
    sigmoid,
    beta_linear_decay,
    warmup_then_constant,
)
from rieVAE.training._terms import (
    likelihood_recon_term,
    manifold_kl_term,
    iso_term,
    global_ordinal_term_factory,
    jvp_vector_term_factory,
)
from rieVAE.training.iso import IsoTrainingPlan
from rieVAE.training.iso_plus_global_order import (
    IsoPlusGlobalOrderTrainingPlan,
)
from rieVAE.training.iso_plus_jvp_legacy import IsoPlusJVPLegacyTrainingPlan
from rieVAE.training.vanilla import VanillaTrainingPlan

__all__ = [
    # Base
    "TrainingPlanBase",
    "Term",
    # Schedules
    "constant",
    "linear_warmup",
    "sigmoid",
    "beta_linear_decay",
    "warmup_then_constant",
    # Term factories
    "likelihood_recon_term",
    "manifold_kl_term",
    "iso_term",
    "global_ordinal_term_factory",
    "jvp_vector_term_factory",
    # Concrete plans
    "IsoTrainingPlan",
    "IsoPlusGlobalOrderTrainingPlan",
    "IsoPlusJVPLegacyTrainingPlan",
    "VanillaTrainingPlan",
]
