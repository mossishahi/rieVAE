"""Reusable term factories for the Certified Riemannian VAE.

Each factory returns a ``Term.fn`` callable matching the
``(model, outputs, batch) -> Tensor`` signature.

Phase-3 design (op47C C.3.2): training plans assemble their objective
out of these reusable building blocks. New objectives are a
``Term(name, fn, schedule)`` triple; subclasses do not need to
override training_step.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Reconstruction term
# ---------------------------------------------------------------------------

def likelihood_recon_term(
    model: nn.Module,
    outputs: dict,
    batch: dict,
) -> torch.Tensor:
    """L_rec = -E_{q_phi(z|x)}[log p_theta(x|z)] (likelihood-aware).

    The model's likelihood plug-in supplies ``log_prob(x, params)``;
    we sum over the feature axis (per-feature log-probs) and average
    over the batch.
    """
    log_p = model.likelihood.log_prob(batch["x"], outputs["likelihood_params"])
    if log_p.dim() > 1:
        log_p = log_p.sum(dim=-1)
    return -log_p.mean()


# ---------------------------------------------------------------------------
# KL term
# ---------------------------------------------------------------------------

def manifold_kl_term(
    model: nn.Module,
    outputs: dict,
    batch: dict,
) -> torch.Tensor:
    """L_KL = D_KL(q_phi(z|x) || p(z)) using the manifold's closed form."""
    return model.manifold.kl_to_prior(outputs["mu"], outputs["var"])


# ---------------------------------------------------------------------------
# Iso term
# ---------------------------------------------------------------------------

def iso_term(
    model: nn.Module,
    outputs: dict,
    batch: dict,
) -> torch.Tensor:
    """L_iso = mean over E* of (d_{M_z}(mu_i, mu_j) - tilde_w_ij)^2.

    Uses the manifold's geodesic distance directly so the optimised
    quantity matches the certificate's C1' (delta_iso) by
    construction (op47C bug B.3).
    """
    edge_index = batch.get("edge_index")
    tilde_w = batch.get("tilde_w")
    if edge_index is None or tilde_w is None or edge_index.numel() == 0:
        return outputs["mu"].new_zeros(())
    mu = outputs["mu"]
    src, dst = edge_index[0], edge_index[1]
    pred = model.manifold.distance(mu[src], mu[dst])
    target = tilde_w.detach().to(pred.device).to(pred.dtype)
    return (pred - target).pow(2).mean()


# ---------------------------------------------------------------------------
# Global ordinal term (Section sec:pe / 3sum.md Section 3.1)
# ---------------------------------------------------------------------------

def global_ordinal_term_factory(
    k_near: int = 5,
    k_far: int = 16,
    batch_size: int = 128,
) -> Callable[[nn.Module, dict, dict], torch.Tensor]:
    """Return a term function that computes the global-ordinal rank
    loss on a fresh random batch of nodes.

    The loss is RankNet-shaped (Burges 2005); see
    ``rieVAE.geometry.global_order.global_ordinal_loss``.
    Driving it to zero gives Spearman(latent, Psi) -> 1, and since
    Spearman(Psi, d^M) ~ 0.99 (empirical), this implies ordinal
    isometry globally. No absolute scale target -> no gradient
    conflict with L_iso.

    The factory captures the rank-loss hyperparameters; the returned
    term reads ``batch["psi_full_batch"]`` (the global-ordinal oracle)
    or, when absent (e.g. when use_global_order is off), returns
    a zero tensor so the term is a structural no-op.
    """
    from rieVAE.geometry.global_order import global_ordinal_loss

    def _term(model: nn.Module, outputs: dict, batch: dict) -> torch.Tensor:
        psi_full = batch.get("psi_full_batch")
        if psi_full is None:
            return outputs["mu"].new_zeros(())
        # We sample a fresh subset within the batch's nodes; if the
        # batch is smaller than ``batch_size`` we simply use all of it.
        mu = outputs["mu"]
        B = mu.shape[0]
        if B < 4:
            return mu.new_zeros(())
        if B > batch_size:
            idx = torch.randperm(B, device=mu.device)[:batch_size]
            mu_b = mu[idx]
            psi_b = psi_full[idx]
        else:
            mu_b = mu
            psi_b = psi_full
        return global_ordinal_loss(
            mu_batch=mu_b,
            psi_batch=psi_b.to(mu_b.device),
            k_near=int(k_near),
            k_far=int(k_far),
        )

    return _term


# ---------------------------------------------------------------------------
# Legacy JVP terms (used by IsoPlusJVPLegacyTrainingPlan only)
# ---------------------------------------------------------------------------

def jvp_vector_term_factory() -> Callable[[nn.Module, dict, dict], torch.Tensor]:
    """JVP-architecture vector-residual term. Used only by the
    legacy plan kept for paper ablations (Phase-3 stub).

    Returns 0 in this Phase; reactivated in a future version when the
    JVP-pullback edge head returns to the public surface.
    """
    def _term(model: nn.Module, outputs: dict, batch: dict) -> torch.Tensor:
        return outputs["mu"].new_zeros(())
    return _term
