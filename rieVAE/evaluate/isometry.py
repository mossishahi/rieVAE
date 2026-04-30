"""Chart-isometry diagnostics for the certified Riemannian VAE.

Phase-1 deletion (op47C C.1.3): the legacy ``compare_encoder_decoder_isometry``
helper, which compared encoder Euclidean distances to decoder Riemannian
JVP-pullback distances on a Riemannian-rebuilt KNN graph, has been
removed. It was the JVP-architecture's evaluation entry point and
relied on ``riemannian_knn_graph`` / ``torus_riemannian_knn_graph``,
all deleted alongside it. Under the static-graph iso architecture the
relevant evaluation lives on the trainer (``ProximalSCRVAETrainer.evaluate_isometry``)
and uses the certified scalar edge head directly (no JVP graph
rebuild). The unified manifold abstraction of Phase 2 will replace
this entire module with a manifold-aware evaluation suite.

Surviving public surface:

  * :func:`estimate_chart_isometry_residual` -- the empirical witness
    sup_i ||f_theta(g_phi(x_i)) - x_i|| of Step 1 of the corrected
    proof of Theorem thm:isometry_main.
  * :func:`verify_chart_isometry` -- packaging helper that returns
    chart-isometry diagnostics in one call.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


@torch.no_grad()
def estimate_chart_isometry_residual(
    model: nn.Module,
    x: torch.Tensor,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Empirical witness of Step 1 of the corrected isometry proof.

    Returns the chart-isometry residual

        chart_isometry_max  = sup_i  || f_theta( g_phi(x_i) ) - x_i ||,
        chart_isometry_mean = E_i   || f_theta( g_phi(x_i) ) - x_i ||,

    using the encoder POSTERIOR MEAN (not the reparameterised z), so
    that this is the deterministic round-trip residual the proof
    requires. It is closely related to but distinct from the certificate
    quantity ``delta_rec`` in
    :func:`rieVAE.evaluate.certificate.estimate_delta_rec`, which uses
    the reparameterised z (and so includes a sampling component).

    Parameters
    ----------
    model : nn.Module
        SCR-VAE-like model with ``encode_nodes`` and ``decode_nodes``.
    x : (N, G) ambient samples.
    device : torch.device or None.

    Returns
    -------
    dict with keys
      'chart_isometry_max'  -- sup_i ||f(g(x_i)) - x_i||
      'chart_isometry_mean' -- mean_i ||f(g(x_i)) - x_i||
      'n_samples'           -- number of samples used.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    x = x.to(device)
    mu_node, _ = model.encode_nodes(x)
    x_round_trip = model.decode_nodes(mu_node)
    residuals = (x_round_trip - x).norm(dim=-1)
    return {
        "chart_isometry_max": float(residuals.max().item()),
        "chart_isometry_mean": float(residuals.mean().item()),
        "n_samples": int(x.shape[0]),
    }


@torch.no_grad()
def verify_chart_isometry(
    model: nn.Module,
    x: torch.Tensor,
    L_phi: float,
    kappa_phi: float,
    Lambda_max: float,
    r_n: float,
    *,
    chart_isometry_threshold: float | None = None,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Package the three steps of the corrected isometry derivation.

    Implements the diagnostic version of the chain in App. app:isometry
    "From d_g to dM":

      Step 1: chart_isometry_residual = sup ||f(g(x)) - x||
              should be O(r_n^2)  (Step 1 of the proof).
      Step 2: encoder_jacobian_inverse_norm proxy
              = L_phi * Lambda_max^{1/2}  (operator norm of g_phi^{-1}).
      Step 3: passes_step3_taylor : bool
              = chart_isometry_residual <= threshold
              (defaults to L_phi * Lambda_max^{1/2} * r_n^2).

    All three are observable at a checkpoint; together they certify
    that the encoder is a chart isometry up to O(r_n^2). For
    statements at the geodesic-distance level the relevant diagnostic
    is the trainer's ``evaluate_isometry`` method (which uses the
    certified scalar edge head, not a JVP graph rebuild).

    Parameters
    ----------
    model : nn.Module
        SCR-VAE-like model.
    x : (N, G) ambient samples.
    L_phi : float
        Encoder Lipschitz upper bound (from
        :func:`rieVAE.geometry.encoder_regularity.estimate_encoder_regularity`).
    kappa_phi : float
        Encoder Hessian upper bound (same source).
    Lambda_max : float
        Upper bound lambda_max(G^*) on the decoder pullback metric.
    r_n : float
        Connectivity radius rate (log n / n)^{1/d}.
    chart_isometry_threshold : float, optional
        Override threshold; default is L_phi * sqrt(Lambda_max) * r_n^2.
    device : torch.device or None.

    Returns
    -------
    dict containing
        'chart_isometry_max', 'chart_isometry_mean',
        'jacobian_inverse_norm_proxy',
        'chart_isometry_threshold',
        'passes_step3_taylor',
        'n_samples'.
    """
    chart = estimate_chart_isometry_residual(model, x, device=device)
    jac_inv_norm = float(L_phi * (max(Lambda_max, 0.0) ** 0.5))
    if chart_isometry_threshold is None:
        chart_isometry_threshold = jac_inv_norm * (r_n ** 2)
    passes = chart["chart_isometry_max"] <= chart_isometry_threshold
    return {
        "chart_isometry_max": chart["chart_isometry_max"],
        "chart_isometry_mean": chart["chart_isometry_mean"],
        "jacobian_inverse_norm_proxy": jac_inv_norm,
        "chart_isometry_threshold": float(chart_isometry_threshold),
        "passes_step3_taylor": bool(passes),
        "n_samples": int(chart["n_samples"]),
    }
