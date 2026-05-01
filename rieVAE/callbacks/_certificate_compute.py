"""Shared certificate-compute helpers for the Phase-3 callbacks.

Extracts the certificate computation that previously lived inside
``ProximalSCRVAETrainer._compute_global_certificate`` and
``_estimate_pullback_spectrum`` into pure functions that take the
model + data artefacts as inputs. The Phase-3
:class:`CertificateObserverCallback` and
:class:`PostHocCalibrationCallback` consume these helpers.

The certificate is the runtime-checkable predicate of
Definition def:cert (post-Phase-0 form): C1' (encoder isometry on
E*), C2 (likelihood-aware reconstruction floor), C3 (mu_hat_1 > 0);
C4 is logged as a diagnostic only. The shared helpers below produce
all the scalars that go into the predicate.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from rieVAE.geometry.log_map import riemannian_log_maps_batched
from rieVAE.geometry.strong_convexity import verify_restricted_sc_output_layer
from rieVAE.geometry.encoder_regularity import (
    encoder_lipschitz_bound, ACTIVATION_BOUNDS,
)
from rieVAE.train.loss import compute_delta_iso, compute_delta_edge_scalar


def estimate_pullback_spectrum(
    model: nn.Module,
    mu_eval: torch.Tensor,
    n_pts: int,
) -> tuple[float, float]:
    """Estimate (lambda_min, lambda_max) of the decoder pullback metric
    G^* = J_f^T J_f at a sample of latent points.

    Uses the COMPOSED map ``embed_for_decoder o decoder`` so the
    pullback metric is on the chart, matching op47C option (ii)'s
    chart-coordinate convention. Returns ``(lam0, Lam_max)`` averaged
    over the sample; falls back to SVD if the eigvalsh solver fails
    on ill-conditioned matrices (typical at step 0).
    """
    if mu_eval.shape[0] < 1 or n_pts < 1:
        return 0.0, 1.0
    z_sub = mu_eval[: min(n_pts, mu_eval.shape[0])].detach()

    manifold = model.manifold
    node_decoder = model.node_decoder

    class _DecoderThroughManifold(nn.Module):
        def __init__(_self):
            super().__init__()
            _self.node_decoder = node_decoder
            _self.manifold = manifold

        def forward(_self, z):
            return _self.node_decoder(_self.manifold.embed_for_decoder(z))

    decoder_through_chart = _DecoderThroughManifold()
    eigs = []
    for zi in z_sub:
        jvp_cols = []
        for k in range(zi.shape[0]):
            e_k = torch.zeros_like(zi)
            e_k[k] = 1.0
            jv = riemannian_log_maps_batched(
                decoder_through_chart,
                zi.unsqueeze(0),
                e_k.unsqueeze(0),
                decoder_grad_weight=0.0,
            )
            jvp_cols.append(jv.squeeze(0))
        J = torch.stack(jvp_cols, dim=-1)
        G_star = J.T @ J
        try:
            ev = torch.linalg.eigvalsh(G_star)
            eigs.append((float(ev.min().item()), float(ev.max().item())))
        except Exception:
            try:
                sv = torch.linalg.svdvals(J)
                eigs.append((float(sv.min().item()) ** 2, float(sv.max().item()) ** 2))
            except Exception:
                continue
    if not eigs:
        return 0.0, 1.0
    lam0    = float(np.mean([e[0] for e in eigs]))
    Lam_max = float(np.mean([e[1] for e in eigs]))
    return max(lam0, 0.0), max(Lam_max, 1e-8)


def rn_estimate(n: int, dim_latent: int) -> float:
    """r_n = (log n / n)^{1/d}."""
    d = max(int(dim_latent), 1)
    n_eff = max(int(n), 2)
    return float((math.log(n_eff) / n_eff) ** (1.0 / d))


def compute_global_certificate(
    *,
    model: nn.Module,
    artefacts,
    pe_feat: Optional[torch.Tensor] = None,
    alpha_pe: float = 0.0,
    cert_subsample: Optional[int] = 2048,
    cert_pullback_nodes: int = 32,
    delta_pe_aux_sup: Optional[float] = None,
    activation: str = "silu",
    rng_seed: int = 0,
    force_global: bool = False,
    gamma_t: float = 0.0,
    chart_regime: str = "general",
) -> dict:
    """Compute the runtime certificate for one checkpoint.

    Parameters
    ----------
    model : the unified RiemannianVAE.
    artefacts : :class:`SpectralArtefacts` from the preprocessor.
    pe_feat : (n_active, K) PE features when use_pe is on (defaults
        to ``artefacts.pe_feat``).
    alpha_pe : current PE gate value.
    cert_subsample : if not None and < n_active, evaluate the cert on
        a random subsample (sup over E* edges with both endpoints in
        the subsample is then a *biased low* estimate; the Phase-1
        bug B.9 fix exposes ``is_global=False`` in this case).
    cert_pullback_nodes : number of points at which to estimate
        the pullback spectrum.
    delta_pe_aux_sup : the post-hoc-fitted PE residual (filled in by
        :class:`PEAuxFitCallback`); pass-through here.
    activation : activation name (drives sigma'_max in
        ``encoder_lipschitz_bound``).
    rng_seed : seed for the optional subsample.
    force_global : when True, always evaluate over the full active set
        (used at the end of training for the headline cert).
    gamma_t : the L_iso schedule weight; logged as the C4 diagnostic
        (post-Phase-0 honest naming -- ``cert['gamma_t']`` and
        ``cert['lambda_t']`` carry the same value, ``lambda_t`` is a
        legacy alias).
    chart_regime : 'general' (p=1) or 'flat' (p=2 = topology-matched).

    Returns
    -------
    dict with all certificate scalars. Keys:
      ``delta_rec``, ``delta_iso``, ``delta_edge_scalar``,
      ``mu_hat_1``, ``gamma_t``, ``lambda_t``, ``lambda_cross``, ``r_n``,
      ``L_phi_observed``, ``c1_ok``, ``c2_ok``, ``c3_ok``, ``c4_ok``,
      ``isometry_holds``, ``envelope_C1_rn``, ``is_global``,
      ``global_n_used``, ``delta_pe_aux_sup``, ``r_n_for_pe``,
      ``edge_scale``.
    """
    from rieVAE.evaluate.certificate import (
        compute_certificate, CertificateThresholds,
    )
    device = next(model.parameters()).device
    n_active = int(artefacts.n_active)
    edge_index = artefacts.edge_index.to(device)
    edge_weight = artefacts.edge_weight.to(device)
    x_active = artefacts.x_active.to(device)
    if pe_feat is None:
        pe_feat = artefacts.pe_feat
    if pe_feat is not None:
        pe_feat = pe_feat.to(device, x_active.dtype)

    model.eval()
    try:
        # Decide subsample scope.
        use_global = (
            force_global
            or not cert_subsample
            or cert_subsample >= n_active
        )
        if use_global:
            node_sub = torch.arange(n_active, device=device)
        else:
            rng = np.random.default_rng(int(rng_seed))
            node_sub = torch.from_numpy(
                rng.choice(n_active, size=int(cert_subsample), replace=False)
            ).long().to(device)

        x_eval = x_active[node_sub]
        if pe_feat is not None and getattr(model, "use_pe", False):
            pe_eval = pe_feat[node_sub]
            mu_eval, _ = model.encode_nodes(
                x_eval, pe_feat=pe_eval, alpha_pe=float(alpha_pe),
            )
        else:
            mu_eval, _ = model.encode_nodes(x_eval)
        # delta_rec via likelihood-aware expected_value (Phase-2).
        x_hat_eval = model.decode_nodes(mu_eval)
        recon_norms = (x_hat_eval - x_eval).norm(dim=-1)
        delta_rec_global = float(recon_norms.max().item())

        # Subset edges to those with both endpoints in node_sub.
        node_set = set(int(v.item()) for v in node_sub)
        src_np = edge_index[0].cpu().numpy()
        dst_np = edge_index[1].cpu().numpy()
        keep_np = np.array(
            [(int(s) in node_set) and (int(d) in node_set)
             for s, d in zip(src_np, dst_np)],
            dtype=bool,
        )
        delta_iso_global = 0.0
        delta_edge_scalar_global = 0.0
        if keep_np.sum() > 0:
            node_to_local = {int(v): i for i, v in enumerate(node_sub.cpu().numpy())}
            sub_src_c = np.array([node_to_local[int(s)] for s in src_np[keep_np]], dtype=np.int64)
            sub_dst_c = np.array([node_to_local[int(d)] for d in dst_np[keep_np]], dtype=np.int64)
            sub_ei = torch.from_numpy(np.stack([sub_src_c, sub_dst_c], 0)).long().to(device)
            keep_idx_t = torch.from_numpy(np.where(keep_np)[0]).long().to(device)
            sub_tilde_w = edge_weight[keep_idx_t]
            delta_iso_global = compute_delta_iso(
                mu=mu_eval, edge_index=sub_ei, tilde_w=sub_tilde_w,
                reduction="max", latent_distance_fn=model.manifold.distance,
            )
            if getattr(model, "edge_decoder_type", None) == "scalar":
                delta_edge_scalar_global = compute_delta_edge_scalar(
                    edge_decoder=model.edge_decoder, mu=mu_eval,
                    edge_index=sub_ei, tilde_w=sub_tilde_w, reduction="max",
                )
            ei_for_sc = sub_ei
            z_for_sc = mu_eval
        else:
            ei_for_sc = torch.zeros(2, 1, dtype=torch.long, device=device)
            z_for_sc = mu_eval[:2] if mu_eval.shape[0] >= 2 else mu_eval

        # Pullback spectrum + restricted SC.
        lam0, Lam_max = estimate_pullback_spectrum(
            model, mu_eval, n_pts=int(cert_pullback_nodes),
        )
        try:
            sc = verify_restricted_sc_output_layer(
                model=model, z_mu=z_for_sc, edge_index=ei_for_sc,
                lambda_0=lam0, Lambda_max=Lam_max, device=device,
            )
            mu_hat_L = float(sc["mu_1_output_layer"])
        except Exception:
            mu_hat_L = 0.0

        # L_phi_observed: encoder Lipschitz upper bound (op47C C.1.2).
        try:
            _act = str(activation or "silu").lower()
            _bounds = ACTIVATION_BOUNDS.get(_act, ACTIVATION_BOUNDS["silu"])
            sigma_prime_max = float(_bounds[0])
            L_phi_obs = float(encoder_lipschitz_bound(
                model.node_encoder, sigma_prime_max=sigma_prime_max,
            ))
        except Exception:
            L_phi_obs = float("nan")
    finally:
        model.train()

    # Mo2 fix: use artefacts.intrinsic_dim (Two-NN MLE) as d for r_n,
    # falling back to model.dim_latent only when not available.
    intrinsic_dim = int(
        getattr(artefacts, "intrinsic_dim", None)
        or getattr(model, "dim_latent", 2)
    )
    r_n = rn_estimate(n_active, intrinsic_dim)
    lambda_cross = (r_n ** 2) / max(mu_hat_L, 1e-12)

    cert: dict = {
        "delta_rec":         delta_rec_global,
        "delta_iso":         delta_iso_global,
        "delta_edge_scalar": delta_edge_scalar_global,
        "mu_hat_1":          mu_hat_L,
        "gamma_t":           float(gamma_t),
        "lambda_t":          float(gamma_t),  # legacy alias of gamma_t
        "lambda_cross":      lambda_cross,
        "r_n":               r_n,
        "L_phi_observed":    L_phi_obs,
        "delta_pe_aux_sup":  (
            float(delta_pe_aux_sup) if delta_pe_aux_sup is not None else float("nan")
        ),
        "r_n_for_pe":        r_n,
        "is_global":         bool(use_global),
        "global_n_used":     int(node_sub.numel()) if use_global else None,
    }

    # Edge scale for Mo5 guard.
    edge_scale_val: Optional[float] = None
    if getattr(model, "edge_decoder_type", None) == "scalar":
        try:
            edge_scale_val = float(model.edge_decoder.scale.detach().item())
        except Exception:
            pass

    # compute_certificate -> isometry_holds = c1' AND c2 AND c3.
    try:
        rec_thr = float(getattr(artefacts, "rec_threshold", 0.0))
        thresholds = CertificateThresholds.for_chart_regime(
            chart_regime, rec_threshold=rec_thr,
        )
        report = compute_certificate(
            n=n_active, d=intrinsic_dim,
            delta_rec=delta_rec_global,
            delta_edge=delta_edge_scalar_global,
            mu_hat_1=mu_hat_L, mu_hat_1_output_layer=mu_hat_L,
            lambda_t=float(gamma_t),
            thresholds=thresholds,
            edge_scale=edge_scale_val,
            is_global=bool(use_global),
            global_n_used=(int(node_sub.numel()) if use_global else None),
        )
        cert["isometry_holds"]    = bool(report.isometry_holds)
        cert["c1_ok"]             = bool(report.c1_ok)
        cert["c1_scale_ok"]       = bool(report.c1_scale_ok)
        cert["c2_ok"]             = bool(report.c2_ok)
        cert["c3_ok"]             = bool(report.c3_ok)
        cert["c4_ok"]             = bool(report.c4_ok)
        cert["envelope_C1_rn"]    = float(report.envelope_C1_rn)
        cert["intrinsic_dim_used"] = intrinsic_dim
        cert["e_star_connected"]   = bool(
            getattr(artefacts, "e_star_connected", True)
        )
    except Exception:
        cert["isometry_holds"] = False

    if (
        hasattr(model.edge_decoder, "scale")
        and getattr(model, "edge_decoder_type", None) == "scalar"
    ):
        try:
            cert["edge_scale"] = float(model.edge_decoder.scale.detach().item())
        except Exception:
            pass

    return cert
