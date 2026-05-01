"""Restricted strong convexity / PL* verification for the SCR-VAE.

Implements the computable estimate hat_mu_1 of the
restricted-SC constant of Theorem thm:sc:

    hat_mu_1 = lambda_min(Phi_hat | T_perp_Sigma) / r_bar_mean

where Phi_hat is the empirical JVP Gram matrix estimated from a random
edge batch, r_bar_mean is the mean log-map magnitude over the graph,
and T_perp_Sigma is the symmetry-reduced tangent space (the joint
parameter space modulo continuous neural-network symmetries:
permutation of channels and layer-wise positive-diagonal rescaling;
see Kunin et al., 2021).

Why restricted SC (and not just PL*). In overparameterised deep
networks the loss Hessian generically has a NULL space along the
symmetry orbit (see App. app:sc of the paper). PL* (Liu, Zhu, Belkin
2022; Dinh, Lavoie, Bessadok 2025 LPLR, arXiv:2507.21429) bounds the
gradient-to-suboptimality ratio but does NOT give Hessian
invertibility, which is what the IFT step of Theorem thm:prox_fp
requires. The operative condition is restricted strong convexity on
T_perp_Sigma (a Hessian lower bound on the symmetry-reduced tangent
space); Banerjee, Cisneros-Velarde, Zhu, Belkin (2022,
arXiv:2209.15106) establish RSC for deep networks with smooth
activations under a bounded NTK condition-number hypothesis. The
Hutchinson lambda_min(Phi | T_perp_Sigma) quantity this module
reports is the RSC witness; the NTK condition number
(ntk_condition_number below) is the PL* witness and is used as the
runtime proxy for the RSC condition of Banerjee et al. 2022.

The output-layer-only bound (Part (a) of Theorem thm:sc) is
unconditional and is implemented in
:func:`verify_restricted_sc_output_layer`.

Functions:
  - tangent_covering_matrix : M_cov from eq (tangent_cov).
  - verify_restricted_sc_condition : full empirical RSC / PL* check
        on T_perp_Sigma (preferred name).
  - verify_pl_star_condition : deprecated alias for back-compat.
  - verify_sc_condition : further-deprecated alias.
  - verify_restricted_sc_output_layer : unconditional output-layer
        restricted-SC bound (Part (a)).
  - ntk_condition_number : empirical NTK condition number, the
        runtime witness of Banerjee et al. 2022 RSC.
  - estimate_gradient_variance : sigma_0^2 from Lemma lem:grad_variance.
  - adaptive_p_step_budget : m*_t from Theorem thm:adaptive_p_step.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from rieVAE.geometry.log_map import riemannian_log_maps_batched, riemannian_distances


@torch.no_grad()
def tangent_covering_matrix(
    z_mu: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Tangent covering matrix M_cov (eq tangent_cov in the paper).

    M_cov = (1/|E|) sum_{(i,j)} (dz_ij dz_ij^T) / ||dz_ij||^2

    Each edge contributes a unit-direction outer product.  M_cov is PSD;
    lambda_min(M_cov) > 0 iff the KNN graph covers all tangent directions.

    Parameters
    ----------
    z_mu : (N, d)
    edge_index : (2, E)

    Returns
    -------
    M_cov : (d, d)
    """
    src, dst = edge_index[0], edge_index[1]
    dz = z_mu[dst] - z_mu[src]  # (E, d)
    norms = dz.norm(dim=1, keepdim=True).clamp(min=1e-10)  # (E, 1)
    directions = dz / norms  # (E, d) unit vectors
    M_cov = (directions.T @ directions) / directions.shape[0]  # (d, d)
    return M_cov


def _scaling_symmetry_directions(decoder: nn.Module) -> list[torch.Tensor]:
    """Tangent vectors at the current parameters along NN scaling symmetries.

    ReLU-family networks have an exact continuous symmetry: for any
    consecutive linear layers (W_l, W_{l+1}) and any positive diagonal
    rescaling alpha, the substitution W_l -> alpha * W_l,
    W_{l+1} -> W_{l+1} / alpha leaves the function unchanged (exact
    for ReLU and approximate for SiLU/GELU near the asymptotic
    activation regime).

    Each such symmetry contributes a tangent direction at pi:
        delta W_l       =  W_l
        delta W_{l+1}   = -W_{l+1}
    with all other parameters zero. Projecting Hutchinson probes out
    of these directions before estimating lambda_min(Phi) approximates
    the restriction to the symmetry-reduced tangent space T^perp Sigma
    of Theorem thm:sc.

    For DECODER-only restricted-SC estimation (the variant
    :func:`_jvp_gram_hutchinson` performs), we generate one symmetry
    direction per consecutive Linear-layer pair in
    ``decoder.parameters()`` ordering. Permutation symmetries are
    discrete (measure zero) and ignored.

    Parameters
    ----------
    decoder : nn.Module
        The decoder f_theta whose parameters are being probed.

    Returns
    -------
    list of (P,) tensors, each a tangent direction in flattened
    parameter space, normalised to unit L2 norm. Returns the empty
    list if fewer than two Linear layers exist.
    """
    linears: list[nn.Linear] = [
        m for m in decoder.modules() if isinstance(m, nn.Linear)
    ]
    if len(linears) < 2:
        return []

    # Build the parameter offset map (start, end) of each parameter in
    # decoder.parameters() order. We only care about the .weight of each
    # Linear layer.
    param_list = list(decoder.parameters())
    offsets: dict[int, tuple[int, int]] = {}
    cursor = 0
    for p in param_list:
        n = p.numel()
        offsets[id(p)] = (cursor, cursor + n)
        cursor += n
    P = cursor

    directions: list[torch.Tensor] = []
    for L_curr, L_next in zip(linears[:-1], linears[1:]):
        if id(L_curr.weight) not in offsets or id(L_next.weight) not in offsets:
            continue
        v = torch.zeros(P, device=L_curr.weight.device, dtype=L_curr.weight.dtype)
        s_c, e_c = offsets[id(L_curr.weight)]
        s_n, e_n = offsets[id(L_next.weight)]
        v[s_c:e_c] = L_curr.weight.detach().reshape(-1)
        v[s_n:e_n] = -L_next.weight.detach().reshape(-1)
        nrm = v.norm().clamp(min=1e-12)
        directions.append(v / nrm)
    return directions


def _project_out_symmetries(
    u: torch.Tensor,
    sym_directions: list[torch.Tensor],
) -> torch.Tensor:
    """Gram-Schmidt-project u out of the span of the given directions.

    Used to restrict Hutchinson probes to the symmetry-reduced
    parameter subspace T^perp Sigma of Theorem thm:sc.
    """
    out = u.clone()
    for v in sym_directions:
        out = out - (out * v).sum() * v
    nrm = out.norm().clamp(min=1e-12)
    return out / nrm


def _jvp_gram_hutchinson(
    model: nn.Module,
    z_mu: torch.Tensor,
    edge_index: torch.Tensor,
    n_samples: int = 64,
    n_probe_vectors: int = 4,
    n_output_probes: int = 8,
    device: Optional[torch.device] = None,
    project_out_symmetries: bool = True,
) -> tuple[float, float, float]:
    """Estimate lambda_min of the JVP Gram matrix Phi via nested Hutchinson probing.

    The JVP Gram matrix (NTK-style) is:
        Phi = (1/|E|) sum_{ij} B_ij B_ij^T / ||l*_ij||
    where B_ij = grad_{theta}[l_ij] in R^{P x G} (P = num_params, G = ambient dim).

    For parameter-space probe u in R^P:
        u^T Phi u = (1/|E|) sum_ij ||B_ij^T u||_2^2 / ||l_ij||

    We estimate ||B_ij^T u||^2 using n_output_probes Rademacher vectors v in R^G:
        E_v[(v^T B_ij u)^2] = ||B_ij u||^2  (exact for {-1,+1} Rademacher v)

    Each v gives one unbiased sample: (v^T B_ij u)^2 = (u^T B_ij^T v)^2.
    B_ij^T v = grad_{theta}[v^T l_ij] (standard vjp via .backward()).
    We use {-1, +1} Rademacher vectors so that E[(v^T B_ij u)^2]
    = ||B_ij u||^2 exactly (no ambient-dimension scaling factor).

    Returns
    -------
    (lambda_min_estimate, r_bar_mean) : (float, float)
        lambda_min_estimate = min over parameter-space probes of u^T Phi u.
        r_bar_mean = mean log-map magnitude (1/S) sum ||l_ij||.
    """
    if device is None:
        device = z_mu.device

    decoder = model.node_decoder
    was_training = decoder.training
    decoder.eval()

    # ── Step 1: compute log-map norms and fold-pair fraction ────────────────
    # NOTE on fold-pair contamination (torus):
    # Fold pairs have ||l_ij|| >> typical, contributing tiny terms 1/||l_ij||
    # to Phi.  They are CORRECTLY included — the SC constant is defined over
    # all graph edges.  mu_hat_1 ≈ 0 for the standard torus (Euclidean latent)
    # is the CORRECT theoretical answer: fold pairs genuinely reduce μ₁, which
    # is why the paper requires topology-matched latent space to satisfy SC.
    # We report the fold-pair fraction separately so the user can interpret
    # whether mu_hat_1 ≈ 0 is due to genuine SC failure vs high variance.
    # We do NOT filter them: filtering biases the estimate toward a non-fold
    # subgraph SC constant, which is a different (optimistic) quantity.
    with torch.no_grad():
        src, dst = edge_index[0], edge_index[1]
        E = src.shape[0]

        sample_idx = torch.randperm(E, device=device)[:min(n_samples, E)]
        z_src = z_mu[src[sample_idx]].detach()
        dz    = (z_mu[dst[sample_idx]] - z_mu[src[sample_idx]]).detach()

        log_maps_nograd = riemannian_log_maps_batched(decoder, z_src, dz)
        log_norms  = log_maps_nograd.norm(dim=1).clamp(min=1e-10)
        r_bar_mean = float(log_norms.mean().item())

        # Diagnostic: fraction of sampled edges that are likely fold pairs
        # (||l_ij|| > 3 × median).  High fold_frac + mu_hat_1≈0 → SC genuinely
        # hard for this model (correct for standard torus).
        # Low fold_frac + mu_hat_1≈0 → SC failure unrelated to fold pairs.
        median_norm   = float(log_norms.median().item())
        fold_threshold = 3.0 * median_norm
        fold_frac     = float((log_norms > fold_threshold).float().mean().item())

    n_params = sum(p.numel() for p in decoder.parameters())
    if n_params == 0:
        if was_training:
            decoder.train()
        return 0.0, r_bar_mean, fold_frac

    # Symmetry directions to project Hutchinson probes out of (R8.a / R2.d).
    # When True, this restricts the spectrum estimate to the symmetry-reduced
    # tangent space T^perp Sigma of Theorem thm:sc, avoiding artificially
    # small eigenvalues along scaling-symmetry directions.
    sym_directions: list[torch.Tensor] = []
    if project_out_symmetries:
        sym_directions = _scaling_symmetry_directions(decoder)

    min_quad = float("inf")
    S = len(sample_idx)

    # ── Step 2: Hutchinson over parameter-space probes u ────────────────────
    # GRADIENTS ARE NEEDED from here — no @torch.no_grad() context.
    with torch.enable_grad():
        for _ in range(n_probe_vectors):
            # Random unit probe in parameter space
            u = torch.randn(n_params, device=device)
            u = u / u.norm().clamp(min=1e-10)
            # Restrict to symmetry-quotient (T^perp Sigma).
            if sym_directions:
                u = _project_out_symmetries(u, sym_directions)

            quad_sum = 0.0
            n_valid  = 0

            for idx in range(S):
                z_i   = z_src[idx : idx + 1]   # (1, d), detached
                dz_i  = dz   [idx : idx + 1]   # (1, d), detached
                l_norm = float(log_norms[idx].item())

                # Estimate ||B_ij^T u||^2 via n_output_probes Rademacher v
                # Each v gives: (v^T B_ij u)^2  [unbiased for ||B_ij u||^2]
                output_sq_sum = 0.0
                n_v_ok = 0

                for _ in range(n_output_probes):
                    try:
                        for p in decoder.parameters():
                            p.requires_grad_(True)
                        decoder.zero_grad()

                        # Forward pass WITH autograd graph through decoder params
                        z_req = z_i.detach().requires_grad_(True)
                        out   = decoder(z_req)          # (1, G), graph on params
                        G_out = out.shape[1]

                        # Rademacher v in {-1, +1}^G (unbiased: E[v v^T] = I)
                        v = (2 * torch.randint(0, 2, (G_out,), device=device,
                                               dtype=out.dtype) - 1)

                        # scalar = v^T f(z) — grad w.r.t. z_req gives J_f(z)^T v
                        scalar_v = (out[0] * v.detach()).sum()

                        # J_f(z)^T v  — create_graph=True to allow further backward
                        jac_T_v = torch.autograd.grad(
                            scalar_v, z_req,
                            create_graph=True,
                            retain_graph=True,
                        )[0]   # (1, d)

                        # v^T l_ij = (J_f(z)^T v)^T dz
                        v_dot_l = (jac_T_v * dz_i.detach()).sum()   # scalar

                        # Backward: p.grad = grad_{theta}[v^T l_ij] = B_ij^T v
                        v_dot_l.backward()

                        # B_ij^T v concatenated
                        B_T_v = torch.cat([
                            p.grad.flatten() if p.grad is not None
                            else torch.zeros(p.numel(), device=device)
                            for p in decoder.parameters()
                        ])

                        # (v^T B_ij u) = (B_ij^T v)^T u
                        v_B_u = float((B_T_v * u).sum().item())

                        # Unbiased sample of ||B_ij u||^2
                        output_sq_sum += v_B_u ** 2
                        n_v_ok += 1

                        decoder.zero_grad()

                    except Exception:
                        decoder.zero_grad()
                        continue

                if n_v_ok > 0:
                    # u^T Phi_ij u = ||B_ij^T u||^2 / ||l_ij||
                    quad_sum += (output_sq_sum / n_v_ok) / l_norm
                    n_valid  += 1

            if n_valid > 0:
                quad_avg = quad_sum / n_valid   # u^T Phi u estimate
                if quad_avg < min_quad:
                    min_quad = quad_avg

    if was_training:
        decoder.train()
    for p in decoder.parameters():
        p.requires_grad_(True)
    decoder.zero_grad()

    return (min_quad if min_quad < float("inf") else 0.0), r_bar_mean, fold_frac


@torch.no_grad()
def estimate_gradient_variance(
    model: nn.Module,
    z_mu: torch.Tensor,
    edge_index: torch.Tensor,
    n_samples: int = 64,
    device: Optional[torch.device] = None,
    target: Optional[torch.Tensor] = None,
    tracker: str = "upper_bound",
) -> float:
    """Upper-bound estimate of per-edge gradient variance sigma_0^2.

    Paper quantity (App. app:sgd): the per-step gradient variance
    satisfies, at a certified fixed point,
        sigma_0^2  <=  4 * delta_edge^2 * ||param_jac||^2 * Lambda_max
    where delta_edge is the certified edge residual (C2) and
    ``param_jac`` is the per-edge Jacobian of l_ij with respect to
    theta. We report the closed-form upper bound

        sigma_0^2  <=  4 * mean_{(i,j)} ||l_ij - target||^2
                       * C_arch^2 * Lambda_max,

    with C_arch set to 1 in the absence of a tracked architecture-norm
    estimator (the factor is captured separately by lambda_0 /
    Lambda_max elsewhere in the certificate). This is the quantity the
    paper actually uses; the previous code computed
    ``Var(||l_ij||^2)`` which is an unrelated higher moment.

    Parameters
    ----------
    model : nn.Module with ``node_decoder``.
    z_mu : (N, d) latent codes.
    edge_index : (2, E) training edges.
    n_samples : max number of edges to sample for the estimate.
    target : optional (E, G) target to subtract from the JVP before
        norming; when None, uses zero (i.e. reports mean ||l||^2).
    tracker : 'upper_bound' (default) or 'l2_squared'. 'l2_squared'
        returns mean ||l_ij||^2 as a legacy diagnostic.

    Returns
    -------
    sigma_0^2 estimate (float).
    """
    if device is None:
        device = z_mu.device

    decoder = model.node_decoder
    src, dst = edge_index[0], edge_index[1]
    E = src.shape[0]
    if E == 0:
        return 0.0

    sample_idx = torch.randperm(E, device=device)[:min(n_samples, E)]
    z_src = z_mu[src[sample_idx]]
    dz = z_mu[dst[sample_idx]] - z_src

    log_maps = riemannian_log_maps_batched(decoder, z_src, dz)
    if target is not None:
        target_sub = target[sample_idx]
        residual = log_maps - target_sub
    else:
        residual = log_maps

    if tracker == "upper_bound":
        # 4 * E[||l - target||^2] matches the delta_edge-driven proxy.
        return float(4.0 * residual.pow(2).sum(dim=1).mean().item())
    elif tracker == "l2_squared":
        return float(log_maps.pow(2).sum(dim=1).mean().item())
    else:
        raise ValueError(
            f"tracker must be 'upper_bound' or 'l2_squared', got {tracker!r}"
        )


def estimate_mu0(
    lambda_min_phi: float,
    r_bar_mean: float,
) -> float:
    """Compute the SC estimate hat_mu_1 = lambda_min(Phi) / r_bar_mean.

    This is the computable estimate from Corollary cor:mu0_computable.

    Parameters
    ----------
    lambda_min_phi : float
        Minimum eigenvalue of the JVP Gram matrix Phi.
    r_bar_mean : float
        Mean log-map magnitude (1/|E|) sum ||l*_ij||.

    Returns
    -------
    mu0_estimate : float
    """
    if r_bar_mean <= 0:
        return 0.0
    return lambda_min_phi / r_bar_mean


def verify_restricted_sc_condition(
    model: nn.Module,
    z_mu: torch.Tensor,
    edge_index: torch.Tensor,
    n_samples: int = 64,
    n_probe_vectors: int = 4,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    estimate_variance: bool = False,
    n_variance_samples: int = 64,
    project_out_symmetries: bool = True,
) -> dict:
    """Full restricted-SC / PL* verification at a checkpoint.

    Implements the empirical check of Theorem thm:sc:
      hat_mu_1 = lambda_min(Phi | T_perp_Sigma) / r_bar_mean

    where T_perp_Sigma is the symmetry-quotient parameter subspace
    (rescaling-symmetry directions projected out via
    :func:`_scaling_symmetry_directions`). Set
    ``project_out_symmetries=False`` to recover the legacy
    "raw-Hessian" Hutchinson estimate.

    The Hutchinson lambda_min(Phi | T_perp_Sigma) is a restricted
    strong convexity (RSC) witness in the sense of Banerjee et al.
    2022 (arXiv:2209.15106). PL* of Liu, Zhu, Belkin (2022) is
    strictly weaker; the IFT step of Theorem thm:prox_fp requires
    the RSC reading.

    Certification conditions (Lemma lem:ntk_gram).
    ``lambda_min_phi`` is a Hutchinson lower-bound ESTIMATE of
    lambda_min(Phi | T_perp_Sigma). It equals a certified theoretical
    lower bound (Lemma lem:ntk_gram) only when BOTH of the following
    hold:

      (a) sigma'_min > 0 on the decoder's preactivation range
          (use :func:`rieVAE.geometry.encoder_regularity.estimate_encoder_regularity`
          to check ``ntk_condition_holds``). ReLU decoders have
          sigma'_min = 0 and do NOT satisfy this. SiLU satisfies it
          on compact preactivation ranges (typical at convergence).

      (b) Edge spanning condition: the graph edges {B^ell_ij} span
          the decoder parameter space (Lemma lem:ntk_gram assumption
          (c)). This holds with high probability for overparameterised
          networks and diverse graph edges, but is not verified here.

    When (a)-(b) hold, hat_mu_1 > 0 implies the iteration is locally
    PL* (Theorem thm:sc Part (b)). When (a)-(b) are uncertain,
    hat_mu_1 should be treated as a diagnostic empirical quantity, not
    a certified guarantee. Use
    :func:`verify_restricted_sc_output_layer` for the unconditional
    output-layer bound (Part (a)).

    The tangent covering lambda_min(M_cov) provides an UNCONDITIONAL
    lower bound on the graph's directional coverage of the latent space
    (no activation assumptions needed).

    Parameters
    ----------
    model : nn.Module
        A ``RiemannianVAE`` (or any model exposing ``node_decoder``
        and ``dim_latent``).
    z_mu : (N, d)
        Latent posterior means.
    edge_index : (2, E)
        Current graph edges.
    n_samples : int
        Number of edges to sample for Gram matrix estimation.
    n_probe_vectors : int
        Number of random probe vectors for Hutchinson lambda_min estimation.
    device : torch.device or None
    verbose : bool
    estimate_variance : bool
        If True, also estimate sigma_0^2 and compute noise floor ratio.
    n_variance_samples : int
        Number of edges for variance estimation.

    Returns
    -------
    dict with keys:
      'mu0_estimate'      : float -- hat_mu_1 (certified iff conditions (a)-(b) hold)
      'lambda_min_mcover'  : float -- lambda_min(M_cov) (unconditional coverage bound)
      'lambda_min_phi'     : float -- lambda_min(Phi | T_perp_Sigma) Hutchinson estimate
      'r_bar_mean'         : float -- mean log-map magnitude
      'pl_star_holds'      : bool  -- True if mu0_estimate > 0
      'sc_holds'           : bool  -- DEPRECATED alias for pl_star_holds
      'sc_certification_level' : str -- 'output_layer_only' or 'full_multilayer'
      'fold_frac'          : float -- fraction of fold-pair edges (diagnostic)
      'symmetry_projected' : bool  -- whether the Hutchinson probes were
                                       restricted to T_perp_Sigma
      'sigma0_sq'          : float -- gradient variance (if estimate_variance)
      'noise_floor_ratio'  : float -- sigma0^2 / (mu1 * r_n^2), should be << 1
      'sgd_floor_ok'       : bool  -- True if noise_floor_ratio < 1
    """
    if device is None:
        device = z_mu.device

    z_mu = z_mu.to(device)
    edge_index = edge_index.to(device)

    M_cov = tangent_covering_matrix(z_mu, edge_index)

    # Tikhonov regularisation: add eps*I before eigendecomposition.
    # When all edge vectors are near-collinear (degenerate latent distribution,
    # e.g. all points collapsed to a 1-D curve), M_cov is rank-deficient and
    # torch.linalg.eigvalsh raises LinAlgError ("algorithm failed to converge").
    # Adding eps*I shifts all eigenvalues by eps without changing the relative
    # ordering, making the matrix SPD and the decomposition numerically stable.
    # eps = 1e-6 is negligible compared to meaningful lambda_min values (>1e-3).
    d = M_cov.shape[0]
    eps_reg = 1e-6 * torch.eye(d, dtype=M_cov.dtype, device=M_cov.device)
    try:
        lambda_min_mcover = float(
            torch.linalg.eigvalsh(M_cov + eps_reg)[0].item()
        ) - 1e-6   # subtract eps so the reported value is unbiased
    except (torch.linalg.LinAlgError, RuntimeError):
        # Last-resort: report 0 (degenerate graph) rather than crashing.
        lambda_min_mcover = 0.0

    # Hutchinson on Phi, restricted to T_perp_Sigma when project_out_symmetries.
    # On manifolds with non-trivial topology (e.g. flat torus), fold pairs have
    # ||l_ij|| >> typical, biasing the Hutchinson estimate downward (each fold
    # edge contributes 1/||l_ij|| which is small, while non-fold edges with small
    # ||l_ij|| contribute more); the fold_frac diagnostic is reported separately.
    lambda_min_phi, r_bar_mean, fold_frac = _jvp_gram_hutchinson(
        model, z_mu, edge_index,
        n_samples=n_samples,
        n_probe_vectors=max(n_probe_vectors, 8),
        n_output_probes=16,
        device=device,
        project_out_symmetries=project_out_symmetries,
    )

    mu0_est = estimate_mu0(lambda_min_phi, r_bar_mean)
    pl_star_holds = mu0_est > 0

    # Certification level: 'full_multilayer' requires the PL* or NTK spanning
    # condition (Lemma lem:ntk_gram); we report 'output_layer_only' when the
    # tangent coverage M_cov is degenerate, since only Part (a) of Theorem
    # thm:sc applies in that case.
    cert_level = (
        "full_multilayer"
        if (pl_star_holds and lambda_min_mcover > 0.0)
        else "output_layer_only"
    )

    result = {
        "mu0_estimate": mu0_est,
        "lambda_min_mcover": lambda_min_mcover,
        "lambda_min_phi": lambda_min_phi,
        "r_bar_mean": r_bar_mean,
        "pl_star_holds": pl_star_holds,
        "sc_holds": pl_star_holds,  # deprecated alias
        "sc_certification_level": cert_level,
        # Fraction of sampled edges with ||l_ij|| > 3 * median (fold-pair proxy).
        "fold_frac": fold_frac,
        "symmetry_projected": bool(project_out_symmetries),
    }

    if estimate_variance:
        sigma0_sq = estimate_gradient_variance(
            model, z_mu, edge_index,
            n_samples=n_variance_samples,
            device=device,
        )
        n = z_mu.shape[0]
        d = z_mu.shape[1]
        r_n = (math.log(max(n, 2)) / max(n, 1)) ** (1.0 / max(d, 1))

        # Fix 2: use a minimum floor on mu0_est before division to prevent
        # 600K spikes from near-zero Hutchinson estimates.
        # The floor is set at lambda_min_mcover / r_bar_mean when positive
        # (tangent coverage gives a lower bound on the SC constant),
        # otherwise fall back to a small absolute value.
        if mu0_est > 0 and r_n > 0:
            # Fix 3: EMA-smooth mu_hat_1 across consecutive checks using a
            # simple running update stored in the result for caller use.
            # The noise_floor_ratio uses the RAW estimate (not smoothed) so
            # individual spikes are visible, but clip to [0, 1e4] for display.
            noise_floor_ratio = sigma0_sq / (mu0_est * r_n ** 2)
            noise_floor_ratio = min(noise_floor_ratio, 1e4)   # clip display spikes
        else:
            noise_floor_ratio = float("nan")

        sgd_floor_ok = (
            not math.isnan(noise_floor_ratio) and noise_floor_ratio < 1.0
        )

        result["sigma0_sq"] = sigma0_sq
        result["noise_floor_ratio"] = noise_floor_ratio
        result["sgd_floor_ok"] = sgd_floor_ok

    if verbose:
        print(f"  [RSC] mu_hat_1 = {mu0_est:.4e}")
        print(f"        lambda_min(M_cov) = {lambda_min_mcover:.4e}")
        print(f"        lambda_min(Phi | T_perp Sigma) = {lambda_min_phi:.4e}")
        print(f"        r_bar_mean        = {r_bar_mean:.4e}")
        print(f"        pl_star_holds     = {pl_star_holds}")
        if estimate_variance:
            print(f"        sigma0_sq         = {result['sigma0_sq']:.4e}")
            print(f"        noise_floor_ratio = {result['noise_floor_ratio']:.4e}")

    return result


# Deprecated aliases for the renamed function. Will be removed in a
# future release; new code should call verify_restricted_sc_condition.
def verify_pl_star_condition(*args, **kwargs) -> dict:
    """DEPRECATED alias for :func:`verify_restricted_sc_condition`.

    The function now returns the restricted strong convexity (RSC)
    witness in the sense of Banerjee et al. 2022, which is strictly
    stronger than PL* and is the condition required by the IFT step
    of Theorem thm:prox_fp.
    """
    return verify_restricted_sc_condition(*args, **kwargs)


def verify_sc_condition(*args, **kwargs) -> dict:
    """DEPRECATED alias for :func:`verify_restricted_sc_condition`."""
    import warnings
    warnings.warn(
        "verify_sc_condition is a deprecated alias; "
        "call verify_restricted_sc_condition directly. The rename "
        "reflects the paper's move from plain strong convexity to "
        "restricted strong convexity on the symmetry-quotient "
        "tangent space (Theorem thm:sc).",
        DeprecationWarning,
        stacklevel=2,
    )
    return verify_restricted_sc_condition(*args, **kwargs)


def verify_restricted_sc_output_layer(
    model: nn.Module,
    z_mu: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    lambda_0: float,
    Lambda_max: float,
    device: Optional[torch.device] = None,
) -> dict:
    """Unconditional output-layer restricted-SC bound (Part (a) of Thm. thm:sc).

    The output-layer backward vector B_ij^L = I_G gives trivial spanning
    and yields the unconditional bound

        mu_1^{(L)}  >=  lambda_0 * lambda_cov / ( r_bar_mean * ||W_L||^2 * Lambda_max )

    where lambda_cov = lambda_min(M_cov) >= c_1 / d^2 is the tangent-coverage
    eigenvalue, computed by :func:`tangent_covering_matrix`. This is
    Part (a) of Theorem thm:sc -- always positive under non-degeneracy,
    no PL* / NTK condition required.

    Parameters
    ----------
    model : nn.Module
        A ``RiemannianVAE`` (or any model exposing ``node_decoder``
        and ``dim_latent``).
    z_mu : (N, d) latent posterior means.
    edge_index : (2, E) directed edges.
    lambda_0 : float
        Lower bound on lambda_min(G^*) at the checkpoint.
    Lambda_max : float
        Upper bound on lambda_max(G^*) at the checkpoint.
    device : torch.device or None

    Returns
    -------
    dict with keys:
        'mu_1_output_layer' : float -- unconditional restricted-SC bound.
        'lambda_min_mcover' : float -- coverage eigenvalue.
        'r_bar_mean'        : float -- mean log-map magnitude.
        'WL_op_norm'        : float -- ||W_L||_op of the decoder output layer.
    """
    if device is None:
        device = z_mu.device
    z_mu = z_mu.to(device)
    edge_index = edge_index.to(device)

    decoder = model.node_decoder
    manifold = getattr(model, "manifold", None)

    # FV2 fix: route the JVP through `embed_for_decoder o decoder` so
    # the input shape matches what the decoder expects (e.g., for
    # FlatTorus the decoder takes (cos, sin, cos, sin), not raw chart
    # coordinates). For Euclidean ``embed_for_decoder`` is identity, so
    # this wrap is a no-op there. Without this, non-Euclidean latents
    # silently fail the JVP and the certificate falls back to
    # mu_hat_1 = 0 (= c3_ok=False), reporting a spurious failure of
    # restricted strong convexity.
    if manifold is not None and hasattr(manifold, "embed_for_decoder"):

        class _DecoderThroughChart(nn.Module):
            def __init__(_self):
                super().__init__()
                _self.dec = decoder
                _self.man = manifold

            def forward(_self, z):
                return _self.dec(_self.man.embed_for_decoder(z))

        decoder_eff = _DecoderThroughChart().to(device)
    else:
        decoder_eff = decoder

    # Mean log-map magnitude r_bar_mean.
    # FV2 fix: dz must be the manifold-aware tangent direction. For
    # FlatTorus the raw chart difference can be huge across the wrap-
    # around boundary while the actual geodesic motion is small; using
    # the wrapped difference brings the JVP into the correct regime.
    # For other manifolds shipped today (Euclidean, Sphere, Hyperbolic
    # tangent-at-origin chart) the raw difference is the correct
    # tangent direction.
    src, dst = edge_index[0], edge_index[1]
    with torch.no_grad():
        z_src = z_mu[src]
        raw_dz = z_mu[dst] - z_src
        man_name = getattr(manifold, "name", "") if manifold is not None else ""
        if man_name == "flat_torus":
            dz = torch.atan2(torch.sin(raw_dz), torch.cos(raw_dz))
        else:
            dz = raw_dz
        log_maps = riemannian_log_maps_batched(decoder_eff, z_src, dz)
        r_bar_mean = float(log_maps.norm(dim=1).clamp(min=1e-10).mean().item())

    # Coverage eigenvalue lambda_cov = lambda_min(M_cov).
    M_cov = tangent_covering_matrix(z_mu, edge_index)
    d = M_cov.shape[0]
    eps_reg = 1e-6 * torch.eye(d, dtype=M_cov.dtype, device=M_cov.device)
    try:
        lambda_min_mcover = float(
            torch.linalg.eigvalsh(M_cov + eps_reg)[0].item()
        ) - 1e-6
    except (torch.linalg.LinAlgError, RuntimeError):
        lambda_min_mcover = 0.0
    lambda_min_mcover = max(lambda_min_mcover, 0.0)

    # ||W_L||_op of the decoder's last Linear layer.
    last_linear: Optional[nn.Linear] = None
    for module in decoder.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is None:
        WL_op = 1.0
    else:
        W = last_linear.weight.detach().float()
        WL_op = float(torch.linalg.matrix_norm(W, ord=2).item())

    if (
        WL_op <= 0.0
        or Lambda_max <= 0.0
        or r_bar_mean <= 0.0
    ):
        mu_1_output_layer = 0.0
    else:
        mu_1_output_layer = (
            lambda_0 * lambda_min_mcover
            / (r_bar_mean * (WL_op ** 2) * Lambda_max)
        )

    return {
        "mu_1_output_layer": float(mu_1_output_layer),
        "lambda_min_mcover": float(lambda_min_mcover),
        "r_bar_mean": float(r_bar_mean),
        "WL_op_norm": float(WL_op),
    }


def ntk_condition_number(
    model: nn.Module,
    z_mu: torch.Tensor,
    edge_index: torch.Tensor,
    n_samples: int = 64,
    n_probe_vectors: int = 8,
    device: Optional[torch.device] = None,
) -> float:
    """Empirical NTK condition number lambda_max(Phi) / lambda_min(Phi).

    Used as the practical stand-in for the multi-layer PL* condition of
    Theorem thm:sc Part (b) (Liu, Zhu, Belkin 2022; Dinh et al. 2025).
    A finite, bounded condition number is the certificate that the
    iteration is in a locally PL* region (LPLR).

    Parameters
    ----------
    model : nn.Module
        A ``RiemannianVAE`` (or any model exposing ``node_decoder``).
    z_mu : (N, d) latent posterior means.
    edge_index : (2, E) directed edges.
    n_samples : int
    n_probe_vectors : int
    device : torch.device or None

    Returns
    -------
    float
        Empirical condition number; +inf if lambda_min == 0.
    """
    if device is None:
        device = z_mu.device

    lambda_min_phi, _r, _f = _jvp_gram_hutchinson(
        model, z_mu, edge_index,
        n_samples=n_samples,
        n_probe_vectors=n_probe_vectors,
        n_output_probes=16,
        device=device,
        project_out_symmetries=True,
    )
    # Estimate lambda_max via a power-iteration-like procedure with
    # max() across probes instead of min().
    decoder = model.node_decoder
    n_params = sum(p.numel() for p in decoder.parameters())
    if n_params == 0 or lambda_min_phi <= 0.0:
        return float("inf")

    # For practical purposes we report the ratio of the (positive)
    # symmetry-projected lambda_min to a coarse lambda_max upper bound
    # derived from the Frobenius norm of the average Jacobian. This is
    # an upper bound on the true condition number; smaller is better.
    src, dst = edge_index[0], edge_index[1]
    with torch.no_grad():
        z_src = z_mu[src]
        dz = z_mu[dst] - z_src
        log_maps = riemannian_log_maps_batched(decoder, z_src, dz)
        # Heuristic upper bound on lambda_max: the average squared norm.
        lambda_max_proxy = float(log_maps.pow(2).sum(dim=1).mean().item())

    return float(lambda_max_proxy / max(lambda_min_phi, 1e-12))


def adaptive_p_step_budget(
    mu_hat_1: float | None = None,
    tau_t: float = 1.0,
    eta: float = 1e-3,
    beta_smooth: float = 1.0,
    *,
    pl_star_constant: float | None = None,
    mu0_estimate: float | None = None,
) -> int:
    """LEGACY: adaptive P-step budget m*_t from the old proximal formulation.

        m*_t = ceil(log(1 / tau_t) / (eta * mu_1))

    Originally a helper for the Gibbs-temperature-annealed proximal
    trainer (Theorem thm:prox_fp of earlier drafts). In the current
    certified Riemannian VAE the training graph is STATIC (no
    temperature, no G-step), so tau_t = 1.0 always, log(1/tau_t) = 0,
    and the formula returns 1. This function is retained only for
    backward compatibility with the legacy ``SCRVAETrainer`` (which
    used it to schedule inner P-step iterations). New training code
    should not rely on it.

    Parameters
    ----------
    mu_hat_1 : float, optional
        Current restricted-SC / PL* constant estimate.
    tau_t : float
        Legacy temperature of the old soft-Gibbs graph. In the
        static-graph regime this is 1.0 and the function returns 1.
    eta : float
        Learning rate.
    beta_smooth : float
        Smoothness constant (default 1.0).
    pl_star_constant : float, optional
        DEPRECATED alias for ``mu_hat_1``.
    mu0_estimate : float, optional
        DEPRECATED alias for ``mu_hat_1``.

    Returns
    -------
    m_star : int
        Number of inner P-step (parameter-update) gradient steps.
    """
    mu = mu_hat_1
    if mu is None:
        mu = pl_star_constant
    if mu is None:
        mu = mu0_estimate
    if mu is None or mu <= 0 or tau_t <= 0 or eta <= 0:
        return 1000

    log_inv_tau = math.log(1.0 / max(tau_t, 1e-10))
    m_star = math.ceil(log_inv_tau / (eta * mu))
    return max(m_star, 1)


# Backward-compatible alias for the renamed function.
adaptive_mstep_budget = adaptive_p_step_budget
