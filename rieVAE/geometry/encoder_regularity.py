"""Encoder Regularity Verification for the SCR-VAE.

Implements Theorem thm:aec_verification and Corollary cor:aec_computable from the
paper (Section sec:enc_regularity).

For any MLP encoder g_phi with bounded weight matrices and C^2-smooth activations,
the regularity constants required by Assumptions A-EC and A-EC2 are automatically
finite and computable:

  C_g  <= L_phi  = sigma'_max^(L-1) * prod_l ||W_l||_op   (Lemma lem:enc_lipschitz)
  kappa_enc <= kappa_phi = (L-1) * sigma''_max * sigma'_max^{L-2} * ||W_max||_op^{L+1}
                                                        (Lemma lem:enc_hessian_bound)

  Note: L-1 activation-derivative factors, not L, because the output layer is
  linear (no activation after W_L).  The function encoder_lipschitz_bound()
  implements this correctly; the paper text (Lemma lem:enc_lipschitz) needs
  the same correction (sigma'_max^L -> sigma'_max^(L-1)).

These certified upper bounds make the isometry constant C_1 in Theorem thm:isometry
fully computable from data and architecture.

Main entry points:
  estimate_encoder_regularity(encoder, activation_info)
  measure_preactivation_range(decoder, z_samples)   -- for Assumption A-BPA
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Activation derivative bounds (sigma'_max, sigma''_max)
# ---------------------------------------------------------------------------

# Activation bounds: (sigma'_max, sigma''_max, sigma'_min_on_compact)
#
# sigma'_max and sigma''_max are global suprema, used in Lipschitz and
# Hessian bounds for Theorem thm:aec_verification.
#
# sigma'_min_on_compact is the infimum of sigma'(x) on a COMPACT preactivation
# range, required by Lemma lem:ntk_gram:
#   - Lemma 5 requires sigma'_min := inf_{x in R} sigma'(x) > 0 for the NTK
#     backward-layer bound  ||D_ell v|| >= sigma'_min * ||v||  to hold.
#   - This requires sigma'(x) > 0 for ALL x in the preactivation range.
#
# IMPORTANT — SiLU and GELU have NEGATIVE derivatives on part of the real line:
#   SiLU'(x)  < 0  for  x in (-inf, -1.278)  [minimum ≈ -0.0998 at x ≈ -2.4]
#   GELU'(x)  < 0  for  x in (-inf, -0.751)  [minimum ≈ -0.1289 at x ≈ -1.41]
#
# A negative sigma'_min means:
#   1. sigma'_min_compact < 0 → ntk_condition_holds = False (correct)
#   2. The per-layer backward bound alternates in sign for odd-depth steps,
#      making the SC proof INVALID for SiLU/GELU via the NTK route.
#   3. The paper's Lemma 5 incorrectly lists SiLU as satisfying sigma'_min > 0.
#
# The NTK SC proof is valid only for:
#   (a) Activations with sigma'(x) > 0 globally: tanh, sigmoid, softplus
#   (b) Any activation IF preactivations are known to be bounded in a range
#       where sigma'(x) > 0 throughout (e.g., SiLU with preact > -1.278).
#
# hat_mu_1 from verify_sc_condition() is always a valid EMPIRICAL estimate
# (it measures the actual Gram matrix).  The NTK-based THEORETICAL lower bound
# only applies when ntk_condition_holds = True.
#
# Format: (sigma'_max, sigma''_max, sigma'_min_compact)
ACTIVATION_BOUNDS: dict[str, tuple[float, float, float]] = {
    "relu":       (1.0,  0.0,   0.0),   # sigma'_min = 0 at x=0 (kink)
    "leaky_relu": (1.0,  0.0,   0.01),  # min slope = alpha (default 0.01); > 0 globally
    "tanh":       (1.0,  0.770, 0.007), # sigma'_min on [-5,5] ≈ sech²(5) ≈ 0.007; > 0 globally
    "sigmoid":    (0.25, 0.096, 0.007), # sigma'_min on [-5,5] ≈ sig(5)(1-sig(5)) ≈ 0.007; > 0 globally
    "gelu":       (1.10, 0.55,  0.0),   # GELU'(x) < 0 for x < -0.751; zero crossing at x≈-0.751
                                         # min on [-5,5] ≈ -0.1289 at x≈-1.41 → NTK NOT certified
    "silu":       (1.10, 0.40,  0.0),   # SiLU'(x) < 0 for x < -1.278; zero crossing at x≈-1.278
                                         # min on [-5,5] ≈ -0.0998 at x≈-2.4 → NTK NOT certified
    "elu":        (1.0,  1.0,   0.0),   # sigma'_min = 0 at the kink (x=0)
    "softplus":   (1.0,  0.25,  0.007), # log(1+e^x)': sigma(x) in (0,1); > 0 globally
    "mish":       (1.10, 0.50,  0.0),   # Mish'(x) can be negative; NTK NOT certified
}


@torch.no_grad()
def measure_preactivation_range(
    decoder: nn.Module,
    z_samples: torch.Tensor,
) -> tuple[float, float]:
    """Measure the min/max pre-activation values inside the decoder MLP.

    Used to verify Assumption A-BPA (bounded preactivations) from the paper.
    If the returned preact_min is above -1.278 (for SiLU), then sigma'_min > 0
    on the observed range and the NTK SC proof is certified for this checkpoint.

    Parameters
    ----------
    decoder : nn.Module
        The node decoder (f_theta).  Must contain nn.Linear layers followed by
        activations.  Works with any Sequential or custom MLP architecture.
    z_samples : (N, d) tensor
        Latent codes at the current checkpoint (e.g. posterior means mu_i).

    Returns
    -------
    (preact_min, preact_max) : (float, float)
        Global min and max of all pre-activation values
        h_ell = W_ell * a_{ell-1} + b_ell across all layers ell and all samples.
        These bound the preactivation range for Assumption A-BPA.
    """
    preact_min = float("inf")
    preact_max = float("-inf")

    hooks = []
    captured: list[torch.Tensor] = []

    def _hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        captured.append(out.detach().cpu())

    # Register hooks on all Linear layers to capture their pre-activation output
    # (the raw linear output before any activation is applied).
    for module in decoder.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(_hook))

    try:
        decoder.eval()
        _ = decoder(z_samples.detach())
        for tensor in captured:
            v_min = float(tensor.min().item())
            v_max = float(tensor.max().item())
            if v_min < preact_min:
                preact_min = v_min
            if v_max > preact_max:
                preact_max = v_max
    finally:
        for h in hooks:
            h.remove()

    if preact_min == float("inf"):
        return 0.0, 0.0
    return preact_min, preact_max


def activation_bounds(activation: str | nn.Module) -> tuple[float, float, float]:
    """Return (sigma'_max, sigma''_max, sigma'_min_compact) for a given activation.

    sigma'_min_compact is the infimum of |sigma'(x)| on the typical compact
    preactivation range [-5, 5] of a trained MLP.  It is used by Lemma
    lem:ntk_gram (Issue 2) to verify the hypothesis sigma'_min > 0 that is
    needed for the NTK lower bound on lambda_min(Phi).  Activations with
    sigma'_min_compact = 0 (ReLU, ELU) do NOT satisfy Lemma 5's hypotheses
    strictly; for those, the SC estimate hat_mu_1 should be treated as purely
    empirical rather than as a certified lower bound.

    Parameters
    ----------
    activation : str or nn.Module

    Returns
    -------
    (sigma_prime_max, sigma_double_prime_max, sigma_prime_min_compact)
    """
    if isinstance(activation, str):
        key = activation.lower().replace("-", "_")
    else:
        key = type(activation).__name__.lower().replace("-", "_")
        # Map PyTorch class names to keys
        key = {
            "relu": "relu",
            "leakyrelu": "leaky_relu",
            "tanh": "tanh",
            "sigmoid": "sigmoid",
            "gelu": "gelu",
            "silu": "silu",
            "elu": "elu",
            "softplus": "softplus",
            "mish": "mish",
        }.get(key, key)

    if key not in ACTIVATION_BOUNDS:
        # Default conservative bound for unknown activations
        return (2.0, 2.0, 0.0)
    return ACTIVATION_BOUNDS[key]


# ---------------------------------------------------------------------------
# 1. Encoder Lipschitz constant (Lemma lem:enc_lipschitz)
# ---------------------------------------------------------------------------

def _mu_path_layers(encoder: nn.Module) -> list[nn.Linear]:
    """Extract the Linear layers on the input -> mu path only.

    For a standard NodeEncoder this is backbone layers + mu_head.
    The logvar_head is on a separate branch and must be excluded.
    """
    layers: list[nn.Linear] = []
    backbone = getattr(encoder, "backbone", None)
    mu_head = getattr(encoder, "mu_head", None)

    if backbone is not None:
        for module in backbone.modules():
            if isinstance(module, nn.Linear):
                layers.append(module)
        if mu_head is not None and isinstance(mu_head, nn.Linear):
            layers.append(mu_head)
    else:
        for module in encoder.modules():
            if isinstance(module, nn.Linear):
                layers.append(module)
    return layers


def encoder_lipschitz_bound(
    encoder: nn.Module,
    sigma_prime_max: float = 1.1,
) -> float:
    """Compute the upper bound L_phi on the encoder Lipschitz constant.

    L_phi = sigma'_max^(L-1) * prod_l ||W_l||_op

    Only layers on the input -> mu path are included (logvar_head excluded).
    The final mu_head has no activation, so the activation count is L-1
    where L is the number of Linear layers on the mu path.

    Parameters
    ----------
    encoder : nn.Module
        The encoder network (any MLP with Linear layers).
    sigma_prime_max : float
        Upper bound on |sigma'(x)| for the activation. Default 1.1 (SiLU).

    Returns
    -------
    L_phi : float
        Upper bound on the encoder Lipschitz constant C_g.
    """
    mu_layers = _mu_path_layers(encoder)
    if not mu_layers:
        return 1.0

    L_phi = 1.0
    for layer in mu_layers:
        W = layer.weight.detach().float()
        sigma_max = float(torch.linalg.matrix_norm(W, ord=2).item())
        L_phi *= sigma_max

    n_activations = max(0, len(mu_layers) - 1)
    L_phi *= (sigma_prime_max ** n_activations)
    return float(L_phi)


# ---------------------------------------------------------------------------
# 2. Encoder Hessian bound (Lemma lem:enc_hessian_bound)
# ---------------------------------------------------------------------------

def encoder_hessian_bound(
    encoder: nn.Module,
    sigma_prime_max: float = 1.1,
    sigma_double_prime_max: float = 0.40,
) -> float:
    """Compute the upper bound kappa_phi on the encoder Hessian operator norm.

    kappa_phi = L * sigma''_max * sigma'_max^{L-2} * ||W_max||_op^{L+1}

    This is Lemma lem:enc_hessian_bound: at any training checkpoint,
    kappa_enc <= kappa_phi, and A-EC2 holds (for C^2 activations).

    Parameters
    ----------
    encoder : nn.Module
        The encoder network.
    sigma_prime_max : float
        Upper bound on |sigma'|. Default 1.1 (SiLU).
    sigma_double_prime_max : float
        Upper bound on |sigma''|. Default 0.40 (SiLU).
        Pass 0.0 for ReLU (piecewise linear; Hessian is zero a.e.).

    Returns
    -------
    kappa_phi : float
        Upper bound on kappa_enc (encoder Hessian operator norm).
    """
    mu_layers = _mu_path_layers(encoder)
    weight_op_norms = []
    for layer in mu_layers:
        W = layer.weight.detach().float()
        sigma_max = float(torch.linalg.matrix_norm(W, ord=2).item())
        weight_op_norms.append(sigma_max)

    L = len(weight_op_norms)
    if L == 0:
        return 0.0

    W_max = max(weight_op_norms)

    if sigma_double_prime_max == 0.0:
        # ReLU: Hessian is 0 a.e. (piecewise linear network)
        return 0.0

    # Tighter bound using the PRODUCT of all weight spectral norms,
    # not W_max^(L+1).  The chain-rule Hessian bound is:
    #   kappa_enc ≤ (L-1) × σ''_max × σ'_max^(L-2) × prod_{l} ||W_l||_op
    #
    # L-1 (not L) because the output layer has NO activation; only the L-1
    # hidden layers can contribute a σ'' Hessian term.  Each such term has:
    #   - one σ'' factor (from the layer where the Hessian is taken)
    #   - L-2 σ' factors (from all other hidden layers)
    #   - prod_W weight-norm factors (L matrices, bilinear structure)
    # The former version used L here, which overcounts by 1.
    prod_W = 1.0
    for norm in weight_op_norms:
        prod_W *= norm

    n_hessian_terms = max(0, L - 1)  # only hidden layers contribute σ''
    kappa_phi = (
        n_hessian_terms
        * sigma_double_prime_max
        * (sigma_prime_max ** max(0, L - 2))
        * prod_W
    )
    return float(kappa_phi)


# ---------------------------------------------------------------------------
# 3. Full Encoder Regularity Diagnostic (Thm. thm:aec_verification)
# ---------------------------------------------------------------------------


def measure_ab_norm_min(
    encoder: nn.Module,
    x_samples: torch.Tensor,
) -> float:
    """Phase-1 helper preserved as a no-op for compatibility.

    The pre-Phase-2 ``TorusNodeEncoder`` (deleted in Phase 2 of op47C)
    had a 4-output ``circle_head`` whose minimum (a, b) norm gated
    the atan2 Lipschitz constant. The unified ``RiemannianVAE`` of
    Phase 2 emits raw chart coordinates without an architectural
    atan2, so this measurement is no longer meaningful. The function
    returns 0.0 unconditionally and will be removed in Phase 4.
    """
    return 0.0
    encoder.eval()
    h = encoder.backbone(x_samples)
    circle = encoder.circle_head(h)            # (N, 4)
    a1, b1 = circle[:, 0], circle[:, 1]
    a2, b2 = circle[:, 2], circle[:, 3]
    norm1 = (a1.pow(2) + b1.pow(2)).sqrt()
    norm2 = (a2.pow(2) + b2.pow(2)).sqrt()
    return float(torch.min(torch.cat([norm1, norm2])).item())


def estimate_encoder_regularity(
    encoder: nn.Module,
    activation: str | nn.Module = "silu",
    Lambda_max: float = 1.0,
    kappa_decoder: float = 1.0,
    verbose: bool = False,
    x_samples: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute certified upper bounds on encoder regularity constants.

    Implements Theorem thm:aec_verification and Corollary cor:aec_computable.
    Returns computable bounds hat_L_phi and hat_kappa_phi that certify
    Assumptions A-EC and A-EC2.

    Parameters
    ----------
    encoder : nn.Module
        The encoder network g_phi.
    activation : str or nn.Module
        Activation function name or instance (determines sigma'_max, sigma''_max).
    Lambda_max : float
        Upper bound on lambda_max(G*(z)) -- the decoder metric upper bound
        (Assumption A1). Used to compute the combined constant in Eq. eq:aec_combined.
    kappa_decoder : float
        Decoder Hessian bound kappa (from Assumption A2). Used in eq:aec_combined.
    verbose : bool
        If True, print a diagnostic table.

    Returns
    -------
    dict with keys:
      'L_phi'             : float -- upper bound on C_g (encoder Lipschitz)
      'kappa_phi'         : float -- upper bound on kappa_enc (encoder Hessian)
      'sigma_prime_max'   : float -- activation bound used
      'sigma_double_prime_max' : float
      'n_linear_layers'   : int   -- depth L
      'W_max_op_norm'     : float -- max weight spectral norm
      'combined_bound'    : float -- L_phi*kappa_decoder + kappa_phi*Lambda_max
                                     (appears in lem:jac_bound's Hessian composition)
      'aec_holds'         : bool  -- L_phi < inf (always True for bounded weights)
      'aec2_holds'        : bool  -- kappa_phi < inf (True for C^2 activations)
    """

    sp_max, spp_max, sp_min_compact = activation_bounds(activation)

    L_phi = encoder_lipschitz_bound(encoder, sigma_prime_max=sp_max)
    kappa_phi = encoder_hessian_bound(
        encoder,
        sigma_prime_max=sp_max,
        sigma_double_prime_max=spp_max,
    )

    # Count linear layers
    n_linear = sum(1 for m in encoder.modules() if isinstance(m, nn.Linear))

    # Max weight spectral norm
    W_op_norms = []
    for module in encoder.modules():
        if isinstance(module, nn.Linear):
            W = module.weight.detach().float()
            W_op_norms.append(float(torch.linalg.matrix_norm(W, ord=2).item()))
    W_max_op = max(W_op_norms) if W_op_norms else 0.0

    # Combined constant from Eq. eq:aec_combined (in lem:jac_bound Step 1)
    combined_bound = L_phi * kappa_decoder + kappa_phi * Lambda_max

    # Lemma lem:ntk_gram condition: sigma'_min > 0 required for the NTK
    # lower bound on lambda_min(Phi) to be a certified guarantee.
    # For activations where sigma'_min_compact = 0 (e.g. ReLU), the SC
    # estimate hat_mu_1 from verify_sc_condition() is purely empirical.
    ntk_condition_holds = sp_min_compact > 0.0

    result = {
        "L_phi": float(L_phi),
        "kappa_phi": float(kappa_phi),
        "sigma_prime_max": float(sp_max),
        "sigma_double_prime_max": float(spp_max),
        "sigma_prime_min_compact": float(sp_min_compact),
        "ntk_condition_holds": ntk_condition_holds,
        "n_linear_layers": int(n_linear),
        "W_max_op_norm": float(W_max_op),
        "combined_bound": float(combined_bound),
        "aec_holds": math.isfinite(L_phi),
        "aec2_holds": math.isfinite(kappa_phi),
    }

    if verbose:
        print("\n[Encoder Regularity (Thm. thm:aec_verification)]")
        print(f"  Activation:         {activation} "
              f"(sigma'_max={sp_max}, sigma''_max={spp_max}, "
              f"sigma'_min_compact={sp_min_compact})")
        print(f"  Depth L:            {n_linear}")
        print(f"  Max ||W_l||_op:     {W_max_op:.4f}")
        print(f"  L_phi (A-EC bound): {L_phi:.4f}  "
              f"{'OK -- A-EC holds' if result['aec_holds'] else 'FAIL'}")
        print(f"  kappa_phi (A-EC2):  {kappa_phi:.4f}  "
              f"{'OK -- A-EC2 holds' if result['aec2_holds'] else 'FAIL (ReLU?)'}")
        print(f"  NTK cond. (Lem.5):  {'OK -- sigma_min>0' if ntk_condition_holds else 'NOT CERTIFIED -- sigma_min=0 (ReLU/ELU); SC estimate is empirical only'}")
        print(f"  Combined Hessian:   {combined_bound:.4f}  "
              f"(= L_phi*kappa_dec + kappa_phi*Lambda_max)")

    return result
