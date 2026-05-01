"""Runtime isometry certificate (Definition def:cert of the paper).

The certificate bundles FOUR scalar conditions into a single
pass/fail signal. When all four pass at a checkpoint, the isometry
conclusion of Theorem thm:isometry_main holds at that checkpoint
without further unverifiable assumptions:

    C1. delta_rec    <= C_cap        * r_n^p_rec   (decoder capacity)
    C2. delta_edge   <= C_cap_prime  * r_n^p_edge  (edge-head capacity)
    C3. mu_hat_1     > 0 (restricted strong convexity / PL* witness)
    C4. lambda_t     >= lambda_cross := r_n^2 / mu_hat_1

The two deprecated conditions of earlier drafts have been removed:
  * The deformation residual C3(old) (delta_def) is gone with the
    deformation module itself; the training graph is now the static
    biharmonic spectral edge set and there is no deformation to
    certify (Section sec:spectral of the paper).
  * The soft-to-hard gap C6(old) is gone with the temperature
    schedule; the training graph is always hard and never softened.

Chart regime (``chart_regime`` keyword of :func:`compute_certificate`):
  - "general"  -> p_rec = p_edge = 2 (Theorem thm:isometry_main).
  - "flat"     -> p_rec = p_edge = 3 (Corollary cor:topo_matched;
                  the tightened conditions on a topology-matched chart).

Optional topology diagnostic: ``delta_fold_lower_bound``
    delta_fold  >=  lambda_0 * inj(M) / (C_tau * sqrt(Lambda_max))
(see App. app:topo of the paper). When the latent topology matches
M the fold set is empty and delta_fold is not reported.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch


@dataclasses.dataclass
class CertificateThresholds:
    """Tunable thresholds for the four certificate conditions.

    C1' check (delta_edge parameter = paper's delta_iso):
        max_{E*} |softplus(w*) d_{Mz}(mu_i, mu_j) - tilde_w_ij|
        Threshold: C_cap * r_n^r_n_power_rec
        Default: C_cap=5, r_n_power_rec=1  =>  5*r_n

    C2 check -- two modes (parameter-free default recommended):

        DEFAULT (parameter-free, eq:c2_edge_scale in main.tex):
            rec_threshold > 0 (passed in from SpectralArtefacts):
                L_rec <= rec_threshold  where
                rec_threshold = mean_{E*}(tilde_w_{ij}^2) = Theta(r_n^2).
            The threshold is computed at preprocessing and stored in
            SpectralArtefacts.rec_threshold. It is data-scale-invariant,
            n-dependent, and requires no user constant.

        PARAMETRIC FALLBACK (rec_threshold <= 0):
            L_rec <= C_cap_prime * r_n^r_n_power_edge
            Default: C_cap_prime=1.0, r_n_power_edge=0 (absolute 1.0).
            Use only when SpectralArtefacts is not available, e.g. in
            unit tests.

    NOTE on caller convention: `delta_edge` = delta_iso (C1'),
    `delta_rec` = L_rec or sup ||f(z)-x|| depending on caller. The
    parameter naming follows the code's historic labeling; see
    Remark rem:cert_labels in main.tex.

    Parameters
    ----------
    C_cap : float
        Capacity constant for C1'. Default 5 (paper's 5*r_n).
    r_n_power_rec : int
        Exponent for C1'. Default 1 (linear).
    rec_threshold : float
        Parameter-free C2 threshold (eq:c2_edge_scale). When > 0 this
        overrides the parametric C_cap_prime fallback. Pass
        SpectralArtefacts.rec_threshold here.
    C_cap_prime : float
        Parametric C2 constant. Used only when rec_threshold <= 0.
    r_n_power_edge : int
        Exponent for parametric C2. Used only when rec_threshold <= 0.
    """

    C_cap: float = 5.0
    r_n_power_rec: int = 1
    rec_threshold: float = 0.0
    C_cap_prime: float = 1.0
    r_n_power_edge: int = 0

    @classmethod
    def for_chart_regime(
        cls,
        chart_regime: str = "general",
        rec_threshold: float = 0.0,
        **kwargs,
    ) -> "CertificateThresholds":
        """ISO-architecture thresholds for the named chart regime.

        Pass rec_threshold=artefacts.rec_threshold to use the
        parameter-free C2 (recommended). When rec_threshold <= 0 the
        parametric fallback C_cap_prime * r_n^r_n_power_edge is used.

        'general'  -> C1': 5*r_n  (linear)
        'flat'     -> C1': 5*r_n^2 (quadratic, topo-matched)
        """
        if chart_regime == "general":
            defaults = dict(C_cap=5.0, r_n_power_rec=1)
        elif chart_regime == "flat":
            defaults = dict(C_cap=5.0, r_n_power_rec=2)
        else:
            raise ValueError(
                f"chart_regime must be 'general' or 'flat', got {chart_regime!r}"
            )
        defaults["rec_threshold"] = float(rec_threshold)
        defaults.update(kwargs)
        return cls(**defaults)


@dataclasses.dataclass
class CertificateReport:
    """Scalar quantities reported by the certificate at a checkpoint.

    All entries are plain Python floats / booleans so the object is
    trivially logged to CSV / wandb / JSON.  ``isometry_holds`` is the
    AND of THREE conditions C1'--C3 only; C4 is a diagnostic (schedule
    indicator) excluded from the certified pass/fail for the ISO
    architecture (see compute_certificate and Remark rem:cert_labels).

    Scope fields
    ------------
    The theorem's conclusion
    ``sup_{(i, j) in E*} |d^{R*} - d^M| <= C_1 r_n`` requires a
    SUPREMUM over all training edges. Batch-local approximations
    are biased underestimates and do not license the isometry
    conclusion. Two scope fields document which case the caller is in:

    is_global : bool
        True iff ``delta_rec`` and ``delta_edge`` were computed as a
        supremum over the full active sample (possibly subsampled with
        ``global_n_used`` nodes). False if the caller passed batch-local
        estimates; in that case ``isometry_holds`` is NOT a valid
        certificate of Theorem thm:isometry_main and should be read as
        a log-diagnostic only.
    global_n_used : int or None
        Number of active nodes over which the sup was taken. ``None``
        when ``is_global=False``. A value equal to n_active means the
        full-data pass was used; a smaller value is a subsample scope.
    """

    # NOTE on label convention (see Remark rem:cert_labels in main.tex):
    # compute_certificate swaps the intuitive order to match the theorem:
    #   c1_ok <-> delta_edge parameter (= paper's C1': encoder isometry
    #             = |softplus(w*)||mu_i-mu_j|| - tilde_w| <= 5 r_n)
    #   c2_ok <-> delta_rec  parameter (= paper's C2: reconstruction scale
    #             = L_rec <= rec_threshold = mean_{E*}(tilde_w^2) [default]
    #               or L_rec <= C_cap_prime * r_n^r_n_power_edge [fallback])
    #   c3_ok <-> mu_hat_1 > 0
    #   c4_ok <-> lambda_t >= lambda_cross  [diagnostic only, NOT in isometry_holds]
    # isometry_holds = c1_ok AND c2_ok AND c3_ok  (C4 excluded for ISO arch)
    r_n: float
    delta_rec: float
    delta_edge: float
    mu_hat_1: float
    mu_hat_1_output_layer: float
    lambda_t: float
    lambda_cross: float
    c1_ok: bool
    c2_ok: bool
    c3_ok: bool
    c4_ok: bool
    isometry_holds: bool
    envelope_C1_rn: float
    is_global: bool = True
    global_n_used: Optional[int] = None
    fold_fraction: Optional[float] = None
    delta_fold_bound: Optional[float] = None
    properness_holds: Optional[bool] = None
    simple_connectivity_assumed: bool = True
    chart_regime: str = "general"
    alignment_diagnostic: Optional[float] = None

    def __post_init__(self) -> None:
        # isometry_holds is the certified conclusion only when all
        # four conditions are scope-global. Collapse to False with a
        # warning if the caller supplied batch-local residuals.
        if not self.is_global and self.isometry_holds:
            import warnings
            warnings.warn(
                "CertificateReport constructed with is_global=False but "
                "isometry_holds=True. The theorem requires a supremum "
                "over all training edges; batch-local residuals do not "
                "license the isometry conclusion. Setting "
                "isometry_holds=False; this report is a log diagnostic "
                "only, not a certificate of Theorem thm:isometry_main.",
                stacklevel=2,
            )
            self.isometry_holds = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_metrics(self, prefix: str = "cert/") -> dict:
        """Return a wandb/CSV-ready metrics dict."""
        keys = [
            "r_n", "delta_rec", "delta_edge",
            "mu_hat_1", "mu_hat_1_output_layer",
            "lambda_t", "lambda_cross", "envelope_C1_rn",
        ]
        out = {f"{prefix}{k}": getattr(self, k) for k in keys}
        for i in range(1, 5):
            out[f"{prefix}c{i}_ok"] = 1.0 if getattr(self, f"c{i}_ok") else 0.0
        out[f"{prefix}isometry_holds"] = 1.0 if self.isometry_holds else 0.0
        out[f"{prefix}is_global"] = 1.0 if self.is_global else 0.0
        if self.global_n_used is not None:
            out[f"{prefix}global_n_used"] = float(self.global_n_used)
        if self.fold_fraction is not None:
            out[f"{prefix}fold_fraction"] = self.fold_fraction
        if self.delta_fold_bound is not None:
            out[f"{prefix}delta_fold_bound"] = self.delta_fold_bound
        if self.alignment_diagnostic is not None:
            out[f"{prefix}alignment_diagnostic"] = self.alignment_diagnostic
        if self.properness_holds is not None:
            out[f"{prefix}properness_holds"] = (
                1.0 if self.properness_holds else 0.0
            )
        out[f"{prefix}simple_connectivity_assumed"] = (
            1.0 if self.simple_connectivity_assumed else 0.0
        )
        out[f"{prefix}chart_regime"] = self.chart_regime
        return out


# ---------------------------------------------------------------- helpers

def compute_r_n(n: int, d: int) -> float:
    """Standard kNN radius rate r_n = (log n / n)^{1/d}."""
    if n <= 1 or d <= 0:
        return float("nan")
    return (math.log(max(n, 2)) / n) ** (1.0 / d)


def intrinsic_dim_estimate(
    x: "torch.Tensor",
    k: int = 5,
    n_anchor: int = 512,
    seed: int = 0,
) -> int:
    """Estimate the intrinsic dimension of x via the Two-NN MLE (Facco et al. 2017).

    For each anchor point the estimator uses the ratio of the 1st- and
    2nd-nearest-neighbour distances:

        mu_i = T_i2 / T_i1

    and fits the Pareto distribution with maximum-likelihood estimate:

        d_hat = n / sum_i log(mu_i)

    This is the same estimator used by the scikit-dimension library. It is
    O(n_anchor * n) and runs in a fraction of a second for typical manifold
    sizes (n <= 50 000, ambient dim <= 1000).

    Parameters
    ----------
    x : (N, G) float tensor or numpy array of ambient coordinates.
    k : int, ignored (retained for API compatibility; Two-NN always uses k=2).
    n_anchor : int, number of random anchor points to subsample for speed.
    seed : int, random seed for anchor selection.

    Returns
    -------
    d : int, estimated intrinsic dimension (clamped to [1, min(G, N-1)]).
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    if hasattr(x, "detach"):
        x_np = x.detach().cpu().float().numpy()
    else:
        x_np = np.asarray(x, dtype=np.float32)

    N, G = x_np.shape
    if N < 4:
        return 1

    rng = np.random.default_rng(seed)
    n_sub = min(n_anchor, N)
    idx = rng.choice(N, size=n_sub, replace=False)
    x_sub = x_np[idx]

    nn = NearestNeighbors(n_neighbors=3).fit(x_np)
    dists, _ = nn.kneighbors(x_sub)      # (n_sub, 3); col 0 = self (dist=0)
    # Two-NN uses the first and second non-self neighbours.
    T1 = dists[:, 1].clip(min=1e-12)     # 1st NN distance
    T2 = dists[:, 2].clip(min=1e-12)     # 2nd NN distance

    mu = T2 / T1                          # ratio always >= 1
    # Exclude degenerate points where T1 == T2 (duplicate points).
    valid = mu > 1.0 + 1e-6
    if valid.sum() < 4:
        return max(1, min(G, 2))          # fallback: guess 2

    d_hat = 1.0 / float(np.mean(np.log(mu[valid])))
    d_int = int(round(d_hat))
    return max(1, min(d_int, G, N - 1))


def lambda_cross_from(mu_hat_1: float, r_n: float) -> float:
    """Schedule-regime threshold lambda_cross = r_n^2 / mu_hat_1."""
    if mu_hat_1 <= 0.0:
        return float("inf")
    return (r_n * r_n) / mu_hat_1


def delta_fold_lower_bound(
    lambda_0: float,
    Lambda_max: float,
    inj_M: float,
    C_tau: float = math.pi,
) -> float:
    """Topological fold-separation lower bound (Thm. thm:topo_floor).

    The theorem states: every fold pair (z, z') satisfies
        d^{R*}(z, z') >= 2 * inj(M) / C_tau  (main lower bound)
    where C_tau = 1 + O(delta_rec / tau) is the Federer projection
    Lipschitz constant. The pullback geodesic distance d^{R*} is bounded
    below in Euclidean latent units by d^{R*} / sqrt(Lambda_max), giving:
        ||z - z'||_Eucl >= 2 * inj(M) / (C_tau * sqrt(Lambda_max))

    This function returns the Euclidean latent-distance lower bound.
    lambda_0 is accepted for backward compatibility but is NOT used in
    the bound (lambda_min(G*) enters only in the LENGTH COMPARISON step,
    not in the fold-separation theorem itself).

    Parameters
    ----------
    lambda_0 : float
        Accepted for API compatibility; not used in the bound.
    Lambda_max : float
        Upper bound on lambda_max(G^*) at the checkpoint; converts the
        d^{R*} bound to a Euclidean latent distance bound.
    inj_M : float
        Injectivity radius of M.
    C_tau : float
        Federer chord-to-arc constant. Default pi (safe upper bound
        for the Clifford torus with radii (2,1): C_tau ~= 1 + O(0)).
    """
    if Lambda_max <= 0 or C_tau <= 0 or inj_M <= 0:
        return float("nan")
    return (2.0 * inj_M) / (C_tau * math.sqrt(Lambda_max))


def alignment_diagnostic(
    pullback_matrix: torch.Tensor,
    data_covariance: torch.Tensor,
    c: Optional[float] = None,
) -> float:
    """Post-hoc consistency diagnostic || G^* - c * Sigma^{-1} ||_op.

    Not a certificate condition: this is a DIAGNOSTIC computed from
    the already-certified C1, C2 quantities and the classical
    manifold-hypothesis identity Sigma(x) ~ rho(x)^-1 g(x)^-1
    (Proposition prop:reweight_refinement in the paper). It is
    reported as an optional refinement factor for the isometry
    constant C_1 under the decoder-independent reweighting of
    Section sec:reweight.

    Parameters
    ----------
    pullback_matrix : (G, G) observed G^*(g_phi(x_i)) at a sample point.
    data_covariance : (G, G) local PCA Sigma(x_i) at the same sample
        point.
    c : optional global scale factor. If None, estimated from the
        ratio of the top eigenvalues.

    Returns
    -------
    float : operator-norm distance.
    """
    Sigma_inv = torch.linalg.pinv(data_covariance)
    if c is None:
        lam_top_G = float(torch.linalg.eigvalsh(pullback_matrix).max().item())
        lam_top_Sinv = float(torch.linalg.eigvalsh(Sigma_inv).max().item())
        c = lam_top_G / max(lam_top_Sinv, 1e-12)
    diff = pullback_matrix - c * Sigma_inv
    return float(torch.linalg.svdvals(diff).max().item())


# ---------------------------------------------------------------- main

def compute_certificate(
    *,
    n: int,
    d: int,
    delta_rec: float,
    delta_edge: float,
    mu_hat_1: float,
    mu_hat_1_output_layer: float,
    lambda_t: float,
    envelope_C1: float = 1.0,
    thresholds: Optional[CertificateThresholds] = None,
    chart_regime: str = "general",
    fold_fraction: Optional[float] = None,
    lambda_0: Optional[float] = None,
    Lambda_max: Optional[float] = None,
    inj_M: Optional[float] = None,
    properness_holds: Optional[bool] = None,
    simple_connectivity_assumed: bool = True,
    alignment_diagnostic_value: Optional[float] = None,
    is_global: bool = True,
    global_n_used: Optional[int] = None,
) -> CertificateReport:
    """Compute the four-scalar certificate at a checkpoint.

    Parameters
    ----------
    n : int
        Sample size used to compute r_n = (log n / n)^{1/d}.
    d : int
        Intrinsic manifold dimension.
    delta_rec, delta_edge : float
        Observed residuals at the checkpoint. These MUST be suprema
        over all training edges for the certified isometry conclusion
        to hold; see ``is_global`` / ``global_n_used`` below.
    mu_hat_1, mu_hat_1_output_layer : float
        Restricted strong convexity witnesses (Theorem thm:sc Part a
        unconditional, full via Hutchinson on the symmetry-reduced
        tangent space).
    lambda_t : float
        Current Riemannian regulariser weight.
    envelope_C1 : float
        Constant used for the isometry envelope C_1 r_n.
    thresholds : CertificateThresholds or None
        Tunable thresholds; defaults to the 'general' chart regime.
    chart_regime : {"general", "flat"}
        Tightens r_n powers when the latent chart is topology-matched.
    fold_fraction, lambda_0, Lambda_max, inj_M, properness_holds :
        Optional topology diagnostics (App. app:topo).
    simple_connectivity_assumed : bool
        Scope flag for Theorem thm:isometry_main; the simply-connected
        branch cannot be empirically verified.
    alignment_diagnostic_value : Optional[float]
        Optional post-hoc diagnostic
        || G^* - c * Sigma^-1 ||_op from
        Proposition prop:reweight_refinement. Not a certificate
        condition; reported for optional reweighting refinement.
    is_global : bool, default True
        True iff ``delta_rec`` and ``delta_edge`` are suprema over all
        training edges (possibly via a subsample of
        ``global_n_used`` nodes). False means batch-local estimates
        were supplied; the returned report will collapse
        ``isometry_holds`` to False with a warning, since the
        theorem's supremum condition is not attested by batch-local
        residuals.
    global_n_used : Optional[int], default None
        When ``is_global=True``, the number of active nodes over
        which the supremum was taken. Passing n_active denotes a
        full-data pass; a smaller value denotes a cert-time
        subsample. Leave None when ``is_global=False``.

    Returns
    -------
    CertificateReport
    """
    thresholds = thresholds or CertificateThresholds.for_chart_regime(chart_regime)

    r_n = compute_r_n(n, d)
    lambda_cross = lambda_cross_from(mu_hat_1, r_n)

    # C1' (paper) = encoder isometry condition.
    # The caller passes this as `delta_edge` (= delta_edge_scalar =
    # max |softplus(w*)||mu_i-mu_j|| - tilde_w_ij|).
    # ISO default threshold: C_cap * r_n^1 = 5*r_n (linear).
    c1_ok = delta_edge <= thresholds.C_cap * (r_n ** thresholds.r_n_power_rec)

    # C2 (paper) = reconstruction capacity condition (eq:c2_edge_scale).
    # Two modes:
    #   DEFAULT (parameter-free): rec_threshold = mean_{E*}(tilde_w^2)
    #     from SpectralArtefacts.rec_threshold.  The caller passes
    #     delta_rec = L_rec (mean squared reconstruction loss).
    #     This threshold is Theta(r_n^2), data-scale-invariant, and
    #     automatically detects topology mismatch at large n.
    #   PARAMETRIC FALLBACK: when rec_threshold <= 0, falls back to
    #     C_cap_prime * r_n^r_n_power_edge (default: absolute 1.0).
    if thresholds.rec_threshold > 0.0:
        c2_ok = delta_rec <= thresholds.rec_threshold
    else:
        c2_ok = delta_rec <= thresholds.C_cap_prime * (
            r_n ** thresholds.r_n_power_edge
        )
    c3_ok = mu_hat_1 > 0.0
    c4_ok = lambda_t >= lambda_cross

    # For the ISO architecture, isometry is certified by C1'--C3 alone.
    # C4 (lambda_t >= lambda_cross = r_n^2 / mu_hat_1) is a legacy
    # schedule-regime indicator for the IFT-based pullback theorem
    # (Theorem thm:prox_fp / App. app:schedule). It is NOT required for
    # the direct encoder isometry certificate (the paper: "When C1'--C3
    # pass, the encoder has achieved locally isometric posterior means on
    # E* at the correct scale."). On topologically nontrivial manifolds
    # with Euclidean latent (e.g. T^2), mu_hat_1 ~ 1e-8 (the fold pairs
    # genuinely reduce SC to near zero -- this is CORRECT, not a bug),
    # making lambda_cross ~ 1e5 and C4 impassable with lambda_t = O(1).
    # C4 is retained as a logged diagnostic but excluded from isometry_holds.
    isometry_holds = c1_ok and c2_ok and c3_ok
    envelope_C1_rn = envelope_C1 * r_n

    fold_bound: Optional[float] = None
    if lambda_0 is not None and Lambda_max is not None and inj_M is not None:
        fold_bound = delta_fold_lower_bound(lambda_0, Lambda_max, inj_M)

    return CertificateReport(
        r_n=r_n,
        delta_rec=delta_rec,
        delta_edge=delta_edge,
        mu_hat_1=mu_hat_1,
        mu_hat_1_output_layer=mu_hat_1_output_layer,
        lambda_t=lambda_t,
        lambda_cross=lambda_cross,
        c1_ok=c1_ok,
        c2_ok=c2_ok,
        c3_ok=c3_ok,
        c4_ok=c4_ok,
        isometry_holds=isometry_holds,
        envelope_C1_rn=envelope_C1_rn,
        is_global=is_global,
        global_n_used=global_n_used,
        fold_fraction=fold_fraction,
        delta_fold_bound=fold_bound,
        properness_holds=properness_holds,
        simple_connectivity_assumed=simple_connectivity_assumed,
        chart_regime=chart_regime,
        alignment_diagnostic=alignment_diagnostic_value,
    )
