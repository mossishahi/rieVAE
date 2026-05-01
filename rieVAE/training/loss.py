"""Loss functions for the Certified Riemannian VAE (iso architecture).

Effective objective (cf. main.tex eq:loss):

    L(t) = L_rec / s_rec
         + beta(t)  * L_KL  / s_kl
         + gamma(t) * L_iso / s_iso

Term semantics:

    L_rec : reconstruction term. In Phase 1 this is MSE (Gaussian
            likelihood); the unified manifold-VAE template of Phase 2
            replaces it with -E_{q(z|x)}[log p_theta(x|z)] for a
            pluggable likelihood.
    L_KL  : KL divergence q_phi(z|x) || p(z). With kl_mode='partial'
            (the iso default) the gradient on mu is zero and L_KL
            regularises sigma only; the geometric shape of the latent
            is shaped exclusively by L_iso and L_rec.
    L_iso : direct latent-isometry term against precomputed Varadhan
            heat-kernel targets on the static spectral edge set E*.
            Gradient flows to encoder means only (decoder receives
            zero gradient from L_iso, breaking the self-consistency
            failure of JVP-based Riemannian regularisers).

Scale balancing (initial-scale normalisation):

    s_rec, s_kl, s_iso : recorded ONCE on the warm-up batch at step 0
    and held fixed for the rest of training. The certificate residuals
    delta_rec / delta_iso are defined on the RAW forward outputs, not
    on the scaled loss.

Phase-1 history:

    Pre-R4 the file additionally carried a JVP-architecture loss
    bundle SCRVAELoss with L_vector / L_anchor / decoder-Hessian /
    metric-floor / free-bits-band / log-mean-square / dependence-(DI)
    helpers. The static-graph iso architecture (Section sec:method of
    the post-R4 paper) does not need any of that machinery. Phase 1
    of op47C deletes it; the JVP path returns in Phase 3 of op47C as
    the dedicated IsoPlusJVPLegacyTrainingPlan if/when ablation runs
    need it again. The only public surface in this file is now:

        - node_reconstruction_loss
        - node_kl_loss
        - iso_loss
        - calibrate_edge_decoder_scale
        - compute_delta_iso
        - compute_delta_edge_scalar
        - IsoVAELoss
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- Node

def node_reconstruction_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error reconstruction loss (Gaussian likelihood)."""
    return nn.functional.mse_loss(x_hat, x, reduction="mean")


def node_kl_loss(
    mu: torch.Tensor,
    var: torch.Tensor,
    flat_prior: bool = False,
    kl_mode: str = "auto",
    free_bits: float = 0.0,
) -> torch.Tensor:
    """KL divergence for the node posterior, averaged over nodes.

    Three base modes (kl_mode):
    - 'standard': 0.5*(mu^2 + var - 1 - log var)
      Gradient wrt mu: mu (pulls mu->0); wrt var: 0.5*(1-1/var) (pulls var->1)
    - 'flat': -0.5*(1 + log var)
      Entropy-only; no attracting fixed point for var (var->inf).
    - 'partial': 0.5*(var - 1 - log var)
      Drops mu^2; var still attracted to sigma=1; mu shaped only by other
      losses. This is the formula the manuscript names the "partial KL"
      (cf. main.tex sec:method) and is the recommended mode for the
      iso-architecture, where L_iso provides geometric regularisation on
      mu so memorization collapse is prevented by the isometry constraint
      rather than by the KL mean term.

    Free-bits modifier (Kingma et al. 2016):
    When free_bits > 0 AND kl_mode='standard', per-dimension KL is clamped
    from below at free_bits (in nats):

        kl_k = max(free_bits, 0.5*(mu_k^2 + var_k - 1 - log var_k))

    Inside the free zone (KL_k < free_bits): gradient is zero -- encoder
    is unconstrained geometrically.  Outside: standard KL activates --
    compactness enforced, memorization collapse prevented.

    At var=1 the free zone covers |mu_k| <= sqrt(2*free_bits).
    Recommended: free_bits=1.0 for d=2 latent on S^2.
    """
    if kl_mode == "auto":
        mode = "flat" if flat_prior else "standard"
    else:
        mode = kl_mode

    if mode == "flat":
        kl_per_dim = -0.5 * (1.0 + var.log())
    elif mode == "partial":
        kl_per_dim = 0.5 * (var - 1.0 - var.log())
    elif mode == "standard":
        kl_per_dim = 0.5 * (mu.pow(2) + var - 1.0 - var.log())
        if free_bits > 0.0:
            kl_per_dim = kl_per_dim.clamp(min=float(free_bits))
    else:
        raise ValueError(
            f"Unknown kl_mode {kl_mode!r}; expected 'auto', 'standard', "
            f"'flat', or 'partial'."
        )
    return kl_per_dim.sum(dim=-1).mean()


def iso_loss(
    mu: torch.Tensor,
    edge_index: torch.Tensor,
    tilde_w: torch.Tensor,
    latent_distance_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Direct latent-isometry loss against precomputed spectral targets.

    L_iso = mean over edges of ( d_Z(mu_i, mu_j) - tilde_w_{ij} )^2

    where tilde_w_{ij} is the static, precomputed spectral distance from
    the data Laplacian eigenpairs -- by default the Varadhan heat-kernel
    distance sqrt(-4t * log K_t(i,j)) (see :func:`compute_varadhan_edge_distances`),
    which converges to d^M(x_i, x_j) as t->0 and n->inf (Varadhan 1967).
    No decoder, no JVP; the target is decoder-independent and fixed for
    the entire run.

    The latent distance d_Z is the Euclidean norm by default, which is
    the correct choice when the latent space is contractible R^d. For
    architecturally-compactified latents (e.g., S^1 x S^1 in the
    topology-matched regime of Cor. cor:topo_matched), pass a
    ``latent_distance_fn`` that computes the wrapped/intrinsic geodesic
    distance on the latent manifold; the gradient then flows through
    that wrapped distance to the encoder means.

    Parameters
    ----------
    mu : (N, d)
        Encoder posterior means.
    edge_index : (2, E) long
        Directed edges (src, dst) into ``mu``.
    tilde_w : (E,)
        Precomputed Varadhan / biharmonic distances per edge.
    latent_distance_fn : callable, optional
        Function (z_src, z_dst) -> (E,) tensor of latent distances. If
        None, the Euclidean norm is used (the contractible-latent
        default).

    Returns
    -------
    Scalar loss tensor.
    """
    src, dst = edge_index[0], edge_index[1]
    if latent_distance_fn is None:
        pred = (mu[dst] - mu[src]).norm(dim=-1)
    else:
        pred = latent_distance_fn(mu[src], mu[dst])
    target = tilde_w.detach().to(pred.device).to(pred.dtype)
    return (pred - target).pow(2).mean()


@torch.no_grad()
def calibrate_edge_decoder_scale(
    edge_decoder: nn.Module,
    mu: torch.Tensor,
    edge_index: torch.Tensor,
    tilde_w: torch.Tensor,
    latent_distance_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> float:
    """Closed-form post-hoc calibration of the scalar edge decoder's scale w.

    Finds w* = argmin sum( softplus(w) * d_Z(mu_i, mu_j) - tilde_w )^2 over E*.
    The OLS solution is:

        softplus(w*) = sum(tilde_w * d_Z) / sum(d_Z^2)

    This is called ONCE after the encoder has converged via L_iso training.
    Because L_iso drives d_Z(mu_i, mu_j) -> tilde_w, the calibrated value is
    softplus(w*) ~= 1 at convergence.

    Parameters
    ----------
    edge_decoder : ScalarEdgeDecoder (has .set_scale_from_value method)
    mu           : (N, d) encoder posterior means
    edge_index   : (2, E) directed edges
    tilde_w      : (E,)  chord-arc rescaled spectral targets
    latent_distance_fn : callable, optional
        Latent distance d_Z; Euclidean by default. For compact latents
        (S^1 x S^1 etc.) pass a wrapping-aware callable.

    Returns
    -------
    Calibrated softplus(w*) value.
    """
    if not hasattr(edge_decoder, "set_scale_from_value"):
        return float("nan")
    src, dst = edge_index[0], edge_index[1]
    if src.numel() == 0:
        return float("nan")
    if latent_distance_fn is None:
        norms = (mu[dst] - mu[src]).norm(dim=-1)          # (E,)
    else:
        norms = latent_distance_fn(mu[src], mu[dst])
    tw    = tilde_w.detach().to(norms.device).to(norms.dtype)
    numerator   = float((tw * norms).sum().item())
    denominator = float((norms * norms).sum().item())
    if denominator < 1e-12:
        return float("nan")
    scale = numerator / denominator
    edge_decoder.set_scale_from_value(scale)
    return scale


@torch.no_grad()
def compute_delta_iso(
    mu: torch.Tensor,
    edge_index: torch.Tensor,
    tilde_w: torch.Tensor,
    reduction: str = "max",
    latent_distance_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> float:
    """Sup-norm latent-isometry residual: max | d_Z(mu_i, mu_j) - tilde_w |.

    The certificate scalar of C1' in the iso-architecture (cf. main.tex
    Definition def:cert). Requires no JVP and no decoder evaluation.
    For compact latents pass ``latent_distance_fn`` so the residual
    measures the wrapped/intrinsic distance rather than the Euclidean
    norm of raw angular coordinates.
    """
    src, dst = edge_index[0], edge_index[1]
    if edge_index.numel() == 0 or mu.numel() == 0:
        return 0.0
    if latent_distance_fn is None:
        pred = (mu[dst] - mu[src]).norm(dim=-1)
    else:
        pred = latent_distance_fn(mu[src], mu[dst])
    target = tilde_w.detach().to(pred.device).to(pred.dtype)
    residuals = (pred - target).abs()
    if residuals.numel() == 0:
        return 0.0
    if reduction == "max":
        return float(residuals.max().item())
    if reduction == "mean":
        return float(residuals.mean().item())
    raise ValueError(f"Unknown reduction {reduction!r}; expected 'max' or 'mean'.")


@torch.no_grad()
def compute_delta_edge_scalar(
    edge_decoder: nn.Module,
    mu: torch.Tensor,
    edge_index: torch.Tensor,
    tilde_w: torch.Tensor,
    reduction: str = "max",
) -> float:
    """Sup-norm scalar edge-head residual: max | F_phi(mu_i, mu_j) - tilde_w |.

    Companion certificate scalar to :func:`compute_delta_iso`. With the
    ScalarEdgeDecoder this is exactly equal to
    | softplus(w) * d_{M_z}(mu_i, mu_j) - tilde_w |, where d_{M_z} is the
    latent_distance_fn installed on the edge decoder (Phase-1 bug B.3
    fix). For Euclidean latents the closure is None and the formula
    reduces to | softplus(w) * ||mu_i - mu_j|| - tilde_w |.
    """
    src, dst = edge_index[0], edge_index[1]
    if edge_index.numel() == 0 or mu.numel() == 0:
        return 0.0
    z_src = mu[src]
    z_dst = mu[dst]
    pred = edge_decoder(z_src, z_dst)
    target = tilde_w.detach().to(pred.device).to(pred.dtype)
    residuals = (pred - target).abs()
    if residuals.numel() == 0:
        return 0.0
    if reduction == "max":
        return float(residuals.max().item())
    if reduction == "mean":
        return float(residuals.mean().item())
    raise ValueError(f"Unknown reduction {reduction!r}; expected 'max' or 'mean'.")


# --------------------------------------------------------------------------- IsoVAELoss

class IsoVAELoss(nn.Module):
    """Loss bundle for the iso-architecture.

    Effective objective:

        L = L_rec / s_rec
            + beta(t) * L_KL / s_kl
            + gamma(t) * L_iso / s_iso

    where

        L_iso = mean over E* of ( d_{M_z}(mu_i, mu_j) - tilde_w )^2

    Gradient flow:
        - L_rec  : encoder (via reparam), decoder
        - L_KL   : encoder (sigma only when kl_mode='partial')
        - L_iso  : encoder means ONLY -- no gradient to edge decoder scale w

    The scalar edge decoder weight w is NOT trained here.  It is calibrated
    once as a closed-form post-processing step (see
    :func:`calibrate_edge_decoder_scale`) after the encoder has converged.
    This strict separation prevents the compensating-mechanism failure mode
    where w shrinks to absorb encoder scale drift, which was observed in the
    pre-R4 ``iso_edge_loss`` design (now deleted).

    Parameters
    ----------
    beta : float
        KL coefficient. With kl_mode='partial' (default) this only
        regularises sigma^2 -- no gradient on mu from the KL term.
    gamma_init : float
        Initial geometric weight; trainer sets gamma(t) via :meth:`set_gamma`.
    kl_mode : str
        KL form: 'partial' (default, recommended -- zero gradient on mu
        so geometry is shaped exclusively by L_iso and L_rec), 'flat'
        (entropy-only, no stable sigma fixed point -- avoid for the
        Euclidean-Gaussian instantiation), 'standard'.
    use_initial_scale_norm : bool
        Divide each term by its Phase-2-entry value (fixed after first call
        to :meth:`init_scale_from_batch`).
    scale_eps : float
        Minimum scale denominator.
    latent_distance_fn : callable, optional
        Latent distance d_{M_z}(z_src, z_dst). When None the Euclidean
        norm is used (the contractible-latent default of Theorem
        thm:encoder_isometry); compact latents under
        Cor. cor:topo_matched require a wrapping-aware callable.
    free_bits : float
        Free-bits regularisation threshold (Kingma et al. 2016). Only
        active when kl_mode='standard'; cf. node_kl_loss.
    """

    def __init__(
        self,
        beta: float = 0.01,
        gamma_init: float = 0.0,
        kl_mode: str = "partial",
        use_initial_scale_norm: bool = True,
        scale_eps: float = 1e-6,
        latent_distance_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        free_bits: float = 0.0,
        manifold: Optional[nn.Module] = None,
        likelihood: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.gamma_t = float(gamma_init)
        self.kl_mode = str(kl_mode)
        self.use_initial_scale_norm = bool(use_initial_scale_norm)
        self.scale_eps = float(scale_eps)
        # Phase-1 bug B.6 fix: previously hard-coded to 0.0 in forward,
        # which made TrainingConfig.free_bits a dead flag. Now plumbed
        # through so users can request free-bits regularisation on
        # bounded latents (e.g. d=2 on S^2). Effective only when
        # kl_mode='standard' per node_kl_loss's contract.
        self.free_bits = float(free_bits)
        # Optional latent distance function for compact topologies (e.g.,
        # S^1 x S^1 wrapped distance). When None the Euclidean norm is
        # used; this is the contractible-latent default of Theorem
        # thm:encoder_isometry. Compact latents under
        # Cor. cor:topo_matched require a wrapping-aware callable.
        # NOTE (Phase 2): when ``manifold`` is supplied, the iso loss
        # uses ``manifold.distance`` directly and ``latent_distance_fn``
        # is ignored. This keeps backward compat with Phase-1 callers
        # while giving the unified RiemannianVAE a single source of
        # truth for the latent distance.
        self.latent_distance_fn = latent_distance_fn
        # Phase-2 plug-ins (op47C C.2). When supplied, IsoVAELoss
        # routes the KL through ``manifold.kl_to_prior`` and the
        # reconstruction term through ``-likelihood.log_prob`` instead
        # of the Phase-1 fallbacks (``node_kl_loss`` / MSE). When None,
        # behaviour is byte-for-byte identical to Phase 1, which is
        # what the trainer's regression test checks.
        self.manifold = manifold
        self.likelihood = likelihood

        self.register_buffer("_scale_rec",      torch.tensor(1.0))
        self.register_buffer("_scale_kl",       torch.tensor(1.0))
        self.register_buffer("_scale_iso_edge", torch.tensor(1.0))
        self.register_buffer("_scale_initialised", torch.tensor(0.0))

    # --- knobs callable by the trainer --------------------------------

    def set_gamma(self, value: float) -> None:
        """Update the geometric weight gamma(t)."""
        self.gamma_t = float(value)

    def set_beta(self, value: float) -> None:
        """Update the KL coefficient beta(t)."""
        self.beta = float(value)

    def init_scale_from_batch(
        self,
        L_rec: float,
        L_kl: float,
        L_iso_edge: float,
        # legacy kwargs accepted silently for backward compat with
        # callers carried over from the pre-R4 code.
        L_iso: float = 0.0,
        L_edge: float = 0.0,
    ) -> None:
        """Record Phase-2-entry raw losses as fixed normalisation denominators.

        Called ONCE before Phase 2 begins. Subsequent calls are no-ops.
        """
        eps = self.scale_eps
        if float(self._scale_initialised.item()) < 0.5:
            self._scale_rec.fill_(max(abs(L_rec), eps))
            self._scale_kl.fill_(max(abs(L_kl), eps))
            self._scale_iso_edge.fill_(max(abs(L_iso_edge), eps))
            self._scale_initialised.fill_(1.0)

    # --- forward ------------------------------------------------------

    def forward(
        self,
        outputs: dict,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_decoder: nn.Module | None = None,
        tilde_w: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the total iso loss.

        Parameters
        ----------
        outputs : dict
            Model forward output. Must contain 'mu', 'var', 'x_hat',
            and (when ``self.likelihood`` is supplied)
            'likelihood_params'.
        x : (N, G)
            Target features for reconstruction.
        edge_index : (2, E) long, optional
            Directed edges. Geometric term is skipped when None/empty.
        edge_decoder : nn.Module, optional
            Scalar edge head. Accepted for backward-compatible call
            signature; the iso loss does not pass gradient through it
            (the edge head's scale is calibrated post hoc).
        tilde_w : (E,), optional
            Chord-arc-rescaled spectral targets per edge.
        """
        # Reconstruction term. Phase-2 plug-in path: if a likelihood
        # is supplied AND outputs carry the parsed params, use the
        # likelihood's log_prob. Phase-1 fallback: MSE on x_hat.
        if self.likelihood is not None and "likelihood_params" in outputs:
            log_p = self.likelihood.log_prob(x, outputs["likelihood_params"])
            # Sum over feature axis (the likelihood returns per-feature
            # log-prob), then mean over batch.
            if log_p.dim() > 1:
                log_p = log_p.sum(dim=-1)
            L_rec = -log_p.mean()
        else:
            L_rec = node_reconstruction_loss(outputs["x_hat"], x)

        # KL term. Phase-2 plug-in path: if a manifold is supplied,
        # delegate to ``manifold.kl_to_prior``. Phase-1 fallback:
        # ``node_kl_loss`` with the configured kl_mode / free_bits.
        if self.manifold is not None:
            L_kl = self.manifold.kl_to_prior(
                outputs["mu"], outputs["var"],
                kl_mode=self.kl_mode if self.kl_mode != "auto" else None,
                free_bits=self.free_bits,
            )
        else:
            L_kl = node_kl_loss(
                outputs["mu"], outputs["var"],
                kl_mode=self.kl_mode,
                free_bits=self.free_bits,
            )

        # Geometric loss: gradient flows to encoder means ONLY.
        # No edge decoder involvement during training. The latent
        # distance function used here MUST match the one installed on
        # the edge head's ``ScalarEdgeDecoder`` (Phase-1 bug B.3); the
        # trainer guarantees this when both come from the same
        # manifold. We prefer ``manifold.distance`` over the legacy
        # ``latent_distance_fn`` when the manifold is available.
        L_geo_t: Optional[torch.Tensor] = None
        if (
            edge_index is not None
            and tilde_w is not None
            and edge_index.numel() > 0
        ):
            distance_fn = (
                self.manifold.distance if self.manifold is not None
                else self.latent_distance_fn
            )
            L_geo_t = iso_loss(
                mu=outputs["mu"],
                edge_index=edge_index,
                tilde_w=tilde_w,
                latent_distance_fn=distance_fn,
            )

        diagnostics: dict = {
            "L_rec_raw":      float(L_rec.detach().item()),
            "L_KL_raw":       float(L_kl.detach().item()),
            "L_iso_raw":      float(L_geo_t.detach().item()) if L_geo_t is not None else 0.0,
            # legacy aliases for callers that keyed off the pre-R4 names
            "L_iso_edge_raw": float(L_geo_t.detach().item()) if L_geo_t is not None else 0.0,
            "L_edge_raw":     0.0,
        }

        eps = self.scale_eps
        if self.use_initial_scale_norm:
            L_rec_eff = L_rec / self._scale_rec.clamp(min=eps)
            L_kl_eff  = L_kl  / self._scale_kl.clamp(min=eps)
        else:
            L_rec_eff = L_rec
            L_kl_eff  = L_kl

        total = L_rec_eff + self.beta * L_kl_eff
        diagnostics["L_rec_eff"] = float(L_rec_eff.detach().item())
        diagnostics["L_KL_eff"]  = float((self.beta * L_kl_eff).detach().item())

        if L_geo_t is not None:
            s = (
                self._scale_iso_edge.clamp(min=eps)
                if self.use_initial_scale_norm
                else torch.tensor(1.0, device=L_geo_t.device, dtype=L_geo_t.dtype)
            )
            L_geo_eff = self.gamma_t * L_geo_t / s
            total = total + L_geo_eff
            diagnostics["L_iso_eff"]      = float(L_geo_eff.detach().item())
            diagnostics["L_iso_edge_eff"] = float(L_geo_eff.detach().item())  # legacy alias
            diagnostics["gamma_t"]        = self.gamma_t

        if self.use_initial_scale_norm:
            diagnostics["scale_rec"]      = float(self._scale_rec.item())
            diagnostics["scale_kl"]       = float(self._scale_kl.item())
            diagnostics["scale_iso"]      = float(self._scale_iso_edge.item())
            diagnostics["scale_iso_edge"] = float(self._scale_iso_edge.item())  # alias
            diagnostics["scale_edge"]     = float(self._scale_iso_edge.item())  # alias

        diagnostics["beta_t"]  = float(self.beta)
        diagnostics["L_total"] = float(total.detach().item())
        return total, diagnostics
