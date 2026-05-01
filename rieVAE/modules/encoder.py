"""Node encoder: maps observed features to variational latent codes.

Architecture: feed-forward MLP with a configurable C^inf-smooth
activation (SiLU by default; see ``rieVAE.modules.activations``).
The mu and log-variance heads are two separate linear layers on top
of the shared backbone, so they can be initialised, regularised,
or frozen independently.

The encoder optionally accepts a positional-encoding feature
``pe_feat`` (see :mod:`rieVAE.geometry.positional_encoding`).
When ``use_pe=True`` the backbone hidden state h is augmented with
a spectral-norm-constrained linear projection of pe_feat, gated by
the scalar ``alpha_pe`` that is set externally by the trainer
(0 -> alpha_max over the warm-up window). Setting ``use_pe=False``
restores the exact pre-PE architecture byte-for-byte, so ablations
toggle a single constructor flag without any other change.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rieVAE.modules.activations import make_activation


class ScaledSpectralLinear(nn.Module):
    """Linear layer whose weight matrix has bounded operator norm.

    Wraps a bias-free Linear with PyTorch's spectral_norm parametrisation
    (which forces ``sigma_max(W_normalised) = 1``), then multiplies the
    output by ``max_lip`` to permit any cap >= 1 on the layer's
    Lipschitz constant in its input.  The bias is a free parameter
    (translation), so the layer is ``max_lip``-Lipschitz in x without
    constraining where the output is centred.

    Used in NodeEncoder when ``encoder_spectral_norm=True`` to bound
    the encoder's global Lipschitz constant; this prevents the
    memorisation regime where the encoder maps nearby data points to
    far-apart latents (which requires unbounded sigma_max).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        max_lip: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        raw = nn.Linear(in_dim, out_dim, bias=False)
        self._lin = nn.utils.parametrizations.spectral_norm(raw)
        self.max_lip = float(max_lip)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._lin(x)
        if self.max_lip != 1.0:
            out = out * self.max_lip
        if self.bias is not None:
            out = out + self.bias
        return out


class NodeEncoder(nn.Module):
    """Point-wise variational encoder x_i -> (mu_i, sigma_i^2).

    Parameters
    ----------
    dim_in : int
        Input feature dimension G.
    dim_latent : int
        Latent dimension d.
    hidden_dims : tuple[int, ...]
        Widths of hidden layers. Minimum one hidden layer recommended.
    dropout : float
        Dropout probability (applied after each hidden activation).
    var_eps : float
        Minimum posterior variance (numerical stability).
    use_pe : bool
        If True, add an optional positional-encoding branch. The
        branch is a single spectral-norm-constrained linear layer
        mapping pe_feat (N, pe_dim) -> hidden_dims[-1], added to the
        backbone hidden state before the mu/logvar heads. Default
        False (no extra parameters, identical to the pre-PE model).
    pe_dim : int, optional
        Dimensionality of the input PE vector. Required iff use_pe.
    """

    def __init__(
        self,
        dim_in: int,
        dim_latent: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.05,
        var_eps: float = 1e-5,
        activation: str = "silu",
        use_pe: bool = False,
        pe_dim: Optional[int] = None,
        encoder_spectral_norm: bool = False,
        encoder_spectral_norm_max_lip: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_latent = dim_latent
        self.var_eps = var_eps
        self.activation = activation
        self.encoder_spectral_norm = bool(encoder_spectral_norm)
        self.encoder_spectral_norm_max_lip = float(encoder_spectral_norm_max_lip)

        # When encoder_spectral_norm is True, every Linear in the backbone
        # and the mu_head is replaced with ScaledSpectralLinear, capping
        # the encoder's global Lipschitz constant by
        # (max_lip ** num_linear_layers) * Lip(activation) ** num_activations.
        # This forbids the memorisation regime that requires sigma_max(W)
        # to diverge.  The logvar_head is intentionally NOT wrapped so the
        # zero-initialised log-variance head is preserved (spectral_norm
        # would require nonzero init for power iteration).
        def _lin(d_in: int, d_out: int) -> nn.Module:
            if self.encoder_spectral_norm:
                return ScaledSpectralLinear(
                    d_in, d_out,
                    max_lip=self.encoder_spectral_norm_max_lip,
                    bias=True,
                )
            return nn.Linear(d_in, d_out)

        dims = (dim_in,) + hidden_dims
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.extend([_lin(d_in, d_out), make_activation(activation)])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)

        # Two separate linear heads on the shared backbone (Item 3 of
        # the original design checklist). Function-equivalent to a
        # single 2d-output linear layer, but cleaner for ablations.
        self.mu_head = _lin(hidden_dims[-1], dim_latent)
        # logvar_head intentionally NOT spectral-norm wrapped (see comment above).
        self.logvar_head = nn.Linear(hidden_dims[-1], dim_latent)

        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

        # Optional PE injection.  The spectral-norm wrap bounds
        # ||W_proj||_2 <= 1, which is the hypothesis we use in the
        # |G - d| = O(r_n) corollary in the paper.  We do NOT
        # zero-initialise the linear weight: spectral_norm normalises
        # by the top singular value, and zero initial weights would
        # divide by zero.  Instead we use the default Kaiming-uniform
        # init and rely on the trainer-side ``alpha_pe(t)`` gate to
        # ramp the PE contribution from zero during warm-up.
        self.use_pe = bool(use_pe)
        if self.use_pe:
            if pe_dim is None or pe_dim < 1:
                raise ValueError(
                    "NodeEncoder(use_pe=True) requires pe_dim >= 1; "
                    f"got pe_dim={pe_dim}."
                )
            self.pe_dim = int(pe_dim)
            raw = nn.Linear(self.pe_dim, hidden_dims[-1], bias=False)
            self.pe_proj = nn.utils.parametrizations.spectral_norm(raw)
        else:
            self.pe_dim = 0
            self.pe_proj = None

    def forward(
        self,
        x: torch.Tensor,
        pe_feat: Optional[torch.Tensor] = None,
        alpha_pe: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of feature vectors.

        Parameters
        ----------
        x : (N, G)
        pe_feat : (N, pe_dim) float tensor, optional
            Per-node positional-encoding features.  Ignored when
            ``use_pe=False``. Required when ``use_pe=True``.
        alpha_pe : float
            Scalar gate multiplying the PE projection; the trainer
            ramps this from 0 to ``pe_gate_max`` over the warm-up
            window.  Ignored when ``use_pe=False``.

        Returns
        -------
        mu : (N, d)  -- posterior means
        var : (N, d) -- posterior variances (sigma^2), bounded below by var_eps
        """
        h = self.backbone(x)
        if self.use_pe and pe_feat is not None:
            # PE branch active.  Shape check (cheap) protects against
            # silent dimensionality mismatches.
            if pe_feat.shape[-1] != self.pe_dim:
                raise ValueError(
                    f"pe_feat last dim {pe_feat.shape[-1]} != pe_dim "
                    f"{self.pe_dim}."
                )
            h = h + float(alpha_pe) * self.pe_proj(pe_feat)
        # NOTE: use_pe=True with pe_feat=None is a legal "diagnostic
        # fallback" path used by some evaluators (properness, isometry
        # reporting) that do not have access to the precomputed PE
        # artefacts.  It is equivalent to alpha_pe = 0 at this step;
        # the aux head on the decoder side is still trained from the
        # training loop.  Callers that need PE-augmented outputs at
        # evaluation time MUST pass pe_feat explicitly.
        mu = self.mu_head(h)
        var = F.softplus(self.logvar_head(h)) + self.var_eps
        return mu, var

    @staticmethod
    def reparameterize(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Draw z ~ N(mu, diag(var)) via the reparameterization trick."""
        if not mu.requires_grad and not var.requires_grad:
            return mu
        std = var.sqrt()
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def kl_divergence(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """KL[ N(mu, diag(var)) || N(0, I) ] per sample, summed over latent dims.

        Returns
        -------
        kl : (N,) -- per-sample KL values
        """
        return 0.5 * (mu.pow(2) + var - 1.0 - var.log()).sum(dim=-1)
