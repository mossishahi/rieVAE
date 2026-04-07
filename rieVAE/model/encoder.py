"""Node encoder: maps observed features to variational latent codes.

Architecture: feed-forward MLP with SiLU activations (C^inf smooth,
required for valid JVP computation in the geometry module).
No graph structure is used -- encoding is purely point-wise.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """

    def __init__(
        self,
        dim_in: int,
        dim_latent: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.05,
        var_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_latent = dim_latent
        self.var_eps = var_eps

        dims = (dim_in,) + hidden_dims
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(d_in, d_out), nn.SiLU()])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)

        self.mu_head = nn.Linear(hidden_dims[-1], dim_latent)
        self.logvar_head = nn.Linear(hidden_dims[-1], dim_latent)

        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of feature vectors.

        Parameters
        ----------
        x : (N, G)

        Returns
        -------
        mu : (N, d)  -- posterior means
        var : (N, d) -- posterior variances (sigma^2), bounded below by var_eps
        """
        h = self.backbone(x)
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
