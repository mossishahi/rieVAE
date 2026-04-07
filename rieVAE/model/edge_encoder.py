"""Edge encoder: maps latent differences to edge codes.

Takes Delta z_ij = mu_j - mu_i (posterior MEANS, not samples) as input.
Using means (not reparameterized samples) reduces variance in the encoder's
input and is consistent with the theory (the optimal code at fixed W is
e*_ij = W^+ l_ij where l_ij = J_f(z_i) Delta z_ij, a deterministic function
of the posterior means through the decoder Jacobian).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EdgeEncoder(nn.Module):
    """MLP edge encoder: Delta z_ij -> e_ij in R^k.

    Parameters
    ----------
    dim_latent : int
        Latent dimension d (dimension of Delta z_ij).
    dim_edge : int
        Edge code dimension k.
    hidden_dims : tuple[int, ...]
        Hidden layer widths.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        dim_latent: int,
        dim_edge: int,
        hidden_dims: tuple[int, ...] = (128,),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.dim_latent = dim_latent
        self.dim_edge = dim_edge

        dims = (dim_latent,) + hidden_dims
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(d_in, d_out), nn.SiLU()])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_dims[-1], dim_edge)
        self.logvar_head = nn.Linear(hidden_dims[-1], dim_edge)

        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(
        self, delta_z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode latent differences into variational edge codes.

        Parameters
        ----------
        delta_z : (E, d) -- mu_j - mu_i for each directed edge (i -> j)

        Returns
        -------
        mu_e : (E, k)  -- posterior edge code means
        var_e : (E, k) -- posterior edge code variances
        """
        h = self.backbone(delta_z)
        mu_e = self.mu_head(h)
        var_e = torch.nn.functional.softplus(self.logvar_head(h)) + 1e-5
        return mu_e, var_e

    @staticmethod
    def reparameterize(
        mu_e: torch.Tensor, var_e: torch.Tensor
    ) -> torch.Tensor:
        """Sample z_edge ~ N(mu_e, diag(var_e)) via reparameterization."""
        std = var_e.sqrt()
        return mu_e + std * torch.randn_like(std)

    @staticmethod
    def kl_divergence(
        mu_e: torch.Tensor, var_e: torch.Tensor
    ) -> torch.Tensor:
        """KL[ N(mu_e, diag(var_e)) || N(0, I) ] per edge, summed over dims.

        Returns
        -------
        kl : (E,) -- per-edge KL values
        """
        return 0.5 * (mu_e.pow(2) + var_e - 1.0 - var_e.log()).sum(dim=-1)
