"""Node decoder.

The former linear Stiefel-constrained edge decoder W has been replaced
by the joint edge decoder F_phi (see rieVAE/model/edge.py). This
module now contains only the node decoder f_theta used to reconstruct
ambient features from latent codes. Activation is configurable
through ``rieVAE.model._activations`` (default SiLU; ReLU is
intentionally not exposed because Lemma 1's curvature remainder
requires C^3).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rieVAE.model._activations import make_activation


class NodeDecoder(nn.Module):
    """Point-wise node decoder z_i -> x_hat_i.

    C^2 smoothness comes from the configurable activation
    (``activation`` kwarg, default ``'silu'``). The decoder must
    support Jacobian-vector products via torch.func.jvp; avoid any
    non-differentiable operations (argmax, rounding, etc.).

    Parameters
    ----------
    dim_latent : int
        Latent dimension d.
    dim_out : int
        Output feature dimension G.
    hidden_dims : tuple[int, ...]
        Hidden layer widths.
    dropout : float
        Dropout probability (set to 0 for JVP stability during analysis).
    activation : str
        One of ``{'silu', 'gelu', 'tanh', 'softplus'}``.
    """

    def __init__(
        self,
        dim_latent: int,
        dim_out: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.dim_latent = dim_latent
        self.dim_out = dim_out
        self.activation = activation

        dims = (dim_latent,) + hidden_dims
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(d_in, d_out), make_activation(activation)])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a batch of latent vectors.

        Parameters
        ----------
        z : (N, d)

        Returns
        -------
        x_hat : (N, G)
        """
        return self.net(z)

    def decode_single(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a single latent vector (shape (d,) -> (G,)).

        Used for JVP computation, which requires scalar-sample functions.
        Dropout is automatically disabled because eval() is set before JVP calls.
        """
        return self.net(z.unsqueeze(0)).squeeze(0)
