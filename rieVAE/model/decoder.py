"""Node decoder and linear edge decoder.

NodeDecoder: z -> x_hat  (must be C^2 smooth for valid JVP / Riemannian log maps).
EdgeDecoder: e -> W e    (linear, NO bias -- required by the Eckart-Young theorem).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NodeDecoder(nn.Module):
    """Point-wise node decoder z_i -> x_hat_i.

    C^2 smoothness is enforced by using SiLU activations throughout.
    The decoder must support Jacobian-vector products via torch.func.jvp;
    avoid any non-differentiable operations (argmax, rounding, etc.).

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
    """

    def __init__(
        self,
        dim_latent: int,
        dim_out: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim_latent = dim_latent
        self.dim_out = dim_out

        dims = (dim_latent,) + hidden_dims
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(d_in, d_out), nn.SiLU()])
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


class EdgeDecoder(nn.Module):
    """Linear edge decoder with Stiefel manifold constraint: e_ij -> W e_ij.

    W is parameterized on the Stiefel manifold St(k, G) = {W : W^T W = I_k}.

    WHY STIEFEL: Proposition 2 of the theory paper requires W^T W = I_k
    (orthonormal principal tangent frame). The naive approach (unconstrained W +
    decorrelation loss on codes) gives W^T W = Lambda_k^{-1} (inverse eigenvalues
    of M), NOT I_k, because Cov(mu_e) = I_k implies W^T M W = (W^T W)^2, and
    when col(W) = top-k(M) the solution is W = U_k Lambda_k^{-1/2} Q.

    The correct fix: enforce W^T W = I_k DIRECTLY via the Stiefel manifold.
    With this constraint, the pseudoinverse is W^+ = W^T (exact), giving
    e^* = W^T l_ij (exact tangent coordinates). The decorrelation loss is
    then unnecessary -- remove it when using this class.

    Implementation: after each gradient step, reproject W onto the Stiefel
    manifold using a thin QR decomposition (a retraction). The gradient step
    is computed in the ambient Euclidean space and then projected back.

    Parameters
    ----------
    dim_edge : int  (k)
    dim_out : int   (G)
    """

    def __init__(self, dim_edge: int, dim_out: int) -> None:
        super().__init__()
        self.dim_edge = dim_edge
        self.dim_out = dim_out
        weight = torch.empty(dim_out, dim_edge)
        nn.init.orthogonal_(weight)
        self._W = nn.Parameter(weight)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Predict Riemannian log maps: l_hat = W e  in R^G.

        Parameters
        ----------
        e : (E, k)

        Returns
        -------
        l_hat : (E, G)
        """
        return e @ self._W.T

    def retract_to_stiefel(self) -> None:
        """Project W back onto St(k, G) via polar decomposition (SVD-based).

        Call this after each optimizer step to maintain W^T W = I_k.
        In-place operation: modifies self._W.data.

        Uses the SVD polar retraction W <- U V^T (where W = U S V^T) which gives
        the nearest orthogonal matrix to W in Frobenius norm. This is more
        numerically stable than QR for tall matrices because it does not depend
        on the ordering of columns and avoids the sign ambiguity of QR.

        For a (G, k) matrix with G >> k, U has shape (G, k) and V has shape (k, k),
        so U V^T has shape (G, k) with orthonormal columns (U^T U = I_k enforced
        by the full SVD).
        """
        with torch.no_grad():
            U, _, Vt = torch.linalg.svd(self._W, full_matrices=False)
            self._W.data.copy_(U @ Vt)

    @property
    def weight(self) -> torch.Tensor:
        """The orthonormal frame W ∈ R^{G × k} with W^T W = I_k."""
        return self._W

    def gram_matrix(self) -> torch.Tensor:
        """W^T W ∈ R^{k × k}.

        Should be I_k when the Stiefel constraint is active.
        Deviations from I_k measure numerical drift between retractions.
        """
        W = self._W
        return W.T @ W
