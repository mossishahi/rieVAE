"""Loss functions for the SCR-VAE.

Three loss terms (equation (3) of the theory paper):
    L = L_node_recon + beta * L_node_KL + lambda_riem * L_Riemannian

The Riemannian loss uses STOP-GRADIENT targets v_ij computed via JVP.
The decoder f_theta receives gradient ONLY from L_node_recon and L_node_KL.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rieVAE.geometry.log_map import riemannian_log_maps_batched


def node_reconstruction_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error reconstruction loss (Gaussian likelihood).

    Parameters
    ----------
    x_hat : (N, G) -- decoder output
    x : (N, G) -- observed features

    Returns
    -------
    loss : scalar
    """
    return nn.functional.mse_loss(x_hat, x, reduction="mean")


def node_kl_loss(
    mu: torch.Tensor,
    var: torch.Tensor,
) -> torch.Tensor:
    """KL[ N(mu, diag(var)) || N(0, I) ] averaged over nodes and latent dims.

    Parameters
    ----------
    mu : (N, d)
    var : (N, d)

    Returns
    -------
    loss : scalar
    """
    kl_per_node = 0.5 * (mu.pow(2) + var - 1.0 - var.log()).sum(dim=-1)
    return kl_per_node.mean()


def riemannian_edge_loss(
    decoder: nn.Module,
    z_mu: torch.Tensor,
    edge_index: torch.Tensor,
    mu_e: torch.Tensor,
    W: torch.Tensor,
) -> torch.Tensor:
    """Riemannian log map reconstruction loss (the core geometric loss).

    Trains W and h_psi to predict Riemannian log maps:
        L_Riem = E_{(i,j) in G} || W mu_e_ij - l_ij ||^2

    where l_ij = J_f(z_i)(z_j - z_i) is computed via JVP with STOP-GRADIENT
    on f_theta (see Remark 1 in the theory paper).

    MEAN-FIELD NOTE: The loss uses mu_e (the posterior MEAN of edge codes),
    not reparameterized samples from q_psi(e_ij | Delta z_ij). This is the
    standard mean-field variational approximation. The paper's notation
    E_{e_ij ~ q}[...] is technically more general, but the mean-field form
    mu_e @ W.T is used in both the forward pass (scrvae.py) and here, and is
    internally consistent with the edge KL that regularizes the same mu_e.
    Using the mean rather than samples reduces gradient variance and is standard
    practice for structured variational inference.

    Parameters
    ----------
    decoder : nn.Module
        Node decoder (must be in eval mode for JVP, temporarily switched).
    z_mu : (N, d)
        Posterior means (NOT reparameterized samples).
    edge_index : (2, E)
        Directed edge indices (src, dst).
    mu_e : (E, k)
        Edge code posterior means from the edge encoder.
    W : (G, k)
        Linear edge decoder weight matrix.

    Returns
    -------
    loss : scalar
    """
    src, dst = edge_index[0], edge_index[1]

    z_src = z_mu[src]
    delta_z = z_mu[dst] - z_mu[src]

    log_maps = riemannian_log_maps_batched(decoder, z_src, delta_z)

    predicted = mu_e @ W.T
    return nn.functional.mse_loss(predicted, log_maps, reduction="mean")


def edge_kl_loss(
    mu_e: torch.Tensor,
    var_e: torch.Tensor,
) -> torch.Tensor:
    """KL[ N(mu_e, diag(var_e)) || N(0, I) ] averaged over edges and dims.

    Parameters
    ----------
    mu_e : (E, k)
    var_e : (E, k)

    Returns
    -------
    loss : scalar
    """
    kl_per_edge = 0.5 * (mu_e.pow(2) + var_e - 1.0 - var_e.log()).sum(dim=-1)
    return kl_per_edge.mean()


def decorrelation_loss(mu_e: torch.Tensor) -> torch.Tensor:
    """Penalty on the empirical covariance of edge posterior means: ||Cov(mu_e) - I_k||_F^2.

    Encourages the edge codes to be uncorrelated with unit variance.
    Applied to the posterior MEANS (not samples), consistent with the
    PCA whitening frame proposition (Proposition 2 of the theory paper).

    Parameters
    ----------
    mu_e : (E, k)

    Returns
    -------
    loss : scalar
    """
    n = mu_e.shape[0]
    if n < 2:
        return mu_e.new_tensor(0.0)

    centered = mu_e - mu_e.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (n - 1)
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    return (cov - eye).pow(2).mean()


class SCRVAELoss(nn.Module):
    """Combined loss for one training step of the SCR-VAE.

    Parameters
    ----------
    beta_node_kl : float
        Weight for the node KL divergence.
    lambda_riem : float
        Weight for the Riemannian log map reconstruction loss.
    beta_edge_kl : float
        Weight for the edge KL divergence (variational rank selection).
    lambda_decorr : float
        Weight for the decorrelation loss (PCA whitening frame).
    """

    def __init__(
        self,
        beta_node_kl: float = 1e-2,
        lambda_riem: float = 1.0,
        beta_edge_kl: float = 1e-3,
        lambda_decorr: float = 1e-2,
    ) -> None:
        super().__init__()
        self.beta_node_kl = beta_node_kl
        self.lambda_riem = lambda_riem
        self.beta_edge_kl = beta_edge_kl
        self.lambda_decorr = lambda_decorr

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu_node: torch.Tensor,
        var_node: torch.Tensor,
        mu_e: torch.Tensor,
        var_e: torch.Tensor,
        decoder: nn.Module,
        z_mu: torch.Tensor,
        edge_index: torch.Tensor,
        W: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all loss components and the total weighted loss.

        Returns
        -------
        dict with keys: 'total', 'node_recon', 'node_kl',
                        'riemannian', 'edge_kl', 'decorr'
        """
        l_recon = node_reconstruction_loss(x_hat, x)
        l_node_kl = node_kl_loss(mu_node, var_node)
        l_riem = riemannian_edge_loss(decoder, z_mu, edge_index, mu_e, W)
        l_edge_kl = edge_kl_loss(mu_e, var_e)

        # Note: decorrelation loss is intentionally OMITTED.
        # With the Stiefel manifold parameterization (W^T W = I_k enforced
        # by QR retraction), the decorrelation loss Cov(mu_e) = I_k would
        # target the WRONG quantity: it would require W^T M W = I_k, but the
        # correct optimum has W^T M W = Lambda_k (eigenvalues of M restricted
        # to col(W)). The KL term on edge codes handles variance calibration.

        total = (
            l_recon
            + self.beta_node_kl * l_node_kl
            + self.lambda_riem * l_riem
            + self.beta_edge_kl * l_edge_kl
        )

        return {
            "total": total,
            "node_recon": l_recon.detach(),
            "node_kl": l_node_kl.detach(),
            "riemannian": l_riem.detach(),
            "edge_kl": l_edge_kl.detach(),
        }
