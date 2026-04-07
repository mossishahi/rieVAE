"""Self-Consistent Riemannian VAE (SCR-VAE).

Implements Algorithm 1 from the theory paper:
  1. Initialize: build Euclidean KNN graph.
  2. M-step: train the VAE on the current graph.
  3. E-step: compute Riemannian distances, rebuild KNN graph.
  4. Repeat until graph topology stabilizes.

The forward pass is used during M-step training.
The graph update is handled by the Trainer (see train/trainer.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rieVAE.model.encoder import NodeEncoder
from rieVAE.model.decoder import NodeDecoder, EdgeDecoder
from rieVAE.model.edge_encoder import EdgeEncoder


class SCRVAE(nn.Module):
    """Self-Consistent Riemannian Variational Autoencoder.

    Parameters
    ----------
    dim_features : int
        Input/output feature dimension G (ambient space).
    dim_latent : int
        Node latent dimension d.
    dim_edge : int
        Edge code dimension k. Must satisfy k <= d.
    encoder_hidden : tuple[int, ...]
        Hidden widths for the node encoder.
    decoder_hidden : tuple[int, ...]
        Hidden widths for the node decoder. Use fewer layers for smoother
        Jacobians (better log map approximation quality).
    edge_hidden : tuple[int, ...]
        Hidden widths for the edge encoder.
    dropout : float
        Dropout rate for encoder and edge encoder. Set to 0 for the decoder
        (JVP requires no stochastic operations in f_theta).
    var_eps : float
        Minimum variance for numerical stability.
    """

    def __init__(
        self,
        dim_features: int,
        dim_latent: int,
        dim_edge: int,
        encoder_hidden: tuple[int, ...] = (256, 256),
        decoder_hidden: tuple[int, ...] = (256, 256),
        edge_hidden: tuple[int, ...] = (128,),
        dropout: float = 0.05,
        var_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        if dim_edge > dim_latent:
            raise ValueError(
                f"dim_edge={dim_edge} must be <= dim_latent={dim_latent}. "
                "The edge codes have lower dimension than the latent space."
            )

        self.dim_features = dim_features
        self.dim_latent = dim_latent
        self.dim_edge = dim_edge

        self.node_encoder = NodeEncoder(
            dim_in=dim_features,
            dim_latent=dim_latent,
            hidden_dims=encoder_hidden,
            dropout=dropout,
            var_eps=var_eps,
        )

        self.node_decoder = NodeDecoder(
            dim_latent=dim_latent,
            dim_out=dim_features,
            hidden_dims=decoder_hidden,
            dropout=0.0,
        )

        self.edge_encoder = EdgeEncoder(
            dim_latent=dim_latent,
            dim_edge=dim_edge,
            hidden_dims=edge_hidden,
            dropout=dropout,
        )

        self.edge_decoder = EdgeDecoder(
            dim_edge=dim_edge,
            dim_out=dim_features,
        )

    def encode_nodes(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode node features to posterior parameters.

        Parameters
        ----------
        x : (N, G)

        Returns
        -------
        mu : (N, d) -- posterior means
        var : (N, d) -- posterior variances
        """
        return self.node_encoder(x)

    def decode_nodes(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to reconstructed features.

        Parameters
        ----------
        z : (N, d)

        Returns
        -------
        x_hat : (N, G)
        """
        return self.node_decoder(z)

    def encode_edges(
        self,
        mu_node: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode directed edge differences to variational edge codes.

        Input to the edge encoder: Delta z_ij = mu_j - mu_i (posterior MEANS).
        This is the critical design choice: using means (not samples) provides
        a deterministic, lower-variance input consistent with the theory.

        Parameters
        ----------
        mu_node : (N, d) -- posterior means of all nodes
        edge_index : (2, E) -- directed edges (src, dst)

        Returns
        -------
        mu_e : (E, k) -- edge code posterior means
        var_e : (E, k) -- edge code posterior variances
        """
        src, dst = edge_index[0], edge_index[1]
        delta_z = mu_node[dst] - mu_node[src]
        return self.edge_encoder(delta_z)

    def predict_log_maps(self, mu_e: torch.Tensor) -> torch.Tensor:
        """Predict Riemannian log maps from edge codes via W.

        l_hat_ij = W e_ij   in R^G

        Parameters
        ----------
        mu_e : (E, k)

        Returns
        -------
        l_hat : (E, G)
        """
        return self.edge_decoder(mu_e)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass for one M-step training iteration.

        Parameters
        ----------
        x : (N, G) -- node features
        edge_index : (2, E) -- current graph edges

        Returns
        -------
        dict with:
          'x_hat'    : (N, G) -- reconstructed features
          'mu_node'  : (N, d) -- node posterior means
          'var_node' : (N, d) -- node posterior variances
          'z_node'   : (N, d) -- reparameterized node latents (training only)
          'mu_e'     : (E, k) -- edge code posterior means
          'var_e'    : (E, k) -- edge code posterior variances
          'l_hat'    : (E, G) -- predicted Riemannian log maps
        """
        mu_node, var_node = self.encode_nodes(x)

        if self.training:
            z_node = NodeEncoder.reparameterize(mu_node, var_node)
        else:
            z_node = mu_node

        x_hat = self.decode_nodes(z_node)

        mu_e, var_e = self.encode_edges(mu_node, edge_index)
        l_hat = self.predict_log_maps(mu_e)

        return {
            "x_hat": x_hat,
            "mu_node": mu_node,
            "var_node": var_node,
            "z_node": z_node,
            "mu_e": mu_e,
            "var_e": var_e,
            "l_hat": l_hat,
        }

    @property
    def frame_W(self) -> torch.Tensor:
        """The learned principal tangent frame W ∈ R^{G × k}."""
        return self.edge_decoder.weight

    def gram_matrix(self) -> torch.Tensor:
        """W^T W ∈ R^{k × k}. Should approach I_k at the decorrelated fixed point."""
        return self.edge_decoder.gram_matrix()

    def count_parameters(self) -> dict[str, int]:
        """Parameter counts by submodule."""
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "node_encoder": count(self.node_encoder),
            "node_decoder": count(self.node_decoder),
            "edge_encoder": count(self.edge_encoder),
            "edge_decoder": count(self.edge_decoder),
            "total": count(self),
        }
