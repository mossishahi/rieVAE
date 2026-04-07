"""Standard (non-Riemannian) VAE trainer: baseline for isometry comparison.

Uses the same SCRVAE architecture (encoder + decoder) but trains WITHOUT:
  - The Riemannian edge loss (lambda_riem = 0)
  - The edge encoder / edge decoder (W is never updated)
  - The self-consistent graph update (single fixed Euclidean KNN graph)

This isolates the effect of Riemannian training: both models have the same
encoder/decoder capacity; the only difference is the training objective and
the graph construction.

After training, isometry is evaluated using the decoder's pullback metric
  d_R(z_i, z_j) = ||J_f(z_i)(z_j - z_i)||
for BOTH the baseline and SCR-VAE. This is the fair comparison: we assess
whether Riemannian training (our method) produces a more isometric decoder
than reconstruction-only training (the baseline).
"""
from __future__ import annotations

import dataclasses
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from rieVAE.model.scrvae import SCRVAE
from rieVAE.model.encoder import NodeEncoder
from rieVAE.train.loss import node_reconstruction_loss, node_kl_loss
from rieVAE.geometry.graph import euclidean_knn_graph


@dataclasses.dataclass
class VanillaConfig:
    """Configuration for baseline VAE training."""

    k_neighbors: int = 8
    n_epochs: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    lr_patience: int = 20
    lr_factor: float = 0.5
    lr_min: float = 1e-5
    grad_clip: float = 1.0
    beta_node_kl: float = 1e-2
    device: str = "cpu"


class VanillaVAETrainer:
    """Trains only the node encoder and decoder, with no Riemannian components.

    This is the baseline: same architecture as SCR-VAE, same number of epochs
    of training, but trained purely for node reconstruction and KL divergence.

    After training, the decoder's pullback metric is used to evaluate isometry,
    providing a fair comparison against the SCR-VAE (which was trained to
    explicitly optimize this metric).

    Parameters
    ----------
    model : SCRVAE
        The same model class as SCR-VAE. Only node_encoder and node_decoder
        are trained; edge_encoder and edge_decoder are ignored.
    config : VanillaConfig
    """

    def __init__(self, model: SCRVAE, config: VanillaConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.device = torch.device(config.device)

        trainable_params = list(model.node_encoder.parameters()) + \
                           list(model.node_decoder.parameters())
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=config.lr_patience,
            factor=config.lr_factor,
            min_lr=config.lr_min,
        )
        self.history: list[dict[str, float]] = []

    def fit(self, x: torch.Tensor) -> None:
        """Train the standard VAE on all data.

        Parameters
        ----------
        x : (N, G)
        """
        x = x.to(self.device)

        print("=" * 60)
        print(f"  Vanilla VAE (Baseline)  N={x.shape[0]}  G={x.shape[1]}")
        print(f"  Epochs: {self.config.n_epochs}")
        print("=" * 60)

        t0 = time.time()
        log_every = max(1, self.config.n_epochs // 10)

        for epoch in range(1, self.config.n_epochs + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            mu, var = self.model.encode_nodes(x)
            z = NodeEncoder.reparameterize(mu, var)
            x_hat = self.model.decode_nodes(z)

            l_recon = node_reconstruction_loss(x_hat, x)
            l_kl = node_kl_loss(mu, var)
            loss = l_recon + self.config.beta_node_kl * l_kl

            loss.backward()
            if self.config.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(
                    list(self.model.node_encoder.parameters()) +
                    list(self.model.node_decoder.parameters()),
                    self.config.grad_clip,
                )
            self.optimizer.step()
            self.scheduler.step(loss.detach())

            self.history.append({
                "epoch": float(epoch),
                "total": float(loss.detach()),
                "node_recon": float(l_recon.detach()),
                "node_kl": float(l_kl.detach()),
            })

            if epoch % log_every == 0 or epoch == self.config.n_epochs:
                print(
                    f"  epoch {epoch:5d}/{self.config.n_epochs}  "
                    f"total={loss.item():.4f}  "
                    f"recon={l_recon.item():.4f}  "
                    f"kl={l_kl.item():.4f}"
                )

        elapsed = time.time() - t0
        print(f"\nTraining complete in {elapsed:.1f}s")
