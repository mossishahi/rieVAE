"""Self-consistent training loop for the SCR-VAE.

Implements Algorithm 1 from the theory paper:
  - M-step: train VAE on fixed graph for n_mstep_epochs epochs.
  - E-step: rebuild Riemannian KNN graph from the current decoder.
  - Repeat for n_iterations (no early stopping -- let it run fully).

Diagnostics logged every iteration:
  - Gram ||W^TW - I||_F  (Stiefel constraint quality)
  - Riemannian distance stats (mean, std, min, max)
  - Gradient norms per parameter group
  - GPU memory usage
  - Graph change fraction
"""
from __future__ import annotations

import dataclasses
import time
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from rieVAE.model.scrvae import SCRVAE
from rieVAE.train.loss import SCRVAELoss
from rieVAE.geometry.graph import (
    euclidean_knn_graph,
    riemannian_knn_graph,
    graph_changed,
    graph_change_fraction,
)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _gpu_mem_mb() -> float:
    """Return current GPU allocated memory in MB (0 if no CUDA)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def _gpu_reserved_mb() -> float:
    """Return current GPU reserved memory in MB (0 if no CUDA)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024**2
    return 0.0


def _grad_norm(params) -> float:
    """Total gradient L2 norm across a list of parameters."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return total ** 0.5


@dataclasses.dataclass
class TrainerConfig:
    """Configuration for the self-consistent training loop."""

    k_neighbors: int = 8
    n_iterations: int = 50
    n_mstep_epochs: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    lr_patience: int = 20
    lr_factor: float = 0.5
    lr_min: float = 1e-5
    grad_clip: float = 1.0
    # Allow the E-step to consider additional random candidate pairs beyond the
    # current graph edges. This prevents permanently losing edges that were
    # spuriously dropped due to linearization artifacts in early iterations.
    #
    # Default is 0 to reproduce the published experiments (sphere_v2 and
    # torus_clifford runs). For new experiments, setting this to 50 is
    # recommended: it allows rediscovery of spuriously dropped edges without
    # significantly increasing compute (50 extra JVP calls per E-step).
    n_extra_candidates: int = 0
    # Clip Riemannian distances at clip_factor * median before KNN ranking.
    # Prevents linearization artifacts on far-away pairs from causing
    # graph oscillation (2-cycle). Set to 0 to disable.
    distance_clip_factor: float = 5.0
    device: str = "cpu"

    beta_node_kl: float = 1e-2
    lambda_riem: float = 0.1
    beta_edge_kl: float = 1e-3
    lambda_decorr: float = 0.0

    # Whether to reset the optimizer (including momentum buffers) and LR
    # scheduler at the start of each self-consistency iteration.
    # When True each M-step starts fresh at learning_rate.
    # When False (old behaviour) the optimizer state accumulates across
    # iterations, causing ReduceLROnPlateau to collapse the LR after iter 1.
    reset_optimizer_each_iteration: bool = True

    # Logging
    log_interval: int = 10
    use_wandb: bool = False
    wandb_project: str = "rieVAE"
    wandb_run_name: Optional[str] = None


class SCRVAETrainer:
    """Self-consistent trainer for SCR-VAE.

    Parameters
    ----------
    model : SCRVAE
    config : TrainerConfig
    """

    def __init__(self, model: SCRVAE, config: TrainerConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.device = torch.device(config.device)

        self.loss_fn = SCRVAELoss(
            beta_node_kl=config.beta_node_kl,
            lambda_riem=config.lambda_riem,
            beta_edge_kl=config.beta_edge_kl,
            lambda_decorr=config.lambda_decorr,
        )

        self.optimizer = AdamW(
            model.parameters(),
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
        self.edge_index: Optional[torch.Tensor] = None
        self.edge_weight: Optional[torch.Tensor] = None

        self._wandb_initialized = False

    def _reset_optimizer(self) -> None:
        """Rebuild the optimizer and LR scheduler from scratch.

        Called at the start of every self-consistency iteration when
        config.reset_optimizer_each_iteration is True. This prevents
        ReduceLROnPlateau from collapsing the LR during iteration 1 and
        keeping it at the floor for all subsequent iterations.

        The model parameters themselves are NOT touched -- only the optimizer
        momentum/variance buffers and the scheduler state are reset, so the
        learned weights are preserved across iterations.
        """
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.config.lr_patience,
            factor=self.config.lr_factor,
            min_lr=self.config.lr_min,
        )

    def _init_wandb(self) -> None:
        if not self.config.use_wandb or not _WANDB_AVAILABLE:
            return
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=dataclasses.asdict(self.config),
            reinit=True,
        )
        # Define custom x-axes so the user can pick any of these in the UI:
        #   "global_epoch" -- monotonically increasing epoch count (0 -> total)
        #   "epoch_in_iter" -- epoch within the current M-step (1 -> n_mstep_epochs)
        #   "iteration"    -- self-consistency iteration index (1 -> n_iterations)
        # All three are logged as Y values AND used as the wandb step.
        wandb.define_metric("global_epoch")
        wandb.define_metric("epoch_in_iter")
        wandb.define_metric("iteration")
        wandb.define_metric("epoch/*", step_metric="global_epoch")
        wandb.define_metric("iter/*",  step_metric="iteration")
        self._wandb_initialized = True

    def _log_wandb(self, metrics: dict, step: int) -> None:
        if self._wandb_initialized:
            wandb.log(metrics, step=step)

    def _gram_error(self) -> float:
        """||W^TW - I_k||_F / k -- Stiefel constraint quality (0 = perfect)."""
        k = self.model.dim_edge
        G = self.model.gram_matrix().detach()
        eye = torch.eye(k, dtype=G.dtype, device=G.device)
        return float((G - eye).norm().item() / k)

    def _riemannian_dist_stats(self) -> dict[str, float]:
        """Stats on current edge weights w_ij = ||J_f(z_i)Dz_ij||."""
        if self.edge_weight is None:
            return {}
        w = self.edge_weight.detach().cpu()
        return {
            "riem_dist_mean": float(w.mean()),
            "riem_dist_std": float(w.std()),
            "riem_dist_min": float(w.min()),
            "riem_dist_max": float(w.max()),
        }

    def _print_header(self, N: int, G: int) -> None:
        print("=" * 70)
        print(f"  SCR-VAE Training   N={N}  G={G}")
        print(f"  d_latent={self.model.dim_latent}  d_edge={self.model.dim_edge}")
        print(f"  k_neighbors={self.config.k_neighbors}")
        print(f"  n_iterations={self.config.n_iterations}  n_mstep_epochs={self.config.n_mstep_epochs}")
        print(f"  lambda_riem={self.config.lambda_riem}  beta_node_kl={self.config.beta_node_kl}")
        print(f"  device={self.config.device}")
        if self.config.use_wandb and _WANDB_AVAILABLE:
            print(f"  wandb: project={self.config.wandb_project}  run={self.config.wandb_run_name}")
        print("=" * 70)

    def fit(self, x: torch.Tensor) -> None:
        """Run the full self-consistent training (no early stopping).

        Parameters
        ----------
        x : (N, G) -- data matrix
        """
        x = x.to(self.device)
        N, G = x.shape

        self._print_header(N, G)
        self._init_wandb()

        print(f"\n  Init GPU memory: {_gpu_mem_mb():.0f} MB alloc  "
              f"{_gpu_reserved_mb():.0f} MB reserved")

        print("\n[Init] Building initial Euclidean KNN graph...")
        self.edge_index, self.edge_weight = euclidean_knn_graph(
            x, k=self.config.k_neighbors
        )
        print(f"       Edges: {self.edge_index.shape[1]:,}")

        gram_init = self._gram_error()
        print(f"\n  [Init] Gram ||W^TW-I||/k = {gram_init:.6f}  (should be 0)")

        global_step = 0

        for iteration in range(1, self.config.n_iterations + 1):
            t0 = time.time()
            print(f"\n{'='*70}")
            print(f"  ITERATION {iteration}/{self.config.n_iterations}")
            print(f"{'='*70}")

            if self.config.reset_optimizer_each_iteration:
                self._reset_optimizer()
                print(f"  [Optimizer reset] LR restored to {self.config.learning_rate:.2e}")

            epoch_logs = self._m_step(x, iteration, global_step)
            global_step += self.config.n_mstep_epochs

            gram_err = self._gram_error()
            dist_stats = self._riemannian_dist_stats()

            print(f"\n[E-step] Rebuilding Riemannian KNN graph...")
            with torch.no_grad():
                mu_node, _ = self.model.encode_nodes(x)

            new_edge_index, new_edge_weight = riemannian_knn_graph(
                decoder=self.model.node_decoder,
                z_mu=mu_node.detach(),
                k=self.config.k_neighbors,
                current_edge_index=self.edge_index,
                n_extra_candidates=self.config.n_extra_candidates,
                distance_clip_factor=self.config.distance_clip_factor,
            )

            frac = graph_change_fraction(self.edge_index, new_edge_index)
            changed = frac > 0.0
            self.edge_index = new_edge_index
            self.edge_weight = new_edge_weight

            new_dist_stats = self._riemannian_dist_stats()

            elapsed = time.time() - t0

            # Compute clip cap for display
            clip_info = ""
            if self.config.distance_clip_factor > 0 and self.edge_weight is not None:
                w = self.edge_weight.detach().cpu().numpy()
                pos = w[w > 0]
                if len(pos) > 0:
                    cap = self.config.distance_clip_factor * float(np.median(pos))
                    n_would_clip = int((w > cap).sum())
                    clip_info = f"  clip_cap={cap:.3f}  n_clipped={n_would_clip}"

            print(f"\n  --- Iteration {iteration} diagnostics ---")
            print(f"  Gram ||W^TW-I||/k  : {gram_err:.6f}  (0 = perfect Stiefel)")
            print(f"  Riemannian dist    : mean={dist_stats.get('riem_dist_mean', float('nan')):.4f}  "
                  f"std={dist_stats.get('riem_dist_std', float('nan')):.4f}  "
                  f"min={dist_stats.get('riem_dist_min', float('nan')):.4f}  "
                  f"max={dist_stats.get('riem_dist_max', float('nan')):.4f}  "
                  f"{clip_info}")
            print(f"  Graph change frac  : {frac:.4f}  edges={self.edge_index.shape[1]:,}")
            print(f"  GPU memory         : {_gpu_mem_mb():.0f} MB alloc  "
                  f"{_gpu_reserved_mb():.0f} MB reserved")
            print(f"  Elapsed            : {elapsed:.1f}s")

            iter_metrics = {
                # Explicit step value for per-iteration charts
                "iteration": float(iteration),
                # Per-iteration diagnostics under "iter/" prefix
                "iter/graph_change_fraction": float(frac),
                "iter/graph_changed": float(changed),
                "iter/n_edges": float(self.edge_index.shape[1]),
                "iter/gram_error": gram_err,
                "iter/gpu_mem_mb": _gpu_mem_mb(),
                "iter/gpu_reserved_mb": _gpu_reserved_mb(),
                "iter/elapsed_s": elapsed,
                **{f"iter/riem_dist_{k.split('_')[-1]}": v
                   for k, v in new_dist_stats.items()},
                # Mean of per-epoch losses (one point per iteration)
                **{f"iter/{k}": v for k, v in epoch_logs.items()},
            }
            self.history.append({
                "iteration": float(iteration),
                "graph_change_fraction": float(frac),
                "graph_changed": float(changed),
                "n_edges": float(self.edge_index.shape[1]),
                "gram_error": gram_err,
                **new_dist_stats,
                **epoch_logs,
            })
            self._log_wandb(iter_metrics, step=global_step)

        print("\n" + "=" * 70)
        print("  Training complete.")
        print(f"  Final Gram error : {self._gram_error():.6f}")
        print(f"  Final GPU memory : {_gpu_mem_mb():.0f} MB alloc")
        print("=" * 70)

    def _m_step(self, x: torch.Tensor, iteration: int,
                global_step_offset: int) -> dict[str, float]:
        """Run n_mstep_epochs of VAE training on the current graph."""
        self.model.train()
        cumulative: dict[str, float] = {}
        n_epochs = self.config.n_mstep_epochs
        log_interval = self.config.log_interval

        for epoch in range(1, n_epochs + 1):
            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(x, self.edge_index)

            losses = self.loss_fn(
                x=x,
                x_hat=outputs["x_hat"],
                mu_node=outputs["mu_node"],
                var_node=outputs["var_node"],
                mu_e=outputs["mu_e"],
                var_e=outputs["var_e"],
                decoder=self.model.node_decoder,
                z_mu=outputs["mu_node"].detach(),
                edge_index=self.edge_index,
                W=self.model.frame_W,
            )

            losses["total"].backward()

            # Gradient norms (before clipping)
            grad_W = _grad_norm([self.model.edge_decoder._W])
            grad_dec = _grad_norm(self.model.node_decoder.parameters())
            grad_enc = _grad_norm(self.model.node_encoder.parameters())

            if self.config.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.optimizer.step()

            # SVD retraction -- enforce W^T W = I_k every step
            self.model.edge_decoder.retract_to_stiefel()

            # Verify Stiefel constraint after retraction (cheap for small k)
            if epoch == 1 or epoch % log_interval == 0:
                gram_err_post = self._gram_error()

            self.scheduler.step(losses["total"].detach())

            for k, v in losses.items():
                cumulative[k] = cumulative.get(k, 0.0) + float(v)

            global_epoch = global_step_offset + epoch
            epoch_metrics = {
                # Explicit step values -- all three selectable as x-axis in wandb
                "global_epoch": float(global_epoch),
                "epoch_in_iter": float(epoch),
                "iteration": float(iteration),
                # Per-epoch loss and gradient metrics
                "epoch/total_loss": float(losses["total"]),
                "epoch/recon_loss": float(losses["node_recon"]),
                "epoch/riem_loss": float(losses["riemannian"]),
                "epoch/node_kl": float(losses["node_kl"]),
                "epoch/edge_kl": float(losses["edge_kl"]),
                "epoch/grad_W": grad_W,
                "epoch/grad_decoder": grad_dec,
                "epoch/grad_encoder": grad_enc,
                "epoch/lr": self.optimizer.param_groups[0]["lr"],
            }
            self._log_wandb(epoch_metrics, step=global_epoch)

            if epoch % log_interval == 0 or epoch == n_epochs:
                print(
                    f"  [M-step iter={iteration}] epoch {epoch:4d}/{n_epochs}  "
                    f"total={losses['total']:.4f}  recon={losses['node_recon']:.4f}  "
                    f"riem={losses['riemannian']:.4f}  "
                    f"gram={gram_err_post:.5f}  "
                    f"gradW={grad_W:.4f}  gradDec={grad_dec:.4f}  "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )

        return {k: v / n_epochs for k, v in cumulative.items()}

    @torch.no_grad()
    def get_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Return posterior means for all nodes."""
        self.model.eval()
        mu, _ = self.model.encode_nodes(x.to(self.device))
        return mu

    @torch.no_grad()
    def get_riemannian_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Return current Riemannian edge weights ||l_ij||."""
        if self.edge_weight is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.edge_weight
