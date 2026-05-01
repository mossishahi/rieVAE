"""Unified experiment logger: wandb + CSV + JSON.

Design principles:
  - ONE logger per training run, shared by trainer + evaluation code.
  - Metric names are enforced by METRIC_REGISTRY to prevent drift.
  - CSV is flushed after every log call → live tail-able.
  - JSON summary written on close() → machine-readable final results.
  - Step counter is always an integer; caller decides what it means.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Canonical metric names — enforced by the logger to prevent naming drift.
# Add new metrics here; if a metric is not registered, the logger warns once.
# ---------------------------------------------------------------------------
METRIC_REGISTRY: dict[str, str] = {
    # Training losses
    "train/loss_total":      "Total loss (recon + KL + riem + decorr)",
    "train/loss_recon":      "Reconstruction loss",
    "train/loss_kl_node":    "Node KL divergence",
    "train/loss_kl_edge":    "Edge KL divergence",
    "train/loss_riem":       "Riemannian distance loss",
    "train/loss_decorr":     "Decorrelation regulariser",
    "train/lr":              "Current learning rate",
    "train/grad_norm":       "Gradient norm (before clipping)",
    "train/gram_error":      "||W^T W - I_k||_F / k",
    # Graph structure
    "graph/n_edges":         "Number of hard-graph edges",
    "graph/change_fraction": "Edge turnover since last G-step",
    "graph/riem_dist_mean":  "Mean Riemannian edge weight",
    "graph/riem_dist_max":   "Max Riemannian edge weight",
    # Evaluation (computed periodically)
    "eval/riem_mae":         "Riemannian pullback MAE vs d^M",
    "eval/eucl_mae":         "Euclidean latent MAE vs d^M",
    "eval/encoder_riem_mae":  "Encoder Riemannian MAE dR*(mu_i,mu_j) vs d^M",
    "eval/encoder_eucl_mae":  "Encoder Euclidean MAE ||mu_i-mu_j|| vs d^M",
    "eval/encoder_mae":      "Encoder MAE (alias for encoder_riem_mae)",
    "eval/decoder_mae":      "Decoder Riemannian MAE vs d^M",
    "eval/encoder_decoder_agreement": "Agreement MAE between encoder and decoder",
    "eval/rho_enc_dec":      "Ratio encoder_MAE / decoder_MAE (theory: -> 1)",
    "eval/spearman_enc":     "Spearman(encoder dists, d^M)",
    "eval/spearman_dec":     "Spearman(decoder dists, d^M)",
    "eval/encoder_spearman": "Spearman(encoder dists, d^M)",
    "eval/decoder_spearman": "Spearman(decoder dists, d^M)",
    "eval/kappa_hat":        "Curvature proxy kappa_hat",
    "eval/kappa_true":       "True sectional curvature",
    "eval/kappa_error":      "|kappa_hat - kappa_true|",
    # Strong convexity (Theorem thm:sc_verification)
    "sc/mu_hat_1":           "Estimated SC constant mu_1",
    "sc/lambda_min_mcover":  "lambda_min(M_cov) — tangent covering",
    "sc/lambda_min_phi":     "lambda_min(Phi) — NTK Gram",
    "sc/r_bar_mean":         "Mean edge length r_bar",
    "sc/adaptive_mstep":        "Adaptive P-step budget m*_t (legacy alias)",
    "sc/adaptive_p_step":       "Adaptive P-step budget m*_t",
    "sc/adaptive_p_step_budget":"Adaptive P-step budget m*_t (full key)",
    "sc/adaptive_mstep_budget": "Adaptive P-step budget m*_t (legacy alias, full key)",
    # SGD noise floor (Theorem thm:sgd_noise_floor)
    "sgd/sigma0_sq":         "Gradient variance sigma_0^2",
    "sgd/noise_floor_ratio": "sigma_0^2 / (mu_1 * r_n^2)",
    "sgd/floor_ok":          "1 if noise_floor_ratio < 1",
    # Encoder regularity (Theorem thm:aec_verification)
    "enc/L_phi":             "Encoder Lipschitz bound (sigma'_max^{L-1} * prod ||W_l||)",
    "enc/kappa_phi":         "Encoder Hessian bound ((L-1)*sigma''_max*sigma'_max^{L-2}*||W_max||^{L+1})",
    "enc/W_max_op_norm":     "Max weight spectral norm",
    "enc/aec_holds":         "1 if A-EC verified",
    "enc/ntk_certified":     "1 if NTK SC proof certified (sigma'_min>0 on preact range, Assumption A-BPA)",
    "enc/preact_min":        "Min decoder preactivation over training pts (for Assumption A-BPA)",
    "enc/preact_max":        "Max decoder preactivation over training pts (for Assumption A-BPA)",
    # Post-training isometry certificate (Definition 1; six conditions)
    "cert/delta_rec_ok":     "1 if reconstruction error <= C_cap * r_n^2 (C1, legacy alias)",
    "cert/sc_ok":            "1 if hat_mu_1 > 0 (C4, legacy alias)",
    "cert/graph_stable":     "1 if graph topology stable (C6, legacy alias)",
    "cert/isometry_holds":   "1 iff all six conditions satisfied (full certificate)",
    "cert/r_n":              "Learning radius r_n = (log n / n)^{1/d}",
    "cert/delta_rec":        "Reconstruction residual max_i ||f(z_i) - x_i||",
    "cert/delta_edge":       "Edge-decoder residual max ||F_phi - J_f Delta z||",
    "cert/delta_def":        "Deformation residual max | bar d^A_psi^2 - || J_f Delta z ||^2 | (C3)",
    "cert/mu_hat_1":         "Strong-convexity estimate hat_mu_1 (alias of sc/mu_hat_1)",
    "cert/mu_hat_1_output_layer":
        "Unconditional output-layer SC lower bound",
    "cert/lambda_t":         "Current Riemannian-loss weight lambda_t",
    "cert/lambda_cross":     "Schedule threshold lambda_cross = r_n^2 / hat_mu_1",
    "cert/fold_fraction":    "Soft-to-hard graph change fraction (last step)",
    "cert/envelope_C1_rn":   "Theoretical isometry envelope C_1 * r_n",
    "cert/c1_ok":            "1 if delta_rec <= C_cap * r_n^2 (Certificate C1)",
    "cert/c2_ok":            "1 if delta_edge <= C_cap' * r_n^2 (Certificate C2)",
    "cert/c3_ok":            "1 if delta_def <= C_cap'' * r_n^2 (Certificate C3)",
    "cert/c4_ok":            "1 if hat_mu_1 >= output-layer bound (Certificate C4)",
    "cert/c5_ok":            "1 if lambda_t >= lambda_cross (Certificate C5)",
    "cert/c6_ok":            "1 if fold fraction within tolerance (Certificate C6)",
    "cert/delta_fold_bound": "Corrected topological fold separation lower bound (optional)",
    # Training loss aliases for the deformation branch
    "train/loss_def":        "Deformation loss L_def on the current graph edges",
    # Graph-size diagnostics for the ambient-ball / deformed-ball graph
    "graph/r_initial":            "Initial ambient-ball radius r (one-shot)",
    "graph/r_initial_method_mst": "1.0 if r_initial came from mst_connectivity_radius, 0.0 if from median_knn or config",
    "graph/n_outliers_dropped":   "Number of MST-flagged outliers dropped at init",
    "graph/n_components_at_r_initial": "Connected components of the kNN graph at the initial radius r",
    "graph/k_safe_used":          "kNN degree used by mst_connectivity_radius (for diagnostics)",
    "graph/n_edges_initial":      "Undirected edges in the initial ambient ball graph",
    "graph/mean_degree_initial":  "Mean degree of the initial ambient ball graph",
    "graph/n_edges":              "Undirected edges in the current hard graph",
    "graph/mean_degree":          "Mean degree (2 * n_edges / N) of the current hard graph",
    "graph/degree_std":           "Std of per-node degree on the current hard graph",
    "graph/max_degree":           "Max per-node degree on the current hard graph (hub detection)",
    "graph/min_degree":           "Min per-node degree on the current hard graph (isolated-node detection)",
    "graph/n_components":         "Connected components of the current hard graph (1 = healthy)",
    "graph/change_fraction":      "Symmetric-difference fraction between consecutive hard graphs",
    # Anchor-batched training (Section 5)
    "graph/batch_n_nodes":        "Closed 1-hop induced subgraph node count per batch",
    "graph/batch_n_edges":        "Closed 1-hop induced subgraph directed edge count per batch",
    "graph/expansion_admit_count":"New (anchor, candidate) pairs admitted via BallTree pool this G-step",
    "graph/expansion_evict_count":"(anchor, slot) pairs evicted to admit new ones",
    "graph/p_min_obs":            "Observed min over data of Def_psi outputs (vs p_min_safe)",
    "graph/p_min_safe_updated":   "p_min_safe value after the most recent pool rebuild",
    # Anchor sampler diagnostics (EpochAnchorSampler)
    "graph/anchor_visit_count_max": "Max per-node anchor visits since training start",
    "graph/anchor_visit_count_min": "Min per-node anchor visits since training start",
    "graph/anchor_visit_count_std": "Std of per-node anchor visit counts",
    "graph/anchor_sampler_epoch":   "EpochAnchorSampler epoch counter (one shuffle per epoch)",
    # Latent-geometry report (eight-metric comparison vs ground truth d^M)
    "eval/latent_pearson":          "Pearson(latent_dist, d_M) on random pairs",
    "eval/latent_spearman":         "Spearman(latent_dist, d_M) on random pairs",
    "eval/latent_kendall":          "Kendall tau(latent_dist, d_M) on random pairs",
    "eval/latent_best_alpha":       "Best linear scale alpha minimising || alpha * d_pred - d_M ||",
    "eval/latent_mae_scaled":       "MAE of (alpha * d_pred) vs d_M",
    "eval/latent_rmse_scaled":      "RMSE of (alpha * d_pred) vs d_M",
    "eval/latent_rel_rmse":         "Relative RMSE: rmse_scaled / mean(d_M)",
    "eval/latent_distortion_max":   "Max-over-min ratio of (d_pred / d_M)",
    "eval/latent_knn_accuracy":     "Fraction of latent kNN that match manifold kNN (k_local)",
    # Proximal-specific
    "proximal/tau":          "Temperature tau_t",
    "proximal/alpha":        "EMA rate alpha_t",
    "proximal/alpha_coupled":"Coupled alpha (Condition C2)",
    "proximal/delta_knn":    "KNN margin delta_KNN",
    "proximal/soft_hard_gap":"Soft-to-hard graph L1 gap",
    "proximal/ema_lag":      "EMA lag (KL or norm)",
    "proximal/c1_ok":        "1 if Condition C1 satisfied",
    "proximal/c2_ok":        "1 if Condition C2 satisfied",
    # Adaptive radius
    "adaptive/radius_mean":  "Mean adaptive radius r_i",
    "adaptive/radius_min":   "Min adaptive radius",
    "adaptive/frac_curved":  "Fraction of nodes with r_i < r_n",
    # Minimax bounds (Theorem thm:minimax_lb)
    "minimax/r_n":           "KNN radius r_n",
    "minimax/upper_bound":   "C_1 * r_n upper bound",
    "minimax/lecam_lb":      "Le Cam lower bound",
    "minimax/lecam_lower_bound": "Le Cam lower bound (alias)",
    "minimax/assouad_lb":    "Fano lower bound (legacy name)",
    "minimax/assouad_lower_bound": "Fano lower bound (legacy alias)",
    "minimax/rate_ratio":    "empirical_MAE / r_n",
    "minimax/optimality_ratio": "MAE / Fano_LB",
    # Frame identification
    "frame/subspace_angle":  "Subspace angle (degrees)",
    "frame/sin_angle":       "sin(subspace angle)",
    "frame/dk_bound":        "Davis-Kahan bound",
    "frame/delta_k":         "Eigenvalue gap delta_k",
    # Loss schedule / warmup
    "warmup/phase":          "Current warmup phase (1/2/3)",
    "warmup/lambda_riem_current": "Current lambda_riem value",
    "warmup/lambda_riem_target":  "Target lambda_riem after calibration",
    "warmup/logmap_ema":     "EMA of mean ||l_ij||^2 (log-map scale)",
    "warmup/norm_recon":     "Normalised reconstruction L_recon/Var(X)",
    # Timing
    "time/step_s":           "Wall-clock per step (seconds)",
    "time/elapsed_s":        "Total elapsed (seconds)",
    # ----------------------------------------------------------------
    # Iso architecture (rieVAE >= R5)
    # ----------------------------------------------------------------
    # Per-step training losses (raw and scale-normalised; total).
    "train/L_rec_raw":       "Raw L_rec (MSE reconstruction)",
    "train/L_KL_raw":        "Raw L_KL (flat prior in iso)",
    "train/L_iso_raw":       "Raw L_iso = mean (||mu_i-mu_j|| - d^bih)^2",
    "train/L_edge_raw":      "Raw L_edge = mean (F_phi(sg[mu_i,mu_j]) - d^bih)^2",
    "train/L_rec_eff":       "L_rec / s_rec",
    "train/L_KL_eff":        "beta(t) * L_KL / s_kl",
    "train/L_iso_eff":       "gamma(t) * L_iso / s_iso",
    "train/L_edge_eff":      "nu(t) * L_edge / s_edge",
    "train/L_total":         "Total scaled iso loss",
    # Iso schedule multipliers.
    "train/gamma_t":         "Current L_iso weight gamma(t)",
    "train/nu_t":            "Current L_edge weight nu(t)",
    "train/beta_t":          "Current KL weight beta(t)",
    "train/phase":           "Training phase id (0 / 1 / 2)",
    "train/phase2_rho":      "Phase-2 fraction rho in [0, 1]",
    "train/edge_scale":      "Current softplus(w) of the scalar edge head",
    # Initial-scale denominators (constant after Phase 2 entry).
    "train/scale_rec":       "Recorded s_rec (Phase 2 entry)",
    "train/scale_kl":        "Recorded s_kl  (Phase 2 entry)",
    "train/scale_iso":       "Recorded s_iso (Phase 2 entry)",
    "train/scale_edge":      "Recorded s_edge (Phase 2 entry)",
    # Iso certificate scalars.
    "cert/delta_iso":           "sup over E* of |||mu_i-mu_j|| - d^bih|",
    "cert/delta_edge_scalar":   "sup over E* of |F_phi(mu_i,mu_j) - d^bih|",
    "cert/edge_scale":          "softplus(w) at certificate checkpoint",
    # Adaptive fine-tuning state machine (Phase 2b).
    # Iso eval metrics (final).
    "eval/mae":              "Mean abs. error of predicted distance vs d^M (E*)",
    "eval/mae_relative":     "Mean of |err| / d^M on E*",
    "eval/max_edge_error":   "sup-norm error on E*",
    "eval/bilip_ratio_edges":"max ratio / min ratio on E* (bi-Lipschitz)",
    "eval/mae_held_out":     "Same MAE on random non-edge held-out pairs",
    "eval/max_held_out_error":"sup-norm error on held-out pairs",
    "eval/bilip_ratio_held_out":"Bi-Lipschitz ratio on held-out pairs",
    "eval/n_pairs":          "Number of training-edge pairs evaluated",
    "eval/n_held_out_pairs": "Number of held-out pairs evaluated",
}


@dataclasses.dataclass
class ExperimentLogger:
    """Unified logger writing to wandb + CSV + JSON simultaneously.

    Parameters
    ----------
    run_dir : str or Path
        Directory for CSV logs and JSON summary.
    experiment_name : str
        Human-readable name (used as wandb run name and CSV filename).
    config : dict
        Full hyperparameter config (logged to wandb + JSON).
    use_wandb : bool
        Whether to initialise wandb.
    wandb_project : str
        wandb project name.
    wandb_group : str or None
        wandb group (for comparing related runs).
    wandb_tags : list[str] or None
        wandb tags.
    csv_filename : str or None
        Override CSV filename (default: metrics.csv).
    strict : bool
        If True, warn on unregistered metric names.
    """
    run_dir: str | Path
    experiment_name: str
    config: dict
    use_wandb: bool = True
    wandb_project: str = "rieVAE"
    wandb_group: Optional[str] = None
    wandb_tags: Optional[list[str]] = None
    csv_filename: Optional[str] = None
    strict: bool = True

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self.run_dir / (self.csv_filename or "metrics.csv")
        self._csv_file = None
        self._csv_writer = None
        self._csv_columns: list[str] = []
        self._csv_rows: list[dict] = []
        self._warned_keys: set[str] = set()
        self._summary: dict[str, Any] = {}
        self._step_count = 0
        self._start_time = time.time()

        # Save config
        config_path = self.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

        # Initialise wandb
        self._wandb_run = None
        self._wandb_ok = False
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.wandb_project,
                    name=self.experiment_name,
                    group=self.wandb_group,
                    tags=self.wandb_tags or [],
                    config=self.config,
                    reinit=True,
                )
                self._wandb_run = wandb.run
                wandb.define_metric("step")
                for prefix in ("train", "eval", "sc", "sgd", "enc",
                               "proximal", "adaptive", "minimax",
                               "graph", "frame", "time", "ablation"):
                    wandb.define_metric(f"{prefix}/*", step_metric="step")
                self._wandb_ok = True
            except Exception:
                self._wandb_ok = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log a dict of metrics at the given step.

        Writes to both wandb and CSV. The CSV is flushed immediately so
        you can `tail -f metrics.csv` during training.
        """
        self._step_count = step

        if self.strict:
            for k in metrics:
                if k not in METRIC_REGISTRY and k not in self._warned_keys:
                    self._warned_keys.add(k)
                    print(f"[ExperimentLogger] WARNING: unregistered metric '{k}' "
                          f"— add it to METRIC_REGISTRY for consistency")

        row = {"step": step, **metrics}

        # wandb
        if self._wandb_ok:
            try:
                import wandb
                wandb.log(row, step=step)
            except Exception:
                pass

        # CSV (lazy init: create header on first call, extend when new keys appear)
        self._append_csv(row)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        """Log final summary metrics (written to JSON and wandb summary)."""
        self._summary.update(metrics)

        if self._wandb_ok and self._wandb_run is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._wandb_run.summary[k] = v

    def close(self) -> None:
        """Flush CSV, write JSON summary, finish wandb."""
        elapsed = time.time() - self._start_time
        self._summary["time/total_elapsed_s"] = elapsed
        self._summary["total_steps"] = self._step_count

        # Write JSON summary
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(self._summary, f, indent=2, default=str)

        # Rewrite CSV with all columns aligned
        self._rewrite_csv()

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

        if self._wandb_ok:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
            self._wandb_ok = False

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _append_csv(self, row: dict) -> None:
        """Append a row to the CSV, extending columns if needed."""
        self._csv_rows.append(row)

        new_keys = [k for k in row if k not in self._csv_columns]
        if new_keys:
            self._csv_columns.extend(new_keys)
            # Rewrite file with updated header
            self._rewrite_csv()
        else:
            # Append single row
            if self._csv_file is None:
                self._csv_file = open(self._csv_path, "a", newline="")
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=self._csv_columns,
                    extrasaction="ignore",
                )
                if self._csv_path.stat().st_size == 0:
                    self._csv_writer.writeheader()
            self._csv_writer.writerow(
                {k: row.get(k, "") for k in self._csv_columns}
            )
            self._csv_file.flush()

    def _rewrite_csv(self) -> None:
        """Rewrite the entire CSV with the current column set."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self._csv_columns, extrasaction="ignore",
            )
            writer.writeheader()
            for r in self._csv_rows:
                writer.writerow({k: r.get(k, "") for k in self._csv_columns})

        # Reopen for appending
        self._csv_file = open(self._csv_path, "a", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=self._csv_columns,
            extrasaction="ignore",
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    @property
    def wandb_run(self):
        return self._wandb_run

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
