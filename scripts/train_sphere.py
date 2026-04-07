"""Validation experiment: sphere S^2.

Ground-truth: geodesic distance = R * arccos(x_i . x_j / R^2)
              sectional curvature K = 1/R^2 (constant, everywhere)

Compares SCR-VAE vs standard VAE baseline on:
  1. Isometry: d_R(z_i, z_j) vs d_true(x_i, x_j)  [lower MAE = more isometric]
  2. Curvature: ambient closure proxy kappa_hat vs K = 1/R^2
  3. Graph convergence: does Riemannian KNN stabilize?
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from rieVAE import SCRVAE
from rieVAE.train.trainer import SCRVAETrainer, TrainerConfig
from rieVAE.train.vanilla_trainer import VanillaVAETrainer, VanillaConfig
from rieVAE.data.synthetic import sphere, compute_true_geodesic_distances
from rieVAE.geometry.log_map import riemannian_log_maps_batched, riemannian_distances
from rieVAE.geometry.curvature import find_triangles, curvature_proxy
from rieVAE.geometry.graph import euclidean_knn_graph


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_points", type=int, default=3000)
    p.add_argument("--ambient_dim", type=int, default=50)
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--noise", type=float, default=0.01)
    # dim_latent=2 matches the intrinsic dimension of S^2 -- critical for isometry
    p.add_argument("--dim_latent", type=int, default=2)
    p.add_argument("--dim_edge", type=int, default=2)
    p.add_argument("--k_neighbors", type=int, default=10)
    p.add_argument("--n_iterations", type=int, default=100,
                   help="Self-consistency iterations (no early stop).")
    p.add_argument("--n_mstep_epochs", type=int, default=1000)
    p.add_argument("--n_baseline_epochs", type=int, default=100000,
                   help="Total epochs for baseline (fair compute budget).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta_node_kl", type=float, default=1e-2)
    p.add_argument("--lambda_riem", type=float, default=0.1,
                   help="Riemannian loss weight. Kept small to not overwhelm recon.")
    p.add_argument("--beta_edge_kl", type=float, default=1e-3)
    p.add_argument("--lambda_decorr", type=float, default=0.0,
                   help="Decorrelation loss (disabled -- Stiefel handles this).")
    p.add_argument("--n_eval_pairs", type=int, default=2000)
    p.add_argument("--max_triangles", type=int, default=1000)
    p.add_argument("--log_interval", type=int, default=100,
                   help="Print/log every N epochs within each M-step.")
    p.add_argument("--no_reset_optimizer", action="store_true", default=False,
                   help="Disable per-iteration optimizer reset (old behaviour).")
    p.add_argument("--distance_clip_factor", type=float, default=5.0,
                   help="Clip Riemannian distances at clip_factor*median before KNN. "
                        "0 = disabled.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="runs/sphere")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    # wandb
    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="rieVAE")
    p.add_argument("--wandb_run_name", type=str, default=None)
    return p.parse_args()


# ─────────────────────── evaluation helpers ────────────────────────────────

def evaluate_isometry(
    decoder: torch.nn.Module,
    z_mu: torch.Tensor,
    params: torch.Tensor,
    radius: float,
    n_pairs: int,
    device: torch.device,
    seed: int,
    use_euclidean_latent: bool = False,
) -> dict[str, float]:
    """Measure isometry against true geodesic distances on S^2.

    Parameters
    ----------
    use_euclidean_latent : bool
        If True, use ||z_j - z_i||_2 (Euclidean in latent space) -- the natural
        distance for a standard VAE.
        If False, use ||J_f(z_i)(z_j-z_i)||_2 (Riemannian pullback) -- the natural
        distance for SCR-VAE which was trained to optimize this quantity.
    """
    rng = np.random.RandomState(seed)
    N = z_mu.shape[0]

    i_idx = torch.from_numpy(rng.choice(N, n_pairs, replace=True)).long()
    j_idx = torch.from_numpy(rng.choice(N, n_pairs, replace=True)).long()
    same = i_idx == j_idx
    j_idx[same] = (j_idx[same] + 1) % N

    z_i = z_mu[i_idx].to(device)
    z_j = z_mu[j_idx].to(device)

    if use_euclidean_latent:
        learned = (z_j - z_i).norm(dim=-1).cpu().numpy()
        dist_name = "euclidean_latent"
    else:
        log_maps = riemannian_log_maps_batched(decoder, z_i, z_j - z_i)
        learned = riemannian_distances(log_maps).cpu().numpy()
        dist_name = "riemannian_pullback"

    true_D = compute_true_geodesic_distances(params, "sphere", R=radius)
    true_vals = true_D[i_idx, j_idx].numpy()

    mask = (true_vals > 0.05) & (true_vals < radius * np.pi * 0.9)
    if mask.sum() < 20:
        return {"mae": float("nan"), "corr": float("nan"), "n_pairs": 0,
                "distance_type": dist_name}

    l, t = learned[mask], true_vals[mask]
    scale = t.mean() / l.mean() if l.mean() > 1e-8 else 1.0
    spearman = float(np.corrcoef(
        np.argsort(np.argsort(l)), np.argsort(np.argsort(t))
    )[0, 1])

    return {
        "mae": float(np.abs(l * scale - t).mean()),
        "corr": float(np.corrcoef(l, t)[0, 1]),
        "spearman": spearman,
        "n_pairs": int(mask.sum()),
        "scale": float(scale),
        "distance_type": dist_name,
    }


def evaluate_curvature(
    decoder: torch.nn.Module,
    z_mu: torch.Tensor,
    edge_index: torch.Tensor,
    radius: float,
    max_triangles: int,
) -> dict[str, float]:
    """Check whether kappa_hat is consistent with K = 1/R^2."""
    triangles = find_triangles(edge_index, max_triangles=max_triangles)
    if triangles.shape[0] == 0:
        return {"mean_kappa": float("nan"), "true_K": 1.0 / radius**2,
                "n_triangles": 0}

    with torch.no_grad():
        kappa = curvature_proxy(decoder, z_mu, triangles)

    return {
        "mean_kappa": float(kappa.mean().item()),
        "std_kappa": float(kappa.std().item()),
        "true_K": float(1.0 / radius**2),
        "n_triangles": int(triangles.shape[0]),
    }


def gram_error(model: SCRVAE) -> float:
    """||W^T W - I_k||_F / k -- measures PCA whitening frame quality."""
    k = model.dim_edge
    G = model.gram_matrix().detach()
    eye = torch.eye(k, dtype=G.dtype, device=G.device)
    return float((G - eye).norm().item() / k)


# ─────────────────────── main ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── wandb (baseline run, separate from SCR-VAE run) ──────────────────────
    _use_wandb = args.use_wandb
    try:
        import wandb as _wandb
    except ImportError:
        _use_wandb = False
        print("[WARN] wandb not found -- logging disabled")

    print("=" * 70)
    print("  SCR-VAE vs Vanilla VAE: Sphere S^2 Isometry Experiment")
    print("=" * 70)
    print(f"  N={args.n_points}  G={args.ambient_dim}  R={args.radius}")
    print(f"  d_latent={args.dim_latent}  d_edge={args.dim_edge}  "
          f"k_neighbors={args.k_neighbors}")
    print(f"  n_iterations={args.n_iterations}  n_mstep_epochs={args.n_mstep_epochs}")
    print(f"  lambda_riem={args.lambda_riem}  beta_node_kl={args.beta_node_kl}")
    print(f"  n_baseline_epochs={args.n_baseline_epochs}")
    print(f"  device={args.device}")
    if _use_wandb:
        print(f"  wandb: project={args.wandb_project}  run={args.wandb_run_name}")

    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB total")

    print("\n[Data] Generating sphere dataset...")
    x, params, A = sphere(
        n_points=args.n_points,
        radius=args.radius,
        ambient_dim=args.ambient_dim,
        noise_std=args.noise,
        seed=args.seed,
    )
    x = x.to(args.device)
    params = params.cpu()
    print(f"  x: {x.shape}  dtype={x.dtype}")
    print(f"  params: {params.shape}  (theta, phi on S^2)")
    print(f"  x norm stats: mean={x.norm(dim=-1).mean():.3f}  "
          f"std={x.norm(dim=-1).std():.4f}  (expected ~{args.radius:.1f})")

    # ── Shared model config ──────────────────────────────────────────────────
    # dim_latent=2 matches S^2 intrinsic dimension: essential for correct isometry
    model_kwargs = dict(
        dim_features=args.ambient_dim,
        dim_latent=args.dim_latent,
        dim_edge=args.dim_edge,
        encoder_hidden=(256, 128),
        decoder_hidden=(128, 256),
        edge_hidden=(64,),
        dropout=0.0,
    )
    total_train_epochs = args.n_iterations * args.n_mstep_epochs
    print(f"\n  SCR-VAE total compute: {args.n_iterations} x {args.n_mstep_epochs} "
          f"= {total_train_epochs:,} epochs")
    print(f"  Baseline total compute: {args.n_baseline_epochs:,} epochs")

    results = {
        "args": vars(args),
        "true_K": 1.0 / args.radius**2,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 1. BASELINE: Standard VAE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  [1/2] Standard VAE (Baseline)")
    print("─" * 64)

    baseline_model = SCRVAE(**model_kwargs).to(args.device)
    baseline_cfg = VanillaConfig(
        n_epochs=total_train_epochs,
        learning_rate=args.lr,
        beta_node_kl=args.beta_node_kl,
        device=args.device,
    )
    baseline_trainer = VanillaVAETrainer(baseline_model, baseline_cfg)

    t0 = time.time()
    baseline_trainer.fit(x)
    baseline_time = time.time() - t0

    baseline_model.eval()
    with torch.no_grad():
        z_baseline, _ = baseline_model.encode_nodes(x)

    ei_euclidean, _ = euclidean_knn_graph(x.cpu(), k=args.k_neighbors)
    ei_euclidean = ei_euclidean.to(args.device)

    iso_baseline_euclid = evaluate_isometry(
        baseline_model.node_decoder, z_baseline, params,
        args.radius, args.n_eval_pairs, torch.device(args.device), args.seed + 1,
        use_euclidean_latent=True,
    )
    iso_baseline_riem = evaluate_isometry(
        baseline_model.node_decoder, z_baseline, params,
        args.radius, args.n_eval_pairs, torch.device(args.device), args.seed + 3,
        use_euclidean_latent=False,
    )
    curv_baseline = evaluate_curvature(
        baseline_model.node_decoder, z_baseline,
        ei_euclidean, args.radius, args.max_triangles,
    )

    print(f"\n  Baseline results:")
    print(f"    Isometry (Euclidean latent):     MAE={iso_baseline_euclid['mae']:.4f}  "
          f"Corr={iso_baseline_euclid['corr']:.4f}  Spearman={iso_baseline_euclid.get('spearman', float('nan')):.4f}")
    print(f"    Isometry (Riemannian pullback):  MAE={iso_baseline_riem['mae']:.4f}  "
          f"Corr={iso_baseline_riem['corr']:.4f}  Spearman={iso_baseline_riem.get('spearman', float('nan')):.4f}")
    kappa_theory = (2.0 / args.radius) * (3.0 / args.ambient_dim) ** 0.5
    print(f"    Mean kappa_hat:    {curv_baseline['mean_kappa']:.4f}")
    print(f"    Theory (recon-regime): {kappa_theory:.4f}  [= (2/R)*sqrt(3/G)]")
    print(f"    Sectional K = 1/R^2:   {curv_baseline['true_K']:.4f}")

    results["baseline"] = {
        "isometry_euclidean": iso_baseline_euclid,
        "isometry_riemannian": iso_baseline_riem,
        "curvature": curv_baseline,
        "training_time_s": baseline_time,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 2. SCR-VAE (our method)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  [2/2] SCR-VAE (Our Method)")
    print("─" * 64)

    scrvae_model = SCRVAE(**model_kwargs).to(args.device)
    run_name = args.wandb_run_name or f"sphere_d{args.dim_latent}_iter{args.n_iterations}_lriem{args.lambda_riem}"
    scrvae_cfg = TrainerConfig(
        k_neighbors=args.k_neighbors,
        n_iterations=args.n_iterations,
        n_mstep_epochs=args.n_mstep_epochs,
        learning_rate=args.lr,
        beta_node_kl=args.beta_node_kl,
        lambda_riem=args.lambda_riem,
        beta_edge_kl=args.beta_edge_kl,
        lambda_decorr=args.lambda_decorr,
        reset_optimizer_each_iteration=not args.no_reset_optimizer,
        distance_clip_factor=args.distance_clip_factor,
        log_interval=args.log_interval,
        device=args.device,
        use_wandb=_use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=run_name + "_scrvae",
    )
    scrvae_trainer = SCRVAETrainer(scrvae_model, scrvae_cfg)

    t0 = time.time()
    scrvae_trainer.fit(x)
    scrvae_time = time.time() - t0

    scrvae_model.eval()
    with torch.no_grad():
        z_scrvae, _ = scrvae_model.encode_nodes(x)

    iso_scrvae_riem = evaluate_isometry(
        scrvae_model.node_decoder, z_scrvae, params,
        args.radius, args.n_eval_pairs, torch.device(args.device), args.seed + 2,
        use_euclidean_latent=False,
    )
    iso_scrvae_euclid = evaluate_isometry(
        scrvae_model.node_decoder, z_scrvae, params,
        args.radius, args.n_eval_pairs, torch.device(args.device), args.seed + 4,
        use_euclidean_latent=True,
    )
    curv_scrvae = evaluate_curvature(
        scrvae_model.node_decoder, z_scrvae,
        scrvae_trainer.edge_index.to(args.device), args.radius, args.max_triangles,
    )
    g_err = gram_error(scrvae_model)

    print(f"\n  SCR-VAE results:")
    print(f"    Isometry (Euclidean latent):     MAE={iso_scrvae_euclid['mae']:.4f}  "
          f"Corr={iso_scrvae_euclid['corr']:.4f}  Spearman={iso_scrvae_euclid.get('spearman', float('nan')):.4f}")
    print(f"    Isometry (Riemannian pullback):  MAE={iso_scrvae_riem['mae']:.4f}  "
          f"Corr={iso_scrvae_riem['corr']:.4f}  Spearman={iso_scrvae_riem.get('spearman', float('nan')):.4f}")
    print(f"    Mean kappa_hat:    {curv_scrvae['mean_kappa']:.4f}  "
          f"(theory={kappa_theory:.4f}, K={curv_scrvae['true_K']:.4f})")
    print(f"    Gram ||W^TW-I||/k: {g_err:.4f}")
    print(f"    Graph iterations:  {len(scrvae_trainer.history)}")

    results["scrvae"] = {
        "isometry_euclidean": iso_scrvae_euclid,
        "isometry_riemannian": iso_scrvae_riem,
        "curvature": curv_scrvae,
        "gram_error": g_err,
        "n_iterations_run": len(scrvae_trainer.history),
        "training_time_s": scrvae_time,
        "training_history": scrvae_trainer.history,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  SUMMARY: Fair Comparison (Euclidean vs Riemannian)")
    print("=" * 64)
    print(f"  {'Metric':<30} {'Baseline':>12} {'SCR-VAE':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Euclidean latent MAE':<30} "
          f"{iso_baseline_euclid['mae']:>12.4f} "
          f"{iso_scrvae_euclid['mae']:>12.4f}")
    print(f"  {'Euclidean latent Spearman':<30} "
          f"{iso_baseline_euclid.get('spearman', float('nan')):>12.4f} "
          f"{iso_scrvae_euclid.get('spearman', float('nan')):>12.4f}")
    print(f"  {'Riemannian pullback MAE':<30} "
          f"{iso_baseline_riem['mae']:>12.4f} "
          f"{iso_scrvae_riem['mae']:>12.4f}")
    print(f"  {'Riemannian pullback Spearman':<30} "
          f"{iso_baseline_riem.get('spearman', float('nan')):>12.4f} "
          f"{iso_scrvae_riem.get('spearman', float('nan')):>12.4f}")
    print(f"  {'Curvature proxy kappa_hat':<30} "
          f"{curv_baseline['mean_kappa']:>12.4f} "
          f"{curv_scrvae['mean_kappa']:>12.4f}  "
          f"(theory={kappa_theory:.4f}, K={1.0/args.radius**2:.4f})")

    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
