"""Validation experiment: flat torus T^2.

Ground-truth: geodesic distance = sqrt((R * dtheta)^2 + (r * dphi)^2)
              where dtheta, dphi are angular differences modulo 2pi.

Key prediction: since the torus has ZERO Gaussian curvature (K = 0),
the ambient closure proxy kappa_hat should be close to zero for all triangles.
This is the NEGATIVE test for our curvature estimator.

Note: the torus has non-trivial topology (fundamental group Z x Z), which
distinguishes it from a flat plane even though K = 0. The SCR-VAE should
still achieve good isometry, and the curvature proxy should correctly
identify the manifold as flat.
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
from rieVAE.data.synthetic import flat_torus, flat_torus_clifford, compute_true_geodesic_distances
from rieVAE.geometry.log_map import riemannian_log_maps_batched, riemannian_distances
from rieVAE.geometry.curvature import find_triangles, curvature_proxy
from rieVAE.geometry.graph import euclidean_knn_graph


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_points", type=int, default=3000)
    p.add_argument("--ambient_dim", type=int, default=50)
    p.add_argument("--major_radius", type=float, default=2.0,
                   help="R: major torus radius (the large ring)")
    p.add_argument("--minor_radius", type=float, default=1.0,
                   help="r: minor torus radius (the tube)")
    p.add_argument("--noise", type=float, default=0.01)
    # dim_latent=2 matches T^2 intrinsic dimension
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
    p.add_argument("--lambda_riem", type=float, default=0.1)
    p.add_argument("--beta_edge_kl", type=float, default=1e-3)
    p.add_argument("--lambda_decorr", type=float, default=0.0)
    p.add_argument("--n_eval_pairs", type=int, default=2000)
    p.add_argument("--max_triangles", type=int, default=1000)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--no_reset_optimizer", action="store_true", default=False,
                   help="Disable per-iteration optimizer reset (old behaviour).")
    p.add_argument("--distance_clip_factor", type=float, default=5.0,
                   help="Clip Riemannian distances at clip_factor*median before KNN. "
                        "0 = disabled.")
    # Use the Clifford torus in R^4 (genuinely flat, K=0 everywhere).
    # Default is True because this is the correct manifold for the K=0 test.
    # Pass --no_clifford to use the old standard embedded torus in R^3.
    p.add_argument("--no_clifford", action="store_true", default=False,
                   help="Use standard torus in R^3 instead of Clifford flat torus in R^4.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="runs/torus_clifford")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    # wandb
    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="rieVAE")
    p.add_argument("--wandb_run_name", type=str, default=None)
    return p.parse_args()


def evaluate_isometry(
    decoder: torch.nn.Module,
    z_mu: torch.Tensor,
    params: torch.Tensor,
    R: float,
    r: float,
    n_pairs: int,
    device: torch.device,
    seed: int,
    use_euclidean_latent: bool = False,
    manifold: str = "clifford_torus",
) -> dict[str, float]:
    """Measure isometry against true geodesic distances on T^2.

    use_euclidean_latent=True  -> ||z_j - z_i||_2  (baseline's natural metric)
    use_euclidean_latent=False -> ||J_f(z_i)(z_j-z_i)||_2  (SCR-VAE's trained metric)
    manifold: 'clifford_torus' (flat, K=0, exact formula) or 'torus' (curved R^3)
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

    true_D = compute_true_geodesic_distances(
        params, manifold, r_torus=r, R_torus=R
    )
    true_vals = true_D[i_idx, j_idx].numpy()

    mask = true_vals > 0.05
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


def evaluate_curvature_torus(
    decoder: torch.nn.Module,
    z_mu: torch.Tensor,
    edge_index: torch.Tensor,
    max_triangles: int,
) -> dict[str, float]:
    """Evaluate the curvature proxy for the flat torus.

    For the Clifford flat torus the intrinsic Gaussian curvature K = 0, but the
    extrinsic second fundamental form h != 0 (it curves through R^4). By
    Proposition 4 of the theory paper, kappa_hat -> 2|H| (mean curvature), NOT
    to 0. The reconstruction-regime prediction for R=2, r=1, G=50 is:
        kappa_hat* = sqrt(1/R^2 + 1/r^2) * sqrt(4/G)
                   = sqrt(5/4) * sqrt(4/50) = sqrt(1/10) ~ 0.316
    Values near 0.316 indicate a well-calibrated decoder; values above reflect
    fold artifacts from the topological obstruction (pi_1(T^2) = Z x Z).
    """
    triangles = find_triangles(edge_index, max_triangles=max_triangles)
    if triangles.shape[0] == 0:
        return {"mean_kappa": float("nan"), "expected_K": 0.0, "n_triangles": 0}

    with torch.no_grad():
        kappa = curvature_proxy(decoder, z_mu, triangles)

    return {
        "mean_kappa": float(kappa.mean().item()),
        "std_kappa": float(kappa.std().item()),
        "expected_K": 0.0,
        "n_triangles": int(triangles.shape[0]),
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 64)
    print("  SCR-VAE vs Vanilla VAE: Flat Torus T^2 Experiment")
    print("=" * 64)
    print(f"  N={args.n_points}  G={args.ambient_dim}")
    R, r, G = args.major_radius, args.minor_radius, args.ambient_dim
    _kappa_theory = (R**-2 + r**-2) ** 0.5 * (4 / G) ** 0.5
    print(f"  R={R}  r={r}  K=0 (Gaussian), kappa_hat* ~{_kappa_theory:.3f} (mean-curv, recon regime)")
    print(f"  d_latent={args.dim_latent}  d_edge={args.dim_edge}")
    print(f"  device={args.device}")

    use_clifford = not args.no_clifford
    manifold_name = "Clifford flat torus (R^4, K=0)" if use_clifford else "Standard torus (R^3, K!=0)"
    print(f"  Manifold: {manifold_name}")

    print("\n[Data] Generating torus dataset...")
    if use_clifford:
        x, params, A = flat_torus_clifford(
            n_points=args.n_points,
            R=args.major_radius,
            r=args.minor_radius,
            ambient_dim=args.ambient_dim,
            noise_std=args.noise,
            seed=args.seed,
        )
        geodesic_manifold = "clifford_torus"
    else:
        x, params, A = flat_torus(
            n_points=args.n_points,
            R=args.major_radius,
            r=args.minor_radius,
            ambient_dim=args.ambient_dim,
            noise_std=args.noise,
            seed=args.seed,
        )
        geodesic_manifold = "torus"
    x = x.to(args.device)
    params = params.cpu()
    print(f"  x: {x.shape}  params: {params.shape}  embedding dim: {A.shape[1]}")

    # dropout=0.0 is used for both models (same as sphere) to avoid
    # an uncontrolled variable in the sphere-vs-torus comparison.
    model_kwargs = dict(
        dim_features=args.ambient_dim,
        dim_latent=args.dim_latent,
        dim_edge=args.dim_edge,
        encoder_hidden=(256, 128),
        decoder_hidden=(128, 256),
        edge_hidden=(64,),
        dropout=0.0,
    )
    total_epochs = args.n_iterations * args.n_mstep_epochs
    results = {"args": vars(args), "expected_K": 0.0}

    # ─── Baseline ────────────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  [1/2] Standard VAE (Baseline)")
    print("─" * 64)

    baseline_model = SCRVAE(**model_kwargs).to(args.device)
    baseline_trainer = VanillaVAETrainer(
        baseline_model,
        VanillaConfig(n_epochs=args.n_baseline_epochs, learning_rate=args.lr,
                      beta_node_kl=args.beta_node_kl, device=args.device),
    )
    t0 = time.time()
    baseline_trainer.fit(x)
    baseline_time = time.time() - t0

    baseline_model.eval()
    with torch.no_grad():
        z_base, _ = baseline_model.encode_nodes(x)

    ei_base, _ = euclidean_knn_graph(x.cpu(), k=args.k_neighbors)
    ei_base = ei_base.to(args.device)

    iso_base_euclid = evaluate_isometry(
        baseline_model.node_decoder, z_base, params,
        args.major_radius, args.minor_radius, args.n_eval_pairs,
        torch.device(args.device), args.seed + 1,
        use_euclidean_latent=True, manifold=geodesic_manifold,
    )
    iso_base_riem = evaluate_isometry(
        baseline_model.node_decoder, z_base, params,
        args.major_radius, args.minor_radius, args.n_eval_pairs,
        torch.device(args.device), args.seed + 3,
        use_euclidean_latent=False, manifold=geodesic_manifold,
    )
    curv_base = evaluate_curvature_torus(
        baseline_model.node_decoder, z_base, ei_base, args.max_triangles,
    )

    print(f"\n  Baseline:")
    print(f"    Euclidean latent:     MAE={iso_base_euclid['mae']:.4f}  "
          f"Spearman={iso_base_euclid.get('spearman', float('nan')):.4f}")
    print(f"    Riemannian pullback:  MAE={iso_base_riem['mae']:.4f}  "
          f"Spearman={iso_base_riem.get('spearman', float('nan')):.4f}")
    print(f"    Curvature kappa_hat:  {curv_base['mean_kappa']:.4f}  "
          f"(theory~{_kappa_theory:.3f}, mean-curv proxy; lower=better)")

    results["baseline"] = {
        "isometry_euclidean": iso_base_euclid,
        "isometry_riemannian": iso_base_riem,
        "curvature": curv_base,
        "training_time_s": baseline_time,
    }

    # ─── SCR-VAE ──────────────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  [2/2] SCR-VAE (Our Method)")
    print("─" * 64)

    _use_wandb = getattr(args, "use_wandb", False)
    try:
        import wandb as _wandb
    except ImportError:
        _use_wandb = False

    scrvae_model = SCRVAE(**model_kwargs).to(args.device)
    run_name = getattr(args, "wandb_run_name", None) or \
        f"torus_d{args.dim_latent}_iter{args.n_iterations}_lriem{args.lambda_riem}"
    scrvae_trainer = SCRVAETrainer(
        scrvae_model,
        TrainerConfig(
            k_neighbors=args.k_neighbors,
            n_iterations=args.n_iterations,
            n_mstep_epochs=args.n_mstep_epochs,
            learning_rate=args.lr,
            beta_node_kl=args.beta_node_kl,
            lambda_riem=args.lambda_riem,
            beta_edge_kl=args.beta_edge_kl,
            lambda_decorr=args.lambda_decorr,
            reset_optimizer_each_iteration=not getattr(args, "no_reset_optimizer", False),
            distance_clip_factor=getattr(args, "distance_clip_factor", 5.0),
            # Pinned to 0 to reproduce published torus_clifford results.
            # Set to 50 for new experiments (allows rediscovery of dropped edges).
            n_extra_candidates=0,
            log_interval=getattr(args, "log_interval", 100),
            device=args.device,
            use_wandb=_use_wandb,
            wandb_project=getattr(args, "wandb_project", "rieVAE"),
            wandb_run_name=run_name + "_scrvae",
        ),
    )
    t0 = time.time()
    scrvae_trainer.fit(x)
    scrvae_time = time.time() - t0

    scrvae_model.eval()
    with torch.no_grad():
        z_scrvae, _ = scrvae_model.encode_nodes(x)

    iso_scrvae_riem = evaluate_isometry(
        scrvae_model.node_decoder, z_scrvae, params,
        args.major_radius, args.minor_radius, args.n_eval_pairs,
        torch.device(args.device), args.seed + 2,
        use_euclidean_latent=False, manifold=geodesic_manifold,
    )
    iso_scrvae_euclid = evaluate_isometry(
        scrvae_model.node_decoder, z_scrvae, params,
        args.major_radius, args.minor_radius, args.n_eval_pairs,
        torch.device(args.device), args.seed + 4,
        use_euclidean_latent=True, manifold=geodesic_manifold,
    )
    curv_scrvae = evaluate_curvature_torus(
        scrvae_model.node_decoder, z_scrvae,
        scrvae_trainer.edge_index.to(args.device), args.max_triangles,
    )

    print(f"\n  SCR-VAE:")
    print(f"    Euclidean latent:     MAE={iso_scrvae_euclid['mae']:.4f}  "
          f"Spearman={iso_scrvae_euclid.get('spearman', float('nan')):.4f}")
    print(f"    Riemannian pullback:  MAE={iso_scrvae_riem['mae']:.4f}  "
          f"Spearman={iso_scrvae_riem.get('spearman', float('nan')):.4f}")
    print(f"    Curvature kappa_hat:  {curv_scrvae['mean_kappa']:.4f}  "
          f"(theory~{_kappa_theory:.3f}, mean-curv proxy; lower=better)")

    results["scrvae"] = {
        "isometry_euclidean": iso_scrvae_euclid,
        "isometry_riemannian": iso_scrvae_riem,
        "curvature": curv_scrvae,
        "n_iterations_run": len(scrvae_trainer.history),
        "training_time_s": scrvae_time,
        "training_history": scrvae_trainer.history,
    }

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  SUMMARY: Clifford T^2 (K=0 Gaussian, kappa_hat* ~{_kappa_theory:.3f})")
    print("=" * 64)
    print(f"  {'Metric':<30} {'Baseline':>12} {'SCR-VAE':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Euclidean latent MAE':<30} "
          f"{iso_base_euclid['mae']:>12.4f} "
          f"{iso_scrvae_euclid['mae']:>12.4f}")
    print(f"  {'Euclidean latent Spearman':<30} "
          f"{iso_base_euclid.get('spearman', float('nan')):>12.4f} "
          f"{iso_scrvae_euclid.get('spearman', float('nan')):>12.4f}")
    print(f"  {'Riemannian pullback MAE':<30} "
          f"{iso_base_riem['mae']:>12.4f} "
          f"{iso_scrvae_riem['mae']:>12.4f}")
    print(f"  {'Riemannian pullback Spearman':<30} "
          f"{iso_base_riem.get('spearman', float('nan')):>12.4f} "
          f"{iso_scrvae_riem.get('spearman', float('nan')):>12.4f}")
    print(f"  {'Curvature proxy (true K=0)':<30} "
          f"{curv_base['mean_kappa']:>12.4f} "
          f"{curv_scrvae['mean_kappa']:>12.4f}")

    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
