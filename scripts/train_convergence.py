"""Graph convergence experiment: validates Proposition (Convergence).

The self-consistent iteration should converge to a fixed point where
G* = KNN(f_theta*). This is measured by the graph change fraction:
    f_t = |E^t △ E^{t-1}| / max(|E^t|, |E^{t-1}|)

Prediction: f_t → 0 as t → ∞.

We run 50 iterations on S^2 (sphere) and T^2 (torus) and track f_t
at each iteration. Convergence is declared when f_t < 0.01 (less than
1% of edges change between consecutive iterations).
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
from rieVAE.data.synthetic import sphere, flat_torus


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifold", choices=["sphere", "torus"], default="sphere")
    p.add_argument("--n_points", type=int, default=3000)
    p.add_argument("--ambient_dim", type=int, default=50)
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--dim_latent", type=int, default=8)
    p.add_argument("--dim_edge", type=int, default=2)
    p.add_argument("--k_neighbors", type=int, default=8)
    p.add_argument("--n_iterations", type=int, default=50)
    p.add_argument("--n_mstep_epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta_node_kl", type=float, default=1e-2)
    p.add_argument("--lambda_riem", type=float, default=10.0)
    p.add_argument("--beta_edge_kl", type=float, default=1e-3)
    p.add_argument("--convergence_threshold", type=float, default=0.01,
                   help="Declare convergence when graph_change_fraction < this.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="runs/convergence")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 64)
    print(f"  Graph Convergence Experiment: {args.manifold.upper()}")
    print(f"  Validates Proposition (Convergence of Self-Consistent Iteration)")
    print("=" * 64)
    print(f"  Max iterations: {args.n_iterations}")
    print(f"  Convergence threshold: {args.convergence_threshold}")

    print("\n[Data] Generating dataset...")
    if args.manifold == "sphere":
        x, params, A = sphere(
            n_points=args.n_points, radius=args.radius,
            ambient_dim=args.ambient_dim, noise_std=args.noise, seed=args.seed,
        )
    else:
        x, params, A = flat_torus(
            n_points=args.n_points, R=2.0, r=1.0,
            ambient_dim=args.ambient_dim, noise_std=args.noise, seed=args.seed,
        )
    print(f"  x: {x.shape}")

    model = SCRVAE(
        dim_features=args.ambient_dim, dim_latent=args.dim_latent,
        dim_edge=args.dim_edge,
        encoder_hidden=(256, 128), decoder_hidden=(128, 256), edge_hidden=(64,),
        dropout=0.05,
    )

    config = TrainerConfig(
        k_neighbors=args.k_neighbors,
        n_iterations=args.n_iterations,
        n_mstep_epochs=args.n_mstep_epochs,
        learning_rate=args.lr,
        beta_node_kl=args.beta_node_kl,
        lambda_riem=args.lambda_riem,
        beta_edge_kl=args.beta_edge_kl,
        lambda_decorr=0.0,
        device=args.device,
    )

    trainer = SCRVAETrainer(model, config)

    t0 = time.time()
    trainer.fit(x.cpu())
    elapsed = time.time() - t0

    fractions = [h["graph_change_fraction"] for h in trainer.history]
    iterations = [h["iteration"] for h in trainer.history]

    converged_at = None
    for i, f in enumerate(fractions):
        if f < args.convergence_threshold and i > 0:
            converged_at = int(iterations[i])
            break

    print("\n" + "=" * 64)
    print("  CONVERGENCE SUMMARY")
    print("=" * 64)
    print(f"  {'Iter':>5}  {'Graph change fraction':>22}  {'Converged?':>12}")
    print(f"  {'-'*44}")
    for it, frac in zip(iterations, fractions):
        conv = "CONVERGED" if frac < args.convergence_threshold else ""
        print(f"  {int(it):>5}  {frac:>22.4f}  {conv:>12}")

    if converged_at is not None:
        print(f"\n  Graph CONVERGED at iteration {converged_at} "
              f"(f < {args.convergence_threshold})")
    else:
        final_frac = fractions[-1]
        print(f"\n  Graph did NOT converge in {args.n_iterations} iterations.")
        print(f"  Final change fraction: {final_frac:.4f}")
        print(f"  Trend: start={fractions[0]:.4f}  "
              f"mid={fractions[len(fractions)//2]:.4f}  "
              f"end={final_frac:.4f}")
        decreasing = fractions[-1] < fractions[0]
        print(f"  Fraction is {'DECREASING' if decreasing else 'NOT DECREASING'} "
              f"{'(convergence direction correct)' if decreasing else '(no convergence trend)'}")

    results = {
        "args": vars(args),
        "convergence_fractions": fractions,
        "iterations": iterations,
        "converged_at": converged_at,
        "converged": converged_at is not None,
        "final_fraction": float(fractions[-1]) if fractions else float("nan"),
        "training_time_s": elapsed,
    }

    out_path = os.path.join(args.out_dir, f"convergence_{args.manifold}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
