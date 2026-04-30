"""Latent-geometry report comparing latent distances to manifold distances.

Given a trained model and a ground-truth manifold-distance matrix
``d_M`` (only available on synthetic datasets), this report quantifies
how well a chosen latent-distance proxy approximates ``d^M``. Each
report returns a dict of eight scalars covering correlation, scale-
adjusted error, distortion, and local kNN trustworthiness.

This is the practical comparison artifact for the "latent distances
you can trust" claim of Section 1; running it on a standard VAE and
on our method side by side on the same data quantifies the gap.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from rieVAE.evaluate.latent_distance import latent_distance


@torch.no_grad()
def latent_geometry_report(
    model,
    x: torch.Tensor,
    d_M: torch.Tensor,
    n_pairs: int = 10000,
    k_local: int = 10,
    distance_mode: str = "euclidean",
    seed: int = 0,
) -> dict:
    """Eight-metric comparison of a latent distance proxy to d^M.

    Parameters
    ----------
    model : trained RiemannianVAE.
    x : (N, G) ambient training samples.
    d_M : (N, N) tensor of ground-truth manifold geodesic distances
        (zero diagonal, symmetric).
    n_pairs : int
        Number of random pairs to sample for the global metrics.
    k_local : int
        kNN size for the local trustworthiness metric.
    distance_mode : str
        One of {'euclidean', 'edge_head', 'jvp', 'jvp_symmetric'};
        the proxy whose fidelity to d^M is being measured.
    seed : int
        Random-pair sampling seed.

    Returns
    -------
    dict with keys:
        eval/latent_pearson, eval/latent_spearman, eval/latent_kendall,
        eval/latent_best_alpha, eval/latent_mae_scaled,
        eval/latent_rmse_scaled, eval/latent_rel_rmse,
        eval/latent_distortion_max, eval/latent_knn_accuracy.
    """
    from scipy.stats import pearsonr, spearmanr, kendalltau

    z, _ = model.encode_nodes(x)
    z = z.detach()
    n = z.shape[0]

    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    if i.size == 0:
        return {"eval/latent_pearson": float("nan")}

    z_i = z[torch.as_tensor(i, dtype=torch.long, device=z.device)]
    z_j = z[torch.as_tensor(j, dtype=torch.long, device=z.device)]
    d_pred = latent_distance(model, z_i, z_j, mode=distance_mode)
    d_pred_np = d_pred.detach().cpu().numpy().astype(np.float64)
    d_true = d_M[i, j].detach().cpu().numpy().astype(np.float64)

    # Pearson, Spearman, Kendall.
    r_p = float(pearsonr(d_pred_np, d_true)[0]) if d_pred_np.size > 1 else float("nan")
    r_s = float(spearmanr(d_pred_np, d_true)[0]) if d_pred_np.size > 1 else float("nan")
    r_k = float(kendalltau(d_pred_np, d_true)[0]) if d_pred_np.size > 1 else float("nan")

    # Best linear scale: alpha = argmin || alpha * d_pred - d_true ||^2.
    denom = float((d_pred_np ** 2).sum())
    alpha = float((d_pred_np * d_true).sum() / max(denom, 1e-12))
    d_scaled = alpha * d_pred_np
    mae = float(np.mean(np.abs(d_scaled - d_true)))
    rmse = float(np.sqrt(np.mean((d_scaled - d_true) ** 2)))
    mean_true = float(d_true.mean())
    rel_rmse = rmse / max(mean_true, 1e-12)

    # Distortion: max ratio over min ratio of (d_pred / d_true).
    ratio = d_pred_np / np.clip(d_true, 1e-12, None)
    pos = ratio[(d_pred_np > 0) & (d_true > 0)]
    if pos.size > 0:
        distortion_max = float(np.max(pos) / max(float(np.min(pos)), 1e-12))
    else:
        distortion_max = float("nan")

    # Local kNN trustworthiness in latent vs manifold.
    knn_acc = _latent_knn_accuracy(z, d_M, k=k_local)

    return {
        "eval/latent_pearson": r_p,
        "eval/latent_spearman": r_s,
        "eval/latent_kendall": r_k,
        "eval/latent_best_alpha": alpha,
        "eval/latent_mae_scaled": mae,
        "eval/latent_rmse_scaled": rmse,
        "eval/latent_rel_rmse": rel_rmse,
        "eval/latent_distortion_max": distortion_max,
        "eval/latent_knn_accuracy": knn_acc,
    }


@torch.no_grad()
def _latent_knn_accuracy(z: torch.Tensor, d_M: torch.Tensor, k: int) -> float:
    """Fraction of latent kNN that are also among the manifold kNN."""
    n = z.shape[0]
    d_lat = torch.cdist(z, z, p=2.0).cpu().numpy()
    d_man = d_M.detach().cpu().numpy()
    knn_lat = np.argsort(d_lat, axis=1)[:, 1:k + 1]
    knn_man = np.argsort(d_man, axis=1)[:, 1:k + 1]
    overlaps = []
    for i in range(n):
        s_lat = set(knn_lat[i].tolist())
        s_man = set(knn_man[i].tolist())
        overlaps.append(len(s_lat & s_man) / k)
    return float(np.mean(overlaps))
