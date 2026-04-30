"""Global ordinal loss for the Certified Riemannian VAE.

Motivation
----------
The L_iso loss (on E*) enforces ABSOLUTE metric values locally:
    ||mu_i - mu_j|| -> tilde_w_ij   for (i,j) in E*

This gives a certified local isometry.  However the PE spectral
embedding Psi has a near-perfect global rank ordering (Spearman ~0.99
with d^M) but DIFFERENT absolute scale due to the chord-vs-arc
non-linearity of the heat-kernel formula at fixed t.  Adding a second
L2 global loss using Varadhan values creates conflicting absolute-scale
gradients and was empirically harmful (see global_varadhan experiments).

This module implements a RANK-ONLY global loss that:
  - Enforces ordinal structure: PE-near pairs should be latent-near
    relative to PE-far pairs, for a random global sample of anchors.
  - Imposes NO absolute scale target -- only relative ordering.
  - Is therefore fully compatible with L_iso (no gradient conflict).
  - Directly minimizes 1 - Spearman(latent, Psi), giving global ordinal
    fidelity Spearman(latent, d^M) ~= 0.99 at convergence.

Loss formulation: pairwise RankNet (Burges et al., 2005).
For each anchor i in the batch, k_near PE-nearest nodes are positives
and k_far PE-farthest are negatives. The soft loss is:
    -log sigma(d_lat_far - d_lat_near)
which is zero when far > near by a large margin and grows when the
latent ordering disagrees with the PE ordering.

Usage in ProximalSCRVAETrainer
-------------------------------
When TrainingConfig.use_global_order=True:
  1. __init__ precomputes self.psi_full = phi * lambda^{-alpha} (N, K).
  2. _phase2_iso_step draws a random global batch of B nodes, runs one
     additional encoder forward pass, and calls global_ordinal_loss.
  3. The resulting gradient is ADDED to the main backward before the
     single optimizer step.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def build_psi(
    eigvecs: torch.Tensor,
    eigvals: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Build the heat-kernel-weighted PE matrix Psi = eigvecs * eigvals^{-alpha}.

    Psi[i, k] = phi_k(x_i) / lambda_k^alpha.

    With alpha=0.5 (Berard-Besson-Gallot 1994), pairwise distances
    ||Psi_i - Psi_j|| are bi-Lipschitz equivalent to d^M globally.

    Parameters
    ----------
    eigvecs : (N, K) float tensor -- Laplacian eigenvectors.
    eigvals : (K,)   float tensor -- corresponding eigenvalues (strictly positive).
    alpha   : float              -- spectral exponent (0.5 default).

    Returns
    -------
    psi : (N, K) float tensor.
    """
    weights = eigvals.clamp(min=1e-12).pow(-float(alpha))  # (K,)
    return eigvecs * weights.unsqueeze(0)                  # (N, K)


def global_ordinal_loss(
    mu_batch: torch.Tensor,
    psi_batch: torch.Tensor,
    k_near: int = 5,
    k_far: int = 16,
) -> torch.Tensor:
    """Pairwise RankNet loss enforcing global ordinal fidelity.

    For each anchor node in the batch, identifies PE-near (positive) and
    PE-far (negative) nodes within the same batch, then penalises latent
    orderings that disagree with the PE ordering via the log-sigmoid
    pairwise ranking loss:

        L_rank = -mean_{(i,j+,j-)} log sigma( d_lat(i,j-) - d_lat(i,j+) )

    where j+ is PE-near and j- is PE-far relative to anchor i.

    Properties:
    - Zero gradient when latent ordering perfectly matches PE ordering.
    - Bounded-magnitude gradients (sigmoid saturation).
    - No absolute scale target -- compatible with L_iso.
    - With alpha=0.5 PE weighting and Spearman(Psi, d^M) ~= 0.99:
      driving L_rank -> 0 implies Spearman(latent, d^M) -> 0.99.

    Parameters
    ----------
    mu_batch  : (B, d) encoder posterior means for the sampled global batch.
    psi_batch : (B, K) PE features for the same batch nodes.
    k_near    : int -- PE-nearest nodes per anchor treated as positives.
    k_far     : int -- PE-farthest nodes per anchor treated as negatives.

    Returns
    -------
    Scalar loss tensor. Returns 0.0 when the batch is too small (B < 4).
    """
    B = int(mu_batch.shape[0])
    if B < 4:
        return mu_batch.new_zeros(())

    k_n = min(k_near, B - 2)   # need at least 1 genuine far (not self)
    k_f = min(k_far,  B - 2)
    if k_n < 1 or k_f < 1:
        return mu_batch.new_zeros(())

    # Pairwise distances within the batch.
    # cdist is O(B^2 * K) and O(B^2 * d) -- trivial for B=128, K=50, d=2.
    pe_d  = torch.cdist(psi_batch.float(), psi_batch.float())  # (B, B)
    lat_d = torch.cdist(mu_batch,           mu_batch)          # (B, B)

    # Mask self-pairs: set pe_d diagonal to inf so self sorts to LAST
    # position in ascending argsort and is never chosen as near.
    # For far: since inf always occupies position B-1 (last), we skip
    # the last position and take positions [B-k_f-1 : B-1] to exclude
    # self from the far set. Without this skip, lat_d[i,i]=0 would enter
    # the "far" bucket, making diff = 0 - d_near < 0 (wrong gradient).
    self_mask = torch.eye(B, dtype=torch.bool, device=mu_batch.device)
    pe_d = pe_d.masked_fill(self_mask, float("inf"))

    # Per-anchor PE-sorted node indices.
    pe_sorted = pe_d.argsort(dim=-1)            # (B, B), ascending
    near_cols = pe_sorted[:, :k_n]              # (B, k_n) PE-nearest (no self)
    far_cols  = pe_sorted[:, -(k_f + 1):-1]    # (B, k_f) PE-farthest, skip self at -1

    # Latent distances to near and far sets.
    rows     = torch.arange(B, device=mu_batch.device).unsqueeze(1)
    lat_near = lat_d[rows, near_cols]       # (B, k_n)
    lat_far  = lat_d[rows, far_cols]        # (B, k_f)

    # All pairwise (near, far) differences for each anchor.
    # lat_near.unsqueeze(2) : (B, k_n, 1)
    # lat_far.unsqueeze(1)  : (B, 1,  k_f)
    # diff                  : (B, k_n, k_f)  -- should be > 0 (far > near)
    diff = lat_far.unsqueeze(1) - lat_near.unsqueeze(2)

    # -log sigma(diff): 0 when correct, log(2) when tied, grows when wrong.
    return -F.logsigmoid(diff).mean()
