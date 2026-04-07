"""Synthetic manifold datasets for validating the SCR-VAE.

Four manifolds with known Riemannian geometry:
  - Sphere S^2: constant positive curvature K = 1/R^2.
  - Flat torus T^2 (Clifford embedding): zero Gaussian curvature K = 0
    everywhere (isometrically embedded in R^4, unlike the standard torus
    in R^3 which has varying nonzero curvature).
  - Standard torus (R^3 embedding): varying curvature, kept for reference.
  - Swiss roll: highly curved 2D manifold in 3D, commonly used in manifold
    learning benchmarks.

All datasets are generated as point clouds in a high-dimensional ambient space
via a random linear projection:
    x_i = A z_i + noise,  A in R^{G x d_intrinsic}
This makes the true Riemannian geometry analytically computable from A.

NOTE: Use flat_torus_clifford() for experiments requiring K=0 validation.
      The old flat_torus() generates the standard embedded torus which has
      NONZERO curvature K(phi) = cos(phi)/[r(R+r*cos(phi))].
"""
from __future__ import annotations

import numpy as np
import torch


def sphere(
    n_points: int,
    radius: float = 1.0,
    ambient_dim: int = 50,
    noise_std: float = 0.01,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Sample points from S^2 (unit sphere) embedded in R^ambient_dim.

    The sphere is parameterized by (theta, phi) and projected into the ambient
    space via a random matrix A in R^{G x 3}.

    Parameters
    ----------
    n_points : int
    radius : float
        Sphere radius R. Gaussian curvature K = 1/R^2.
    ambient_dim : int
        Ambient dimension G.
    noise_std : float
        Isotropic Gaussian noise in ambient space.
    seed : int

    Returns
    -------
    x : (N, G) -- ambient-space data points
    params : (N, 2) -- intrinsic coordinates (theta, phi)
    A : (G, 3) -- embedding matrix (for curvature verification)
    """
    rng = np.random.RandomState(seed)

    # Uniform sampling on S^2: use theta = arccos(1 - 2U) for correct
    # Riemannian volume measure sin(theta) dtheta dphi. Using
    # theta ~ Uniform[0, pi] gives non-uniform density (poles oversampled).
    u = rng.uniform(0, 1, n_points)
    theta = np.arccos(1.0 - 2.0 * u)   # uniform in cos(theta) -> uniform on S^2
    phi = rng.uniform(0, 2 * np.pi, n_points)

    x3d = radius * np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], axis=1)

    A = (rng.randn(ambient_dim, 3) / np.sqrt(3)).astype(np.float32)
    x = (x3d.astype(np.float32) @ A.T).astype(np.float32)
    x += rng.randn(*x.shape).astype(np.float32) * noise_std

    params = np.stack([theta, phi], axis=1)
    return (
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(params.astype(np.float32)),
        A,
    )


def flat_torus(
    n_points: int,
    R: float = 2.0,
    r: float = 1.0,
    ambient_dim: int = 50,
    noise_std: float = 0.01,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Sample points from a torus T^2 embedded in R^ambient_dim.

    The torus has major radius R and minor radius r.
    Gaussian curvature: K(theta, phi) = cos(phi) / (r(R + r cos(phi))),
    which averages to zero (the torus has zero total Gaussian curvature
    by the Gauss-Bonnet theorem, consistent with its trivial Euler characteristic 0).

    Parameters
    ----------
    n_points : int
    R : float  major radius
    r : float  minor radius
    ambient_dim : int
    noise_std : float
    seed : int

    Returns
    -------
    x : (N, G)
    params : (N, 2) -- (theta, phi) angular coordinates
    A : (G, 3)
    """
    rng = np.random.RandomState(seed)

    theta = rng.uniform(0, 2 * np.pi, n_points)
    phi = rng.uniform(0, 2 * np.pi, n_points)

    x3d = np.stack([
        (R + r * np.cos(phi)) * np.cos(theta),
        (R + r * np.cos(phi)) * np.sin(theta),
        r * np.sin(phi),
    ], axis=1).astype(np.float32)

    A = (rng.randn(ambient_dim, 3) / np.sqrt(3)).astype(np.float32)
    x = (x3d @ A.T).astype(np.float32)
    x += rng.randn(*x.shape).astype(np.float32) * noise_std

    params = np.stack([theta, phi], axis=1)
    return (
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(params.astype(np.float32)),
        A,
    )


def flat_torus_clifford(
    n_points: int,
    R: float = 2.0,
    r: float = 1.0,
    ambient_dim: int = 50,
    noise_std: float = 0.01,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Sample points from the CLIFFORD flat torus embedded in R^ambient_dim.

    The Clifford torus is the isometric embedding of the flat torus T^2 into
    R^4 via:
        f(theta, phi) = (R*cos(theta), R*sin(theta), r*cos(phi), r*sin(phi))

    This embedding is GENUINELY FLAT: Gaussian curvature K = 0 everywhere.
    The induced metric is ds^2 = R^2 dtheta^2 + r^2 dphi^2 (constant
    coefficients -- a flat product metric), and the geodesic distance formula
        d = sqrt((R * Delta_theta)^2 + (r * Delta_phi)^2)
    is EXACT for this embedding.

    This is the correct manifold for validating the K=0 test (curvature proxy
    should approach 0). The old flat_torus() function generates the standard
    embedded torus in R^3, which has NONZERO curvature K(phi) =
    cos(phi)/[r(R+r*cos(phi))] varying from +1/3 to -1 for R=2, r=1.

    Parameters
    ----------
    n_points : int
    R : float  major Clifford radius (circle 1 radius)
    r : float  minor Clifford radius (circle 2 radius)
    ambient_dim : int  must be >= 4
    noise_std : float
    seed : int

    Returns
    -------
    x : (N, G) -- ambient-space data points
    params : (N, 2) -- (theta, phi) angular coordinates
    A : (G, 4) -- embedding matrix
    """
    if ambient_dim < 4:
        raise ValueError(
            f"ambient_dim must be >= 4 for Clifford torus, got {ambient_dim}"
        )
    rng = np.random.RandomState(seed)

    theta = rng.uniform(0, 2 * np.pi, n_points)
    phi = rng.uniform(0, 2 * np.pi, n_points)

    # Clifford embedding into R^4: truly flat (K=0 everywhere)
    x4d = np.stack([
        R * np.cos(theta),
        R * np.sin(theta),
        r * np.cos(phi),
        r * np.sin(phi),
    ], axis=1).astype(np.float32)

    # Project into high-dimensional ambient space
    A = (rng.randn(ambient_dim, 4) / np.sqrt(4)).astype(np.float32)
    x = (x4d @ A.T).astype(np.float32)
    x += rng.randn(*x.shape).astype(np.float32) * noise_std

    params = np.stack([theta, phi], axis=1)
    return (
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(params.astype(np.float32)),
        A,
    )


def swiss_roll(
    n_points: int,
    ambient_dim: int = 50,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Sample points from the Swiss roll manifold embedded in R^ambient_dim.

    The Swiss roll is a 2D manifold in R^3: a rolled rectangle [0, 2pi] x [0, H].
    It is commonly used to test manifold learning algorithms because nearby points
    in the embedding space may be far apart on the manifold (adjacent spiral layers).

    Parameters
    ----------
    n_points : int
    ambient_dim : int
    noise_std : float
    seed : int

    Returns
    -------
    x : (N, G)
    params : (N, 2) -- (t, height) unrolled coordinates
    A : (G, 3)
    """
    rng = np.random.RandomState(seed)

    t = 1.5 * np.pi * (1.0 + 2.0 * rng.uniform(0, 1, n_points))
    height = rng.uniform(0, 10.0, n_points)

    x3d = np.stack([
        t * np.cos(t),
        height,
        t * np.sin(t),
    ], axis=1).astype(np.float32)

    x3d = x3d / x3d.std(axis=0, keepdims=True).clip(min=1e-6)

    A = (rng.randn(ambient_dim, 3) / np.sqrt(3)).astype(np.float32)
    x = (x3d @ A.T).astype(np.float32)
    x += rng.randn(*x.shape).astype(np.float32) * noise_std

    params = np.stack([t, height], axis=1)
    return (
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(params.astype(np.float32)),
        A,
    )


def compute_true_geodesic_distances(
    params: torch.Tensor,
    manifold: str,
    R: float = 1.0,
    r_torus: float = 1.0,
    R_torus: float = 2.0,
) -> torch.Tensor:
    """Compute true pairwise geodesic distances for validation.

    Parameters
    ----------
    params : (N, 2)
    manifold : 'sphere' | 'torus'
    R : sphere radius
    r_torus, R_torus : torus radii

    Returns
    -------
    D : (N, N) -- pairwise geodesic distances
    """
    p = params.numpy()
    N = len(p)

    if manifold == "sphere":
        theta, phi = p[:, 0], p[:, 1]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        xyz = np.stack([x, y, z], axis=1)
        dots = np.clip(xyz @ xyz.T, -1.0, 1.0)
        D = R * np.arccos(dots)

    elif manifold in ("torus", "clifford_torus"):
        # Both the standard torus (R^3 embedding) and the Clifford torus (R^4)
        # use the same flat geodesic formula for (theta, phi) coordinates.
        # For the Clifford torus this formula is EXACT (K=0 metric).
        # For the standard torus in R^3 it is an APPROXIMATION (the true metric
        # has position-dependent coefficients (R+r*cos(phi))^2, r^2).
        theta, phi = p[:, 0], p[:, 1]
        dtheta = np.abs(theta[:, None] - theta[None, :])
        dtheta = np.minimum(dtheta, 2 * np.pi - dtheta)
        dphi = np.abs(phi[:, None] - phi[None, :])
        dphi = np.minimum(dphi, 2 * np.pi - dphi)
        D = np.sqrt((R_torus * dtheta) ** 2 + (r_torus * dphi) ** 2)

    else:
        raise ValueError(
            f"Unknown manifold: {manifold!r}. Use 'sphere', 'torus', or 'clifford_torus'."
        )

    return torch.from_numpy(D.astype(np.float32))
