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
    """Sample points from the STANDARD EMBEDDED torus T^2 in R^ambient_dim.

    The torus has major radius R and minor radius r. It is embedded as the
    standard parametric surface in R^3:
        x = (R + r cos(phi)) cos(theta),  y = (R + r cos(phi)) sin(theta),
        z = r sin(phi)
    then projected into R^ambient_dim via a random matrix A.

    WARNING -- local Gaussian curvature is NOT zero:
        K(theta, phi) = cos(phi) / (r * (R + r*cos(phi)))
    which ranges from +1/(r*(R-r)) (outer equator) to -1/(r*(R+r)) (inner
    equator). For R=2, r=1 this is +1 to -1/3. The TOTAL integral of K is
    zero (Gauss-Bonnet), but the local curvature varies strongly.

    For genuine K=0 validation (flat torus everywhere), use flat_torus_clifford()
    instead. This function is kept for compatibility and for experiments that
    intentionally use the standard embedded torus.

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


def triaxial_ellipsoid(
    n_points: int,
    a: float = 2.0,
    b: float = 1.5,
    c: float = 1.0,
    ambient_dim: int = 50,
    noise_std: float = 0.01,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Sample points from a triaxial ellipsoid E(a,b,c) embedded in R^ambient_dim.

    The ellipsoid is parameterized by spherical angles (theta, phi):
        f(theta, phi) = (a*sin(theta)*cos(phi),
                         b*sin(theta)*sin(phi),
                         c*cos(theta))

    with theta in [0, pi], phi in [0, 2*pi).

    WHY THIS MANIFOLD: This is the minimal test case for
    Theorem thm:fixedpoint(c) (Davis-Kahan frame identification).
    It is the only 2D manifold (besides the sphere) in the paper that
    satisfies BOTH conditions simultaneously:
      - M_bar_theta != 0 (non-periodic, position-dependent Jacobian)
      - delta_k > 0     (non-isotropic: eigenvalue gap = |1/a^2 - 1/b^2| > 0)
    Neither the sphere (delta_k = 0) nor the Clifford torus (M_bar = 0)
    satisfies both conditions; the ellipsoid with a != b != c does.

    Parameters
    ----------
    n_points : int
    a, b, c : float
        Semi-axes. The two principal curvatures at a generic point are
        approximately kappa_1 ~ c/a^2 and kappa_2 ~ c/b^2 (at the poles).
        Use a != b to get delta_k > 0.
    ambient_dim : int
        Ambient dimension G (>= 3).
    noise_std : float
        Isotropic Gaussian noise in ambient space.
    seed : int

    Returns
    -------
    x      : (N, G)   -- ambient-space data points
    params : (N, 2)   -- intrinsic coordinates (theta, phi)
    A      : (G, 3)   -- embedding matrix (random linear map from R^3 to R^G)
    """
    rng = np.random.RandomState(seed)

    # Area-weighted sampling: sample proportional to the area element
    # dA = sqrt(det(g)) dtheta dphi so the density is uniform on the manifold.
    # We use rejection sampling with the upper bound on sqrt(det(g)).
    # Upper bound: sqrt(det(g)) <= a*b*c (tight at the equator for a=b; use 2abc).
    max_area = 2.0 * a * b * c
    accepted_theta = []
    accepted_phi = []
    n_extra = max(n_points * 5, 10000)
    while len(accepted_theta) < n_points:
        theta_c = rng.uniform(0, np.pi, n_extra)
        phi_c = rng.uniform(0, 2 * np.pi, n_extra)
        # Metric tensor components:
        g11 = (a * np.cos(theta_c) * np.cos(phi_c))**2 \
            + (b * np.cos(theta_c) * np.sin(phi_c))**2 \
            + (c * np.sin(theta_c))**2
        g22 = (a * np.sin(theta_c) * np.sin(phi_c))**2 \
            + (b * np.sin(theta_c) * np.cos(phi_c))**2
        g12 = (b**2 - a**2) * np.cos(theta_c) * np.sin(theta_c) \
            * np.sin(phi_c) * np.cos(phi_c)
        det_g = np.maximum(g11 * g22 - g12**2, 0.0)
        area_elem = np.sqrt(det_g)
        u = rng.uniform(0, max_area, n_extra)
        mask = u < area_elem
        accepted_theta.extend(theta_c[mask].tolist())
        accepted_phi.extend(phi_c[mask].tolist())

    theta = np.array(accepted_theta[:n_points], dtype=np.float32)
    phi = np.array(accepted_phi[:n_points], dtype=np.float32)

    x3d = np.stack([
        a * np.sin(theta) * np.cos(phi),
        b * np.sin(theta) * np.sin(phi),
        c * np.cos(theta),
    ], axis=1).astype(np.float32)

    A = (rng.randn(ambient_dim, 3) / np.sqrt(3)).astype(np.float32)
    x = (x3d @ A.T).astype(np.float32)
    x += rng.randn(*x.shape).astype(np.float32) * noise_std

    params = np.stack([theta, phi], axis=1)
    return (
        torch.from_numpy(x),
        torch.from_numpy(params),
        A,
    )


def ellipsoid_metric_tensor(
    params: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> np.ndarray:
    """Compute the Riemannian metric tensor g_{ij} at each parameterization point.

    For f(theta,phi) = (a*sin(theta)*cos(phi), b*sin(theta)*sin(phi), c*cos(theta)):
        g_11 = a^2 cos^2(theta) cos^2(phi) + b^2 cos^2(theta) sin^2(phi) + c^2 sin^2(theta)
        g_12 = (b^2 - a^2) cos(theta) sin(theta) sin(phi) cos(phi)
        g_22 = a^2 sin^2(theta) sin^2(phi) + b^2 sin^2(theta) cos^2(phi)

    Parameters
    ----------
    params : (N, 2) -- (theta, phi) per point
    a, b, c : float -- semi-axes

    Returns
    -------
    G_tensors : (N, 2, 2) -- metric tensor at each point
    """
    theta, phi = params[:, 0], params[:, 1]
    g11 = (a * np.cos(theta) * np.cos(phi))**2 \
        + (b * np.cos(theta) * np.sin(phi))**2 \
        + (c * np.sin(theta))**2
    g22 = (a * np.sin(theta) * np.sin(phi))**2 \
        + (b * np.sin(theta) * np.cos(phi))**2
    g12 = (b**2 - a**2) * np.cos(theta) * np.sin(theta) \
        * np.sin(phi) * np.cos(phi)
    N = len(theta)
    G = np.zeros((N, 2, 2), dtype=np.float32)
    G[:, 0, 0] = g11
    G[:, 0, 1] = g12
    G[:, 1, 0] = g12
    G[:, 1, 1] = g22
    return G


def compute_ellipsoid_geodesic_distances_local(
    params: torch.Tensor,
    a: float,
    b: float,
    c: float,
    max_delta: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute approximate pairwise geodesic distances for close pairs on the ellipsoid.

    For pairs with small angular separation, the geodesic distance is well
    approximated by d_approx = sqrt(Dq^T g_mid Dq) where g_mid is the metric
    tensor at the midpoint. This is accurate to O(r^3) where r = ||Dq||.

    Only pairs with ||(Dtheta, Dphi)|| < max_delta are included (near-neighbor
    approximation; distant pairs use a less accurate formula).

    Parameters
    ----------
    params : (N, 2) -- (theta, phi) per point
    a, b, c : float
    max_delta : float -- max parameter distance to include

    Returns
    -------
    i_idx, j_idx : indices of valid pairs
    distances : approximate geodesic distances
    """
    p = params.numpy()
    N = len(p)

    G_tensors = ellipsoid_metric_tensor(p, a, b, c)  # (N, 2, 2)

    row_i, row_j, dists_list = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            dtheta = p[j, 0] - p[i, 0]
            dphi = p[j, 1] - p[i, 1]
            # Wrap phi difference
            dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
            dq = np.array([dtheta, dphi], dtype=np.float32)
            if np.linalg.norm(dq) > max_delta:
                continue
            g_mid = 0.5 * (G_tensors[i] + G_tensors[j])
            d2 = float(dq @ g_mid @ dq)
            if d2 > 0:
                row_i.append(i)
                row_j.append(j)
                dists_list.append(float(np.sqrt(max(d2, 0.0))))

    return (
        torch.tensor(row_i, dtype=torch.long),
        torch.tensor(row_j, dtype=torch.long),
        torch.tensor(dists_list, dtype=torch.float32),
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
