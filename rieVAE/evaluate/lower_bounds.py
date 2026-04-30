"""Minimax Lower Bound Analysis for the SCR-VAE.

Implements the empirical verification of Theorem thm:minimax_lb and
Corollary cor:minimax_optimal from the paper (Section sec:minimax_lb).

The main results:
  - thm:minimax_lb: minimax lower bound Omega(r_n / sqrt(log n)) via Le Cam
  - rem:assouad: exact rate Omega(r_n) via Fano's inequality (no log factor)

The SCR-VAE achieves O(r_n) (Theorem thm:isometry), which exactly matches
the minimax lower bound. This module provides:

1. theoretical_isometry_floor(n, d, k, manifold_params):
   Computes the O(r_n) theoretical prediction for the isometry MAE.

2. check_rate_optimality(empirical_maes, ns, d, k):
   Fits an n^{-1/d} scaling curve to observed MAEs and checks whether
   the SCR-VAE achieves the predicted rate.

3. minimax_lower_bound(n, d, k, c_lb=0.1):
   Returns the lower bound c_lb * r_n / sqrt(log n) from thm:minimax_lb.

4. fano_lower_bound(n, d, k, c_lb_prime=0.05):
   Returns the tighter Fano bound c_lb' * r_n (no log factor).

Note: assouad_lower_bound is kept as an alias for backward compatibility.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. Theoretical predictions from Theorem thm:isometry and thm:minimax_lb
# ---------------------------------------------------------------------------

def knn_radius(n: int, d: int, k: int, log_factor: bool = True) -> float:
    """Compute the theoretical k-NN radius r_n = C_d * (log(n)/n)^{1/d}.

    From Assumption ass:capacity: with k-NN and n samples from a d-dimensional
    manifold, the typical nearest-neighbor radius satisfies
        r_n = O((log n / n)^{1/d}).

    Parameters
    ----------
    n : int
        Number of data points.
    d : int
        Intrinsic manifold dimension.
    k : int
        Number of nearest neighbors.
    log_factor : bool
        If True, include the log(n) factor (tight rate).
        If False, use r_n ~ (k/n)^{1/d} (heuristic rate for fixed k).

    Returns
    -------
    r_n : float
        Theoretical k-NN radius.
    """
    if log_factor:
        return (math.log(n) / n) ** (1.0 / d)
    else:
        return (k / n) ** (1.0 / d)


def theoretical_isometry_floor(
    n: int,
    d: int,
    k: int,
    C1: float = 1.0,
    log_factor: bool = True,
) -> float:
    """Compute the theoretical O(r_n) isometry MAE prediction.

    From Theorem thm:isometry:
        E[|dR* - dM|] <= C1 * r_n  (at the self-consistent fixed point)

    Parameters
    ----------
    n : int
        Number of data points.
    d : int
        Intrinsic manifold dimension.
    k : int
        Number of nearest neighbors.
    C1 : float
        Isometry constant from Theorem thm:isometry.
        Default 1.0 (order-of-magnitude estimate; true value depends on
        manifold geometry via C_Gamma, lambda_0, kappa).
    log_factor : bool
        If True, use r_n = (log n / n)^{1/d} (theoretical).
        If False, use r_n = (k/n)^{1/d} (empirical proxy).

    Returns
    -------
    predicted_mae : float
        Theoretical prediction for isometry MAE.
    """
    r_n = knn_radius(n, d, k, log_factor=log_factor)
    return C1 * r_n


def minimax_lower_bound(
    n: int,
    d: int,
    k: int,
    c_lb: float = 0.1,
) -> float:
    """Compute the Le Cam minimax lower bound from Theorem thm:minimax_lb.

    lower_bound = c_lb * r_n / sqrt(log n)

    This is the information-theoretic lower bound: no estimator based on n
    samples can achieve isometry MAE < c_lb * r_n / sqrt(log n) uniformly
    over the manifold class F.

    Parameters
    ----------
    n : int
    d : int
    k : int
    c_lb : float
        Lower bound constant c_lb = c_bump * c_KL / 2 from the proof.
        Default 0.1 (conservative estimate).

    Returns
    -------
    lower_bound : float
    """
    r_n = knn_radius(n, d, k, log_factor=True)
    return c_lb * r_n / math.sqrt(math.log(n))


def fano_lower_bound(
    n: int,
    d: int,
    k: int,
    c_lb_prime: float = 0.05,
) -> float:
    """Compute the Fano minimax lower bound from Remark rem:assouad.

    lower_bound = c_lb' * r_n  (exact rate, no log factor)

    This is the tighter bound from Fano's inequality with M+1 = Theta(1/r_n^d)
    disjoint bump hypotheses; gives the exact minimax rate Theta(r_n).

    Parameters
    ----------
    n : int
    d : int
    k : int
    c_lb_prime : float
        Fano lower bound constant.

    Returns
    -------
    lower_bound : float
    """
    r_n = knn_radius(n, d, k, log_factor=True)
    return c_lb_prime * r_n


# Backward-compatible alias
assouad_lower_bound = fano_lower_bound


# ---------------------------------------------------------------------------
# 2. Empirical Rate Check: verify SCR-VAE achieves the predicted rate
# ---------------------------------------------------------------------------

def check_rate_optimality(
    empirical_maes: list[float],
    ns: list[int],
    d: int,
    k: int,
    C1_estimate: Optional[float] = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Fit n^{-1/d} scaling to empirical MAEs and verify the predicted rate.

    If the SCR-VAE is optimal, the empirical MAEs should satisfy:
        MAE(n) ≈ C1 * r_n = C1 * (log n / n)^{1/d}

    This function fits C1 via OLS on log-log scale and reports:
      - fitted_C1: the empirical isometry constant
      - fitted_slope: should be ≈ -1/d for optimal rate
      - rate_check: |fitted_slope - (-1/d)| / (1/d) (relative error)
      - is_optimal: True if rate_check < 0.3 (within 30% of predicted)

    Parameters
    ----------
    empirical_maes : list[float]
        Observed isometry MAEs at each n value.
    ns : list[int]
        Data sizes corresponding to each MAE.
    d : int
        Intrinsic manifold dimension.
    k : int
        Number of neighbors.
    C1_estimate : float or None
        If provided, use this as the theoretical C1; else estimate from data.
    verbose : bool

    Returns
    -------
    dict with keys:
      'fitted_C1': float
      'fitted_slope': float
      'theoretical_slope': float  (-1/d)
      'rate_check': float         (relative error from theoretical slope)
      'is_optimal': bool
      'lower_bound_ns': list[float]  (Assouad lower bound at each n)
      'upper_bound_ns': list[float]  (theoretical prediction at each n)
    """
    assert len(empirical_maes) == len(ns), "Need equal number of MAEs and n values"
    ns_arr = np.array(ns, dtype=float)
    maes_arr = np.array(empirical_maes, dtype=float)

    # Log-log regression: log(MAE) = log(C1) + slope * log(r_n)
    # where r_n = (log n / n)^{1/d}, so log(r_n) = (1/d) * (log(log n) - log(n))
    log_rn = np.array([
        (1.0 / d) * (math.log(math.log(n)) - math.log(n)) for n in ns
    ])
    log_mae = np.log(maes_arr)

    # OLS: log_mae = a + b * log_rn (note: if MAE = C1 * r_n, then b = 1)
    A = np.stack([np.ones_like(log_rn), log_rn], axis=1)
    result = np.linalg.lstsq(A, log_mae, rcond=None)
    a_fit, b_fit = result[0]
    C1_fit = math.exp(a_fit)

    # The "slope" relative to n is: d(log MAE)/d(log n) = b * (1/d) * (-1) = -b/d
    slope_wrt_n = -b_fit / d

    # Theoretical slope: -1/d
    theoretical_slope = -1.0 / d
    rate_check = abs(slope_wrt_n - theoretical_slope) / abs(theoretical_slope)

    upper_bounds = [theoretical_isometry_floor(n, d, k, C1=C1_estimate or C1_fit) for n in ns]
    lower_bounds = [assouad_lower_bound(n, d, k) for n in ns]

    is_optimal = rate_check < 0.3  # within 30% of predicted rate

    if verbose:
        print(f"\n[Rate Optimality Check (Cor. minimax_optimal)]")
        print(f"  Theoretical rate: MAE ~ n^{{-1/{d}}}  (slope = {theoretical_slope:.3f})")
        print(f"  Fitted slope:     {slope_wrt_n:.3f}  (C1_fit = {C1_fit:.4f})")
        print(f"  Rate error:       {rate_check*100:.1f}%  "
              f"({'OPTIMAL' if is_optimal else 'SUBOPTIMAL -- may not have converged'})")
        print(f"  Assouad lower bound at n={ns[-1]}: {lower_bounds[-1]:.4f}")
        print(f"  Upper bound (theory) at n={ns[-1]}: {upper_bounds[-1]:.4f}")
        print(f"  Empirical MAE at n={ns[-1]}: {empirical_maes[-1]:.4f}")

    return {
        "fitted_C1": float(C1_fit),
        "fitted_slope": float(slope_wrt_n),
        "theoretical_slope": float(theoretical_slope),
        "rate_check": float(rate_check),
        "is_optimal": bool(is_optimal),
        "lower_bound_ns": lower_bounds,
        "upper_bound_ns": upper_bounds,
    }


# ---------------------------------------------------------------------------
# 3. Convenience: compute all bounds for a single (n, d, k) triple
# ---------------------------------------------------------------------------

def isometry_bounds_summary(
    n: int,
    d: int,
    k: int,
    empirical_mae: Optional[float] = None,
    C1: float = 1.0,
) -> dict[str, float]:
    """Return all isometry bounds for a given experimental setup.

    Parameters
    ----------
    n, d, k : int
        Dataset size, intrinsic dimension, neighborhood size.
    empirical_mae : float or None
        Observed MAE (for gap computation).
    C1 : float
        Isometry constant estimate.

    Returns
    -------
    dict with:
      'r_n'                  : k-NN radius (log n/n)^{1/d}
      'upper_bound'          : C1 * r_n   (Theorem thm:isometry)
      'lecam_lower_bound'    : c_lb * r_n / sqrt(log n)
      'assouad_lower_bound'  : c_lb' * r_n  (exact rate, no log)
      'gap_to_lower_bound'   : empirical_mae - assouad_lower_bound (if given)
      'rate_ratio'           : empirical_mae / upper_bound (if given)
    """
    r_n = knn_radius(n, d, k)
    ub = theoretical_isometry_floor(n, d, k, C1=C1)
    lb_lecam = minimax_lower_bound(n, d, k)
    lb_assouad = assouad_lower_bound(n, d, k)

    result = {
        "r_n": r_n,
        "upper_bound": ub,
        "lecam_lower_bound": lb_lecam,
        "assouad_lower_bound": lb_assouad,
    }

    if empirical_mae is not None:
        result["gap_to_lower_bound"] = empirical_mae - lb_assouad
        result["rate_ratio"] = empirical_mae / (ub + 1e-9)
        result["optimality_ratio"] = empirical_mae / (lb_assouad + 1e-9)

    return result
