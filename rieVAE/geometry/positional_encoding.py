"""Heat-kernel-weighted Laplacian positional encoding (optional module).

Given the Phase-1 eigendecomposition (lambda_l, phi_l) of the symmetric
normalised kNN Laplacian (computed by
:mod:`rieVAE.geometry.spectral_premetric`), this module constructs the
per-node positional encoding

    PE(x_i) = s * (phi_1(x_i) / lambda_1^alpha, ..., phi_K(x_i) / lambda_K^alpha)

with heat-kernel weighting alpha = 0.5 by default (see Berard-Besson-
Gallot 1994) and a single global RMS scalar s matching the input
feature magnitude. A single scalar multiplier is an *exact isometry*
of R^K up to a global constant, so bi-Lipschitz equivalence of the
spectral embedding to d^M (Lemma 2) is preserved without any
per-dimension distortion.

The PE enters the training pipeline only when
``TrainingConfig.use_pe = True``; default is False so existing runs
are byte-identical to the pre-PE code. See the PE-augmentation
subsection of sec:method (and its corollary) in the paper for the
theoretical claim that enabling PE tightens the bound
    |G - d|  =  |d^M - ||z_i - z_j|| |  from Omega(1) to O(r_n)
on the raw Euclidean latent-distance proxy.

The one knob that matters for the numerical stability discussion
(sec:method, "PE magnitude handling") is ``alpha``: the heat-kernel
value 0.5 gives a ~30x dynamic range across PE dimensions on the
target manifolds, which is small enough that the global RMS scalar
is sufficient normalisation. The biharmonic value alpha = 1.0
produces ~1000x dynamic range and requires additional per-cluster
normalisation (not currently implemented).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class PEArtifacts:
    """Container for the per-node positional-encoding features.

    Attributes
    ----------
    pe : (N, K) float tensor
        Per-node PE features, already RMS-scaled.
    scale : float
        The global RMS scaling factor applied (for diagnostics).
    alpha : float
        The spectral exponent used (0.5 for heat-kernel, 1.0 for
        biharmonic).
    pe_dim : int
        Equal to K; the second dimension of ``pe``.
    n_used : int
        Equal to N; the number of active nodes PE was computed for.
    """

    pe: torch.Tensor
    scale: float
    alpha: float
    pe_dim: int
    n_used: int

    def to(self, device: torch.device) -> "PEArtifacts":
        return PEArtifacts(
            pe=self.pe.to(device),
            scale=self.scale,
            alpha=self.alpha,
            pe_dim=self.pe_dim,
            n_used=self.n_used,
        )


def canonicalise_eigenvector_signs(phi: torch.Tensor) -> torch.Tensor:
    """Force a canonical sign on each column of ``phi``.

    Laplacian eigenvectors are sign-ambiguous: ARPACK / eigsh can
    return ``phi_l`` or ``-phi_l`` depending on initial-vector and
    graph-ordering conventions, which makes PE features irreproducible
    across machines / versions / reshuffled data. We resolve the
    ambiguity deterministically by requiring the entry of maximum
    absolute value in each column to be non-negative. Ties are
    broken by the lowest-index rule (argmax returns the first
    position of the maximum). See reviewer issue I3.

    This is an isometry of R^{N}, so the downstream pairwise distances
    ``|| Psi(x_i) - Psi(x_j) ||`` are invariant.

    Parameters
    ----------
    phi : (N, K) float tensor
        Eigenvector matrix, columns independent.

    Returns
    -------
    (N, K) float tensor with sign-canonicalised columns.
    """
    if phi.ndim != 2:
        raise ValueError(f"phi must be (N, K), got shape {tuple(phi.shape)}")
    abs_phi = phi.abs()
    # Per-column argmax returns the row with the largest magnitude; if
    # the entry at that row is negative, flip the sign of the whole
    # column. This is a standard convention (e.g. scipy's
    # decomposition.svd uses the same rule on U).
    argmax_rows = abs_phi.argmax(dim=0)                           # (K,)
    col_idx = torch.arange(phi.shape[1], device=phi.device)
    max_entries = phi[argmax_rows, col_idx]                       # (K,)
    signs = torch.where(
        max_entries >= 0.0,
        torch.ones_like(max_entries),
        -torch.ones_like(max_entries),
    )
    return phi * signs.unsqueeze(0)


def compute_pe_features(
    phi: torch.Tensor,
    lambdas: torch.Tensor,
    *,
    alpha: float = 0.5,
    pe_dim: Optional[int] = None,
    x_for_rms: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    canonicalise_signs: bool = True,
) -> PEArtifacts:
    """Compute the heat-kernel-weighted Laplacian PE per node.

    Parameters
    ----------
    phi : (N, K_spec) float tensor
        Eigenvectors of the symmetric normalised Laplacian, as returned
        by :func:`rieVAE.geometry.spectral_premetric.build_biharmonic_distance`
        under the key ``phi`` (or ``eigvecs``).
    lambdas : (K_spec,) float tensor
        Corresponding non-trivial Laplacian eigenvalues, in ascending
        order.  Must be positive (lambda_1 = 0 skipped).
    alpha : float, default 0.5
        Spectral exponent. ``0.5`` gives heat-kernel weighting
        (nearly-isometric embedding of Berard-Besson-Gallot);
        ``1.0`` gives the biharmonic weighting (matches d_bih in
        Phase 1 but produces ~1000x per-dim dynamic range, which
        warrants per-cluster rather than global normalisation).
    pe_dim : int, optional
        Truncation dimension. Default: ``phi.shape[1]``.
        Typical values are 16 - 50.
    x_for_rms : (N, G) float tensor, optional
        If provided, sets the global RMS scalar so that
        ``PE.rms() == x.rms()``. This is the user's original approach
        (equivalent to a single-scalar rescaling; exactly preserves
        bi-Lipschitz).  If omitted, the PE is scaled to unit RMS.
    eps : float
        Numerical floor for the RMS denominator.
    canonicalise_signs : bool, default True
        If True, call :func:`canonicalise_eigenvector_signs` on ``phi``
        first to resolve the sign ambiguity of the Laplacian
        eigendecomposition. Required for PE features to be
        reproducible across machines/versions/graph orderings; see
        reviewer issue I3 and the paper's claim in sec:pe that the
        PE is a sign-free spectral fingerprint.

    Returns
    -------
    PEArtifacts
    """
    if phi.ndim != 2:
        raise ValueError(f"phi must be (N, K_spec), got shape {tuple(phi.shape)}")
    if lambdas.ndim != 1:
        raise ValueError(
            f"lambdas must be (K_spec,), got shape {tuple(lambdas.shape)}"
        )
    if phi.shape[1] != lambdas.shape[0]:
        raise ValueError(
            f"phi has {phi.shape[1]} eigenvectors but lambdas has "
            f"{lambdas.shape[0]} eigenvalues; must match."
        )

    K_spec = phi.shape[1]
    if pe_dim is None:
        pe_dim = K_spec
    pe_dim = int(min(max(pe_dim, 1), K_spec))

    phi_k = phi[:, :pe_dim]
    if canonicalise_signs:
        phi_k = canonicalise_eigenvector_signs(phi_k)
    lambdas_k = lambdas[:pe_dim].clamp_min(eps)

    # Heat-kernel weighting: phi / lambda^alpha.
    weights = lambdas_k.pow(-float(alpha)).unsqueeze(0)           # (1, pe_dim)
    pe_weighted = phi_k * weights                                 # (N, pe_dim)

    # Global RMS scaling: a single positive scalar -> exact isometry
    # of R^{pe_dim} up to a global constant.  Bi-Lipschitz preserved.
    pe_rms = float(pe_weighted.pow(2).mean().sqrt().clamp_min(eps).item())
    if x_for_rms is not None:
        x_rms = float(x_for_rms.pow(2).mean().sqrt().clamp_min(eps).item())
        scale = x_rms / pe_rms
    else:
        scale = 1.0 / pe_rms

    pe_scaled = pe_weighted * float(scale)

    return PEArtifacts(
        pe=pe_scaled.contiguous(),
        scale=float(scale),
        alpha=float(alpha),
        pe_dim=pe_dim,
        n_used=pe_scaled.shape[0],
    )


def resolve_phi_and_lambdas(spec_artefacts: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (phi, lambdas) from the spectral_premetric artefact dict.

    The spec_artefacts dict returned by build_biharmonic_distance may
    use either ``phi`` / ``eigvecs`` and ``lambdas`` / ``eigvals`` as
    key names. This helper accepts both and returns the pair.
    """
    if "phi" in spec_artefacts:
        phi = spec_artefacts["phi"]
    elif "eigvecs" in spec_artefacts:
        phi = spec_artefacts["eigvecs"]
    else:
        raise KeyError(
            "spec_artefacts must contain 'phi' or 'eigvecs' (the "
            "eigenvectors of the Laplacian)."
        )
    if "lambdas" in spec_artefacts:
        lambdas = spec_artefacts["lambdas"]
    elif "eigvals" in spec_artefacts:
        lambdas = spec_artefacts["eigvals"]
    else:
        raise KeyError(
            "spec_artefacts must contain 'lambdas' or 'eigvals' (the "
            "eigenvalues of the Laplacian)."
        )
    if isinstance(phi, np.ndarray):
        phi = torch.from_numpy(phi).float()
    if isinstance(lambdas, np.ndarray):
        lambdas = torch.from_numpy(lambdas).float()
    return phi, lambdas
