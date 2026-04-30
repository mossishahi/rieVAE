"""Runtime properness predicate for the topological-floor theorem.

Theorem thm:topo_floor (App. app:topo of the paper) requires the
composition hat_f := proj_M o f_theta : R^d -> M to be PROPER
(preimages of compact sets are compact). Properness is a global
topological property that cannot in general be verified from finitely
many forward passes; this module provides an EMPIRICAL PROXY that
is necessary, not sufficient:

    On a wide-tail sample z ~ N(0, R^2 I) of the latent prior, the
    decoder image stays within a tubular neighborhood of the training
    data of width bounded by a constant times the data diameter.

A decoder whose image leaves the data tubular neighborhood as ||z||
grows is detected by this check; a decoder that is non-proper in
more exotic ways (e.g., wraps the image in a non-compact cycle with
bounded ambient diameter) might pass this check while still being
non-proper in the topological sense. In practice this check is the
operative runtime witness for the certificate's C7 condition.

Public API:
  - verify_properness(decoder, x_data, ...) -- the thin predicate,
    returns ``(pass, max_distance, data_diameter)``.
  - check_decoder_properness -- the full diagnostic dict used by
    :mod:`rieVAE.evaluate.certificate`. Kept here as the canonical
    implementation; ``certificate.py`` re-exports it for backward
    compatibility.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


@torch.no_grad()
def verify_properness(
    model,
    x_data: torch.Tensor,
    *,
    sample_radius: float = 3.0,
    n_samples: int = 1024,
    tubular_neighborhood_factor: float = 2.0,
    device: Optional[torch.device] = None,
) -> Tuple[bool, float, float]:
    """Minimal properness predicate for use in the certificate.

    Thin wrapper around :func:`check_decoder_properness` returning
    only the three certificate-relevant scalars.

    Parameters
    ----------
    model : nn.Module
        A ``RiemannianVAE`` (or any model exposing ``decode_nodes``
        and ``dim_latent``).
    x_data : (N, G) tensor
        Training data used to compute the reference tubular
        neighborhood diameter.
    sample_radius : float
        Standard deviation of the wide-tail Gaussian probe.
    n_samples : int
        Number of latent samples to evaluate.
    tubular_neighborhood_factor : float
        Multiplier on the data diameter defining the tubular
        neighborhood inside which the decoder image must stay.
    device : torch.device, optional

    Returns
    -------
    tuple
        (is_proper, image_max_distance, data_diameter).
    """
    diag = check_decoder_properness(
        model, x_data,
        sample_radius=sample_radius,
        n_samples=n_samples,
        tubular_neighborhood_factor=tubular_neighborhood_factor,
        device=device,
    )
    return (
        bool(diag["is_proper"]),
        float(diag["image_max_distance"]),
        float(diag["data_diameter"]),
    )


@torch.no_grad()
def check_decoder_properness(
    model,
    x_data: torch.Tensor,
    *,
    sample_radius: float = 3.0,
    n_samples: int = 1024,
    tubular_neighborhood_factor: float = 2.0,
    device: Optional[torch.device] = None,
) -> dict:
    """Empirical proxy for the properness assumption in Thm. thm:topo_floor.

    The topological-floor formula (App. app:topo) requires that
    hat_f := proj_M o f_theta : R^d -> M be PROPER. The proxy here:

      1. Sample z ~ N(0, sample_radius^2 * I) in latent space.
      2. Compute f_theta(z).
      3. Measure the maximum L2 distance from each f_theta(z) to its
         nearest training datum.
      4. Pass iff that maximum is <= tubular_neighborhood_factor *
         (data L2 diameter).

    A failure of this check does NOT prove improperness; passing it
    is the empirical necessary condition the certificate uses.

    Parameters
    ----------
    model : nn.Module
        A ``RiemannianVAE`` (or any model exposing ``decode_nodes``
        and ``dim_latent``).
    x_data : (N, G) tensor
        Training data.
    sample_radius : float
    n_samples : int
    tubular_neighborhood_factor : float
    device : torch.device or None

    Returns
    -------
    dict
        Keys ``is_proper``, ``image_max_distance``, ``data_diameter``.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    x_data = x_data.to(device)
    n_data = x_data.shape[0]
    if n_data == 0:
        return {
            "is_proper": False,
            "image_max_distance": float("inf"),
            "data_diameter": 0.0,
        }

    # Data ambient diameter as the reference scale.
    if n_data <= 4096:
        diff = x_data.unsqueeze(0) - x_data.unsqueeze(1)
        data_diameter = float(diff.norm(dim=-1).max().item())
    else:
        sub = x_data[torch.randperm(n_data, device=device)[:1024]]
        diff = sub.unsqueeze(0) - sub.unsqueeze(1)
        data_diameter = float(diff.norm(dim=-1).max().item())

    # Sample latent from a wide tail.
    d_latent = getattr(model, "dim_latent", None)
    if d_latent is None:
        mu, _ = model.encode_nodes(x_data[:1])
        d_latent = mu.shape[-1]
    z = torch.randn(n_samples, int(d_latent), device=device) * sample_radius
    image = model.decode_nodes(z)

    # Distance from each image to its nearest training datum (chunked).
    chunk = max(1, min(256, n_samples))
    max_dist = 0.0
    for start in range(0, n_samples, chunk):
        sub_image = image[start:start + chunk]
        diffs = sub_image.unsqueeze(1) - x_data.unsqueeze(0)
        dists = diffs.norm(dim=-1)
        nearest = dists.min(dim=-1).values
        max_dist = max(max_dist, float(nearest.max().item()))

    threshold = tubular_neighborhood_factor * data_diameter
    is_proper = max_dist <= threshold
    return {
        "is_proper": bool(is_proper),
        "image_max_distance": float(max_dist),
        "data_diameter": float(data_diameter),
    }
