"""Evaluation utilities for the Certified Riemannian VAE."""
from rieVAE.evaluate.certificate import (
    CertificateReport,
    CertificateThresholds,
    compute_certificate,
    compute_r_n,
    delta_fold_lower_bound,
    lambda_cross_from,
    alignment_diagnostic,
)
try:
    from rieVAE.evaluate.isometry import (
        estimate_chart_isometry_residual,
        verify_chart_isometry,
    )
except Exception:
    pass
try:
    from rieVAE.evaluate.latent_distance import (
        latent_distance,
        compute_pairwise_distances,
        latent_distance_path,
    )
except Exception:
    pass
try:
    from rieVAE.evaluate.latent_geometry_report import latent_geometry_report
except Exception:
    pass
try:
    from rieVAE.evaluate.lower_bounds import (
        assouad_lower_bound,
        check_rate_optimality,
        fano_lower_bound,
        isometry_bounds_summary,
        knn_radius,
        minimax_lower_bound,
        theoretical_isometry_floor,
    )
except Exception:
    pass

__all__ = [
    # Certificate (Definition 1 of the paper)
    "CertificateReport",
    "CertificateThresholds",
    "compute_certificate",
    "compute_r_n",
    "delta_fold_lower_bound",
    "lambda_cross_from",
    "alignment_diagnostic",
]
