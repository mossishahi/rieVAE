"""Geometry primitives for the Certified Riemannian VAE.

Phase-1 deletions (op47C C.1.3):
  - ``riemannian_knn_graph`` (graph.py): G-step / pullback-graph rebuilder
    for the pre-R4 self-consistent iteration; the static-graph iso
    architecture does not call it.
  - ``torus_euclidean_knn_graph``, ``torus_riemannian_knn_graph``
    (topo_graph.py): the parallel torus-side variants of the same.
  - ``_deduplicate_edges`` (topo_graph.py): private helper used only
    by the deleted functions.

The two surviving torus utilities ``torus_latent_delta`` and
``torus_geodesic_distance`` are kept as the manifold-distance closure
for the trainer's iso loss and its certificate. The unified
``LatentManifold`` plug-in of Phase 2 absorbs them behind an interface.
"""
from rieVAE.geometry.log_map import riemannian_log_maps_batched, riemannian_distances
from rieVAE.geometry.graph import (
    euclidean_knn_graph,
    euclidean_ball_graph,
    spectral_ball_graph,
    median_neighbor_radius,
    mst_connectivity_radius,
    graph_changed,
    graph_change_fraction,
)
from rieVAE.geometry.spectral_premetric import (
    build_knn_laplacian,
    solve_laplacian_eigenpairs,
    biharmonic_feature_map,
    pairwise_biharmonic_distance,
    biharmonic_candidate_distances,
    spectral_ball_edges,
    pca_local_reweighting,
    build_biharmonic_distance,
    verify_euclidean_pool_no_miss,
    compute_varadhan_edge_distances,
)
from rieVAE.geometry.curvature import (
    curvature_proxy,
    ambient_closure_vectors,
    find_triangles,
    closure_proxy_per_node,
    adaptive_knn_radii,
)
from rieVAE.geometry.anchor_sampler import EpochAnchorSampler
from rieVAE.geometry.topo_graph import (
    torus_latent_delta,
    torus_geodesic_distance,
)
try:
    from rieVAE.geometry.strong_convexity import (
        tangent_covering_matrix,
        estimate_mu0,
        estimate_gradient_variance,
        verify_restricted_sc_condition,
        verify_pl_star_condition,
        verify_sc_condition,
        verify_restricted_sc_output_layer,
        ntk_condition_number,
        adaptive_p_step_budget,
        adaptive_mstep_budget,
    )
except Exception as _sc_err:
    import warnings
    warnings.warn(f"strong_convexity import failed: {_sc_err}")
from rieVAE.geometry.encoder_regularity import (
    activation_bounds,
    encoder_lipschitz_bound,
    encoder_hessian_bound,
    estimate_encoder_regularity,
)
try:
    from rieVAE.geometry.properness import (
        verify_properness,
        check_decoder_properness,
    )
except Exception:
    pass

__all__ = [
    "riemannian_log_maps_batched", "riemannian_distances",
    "euclidean_knn_graph", "euclidean_ball_graph",
    "spectral_ball_graph",
    "median_neighbor_radius", "mst_connectivity_radius",
    "graph_changed", "graph_change_fraction",
    # Spectral ambient premetric
    "build_knn_laplacian",
    "solve_laplacian_eigenpairs",
    "biharmonic_feature_map",
    "pairwise_biharmonic_distance",
    "biharmonic_candidate_distances",
    "spectral_ball_edges",
    "pca_local_reweighting",
    "build_biharmonic_distance",
    "verify_euclidean_pool_no_miss",
    "compute_varadhan_edge_distances",
    # Curvature proxies
    "curvature_proxy", "ambient_closure_vectors", "find_triangles",
    "closure_proxy_per_node", "adaptive_knn_radii",
    # Anchor sampling
    "EpochAnchorSampler",
    # Wrapped-angular distance utilities for the flat-torus latent
    "torus_latent_delta", "torus_geodesic_distance",
    # Strong-convexity witnesses
    "tangent_covering_matrix", "estimate_mu0", "estimate_gradient_variance",
    "verify_restricted_sc_condition",
    "verify_pl_star_condition",
    "verify_sc_condition",
    "verify_restricted_sc_output_layer",
    "ntk_condition_number",
    "adaptive_p_step_budget", "adaptive_mstep_budget",
    # Encoder regularity bounds
    "activation_bounds", "encoder_lipschitz_bound",
    "encoder_hessian_bound", "estimate_encoder_regularity",
]
