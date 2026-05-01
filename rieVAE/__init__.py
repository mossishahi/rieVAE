"""rieVAE: Certified Riemannian Variational Autoencoder via a Spectral Ambient Premetric.

Phase-3 (op47C C.3) replaces the imperative ``ProximalSCRVAETrainer``
with a PyTorch Lightning training spine:

  * the model is the unified :class:`RiemannianVAE`, parameterised by
    a ``LatentManifold`` plug-in (:mod:`rieVAE.manifolds`) and a
    ``Likelihood`` plug-in (:mod:`rieVAE.likelihoods`);
  * preprocessing is a standalone :class:`SpectralPreprocessor` that
    runs ONCE before training and produces :class:`SpectralArtefacts`
    (CkNN graph, Coifman-Lafon LBO, Varadhan targets, optional PE
    features, optional decoder-independent reweighting);
  * training is driven by ``pytorch_lightning.Trainer`` with a
    :class:`TrainingPlanBase` subclass (Iso / IsoPlusGlobalOrder /
    IsoPlusJVPLegacy / Vanilla);
  * the certificate, post-hoc edge-scale calibration, post-hoc
    PE-aux-head fit, and DDP cert reduction live as Lightning
    callbacks (:mod:`rieVAE.callbacks`);
  * a Hydra YAML config tree + ``python -m rieVAE`` entry point
    drives multi-run ablations.

Theorem references (paper labels):
  - thm:encoder_isometry  : encoder local isometry on E*, runtime-certified
                            by C1' AND C2 AND C3.
  - thm:isometry_main     : decoder pullback isometry, conditional on
                            encoder isometry plus decoder regularity;
                            rate O(r_n^p) with p in {1, 2}.
  - cor:topo_matched      : specialisation p = 2 of the two theorems
                            on topology-matched latents.
  - thm:topo_floor        : Euclidean-LATENT fold floor (architectural).
  - thm:eucl_floor        : Euclidean-graph Omega(1) floor; CkNN +
                            Varadhan circumvents it.
  - thm:minimax           : minimax matching.
  - lem:spec_premetric    : O(1) bi-Lipschitz tilde_w to d^M on E*.
  - lem:lap_convergence   : Coifman-Lafon LBO convergence.
  - thm:impossibility     : O(r_n^2) global accuracy is structurally
                            impossible for poly-K spectral truncation.

Public surface (Phase-3 freeze):

    Model
        RiemannianVAE
        JointEdgeDecoder, ScalarEdgeDecoder
    Manifold registry
        LatentManifold, Euclidean, FlatTorus, Sphere, Hyperbolic,
        StereographicProduct, resolve_manifold
    Likelihood registry
        Likelihood, Gaussian, NegativeBinomial,
        ZeroInflatedNegativeBinomial, Poisson, Bernoulli,
        resolve_likelihood
    Data
        SpectralPreprocessor, SpectralArtefacts, TensorDataModule
    Training plans
        TrainingPlanBase, Term, IsoTrainingPlan,
        IsoPlusGlobalOrderTrainingPlan,
        IsoPlusJVPLegacyTrainingPlan, VanillaTrainingPlan,
        plus schedule helpers (constant, linear_warmup, sigmoid,
        beta_linear_decay, warmup_then_constant)
    Callbacks
        CertificateObserverCallback, PostHocCalibrationCallback,
        PEAuxFitCallback, MultiGPUCertificateReducer
    Certificate
        CertificateReport, CertificateThresholds, compute_certificate
    Spectral premetric
        build_biharmonic_distance, compute_varadhan_edge_distances,
        spectral_ball_edges, pca_local_reweighting

Symbols deleted in Phase 3 (no backward-compat shim):
    ProximalSCRVAETrainer, TrainingConfig.
"""
from rieVAE.model.riemannian_vae import RiemannianVAE
from rieVAE.modules.edge import JointEdgeDecoder, ScalarEdgeDecoder
from rieVAE.manifolds import (
    LatentManifold,
    Euclidean,
    FlatTorus,
    Sphere,
    Hyperbolic,
    StereographicProduct,
    resolve_manifold,
)
from rieVAE.likelihoods import (
    Likelihood,
    Gaussian,
    NegativeBinomial,
    ZeroInflatedNegativeBinomial,
    Poisson,
    Bernoulli,
    resolve_likelihood,
)
from rieVAE.data import (
    SpectralPreprocessor,
    SpectralArtefacts,
    TensorDataModule,
    DatasetModule,
    MmapTensorDataModule,
)
from rieVAE.training import (
    TrainingPlanBase,
    Term,
    constant,
    linear_warmup,
    sigmoid,
    beta_linear_decay,
    warmup_then_constant,
    IsoTrainingPlan,
    IsoPlusGlobalOrderTrainingPlan,
    IsoPlusJVPLegacyTrainingPlan,
    VanillaTrainingPlan,
)
from rieVAE.callbacks import (
    CertificateObserverCallback,
    PostHocCalibrationCallback,
    PEAuxFitCallback,
    MultiGPUCertificateReducer,
)
from rieVAE.evaluate.certificate import (
    CertificateReport,
    CertificateThresholds,
    compute_certificate,
    compute_r_n,
    delta_fold_lower_bound,
    lambda_cross_from,
    alignment_diagnostic,
)
from rieVAE.geometry.spectral_premetric import (
    build_biharmonic_distance,
    compute_varadhan_edge_distances,
    spectral_ball_edges,
    pca_local_reweighting,
)
from rieVAE.geometry.strong_convexity import (
    verify_restricted_sc_condition,
    verify_pl_star_condition,
    verify_sc_condition,
    verify_restricted_sc_output_layer,
    ntk_condition_number,
)


__all__ = [
    # Model
    "RiemannianVAE",
    "JointEdgeDecoder",
    "ScalarEdgeDecoder",
    # Manifold registry
    "LatentManifold",
    "Euclidean",
    "FlatTorus",
    "Sphere",
    "Hyperbolic",
    "StereographicProduct",
    "resolve_manifold",
    # Likelihood registry
    "Likelihood",
    "Gaussian",
    "NegativeBinomial",
    "ZeroInflatedNegativeBinomial",
    "Poisson",
    "Bernoulli",
    "resolve_likelihood",
    # Data
    "SpectralPreprocessor",
    "SpectralArtefacts",
    "TensorDataModule",
    "DatasetModule",
    "MmapTensorDataModule",
    # Training plans
    "TrainingPlanBase",
    "Term",
    "constant",
    "linear_warmup",
    "sigmoid",
    "beta_linear_decay",
    "warmup_then_constant",
    "IsoTrainingPlan",
    "IsoPlusGlobalOrderTrainingPlan",
    "IsoPlusJVPLegacyTrainingPlan",
    "VanillaTrainingPlan",
    # Callbacks
    "CertificateObserverCallback",
    "PostHocCalibrationCallback",
    "PEAuxFitCallback",
    "MultiGPUCertificateReducer",
    # Certificate
    "CertificateReport",
    "CertificateThresholds",
    "compute_certificate",
    "compute_r_n",
    "delta_fold_lower_bound",
    "lambda_cross_from",
    "alignment_diagnostic",
    # Spectral ambient premetric
    "build_biharmonic_distance",
    "compute_varadhan_edge_distances",
    "spectral_ball_edges",
    "pca_local_reweighting",
    # Restricted SC / PL*
    "verify_restricted_sc_condition",
    "verify_pl_star_condition",
    "verify_sc_condition",
    "verify_restricted_sc_output_layer",
    "ntk_condition_number",
]
