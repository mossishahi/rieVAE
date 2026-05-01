"""Unified Certified Riemannian VAE (Phase-2 of op47C).

Replaces the pre-Phase-2 ``SCRVAE`` (Euclidean-only) and
``TopoSCRVAE`` (S^1 x S^1 only) classes with a single class
parameterised by a ``LatentManifold`` plug-in and a ``Likelihood``
plug-in. The encoder emits chart coordinates in ``manifold.chart_dim``
(option (ii) of op47C: "general VAE" convention; the manifold
structure is enforced by the loss and the decoder embedding, not by
an architectural projection on the encoder side).

Theorem references:
  - thm:encoder_isometry   : runtime-certified encoder local isometry
                             on E*; mu_phi(x_i) is the chart coordinate
                             in R^{chart_dim}, d_{M_z} is the
                             manifold's geodesic distance applied to
                             chart coordinates (well-defined on R^k).
  - thm:isometry_main      : conditional decoder pullback isometry;
                             the pullback metric is on the chart, with
                             the decoder factoring through
                             ``manifold.embed_for_decoder``.
  - cor:topo_matched       : specialisation p = 2 when M_z matches the
                             topology of M (e.g., FlatTorus for T^2).
  - cor:pe_euclidean       : applies only when ``manifold.name ==
                             "euclidean"``.

Six concrete instantiations referenced in the paper (Section
sec:method, after eq:loss):

    (i)   Euclidean(d) + Gaussian(...)         -- iso default
    (ii)  Euclidean(d, prior='standard') + Gaussian(...)  -- standard VAE ELBO
    (iii) FlatTorus(d, radii) + Gaussian(...)  -- topology-matched (Cor.)
    (iv)  Sphere(d) + Gaussian(...)            -- spherical latent
    (v)   Hyperbolic(d, K) + Gaussian(...)     -- hyperbolic latent
    (vi)  Euclidean(d) + NegativeBinomial(...) -- generic count-data ELBO
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from rieVAE.manifolds import LatentManifold, resolve_manifold
from rieVAE.likelihoods import Likelihood, resolve_likelihood
from rieVAE.modules.encoder import NodeEncoder
from rieVAE.modules.decoder import NodeDecoder
from rieVAE.modules.edge import JointEdgeDecoder, ScalarEdgeDecoder


class RiemannianVAE(nn.Module):
    """Unified manifold-VAE for the Certified Riemannian VAE.

    Parameters
    ----------
    n_features : int
        Ambient dimension G.
    n_latent : int
        Intrinsic latent dimension d. Drives the manifold's chart_dim
        (the encoder's mu / var output dim) and the decoder's input
        dim through ``manifold.decoder_input_dim``.
    latent_manifold : str or LatentManifold
        One of ``"euclidean"`` (default), ``"torus"`` /
        ``"flat_torus"``, ``"sphere"``, ``"hyperbolic"``,
        ``"stereographic_product"``, or a ``LatentManifold`` instance.
        See ``rieVAE.manifolds`` for the registry. Extra keyword
        arguments accepted by the constructor (e.g.,
        ``radii=(2.0, 1.0)`` for FlatTorus or ``curvature=-1.0`` for
        Hyperbolic) can be forwarded via ``manifold_kwargs``.
    likelihood : str or Likelihood
        One of ``"gaussian"`` (default), ``"nb"`` /
        ``"negative_binomial"``, ``"zinb"``, ``"poisson"``,
        ``"bernoulli"``, or a ``Likelihood`` instance. Extra keyword
        arguments via ``likelihood_kwargs``.
    encoder_hidden, decoder_hidden, edge_decoder_hidden :
        Hidden widths for the encoder, the node decoder, and the
        legacy MLP edge head respectively.
    dropout, var_eps, activation : standard MLP hyperparameters.
    use_pe, pe_dim : optional positional-encoding augmentation
        (Section sec:pe). Currently supported only for Euclidean
        latents; if ``manifold.name != "euclidean"`` and ``use_pe`` is
        True we raise.
    encoder_spectral_norm, encoder_spectral_norm_max_lip :
        Optional spectral-norm cap on the encoder body (bounds the
        encoder Lipschitz constant L_phi reported as the runtime
        diagnostic ``cert['L_phi_observed']``).
    edge_decoder_type :
        ``'scalar'`` (default; the iso architecture's scalar edge
        head) or ``'mlp'`` (the legacy JVP-architecture MLP head,
        retained for the Phase-3 IsoPlusJVPLegacyTrainingPlan).
    edge_decoder_w_init :
        Initial value of the scalar edge head's raw parameter
        (passed through softplus so the effective scale starts at
        softplus(w_init)).
    manifold_kwargs, likelihood_kwargs :
        Extra arguments forwarded to the manifold / likelihood
        constructor when they are specified by string.

    Notes
    -----
    Forward output dict:

        mu, var : (B, manifold.chart_dim)
            Posterior parameters in the chart.
        z : (B, manifold.chart_dim)
            Reparameterised latent (= mu in eval mode).
        z_embedded : (B, manifold.decoder_input_dim)
            ``manifold.embed_for_decoder(z)``; the decoder consumes
            this directly.
        decoder_out : (B, n_features * likelihood.n_decoder_outputs_per_feature)
            Raw decoder output; consumed by likelihood.parse(...).
        x_hat : (B, n_features)
            ``likelihood.expected_value(likelihood.parse(decoder_out))``
            -- the per-sample mean / mode used for the C2
            reconstruction residual.
        likelihood_params : dict
            Output of ``likelihood.parse(decoder_out, scale_factor)``;
            consumed by IsoVAELoss for the log-likelihood term.
        l_hat (optional, only when edge_index is passed and the model
            uses the legacy MLP edge head): (E, G) predicted log maps.

    The model exposes a stable Phase-1-compatible alias:

        outputs["mu"]    -- chart-coordinate posterior mean
        outputs["var"]   -- chart-coordinate posterior variance
        outputs["z"]     -- reparameterised chart-coordinate sample
        outputs["x_hat"] -- likelihood-aware expected reconstruction

    so existing ``IsoVAELoss``/trainer code that keys off these names
    accepts the unified model unchanged.
    """

    def __init__(
        self,
        n_features: int,
        n_latent: int,
        latent_manifold: str | LatentManifold = "euclidean",
        likelihood: str | Likelihood = "gaussian",
        encoder_hidden: tuple[int, ...] = (256, 256),
        decoder_hidden: tuple[int, ...] = (256, 256),
        edge_decoder_hidden: tuple[int, ...] = (256, 256),
        dropout: float = 0.05,
        var_eps: float = 1e-5,
        activation: str = "silu",
        use_pe: bool = False,
        pe_dim: Optional[int] = None,
        encoder_spectral_norm: bool = False,
        encoder_spectral_norm_max_lip: float = 1.0,
        edge_decoder_type: str = "scalar",
        edge_decoder_w_init: float = 0.0,
        manifold_kwargs: Optional[dict] = None,
        likelihood_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()

        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if n_latent < 1:
            raise ValueError(f"n_latent must be >= 1, got {n_latent}")

        # Resolve plug-ins.
        self.manifold = resolve_manifold(
            latent_manifold,
            n_latent=n_latent,
            **(manifold_kwargs or {}),
        )
        self.likelihood = resolve_likelihood(
            likelihood,
            n_features=n_features,
            **(likelihood_kwargs or {}),
        )

        # Remember user-facing fields.
        self.n_features = int(n_features)
        self.n_latent = int(n_latent)
        # dim_features / dim_latent kept as legacy aliases consumed by
        # the trainer's r_n estimate and certificate.
        self.dim_features = int(n_features)
        self.dim_latent = int(self.manifold.dim)
        self.activation = str(activation)
        self.edge_decoder_type = str(edge_decoder_type)
        self.use_pe = bool(use_pe)
        self.pe_dim = int(pe_dim) if (use_pe and pe_dim is not None) else 0

        # Validate PE compatibility: cor:pe_euclidean is silent on
        # non-Euclidean latents, so PE only fires for Euclidean.
        if self.use_pe and self.manifold.name != "euclidean":
            raise ValueError(
                f"use_pe=True is supported only for Euclidean latents "
                f"(Cor. cor:pe_euclidean is silent on non-Euclidean "
                f"latents); got manifold={self.manifold.name!r}."
            )
        if self.use_pe and self.pe_dim < 1:
            raise ValueError(
                "RiemannianVAE(use_pe=True) requires pe_dim >= 1; "
                f"got pe_dim={pe_dim}."
            )

        # Encoder: emits chart coordinates in (B, manifold.chart_dim).
        # Note: NodeEncoder produces both mu and var heads.
        self.node_encoder = NodeEncoder(
            dim_in=int(n_features),
            dim_latent=int(self.manifold.chart_dim),
            hidden_dims=encoder_hidden,
            dropout=dropout,
            var_eps=var_eps,
            activation=activation,
            use_pe=self.use_pe,
            pe_dim=self.pe_dim if self.use_pe else None,
            encoder_spectral_norm=bool(encoder_spectral_norm),
            encoder_spectral_norm_max_lip=float(encoder_spectral_norm_max_lip),
        )

        # Decoder: input dim = manifold.decoder_input_dim,
        # output dim = n_features * likelihood.n_decoder_outputs_per_feature.
        self.node_decoder = NodeDecoder(
            dim_latent=int(self.manifold.decoder_input_dim),
            dim_out=int(self.n_features * self.likelihood.n_decoder_outputs_per_feature),
            hidden_dims=decoder_hidden,
            dropout=0.0,
            activation=activation,
        )

        # Edge head. Two options:
        #   - 'scalar' (default, iso architecture): single learnable
        #     scalar w; F_phi(z_i, z_j) = softplus(w) * d_{M_z}(z_i, z_j).
        #     The latent_distance_fn IS the manifold's distance, so
        #     C1' (delta_iso) and the iso loss measure the same object
        #     by construction (op47C bug B.3).
        #   - 'mlp' (legacy): vector-output MLP for the JVP architecture,
        #     retained for Phase-3 IsoPlusJVPLegacyTrainingPlan.
        if self.edge_decoder_type == "scalar":
            self.edge_decoder = ScalarEdgeDecoder(
                w_init=float(edge_decoder_w_init),
                latent_distance_fn=self.manifold.distance,
            )
        elif self.edge_decoder_type == "mlp":
            self.edge_decoder = JointEdgeDecoder(
                dim_latent=int(self.manifold.chart_dim),
                dim_out=int(n_features),
                hidden_dims=edge_decoder_hidden,
                dropout=0.0,
                activation=activation,
            )
        else:
            raise ValueError(
                f"Unknown edge_decoder_type {edge_decoder_type!r}; "
                "expected 'scalar' or 'mlp'."
            )

        # Optional aux PE head A_psi (Cor. cor:pe_euclidean).
        if self.use_pe:
            raw = nn.Linear(int(self.manifold.chart_dim), self.pe_dim, bias=True)
            self.aux_pe_head = nn.utils.parametrizations.spectral_norm(raw)
        else:
            self.aux_pe_head = None

    # ------------------------------------------------------------------ Nodes

    def encode_nodes(
        self,
        x: torch.Tensor,
        pe_feat: Optional[torch.Tensor] = None,
        alpha_pe: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode node features to posterior parameters (mu, var) in the
        manifold's chart (R^{chart_dim}).

        For non-Euclidean manifolds, pe_feat / alpha_pe are ignored
        (use_pe is forced False at construction).
        """
        if self.use_pe:
            return self.node_encoder(x, pe_feat=pe_feat, alpha_pe=alpha_pe)
        return self.node_encoder(x)

    def decode_nodes(self, z: torch.Tensor) -> torch.Tensor:
        """Decode chart coordinates z to the likelihood's mean (i.e.
        the per-sample expected value E[X | params]).

        Mirrors the Phase-1 ``SCRVAE.decode_nodes`` contract: returns
        a tensor of shape (B, n_features). For likelihoods with extra
        decoder channels (NB / ZINB), we still return E[X] here; full
        likelihood parameters are exposed via ``forward()`` and
        ``decode_likelihood_params``.
        """
        z_embedded = self.manifold.embed_for_decoder(z)
        decoder_out = self.node_decoder(z_embedded)
        params = self.likelihood.parse(decoder_out, scale_factor=None)
        return self.likelihood.expected_value(params)

    def decode_likelihood_params(
        self,
        z: torch.Tensor,
        scale_factor: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Decode chart coordinates z to a dict of likelihood parameters.

        This is what callers should use when they need the full
        parameterisation (e.g., to compute log_prob or sample): the
        ``decode_nodes`` shortcut only returns E[X | params].
        """
        z_embedded = self.manifold.embed_for_decoder(z)
        decoder_out = self.node_decoder(z_embedded)
        return self.likelihood.parse(decoder_out, scale_factor=scale_factor)

    def decode_pe(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_pe or self.aux_pe_head is None:
            raise RuntimeError(
                "decode_pe() called on RiemannianVAE(use_pe=False); "
                "gate on self.use_pe before calling."
            )
        return self.aux_pe_head(z)

    # ------------------------------------------------------------------ Edges

    def predict_log_maps(
        self,
        mu_node: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Predict Riemannian log maps via F_phi(z_i, z_j).

        Shape:
          - 'mlp' edge head: returns (E, n_features) predicted log maps.
          - 'scalar' edge head: returns (E,) predicted scalar distances
            equal to ``softplus(w) * d_{M_z}(mu_i, mu_j)``.
        """
        src, dst = edge_index[0], edge_index[1]
        z_src = mu_node[src]
        z_dst = mu_node[dst]
        return self.edge_decoder(z_src, z_dst)

    def predict_edge_distances(
        self,
        mu_node: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Predict scalar distances per edge via the scalar edge head.

        Requires ``edge_decoder_type='scalar'``; raises otherwise.
        """
        if self.edge_decoder_type != "scalar":
            raise RuntimeError(
                "predict_edge_distances() requires edge_decoder_type="
                f"'scalar'; got {self.edge_decoder_type!r}."
            )
        src, dst = edge_index[0], edge_index[1]
        z_src = mu_node[src]
        z_dst = mu_node[dst]
        return self.edge_decoder(z_src, z_dst)

    # ------------------------------------------------------------------ Forward

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        pe_feat: Optional[torch.Tensor] = None,
        alpha_pe: float = 1.0,
        scale_factor: Optional[torch.Tensor] = None,
    ) -> dict:
        """Full forward pass.

        Parameters
        ----------
        x : (B, n_features)
            Input features.
        edge_index : optional (2, E) directed edges; if given AND the
            model uses the legacy MLP edge head, the predicted log
            maps are included in the output dict under ``l_hat``.
        pe_feat : optional (B, pe_dim) PE features. Required iff
            ``use_pe=True``.
        alpha_pe : PE gate scalar. Ignored when ``use_pe=False``.
        scale_factor : optional (B,) per-sample GLM offset / exposure
            for count likelihoods. Ignored by likelihoods that do not
            consume it (Gaussian, Bernoulli).
        """
        mu, var = self.encode_nodes(x, pe_feat=pe_feat, alpha_pe=alpha_pe)
        if self.training:
            z = self.manifold.reparameterize(mu, var)
        else:
            z = mu
        z_embedded = self.manifold.embed_for_decoder(z)
        decoder_out = self.node_decoder(z_embedded)
        likelihood_params = self.likelihood.parse(
            decoder_out, scale_factor=scale_factor,
        )
        x_hat = self.likelihood.expected_value(likelihood_params)

        out = {
            "mu":                mu,
            "var":               var,
            "z":                 z,
            "z_embedded":        z_embedded,
            "decoder_out":       decoder_out,
            "x_hat":             x_hat,
            "likelihood_params": likelihood_params,
        }
        if edge_index is not None and self.edge_decoder_type == "mlp":
            out["l_hat"] = self.predict_log_maps(mu, edge_index)
        if self.use_pe:
            out["pe_hat"] = self.decode_pe(z)
        return out

    # ------------------------------------------------------------------ Parameter groups

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Split parameters by role for fine-grained optimiser scheduling."""
        groups = {
            "encoder":      list(self.node_encoder.parameters()),
            "decoder":      list(self.node_decoder.parameters()),
            "edge_decoder": list(self.edge_decoder.parameters()),
            "manifold":     list(self.manifold.parameters()),
            "likelihood":   list(self.likelihood.parameters()),
        }
        if self.use_pe and self.aux_pe_head is not None:
            groups["aux_pe_head"] = list(self.aux_pe_head.parameters())
        return groups

    # ==================================================================
    # Phase-4 sklearn-style user-facing API (op47C C.4.1)
    # ==================================================================
    #
    # The methods below provide a few-line user-facing surface
    # (``model.fit(x); z = model.encode(x); cert = model.certificate(x)``)
    # on top of the Phase-3 Lightning core. They are an ergonomic
    # convenience: the underlying training plan + data module + Lightning
    # Trainer + callbacks are all still available for users who want
    # full control (Section D.2 of op47C).
    #
    # The first call to ``fit`` populates the ``_artefacts``,
    # ``_certificate_history``, and ``_calibrated_scale`` caches; the
    # post-training accessors (``encode``, ``decode``, ``certificate``,
    # ``geodesic_distance``) read these caches when appropriate.

    def fit(
        self,
        x,
        *,
        max_steps: int = 50_000,
        plan="iso",
        plan_kwargs: Optional[dict] = None,
        anchor_batch_size: int = 512,
        n_steps_per_epoch: int = 1000,
        max_epochs: Optional[int] = None,
        accelerator: str = "auto",
        devices=1,
        strategy=None,
        cert_every_n_steps: int = 500,
        cert_subsample: int = 2048,
        cert_pullback_nodes: int = 32,
        chart_regime: str = "general",
        post_hoc_calibration: bool = True,
        pe_posthoc_steps: int = 2000,
        pe_posthoc_lr: float = 1.0e-3,
        pe_posthoc_batch: int = 512,
        multi_gpu_cert_reduce: bool = True,
        preprocess_kwargs: Optional[dict] = None,
        seed: int = 0,
        log_every_n_steps: int = 50,
        enable_progress_bar: bool = True,
        callbacks: Optional[list] = None,
    ) -> "RiemannianVAE":
        """Train the model end-to-end on ``x``.

        Sklearn-style convenience that wraps:

          1. :class:`rieVAE.data.SpectralPreprocessor.fit(x)` -- one-shot
             CkNN graph + Coifman-Lafon LBO + Varadhan target build.
          2. :class:`rieVAE.data.TensorDataModule` -- anchor-sampled
             subgraph dataloader.
          3. The chosen :class:`rieVAE.training.TrainingPlanBase` plan
             (``"iso"`` by default).
          4. ``pytorch_lightning.Trainer.fit(plan, datamodule)`` with
             :class:`CertificateObserverCallback`,
             :class:`PostHocCalibrationCallback`,
             :class:`PEAuxFitCallback` (when ``self.use_pe``), and
             :class:`MultiGPUCertificateReducer`.

        After ``fit`` returns, the model's encoder weights are
        post-hoc-calibrated and the certificate history is available
        via ``self.get_certificate_history()`` / ``self.certificate(x)``.

        Parameters
        ----------
        x : torch.Tensor of shape (N, n_features)
            Ambient features. Outliers are filtered by the
            preprocessor; the model trains on the surviving
            ``n_active <= N`` samples.
        max_steps : int
        plan : str or TrainingPlanBase or class
            ``"iso"`` (default), ``"iso_plus_global_order"`` /
            ``"iso_plus_rank"``, ``"iso_plus_jvp_legacy"``,
            ``"vanilla"``, or a TrainingPlanBase instance / subclass.
            Strings are dispatched against the registry.
        plan_kwargs : dict
            Forwarded to the plan constructor when ``plan`` is a
            string or a class.
        anchor_batch_size, n_steps_per_epoch, max_epochs :
            Sampler / Lightning epoch knobs.
        accelerator, devices, strategy : forwarded to ``pl.Trainer``.
        cert_every_n_steps, cert_subsample, cert_pullback_nodes,
        chart_regime :
            CertificateObserverCallback knobs.
        post_hoc_calibration, pe_posthoc_steps, pe_posthoc_lr,
        pe_posthoc_batch :
            Post-hoc calibration / PE-aux-head knobs.
        multi_gpu_cert_reduce : bool
            Whether to attach the DDP cert reducer (no-op
            single-process). Default True.
        preprocess_kwargs : dict
            Forwarded to ``SpectralPreprocessor(**...)``. The
            preprocessor's defaults are sensible for n <= 1e4; for
            larger n switch to a faster ``GraphBuilder``
            (cf. op47C C.4.3) by passing
            ``preprocess_kwargs={'graph_builder': 'faiss', ...}``.
        seed : int
        log_every_n_steps, enable_progress_bar :
            Forwarded to ``pl.Trainer``.
        callbacks : list[pl.Callback] or None
            Extra Lightning callbacks to install in addition to the
            default cert / calibration / PE / DDP-reduce callbacks.

        Returns
        -------
        self
        """
        try:
            import pytorch_lightning as pl
        except ImportError as e:
            raise ImportError(
                "RiemannianVAE.fit requires pytorch_lightning. Install it via "
                "`pip install pytorch-lightning`."
            ) from e

        from rieVAE.data import SpectralPreprocessor, TensorDataModule
        from rieVAE.callbacks import (
            CertificateObserverCallback,
            PostHocCalibrationCallback,
            PEAuxFitCallback,
            MultiGPUCertificateReducer,
        )
        from rieVAE.training import (
            TrainingPlanBase,
            IsoTrainingPlan,
            IsoPlusGlobalOrderTrainingPlan,
            IsoPlusJVPLegacyTrainingPlan,
            VanillaTrainingPlan,
        )

        pl.seed_everything(int(seed), workers=True)

        # ---- 1) Preprocess once. -------------------------------------
        pre_kwargs = dict(preprocess_kwargs or {})
        pre_kwargs.setdefault("use_pe", self.use_pe)
        if self.use_pe:
            pre_kwargs.setdefault("pe_dim", self.pe_dim)
        # Default 'use_global_order=True' if the plan needs psi_full.
        plan_needs_global_order = (
            isinstance(plan, str)
            and plan in ("iso_plus_global_order", "iso_plus_rank")
        ) or (
            isinstance(plan, type)
            and issubclass(plan, IsoPlusGlobalOrderTrainingPlan)
        ) or isinstance(plan, IsoPlusGlobalOrderTrainingPlan)
        if plan_needs_global_order:
            pre_kwargs.setdefault("use_global_order", True)
        preprocessor = SpectralPreprocessor(**pre_kwargs)
        artefacts = preprocessor.fit(x)
        self._artefacts = artefacts

        # ---- 2) Data module. -----------------------------------------
        dm = TensorDataModule(
            x=x,
            artefacts=artefacts,
            anchor_batch_size=int(anchor_batch_size),
            n_steps_per_epoch=int(n_steps_per_epoch),
            seed=int(seed),
        )

        # ---- 3) Plan. -----------------------------------------------
        plan_kwargs = dict(plan_kwargs or {})
        plan_kwargs.setdefault("max_steps", int(max_steps))
        if isinstance(plan, str):
            registry = {
                "iso":                  IsoTrainingPlan,
                "iso_plus_global_order": IsoPlusGlobalOrderTrainingPlan,
                "iso_plus_rank":        IsoPlusGlobalOrderTrainingPlan,  # alias
                "iso_plus_jvp_legacy":  IsoPlusJVPLegacyTrainingPlan,
                "vanilla":              VanillaTrainingPlan,
            }
            key = plan.lower().strip()
            if key not in registry:
                raise ValueError(
                    f"Unknown plan {plan!r}; expected one of "
                    f"{sorted(registry)} or a TrainingPlanBase instance."
                )
            plan_cls = registry[key]
            plan_obj = plan_cls(model=self, **plan_kwargs)
        elif isinstance(plan, type) and issubclass(plan, TrainingPlanBase):
            plan_obj = plan(model=self, **plan_kwargs)
        elif isinstance(plan, TrainingPlanBase):
            plan_obj = plan
        else:
            raise TypeError(
                f"plan must be a string, a TrainingPlanBase subclass, or "
                f"an instance; got {type(plan).__name__}."
            )

        # ---- 4) Callbacks. ------------------------------------------
        cert_cb = CertificateObserverCallback(
            every_n_steps=int(cert_every_n_steps),
            cert_subsample=int(cert_subsample),
            cert_pullback_nodes=int(cert_pullback_nodes),
            force_global_at_end=True,
            chart_regime=str(chart_regime),
            activation=str(self.activation),
        )
        cb_list: list[pl.Callback] = [cert_cb]
        if post_hoc_calibration:
            cb_list.append(PostHocCalibrationCallback())
        if self.use_pe:
            cb_list.append(PEAuxFitCallback(
                n_steps=int(pe_posthoc_steps),
                lr=float(pe_posthoc_lr),
                batch_size=int(pe_posthoc_batch),
            ))
        if multi_gpu_cert_reduce:
            cb_list.append(MultiGPUCertificateReducer(cert_callback=cert_cb))
        if callbacks:
            cb_list.extend(list(callbacks))

        # ---- 5) Trainer + fit. --------------------------------------
        n_active = int(artefacts.n_active)
        if max_epochs is None:
            max_epochs = max(1, math.ceil(int(max_steps) / max(int(n_steps_per_epoch), 1)))
        trainer = pl.Trainer(
            max_epochs=int(max_epochs),
            max_steps=int(max_steps),
            accelerator=str(accelerator),
            devices=devices,
            strategy=("auto" if strategy is None else strategy),
            log_every_n_steps=int(log_every_n_steps),
            enable_progress_bar=bool(enable_progress_bar),
            callbacks=cb_list,
        )
        trainer.fit(plan_obj, datamodule=dm)

        # ---- 6) Cache results for subsequent .certificate(),
        #         .save(), .encode(), etc.
        #
        # Critical: ``plan_obj`` is a LightningModule that holds
        # ``self.model = self`` as a submodule, and ``dm`` is a
        # LightningDataModule (also nn.Module). Assigning them via
        # plain attribute syntax would register them as submodules of
        # ``self`` -- creating a cycle ``self -> plan -> self -> ...``
        # that explodes ``self.eval()`` with infinite recursion. We
        # bypass nn.Module.__setattr__ via ``object.__setattr__``.
        # The list of cert dicts is a plain Python list and is safe.
        object.__setattr__(self, "_fitted_trainer", trainer)
        object.__setattr__(self, "_fitted_plan", plan_obj)
        object.__setattr__(self, "_fitted_datamodule", dm)
        self._certificate_history = list(cert_cb.history)
        return self

    @torch.no_grad()
    def encode(
        self,
        x,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        return_var: bool = False,
    ) -> torch.Tensor:
        """Encode ``x`` to chart-coordinate posterior means.

        Parameters
        ----------
        x : torch.Tensor of shape (N, n_features)
        batch_size : optional batch size; None = single forward pass.
        device : optional torch device override.
        return_var : if True, also return the posterior variance.

        Returns
        -------
        mu : (N, manifold.chart_dim)
        var : (N, manifold.chart_dim), only if ``return_var`` is True.
        """
        device = device or next(self.parameters()).device
        x_t = torch.as_tensor(x).to(device)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        was_training = self.training
        self.eval()
        try:
            pe_feat = (
                self._artefacts.pe_feat.to(device, x_t.dtype)
                if (self.use_pe and getattr(self, "_artefacts", None) is not None
                    and self._artefacts.pe_feat is not None)
                else None
            )
            # When PE is on but no fit was done yet, encoding is
            # still possible (PE was wired through the encoder), but
            # callers must supply pe_feat themselves -- we defer to
            # the encoder's None handling.
            if batch_size is None or batch_size >= x_t.shape[0]:
                mu, var = self.encode_nodes(x_t, pe_feat=pe_feat, alpha_pe=1.0)
            else:
                mus, vars_ = [], []
                for i in range(0, x_t.shape[0], int(batch_size)):
                    sl = slice(i, i + int(batch_size))
                    pe_b = pe_feat[sl] if pe_feat is not None else None
                    mu_b, var_b = self.encode_nodes(x_t[sl], pe_feat=pe_b, alpha_pe=1.0)
                    mus.append(mu_b)
                    vars_.append(var_b)
                mu  = torch.cat(mus, dim=0)
                var = torch.cat(vars_, dim=0)
        finally:
            self.train(was_training)
        if return_var:
            return mu, var
        return mu

    @torch.no_grad()
    def decode(
        self,
        z,
        scale_factor: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Decode chart-coordinate latents to the likelihood's expected
        value E[X | params]."""
        device = device or next(self.parameters()).device
        z_t = torch.as_tensor(z).to(device)
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        was_training = self.training
        self.eval()
        try:
            params = self.decode_likelihood_params(z_t, scale_factor=scale_factor)
            x_hat = self.likelihood.expected_value(params)
        finally:
            self.train(was_training)
        return x_hat

    def reconstruct(
        self,
        x,
        batch_size: Optional[int] = None,
        scale_factor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Round-trip ``decode(encode(x))``. Returns E[X | mu_phi(x_i)]."""
        mu = self.encode(x, batch_size=batch_size)
        return self.decode(mu, scale_factor=scale_factor)

    @torch.no_grad()
    def geodesic_distance(
        self,
        z_a,
        z_b,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Manifold geodesic distance between batches of chart coordinates.

        Parameters
        ----------
        z_a, z_b : (..., manifold.chart_dim) torch.Tensor

        Returns
        -------
        d : (...,) non-negative geodesic distances on the latent manifold.
        """
        device = device or next(self.parameters()).device
        z_a_t = torch.as_tensor(z_a).to(device)
        z_b_t = torch.as_tensor(z_b).to(device)
        return self.manifold.distance(z_a_t, z_b_t)

    def certificate(
        self,
        x=None,
        *,
        cert_subsample: Optional[int] = None,
        cert_pullback_nodes: int = 32,
        chart_regime: str = "general",
        force_global: bool = True,
    ) -> dict:
        """Compute the runtime certificate now.

        When ``x`` is None and ``fit()`` has been called, evaluates the
        certificate on the cached preprocessor artefacts. When ``x`` is
        supplied, runs a fresh preprocessor pass on ``x`` and evaluates
        on its artefacts. Returns the certificate dict (the same
        schema as :class:`CertificateObserverCallback.history` items).
        """
        from rieVAE.callbacks._certificate_compute import compute_global_certificate
        if x is not None:
            from rieVAE.data import SpectralPreprocessor
            preprocessor = SpectralPreprocessor(use_pe=self.use_pe, pe_dim=self.pe_dim if self.use_pe else None)
            artefacts = preprocessor.fit(x)
        else:
            artefacts = getattr(self, "_artefacts", None)
            if artefacts is None:
                raise RuntimeError(
                    "RiemannianVAE.certificate() called without ``x`` "
                    "before ``fit`` was run; either fit the model first "
                    "or supply ``x`` explicitly."
                )
        cert = compute_global_certificate(
            model=self,
            artefacts=artefacts,
            cert_subsample=cert_subsample,
            cert_pullback_nodes=cert_pullback_nodes,
            force_global=bool(force_global),
            gamma_t=0.0,
            chart_regime=str(chart_regime),
            activation=str(self.activation),
        )
        return cert

    def get_certificate_history(self) -> list[dict]:
        """Return the per-checkpoint certificate dicts from the most
        recent ``fit`` call (mid-training cert evaluations + final)."""
        return list(getattr(self, "_certificate_history", []))

    def save(self, path) -> None:
        """Save model state + manifold/likelihood specs + cert history.

        Saves a single ``.pt`` (or ``.ckpt``) checkpoint with:
          - ``state_dict``       : the model weights
          - ``constructor_args`` : the kwargs needed to rebuild the model
          - ``manifold``         : the manifold's class name + repr
          - ``likelihood``       : the likelihood's class name + repr
          - ``certificate_history`` : the cached cert history (if any)

        Use :meth:`RiemannianVAE.load` to restore.
        """
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": self.state_dict(),
            "constructor_args": {
                "n_features":         int(self.n_features),
                "n_latent":           int(self.n_latent),
                "manifold_name":      str(self.manifold.name),
                "likelihood_name":    str(self.likelihood.name),
                "use_pe":             bool(self.use_pe),
                "pe_dim":             int(self.pe_dim) if self.use_pe else None,
                "edge_decoder_type":  str(self.edge_decoder_type),
                "activation":         str(self.activation),
            },
            "manifold_repr":          repr(self.manifold),
            "likelihood_repr":        repr(self.likelihood),
            "certificate_history":    list(getattr(self, "_certificate_history", [])),
        }
        torch.save(ckpt, path)
        print(f"  [Checkpoint] saved -> {path}", flush=True)

    @classmethod
    def load(cls, path, **extra_kwargs) -> "RiemannianVAE":
        """Restore a model from a checkpoint produced by :meth:`save`.

        Parameters
        ----------
        path : path to the .pt checkpoint
        extra_kwargs : forwarded to ``__init__``; lets the caller
            override defaults that were not stored in the checkpoint
            (e.g., ``encoder_hidden=...``). Sensible defaults are used
            for any kwarg not stored.
        """
        from pathlib import Path
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        args = dict(ckpt.get("constructor_args", {}))
        # Reconstruct manifold/likelihood from name; users can override
        # via extra_kwargs.manifold_kwargs / likelihood_kwargs.
        manifold_name = args.pop("manifold_name", "euclidean")
        likelihood_name = args.pop("likelihood_name", "gaussian")
        kwargs = {
            "n_features":      args.get("n_features", 1),
            "n_latent":        args.get("n_latent", 1),
            "latent_manifold": manifold_name,
            "likelihood":      likelihood_name,
            "use_pe":          args.get("use_pe", False),
            "pe_dim":          args.get("pe_dim", None),
            "edge_decoder_type": args.get("edge_decoder_type", "scalar"),
            "activation":      args.get("activation", "silu"),
        }
        kwargs.update(extra_kwargs)
        model = cls(**kwargs)
        try:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        except Exception as e:
            print(
                f"  [Checkpoint] load_state_dict raised {type(e).__name__}: "
                f"{e}; the model architecture may have drifted from the "
                "saved checkpoint. Pass extra_kwargs to override the "
                "defaults used by load.",
                flush=True,
            )
        model._certificate_history = list(ckpt.get("certificate_history", []))
        print(f"  [Checkpoint] loaded <- {path}", flush=True)
        return model
