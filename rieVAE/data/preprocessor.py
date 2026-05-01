"""Standalone spectral preprocessor for the Certified Riemannian VAE.

Extracts the once-per-run preprocessing pipeline that previously lived
in ``ProximalSCRVAETrainer.__init__`` (Phase 1/2) and exposes it as a
plain class. The user runs the preprocessor BEFORE
``pytorch_lightning.Trainer.fit()`` and passes the resulting artefacts
to the data module's constructor (op47C C.3, option (b)). This makes
preprocessing reusable across multiple training runs / training plans
for ablation studies, and decouples the heavy spectral computation
from the Lightning lifecycle.

Pipeline (cf. main.tex Algorithm alg:training, lines 1-3):

  1. ``mst_connectivity_radius`` -- MST-based outlier filter on the
     ambient features.
  2. ``build_biharmonic_distance`` -- CkNN graph + Coifman-Lafon
     alpha=1 LBO + smallest k non-trivial eigenpairs.
  3. ``compute_varadhan_edge_distances`` -- Varadhan heat-kernel
     targets on the CkNN edges; edges with K_t <= 0 are filtered out
     (``varadhan_invalid_frac`` is logged).
  4. Chord-arc rescaling so the Varadhan targets are in chord units
     comparable to the encoder's posterior-mean Euclidean distances.
  5. (Optional) PE features ``Psi`` for the encoder's PE branch and
     for the global-ordinal oracle ``psi_full``.
  6. (Optional) Decoder-independent per-pair reweighting omega.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import numpy as np
import torch

from rieVAE.geometry.spectral_premetric import (
    build_biharmonic_distance,
    spectral_ball_edges,
    pca_local_reweighting,
    compute_varadhan_edge_distances,
)
from rieVAE.geometry.graph import mst_connectivity_radius
from rieVAE.geometry.positional_encoding import (
    compute_pe_features,
    resolve_phi_and_lambdas,
)


@dataclasses.dataclass
class SpectralArtefacts:
    """Bundle of artefacts produced by the spectral preprocessor.

    Stored once and consumed by the data module + the certificate
    callback throughout training. All tensors are on CPU; the user /
    data module moves them to the right device.

    Attributes
    ----------
    x_active : (n_active, G)
        Outlier-filtered ambient features. Used as the unique source
        of training inputs.
    active_idx : (n_active,) long
        Indices of the surviving samples in the original input.
    n_active : int
    n_total  : int
    radius : float
        MST connectivity radius from the original data.
    edge_index : (2, E) long
        Static edge set E^* (CkNN, post-Varadhan-validity filter).
    edge_weight : (E,) float
        Chord-arc-rescaled spectral targets tilde_w on E^*.
    spec_artefacts : dict
        Raw spectral outputs (``eigvals``, ``eigvecs``, ``Psi``,
        ``d_bih``, ``cknn_edges``, ``rho``).
    chord_arc_scale : float
        Empirical mean chord / mean tilde_w on E^*; logged as a
        diagnostic.
    varadhan_t_used : float
    varadhan_invalid_frac : float
    pe_feat : (n_active, K) or None
        Encoder PE features ``Psi`` produced by
        ``compute_pe_features``; None when PE is disabled.
    psi_full : (n_active, K) or None
        Global-ordinal oracle ``Psi = phi * lambda^{-alpha}`` used by
        the rank-only loss; None when ``use_global_order=False``.
    omega : (E,) or None
        Decoder-independent per-pair reweighting (Section sec:reweight);
        None when disabled.
    rec_threshold : float
        Parameter-free C2 reconstruction threshold
        T^{E*}_rec = mean(tilde_w_{ij}^2) over E^* (eq:c2_edge_scale in
        main.tex). Training is certified on C2 when the mean squared
        reconstruction loss L_rec falls below this value.  Computed once
        from edge_weight at the end of .fit() and stored here so it can
        be passed directly to CertificateThresholds.
    intrinsic_dim : int
        Two-NN MLE estimate of the manifold's intrinsic dimension,
        computed from x_active (Facco et al. 2017, same estimator as
        rieVAE.evaluate.certificate.intrinsic_dim_estimate).  Used as
        the dimension d in r_n = (log n / n)^{1/d} for the certificate,
        replacing the unsafe fallback of using model.dim_latent (which
        may differ from the true intrinsic dimension).
    e_star_connected : bool
        True iff the final E^* edge set (after Varadhan validity
        filtering) is a connected graph on the active nodes.  The
        Dijkstra spanning argument of Cor. cor:clgg requires E^* to be
        connected; this flag is False when the Varadhan filter has
        removed enough edges to disconnect the graph (the certificate's
        global-pair conclusion is then not supported by the spanning
        argument).
    """

    x_active: torch.Tensor
    active_idx: torch.Tensor
    n_active: int
    n_total: int
    radius: float
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    spec_artefacts: dict
    chord_arc_scale: float = 1.0
    varadhan_t_used: float = 0.0
    varadhan_invalid_frac: float = 0.0
    pe_feat: Optional[torch.Tensor] = None
    psi_full: Optional[torch.Tensor] = None
    omega: Optional[torch.Tensor] = None
    rec_threshold: float = 0.0
    intrinsic_dim: int = 2
    e_star_connected: bool = True


class SpectralPreprocessor:
    """One-shot spectral preprocessing pipeline.

    Constructed with hyperparameters describing the graph, the
    Varadhan target, and the optional PE / reweighting; called via
    ``.fit(x)`` to produce :class:`SpectralArtefacts`.

    Parameters
    ----------
    knn_k : int
        Internal candidate-pool kNN parameter; passed to
        ``build_biharmonic_distance`` and
        ``mst_connectivity_radius``. Default 16.
    spectral_truncation : int
        Number of non-trivial Laplacian eigenpairs to compute
        (``K`` in the paper). Default 50.
    radius_quantile : float
        Outlier-filter quantile for ``mst_connectivity_radius``.
        Default 0.995.
    laplacian_type : {'cknn', 'binary'}
        ``'cknn'`` (default) builds the Coifman-Lafon (alpha=1) LBO
        Laplacian on a CkNN topology; ``'binary'`` falls back to a
        density-weighted symmetric kNN Laplacian (ablation only).
    cknn_k_cand, cknn_k_sigma : int
        CkNN candidate pool and density-radius neighbour index;
        ignored when ``laplacian_type='binary'``.
    target_mode : {'varadhan', 'geo'}
        ``'varadhan'`` (default) uses the Varadhan heat-kernel target
        ``sqrt(-4 t log K_t)`` with auto-calibrated ``t``;
        ``'geo'`` uses ground-truth geodesic distances supplied via
        ``d_true=`` (for synthetic-data oracle runs only).
    varadhan_t : float or None
        Heat time t for ``compute_varadhan_edge_distances``; None
        triggers the auto heuristic ``0.25 / lambda_mean``.
    use_pe : bool
    pe_dim : int or None
        Number of PE features. Required when ``use_pe=True``.
    pe_alpha : float
        Spectral exponent in the PE map ``Psi = phi * lambda^{-alpha}``
        used by the encoder. Default 0.5 (the heat-kernel exponent of
        Berard-Besson-Gallot 1994).
    use_global_order : bool
    global_order_pe_alpha : float
        Spectral exponent in the global-ordinal oracle ``psi_full``;
        independent of ``pe_alpha``. Default 0.5.
    use_omega_reweight : bool
    omega_k_pca : int
    omega_clip : tuple[float, float]
    """

    def __init__(
        self,
        knn_k: int = 16,
        spectral_truncation: int = 50,
        radius_quantile: float = 0.995,
        laplacian_type: str = "cknn",
        cknn_k_cand: int = 50,
        cknn_k_sigma: int = 7,
        target_mode: str = "varadhan",
        varadhan_t: Optional[float] = None,
        use_pe: bool = False,
        pe_dim: Optional[int] = None,
        pe_alpha: float = 0.5,
        use_global_order: bool = False,
        global_order_pe_alpha: float = 0.5,
        use_omega_reweight: bool = False,
        omega_k_pca: int = 20,
        omega_clip: tuple = (0.25, 4.0),
    ) -> None:
        self.knn_k = int(knn_k)
        self.spectral_truncation = int(spectral_truncation)
        self.radius_quantile = float(radius_quantile)
        self.laplacian_type = str(laplacian_type).lower()
        self.cknn_k_cand = int(cknn_k_cand)
        self.cknn_k_sigma = int(cknn_k_sigma)
        self.target_mode = str(target_mode).lower()
        self.varadhan_t = varadhan_t
        self.use_pe = bool(use_pe)
        self.pe_dim = int(pe_dim) if (use_pe and pe_dim is not None) else 0
        self.pe_alpha = float(pe_alpha)
        self.use_global_order = bool(use_global_order)
        self.global_order_pe_alpha = float(global_order_pe_alpha)
        self.use_omega_reweight = bool(use_omega_reweight)
        self.omega_k_pca = int(omega_k_pca)
        self.omega_clip = tuple(omega_clip)

    def fit(
        self,
        x: torch.Tensor,
        d_true: Optional[torch.Tensor] = None,
    ) -> SpectralArtefacts:
        """Run the preprocessor and return artefacts.

        Parameters
        ----------
        x : (N, G) float tensor
            Ambient features.
        d_true : (N, N) or None
            Ground-truth pairwise geodesic distances. Required iff
            ``target_mode == 'geo'``.

        Returns
        -------
        :class:`SpectralArtefacts`
        """
        x_cpu = x.detach().cpu()
        n_total = int(x_cpu.shape[0])

        # 1) MST connectivity-threshold radius and outlier filter.
        radius, outlier_mask = mst_connectivity_radius(
            x=x_cpu, quantile=self.radius_quantile,
        )
        active_idx = torch.where(~outlier_mask)[0]
        n_active = int(active_idx.numel())
        x_active = x_cpu[active_idx]

        # 2) CkNN + Coifman-Lafon LBO + eigenpairs.
        n_candidates = min(10 * self.knn_k, n_active - 1)
        spec_artefacts = build_biharmonic_distance(
            x=x_active,
            k_nn=self.knn_k,
            k_trunc=min(self.spectral_truncation, n_active - 2),
            k_candidates=n_candidates,
            laplacian_type=self.laplacian_type,
            cknn_k_cand=self.cknn_k_cand,
            cknn_k_sigma=self.cknn_k_sigma,
        )

        # 3) Static edge set E*.
        if (
            self.laplacian_type == "cknn"
            and "cknn_edges" in spec_artefacts
        ):
            cknn_ei = spec_artefacts["cknn_edges"]
            edge_index = torch.from_numpy(cknn_ei).long()
            if edge_index.numel() > 0:
                src_init = edge_index[0].numpy()
                dst_init = edge_index[1].numpy()
                Psi_t = spec_artefacts["Psi"]
                edge_weight = torch.tensor(
                    np.linalg.norm(
                        Psi_t[src_init] - Psi_t[dst_init], axis=1
                    ).astype(np.float32)
                )
            else:
                edge_weight = torch.zeros(0, dtype=torch.float32)
        else:
            edge_index, edge_weight = spectral_ball_edges(
                d_bih=spec_artefacts["d_bih"],
                radius=radius,
                symmetric=True,
            )

        # 4) Chord-arc scale (logged) and Varadhan rescaling.
        chord_arc_scale = 1.0
        if edge_index.numel() > 0:
            src_ca, dst_ca = edge_index[0], edge_index[1]
            chord_e = (x_active[dst_ca] - x_active[src_ca]).norm(dim=-1)
            mean_chord = float(chord_e.mean().item())
            mean_bih = float(edge_weight.mean().item())
            if mean_bih > 1e-12 and mean_chord > 1e-12:
                chord_arc_scale = mean_chord / mean_bih

        varadhan_t_used = 0.0
        varadhan_invalid_frac = 0.0
        if self.target_mode == "varadhan":
            if edge_index.numel() > 0:
                phi_t = torch.from_numpy(spec_artefacts["eigvecs"]).float()
                lam_t = torch.from_numpy(spec_artefacts["eigvals"]).float()
                vdh_e, t_used, valid_mask = compute_varadhan_edge_distances(
                    phi=phi_t, lambdas=lam_t,
                    edge_index=edge_index,
                    t=self.varadhan_t,
                    return_valid_mask=True,
                )
                varadhan_t_used = float(t_used)
                n_before = int(edge_index.shape[1])
                n_invalid = int((~valid_mask).sum().item())
                varadhan_invalid_frac = n_invalid / max(n_before, 1)
                if n_invalid > 0:
                    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
                    edge_index = edge_index[:, valid_idx]
                    vdh_e = vdh_e[valid_idx]
                    if edge_index.numel() > 0:
                        src_f, dst_f = edge_index[0], edge_index[1]
                        chord_e = (x_active[dst_f] - x_active[src_f]).norm(dim=-1)
                        mean_chord = float(chord_e.mean().item())
                    else:
                        mean_chord = 1.0
                    print(
                        f"[varadhan] filtered {n_invalid}/{n_before} edges "
                        f"({100.0 * varadhan_invalid_frac:.1f}%) with "
                        f"K_t <= 0; E* now has {edge_index.shape[1]} edges.",
                        flush=True,
                    )
                mean_vdh = float(vdh_e.mean().item()) if vdh_e.numel() > 0 else 0.0
                vdh_arc_scale = (
                    mean_chord / mean_vdh if (mean_vdh > 1e-12 and mean_chord > 1e-12)
                    else 1.0
                )
                edge_weight = vdh_e * vdh_arc_scale
        elif self.target_mode == "geo":
            if d_true is None:
                raise ValueError(
                    "SpectralPreprocessor(target_mode='geo') requires "
                    "d_true to be passed to .fit(...)."
                )
            if edge_index.numel() > 0:
                src_g = active_idx[edge_index[0]]
                dst_g = active_idx[edge_index[1]]
                geo_e = d_true.cpu().float()[src_g, dst_g]
                edge_weight = geo_e.to(edge_weight.dtype)
        else:
            raise ValueError(
                f"Unknown target_mode {self.target_mode!r}; "
                "expected 'varadhan' or 'geo'."
            )

        # 5) Optional PE features.
        pe_feat: Optional[torch.Tensor] = None
        if self.use_pe:
            if self.pe_dim < 1:
                raise ValueError(
                    "use_pe=True requires pe_dim >= 1; got pe_dim=0."
                )
            phi_all, lambdas_all = resolve_phi_and_lambdas(spec_artefacts)
            pe_artefacts = compute_pe_features(
                phi=phi_all,
                lambdas=lambdas_all,
                alpha=self.pe_alpha,
                pe_dim=self.pe_dim,
                x_for_rms=x_active.detach().cpu(),
            )
            pe_feat = pe_artefacts.pe.to(dtype=x_active.dtype)

        # 6) Optional global-ordinal oracle psi_full.
        psi_full: Optional[torch.Tensor] = None
        if self.use_global_order:
            from rieVAE.geometry.global_order import build_psi
            phi_go, lam_go = resolve_phi_and_lambdas(spec_artefacts)
            psi_full = build_psi(
                eigvecs=phi_go.float(),
                eigvals=lam_go.float(),
                alpha=self.global_order_pe_alpha,
            )

        # 7) Optional decoder-independent reweighting omega.
        omega: Optional[torch.Tensor] = None
        if self.use_omega_reweight:
            omega = pca_local_reweighting(
                x=x_active,
                edge_index=edge_index.cpu(),
                k_pca=self.omega_k_pca,
                omega_min=float(self.omega_clip[0]),
                omega_max=float(self.omega_clip[1]),
            )

        # --- Mo2: intrinsic dimension estimate (Two-NN MLE, Facco 2017) ---
        # Used as d in r_n = (log n / n)^{1/d} for the certificate, instead
        # of the unsafe fallback model.dim_latent.
        try:
            from sklearn.neighbors import NearestNeighbors as _NNS
            _x_np = x_active.detach().cpu().float().numpy()
            _rng = np.random.default_rng(0)
            _n_anchor = min(1024, _x_np.shape[0] - 1)
            _idx = _rng.choice(_x_np.shape[0], size=_n_anchor, replace=False)
            _nbrs = _NNS(n_neighbors=3, algorithm="auto").fit(_x_np)
            _dists, _ = _nbrs.kneighbors(_x_np[_idx])
            _mu = _dists[:, 2] / np.maximum(_dists[:, 1], 1e-12)
            _mu = _mu[_mu > 1.0]
            if len(_mu) > 0:
                _d_hat = int(max(1, round(1.0 / float(np.mean(np.log(_mu))))))
            else:
                _d_hat = int(getattr(x_active, "shape", [None, 2])[1] // 4 or 2)
            intrinsic_dim = max(1, min(_d_hat, x_active.shape[1] - 1))
        except Exception:
            intrinsic_dim = 2
            print("[preprocessor] intrinsic_dim_estimate failed; defaulting to 2",
                  flush=True)

        # --- Mo3: E* connectivity check after Varadhan filter ---
        e_star_connected = True
        if edge_index.numel() > 0:
            try:
                from scipy.sparse import csr_matrix
                from scipy.sparse.csgraph import connected_components
                _n = n_active
                _src = edge_index[0].cpu().numpy()
                _dst = edge_index[1].cpu().numpy()
                _data = np.ones(len(_src), dtype=np.float32)
                _adj = csr_matrix((_data, (_src, _dst)), shape=(_n, _n))
                _n_comp, _ = connected_components(_adj, directed=False)
                e_star_connected = bool(_n_comp == 1)
                if not e_star_connected:
                    print(
                        f"[preprocessor] WARNING: E* is NOT connected after "
                        f"Varadhan filtering ({_n_comp} components). "
                        f"The graph-geodesic spanning argument of Cor. cor:clgg "
                        f"is not supported. Consider increasing K "
                        f"(spectral_truncation) or adjusting varadhan_t. "
                        f"varadhan_invalid_frac={varadhan_invalid_frac:.3f}",
                        flush=True,
                    )
            except Exception:
                e_star_connected = True

        # --- C2 threshold: mean(tilde_w^2) over E* (eq:c2_edge_scale) ---
        if edge_weight.numel() > 0:
            rec_threshold = float(edge_weight.pow(2).mean().item())
        else:
            rec_threshold = 0.0

        return SpectralArtefacts(
            x_active=x_active,
            active_idx=active_idx,
            n_active=n_active,
            n_total=n_total,
            radius=float(radius),
            edge_index=edge_index,
            edge_weight=edge_weight,
            spec_artefacts=spec_artefacts,
            chord_arc_scale=chord_arc_scale,
            varadhan_t_used=varadhan_t_used,
            varadhan_invalid_frac=varadhan_invalid_frac,
            pe_feat=pe_feat,
            psi_full=psi_full,
            omega=omega,
            rec_threshold=rec_threshold,
            intrinsic_dim=intrinsic_dim,
            e_star_connected=e_star_connected,
        )
