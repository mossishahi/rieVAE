"""Hydra entry point for rieVAE training (op47C C.3.5).

Usage
-----

    python -m rieVAE manifold=torus likelihood=nb plan=iso

Or, via the multi-run sweep:

    python -m rieVAE -m \
        manifold=euclidean,torus,sphere \
        likelihood=gaussian,nb \
        plan=iso,iso_plus_rank \
        trainer.devices=4 trainer.strategy=ddp

Programmatic use (no Hydra) is documented at the bottom of this file
(``rieVAE.__main__.run``); the Hydra-driven main() is just a thin shim.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    _HYDRA_AVAILABLE = True
except ImportError:
    _HYDRA_AVAILABLE = False
    DictConfig = Any  # type: ignore[assignment, misc]

try:
    import pytorch_lightning as pl
    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers used by ``data=tensor_from_npy`` configs.
# ---------------------------------------------------------------------------

def _load_npy(path: str) -> torch.Tensor:
    """Load (N, G) tensor from a .npy file. Used by Hydra's
    ``data: tensor_from_npy`` config target."""
    arr = np.load(path)
    return torch.from_numpy(arr).float()


# ---------------------------------------------------------------------------
# Programmatic run (no Hydra).
# ---------------------------------------------------------------------------

def run(
    *,
    x: torch.Tensor,
    n_features: int,
    n_latent: int,
    manifold: Any,
    likelihood: Any,
    model_kwargs: dict,
    preprocess_kwargs: dict,
    plan_factory,           # callable: (model, max_steps) -> TrainingPlan
    trainer_kwargs: dict,
    seed: int = 0,
) -> dict:
    """Programmatic entry point. Builds the data module, the model,
    the training plan, and a ``pl.Trainer``, runs ``fit``, and returns
    a dict with the certificate history.
    """
    if not _PL_AVAILABLE:
        raise ImportError(
            "rieVAE.train.run requires pytorch_lightning."
        )
    pl.seed_everything(int(seed), workers=True)

    from rieVAE import RiemannianVAE
    from rieVAE.data import SpectralPreprocessor, TensorDataModule
    from rieVAE.callbacks import (
        CertificateObserverCallback,
        PostHocCalibrationCallback,
        PEAuxFitCallback,
        MultiGPUCertificateReducer,
    )

    # 1) Preprocessor (op47C option (b): standalone, runs once).
    preprocessor = SpectralPreprocessor(**preprocess_kwargs)
    artefacts = preprocessor.fit(x)
    print(
        f"[preprocess] n_active={artefacts.n_active}, "
        f"|E*|={artefacts.edge_index.shape[1]}, "
        f"varadhan_t={artefacts.varadhan_t_used:.4e}, "
        f"chord_arc={artefacts.chord_arc_scale:.3f}",
        flush=True,
    )

    # 2) Data module.
    dm = TensorDataModule(
        x=x,
        artefacts=artefacts,
        anchor_batch_size=int(trainer_kwargs.get("anchor_batch_size", 512)),
        n_steps_per_epoch=int(trainer_kwargs.get("n_steps_per_epoch", 1000)),
        seed=int(seed),
    )

    # 3) Model.
    model = RiemannianVAE(
        n_features=int(n_features),
        n_latent=int(n_latent),
        latent_manifold=manifold,
        likelihood=likelihood,
        **model_kwargs,
    )

    # 4) Plan.
    max_steps = int(trainer_kwargs.get("max_steps", 50000))
    plan = plan_factory(model, max_steps)

    # 5) Callbacks.
    activation = str(model_kwargs.get("activation", "silu"))
    chart_regime = str(trainer_kwargs.get("cert_chart_regime", "general"))
    cert_cb = CertificateObserverCallback(
        every_n_steps=int(trainer_kwargs.get("cert_every_n_steps", 500)),
        cert_subsample=int(trainer_kwargs.get("cert_subsample", 2048)),
        cert_pullback_nodes=int(trainer_kwargs.get("cert_pullback_nodes", 32)),
        force_global_at_end=bool(trainer_kwargs.get("force_global_at_end", True)),
        chart_regime=chart_regime,
        activation=activation,
    )
    callbacks: list[pl.Callback] = [cert_cb]
    if trainer_kwargs.get("post_hoc_calibration", True):
        callbacks.append(PostHocCalibrationCallback())
    if model_kwargs.get("use_pe", False):
        callbacks.append(PEAuxFitCallback(
            n_steps=int(trainer_kwargs.get("pe_posthoc_steps", 2000)),
            lr=float(trainer_kwargs.get("pe_posthoc_lr", 1.0e-3)),
            batch_size=int(trainer_kwargs.get("pe_posthoc_batch", 512)),
        ))
    if trainer_kwargs.get("multi_gpu_cert_reduce", True):
        callbacks.append(MultiGPUCertificateReducer(cert_callback=cert_cb))

    # 6) pl.Trainer.
    pl_trainer_kwargs = {
        "max_epochs":        int(trainer_kwargs.get("max_epochs", 50)),
        "max_steps":         max_steps,
        "accelerator":       str(trainer_kwargs.get("accelerator", "auto")),
        "devices":           trainer_kwargs.get("devices", 1),
        "strategy":          trainer_kwargs.get("strategy", "auto"),
        "log_every_n_steps": int(trainer_kwargs.get("log_every_n_steps", 50)),
        "deterministic":     bool(trainer_kwargs.get("deterministic", False)),
        "callbacks":         callbacks,
        "enable_progress_bar": True,
    }
    grad_clip = trainer_kwargs.get("gradient_clip_val")
    if grad_clip:
        pl_trainer_kwargs["gradient_clip_val"] = float(grad_clip)
    trainer = pl.Trainer(**pl_trainer_kwargs)

    # 7) Fit.
    trainer.fit(plan, datamodule=dm)

    # 8) Collect outputs.
    return {
        "model":               model,
        "plan":                plan,
        "trainer":             trainer,
        "datamodule":          dm,
        "artefacts":           artefacts,
        "certificate_history": cert_cb.history,
    }


# ---------------------------------------------------------------------------
# Hydra entry point.
# ---------------------------------------------------------------------------

if _HYDRA_AVAILABLE:

    @hydra.main(
        version_base=None,
        config_path="configs",
        config_name="run",
    )
    def main(cfg: DictConfig) -> None:
        from hydra.utils import instantiate
        # Materialise sub-configs.
        x = instantiate(cfg.data)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x).float()
        manifold = instantiate(cfg.manifold)
        likelihood = instantiate(cfg.likelihood)

        model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
        # Inject the resolved manifold/likelihood instances.
        model_kwargs["latent_manifold"] = manifold
        model_kwargs["likelihood"] = likelihood

        preprocess_kwargs = OmegaConf.to_container(cfg.preprocess, resolve=True)
        trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)

        # Plan factory closes over the cfg.plan node so we can pass
        # max_steps in.
        plan_cfg = cfg.plan

        def plan_factory(model, max_steps):
            kwargs = OmegaConf.to_container(plan_cfg, resolve=True)
            target = kwargs.pop("_target_")
            kwargs["max_steps"] = int(max_steps)
            from hydra.utils import get_class
            cls = get_class(target)
            return cls(model=model, **kwargs)

        run(
            x=x,
            n_features=int(cfg.n_features),
            n_latent=int(cfg.n_latent),
            manifold=manifold,
            likelihood=likelihood,
            model_kwargs=model_kwargs,
            preprocess_kwargs=preprocess_kwargs,
            plan_factory=plan_factory,
            trainer_kwargs=trainer_kwargs,
            seed=int(cfg.seed),
        )

else:  # pragma: no cover

    def main(*args, **kwargs):
        raise ImportError(
            "rieVAE.train.main requires Hydra; install via "
            "`pip install hydra-core omegaconf`."
        )


if __name__ == "__main__":
    main()
