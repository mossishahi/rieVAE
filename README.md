# rieVAE

**Certified Riemannian Variational Autoencoder via a Spectral Ambient Premetric.**

A general-purpose manifold-learning VAE whose encoder posterior means form a
runtime-certified `O(r_n^p)`-isometric chart of the unknown data manifold,
with `r_n = (log n / n)^{1/d}` the minimax-optimal manifold-learning rate
and `p ∈ {1, 2}` selected by the latent topology. The certificate is four
batch-computable scalars logged at every checkpoint; when they pass, the
encoder's distances are guaranteed close to manifold geodesic distances on
the training edge set without any held-out labelling.

`rieVAE` is implementation-agnostic to the application domain. Latent
manifolds (Euclidean, FlatTorus, Sphere, Hyperbolic, StereographicProduct)
and observation likelihoods (Gaussian, NegativeBinomial,
ZeroInflatedNegativeBinomial, Poisson, Bernoulli) are first-class plug-ins;
a new manifold or likelihood is a one-file class implementing a small
protocol.

---

## Installation

```bash
pip install -e .
# Optional dependencies:
pip install -e ".[hydra]"   # YAML configs + multi-run sweeps
pip install -e ".[faiss]"   # FAISS-backed kNN graph for n > 1e5
```

Core dependencies: `torch >= 2.1`, `pytorch-lightning >= 2.0`, `numpy`,
`scipy`, `scikit-learn`, `matplotlib`.

---

## Five-line example

```python
import torch
import rieVAE

x = torch.randn(3000, 50)                     # any (N, D) feature tensor
model = rieVAE.RiemannianVAE(
    n_features=50, n_latent=2,
    latent_manifold="euclidean", likelihood="gaussian",
)
model.fit(x, max_steps=5_000, plan="iso")
z = model.encode(x)
report = model.certificate(x)
```

That's it. After `fit`, the model has been trained, post-hoc-calibrated,
and the certificate is logged via the Lightning callbacks. The 5 lines
expand to:

1. Build a `SpectralPreprocessor` and run it once on `x` -- CkNN graph,
   Coifman-Lafon LBO, Varadhan heat-kernel targets.
2. Wrap the resulting artefacts in a `TensorDataModule` that yields
   anchor-sampled subgraph batches.
3. Construct an `IsoTrainingPlan` (the iso architecture's three-term
   manifold-aware ELBO + isometry regulariser).
4. Drive `pytorch_lightning.Trainer.fit(plan, datamodule)` with the
   `CertificateObserverCallback`, `PostHocCalibrationCallback`,
   `PEAuxFitCallback` (when applicable), and `MultiGPUCertificateReducer`.

You can override any step by passing extra kwargs to `model.fit(...)`,
or fall back to the Lightning-level API for full control (see
**Lightning advanced API** below).

---

## Latent manifolds

| Manifold | Constructor | Chart dim | Decoder input dim | Use case |
|---|---|---|---|---|
| Euclidean | `Euclidean(d, prior='partial')` | `d` | `d` | default; contractible latent |
| FlatTorus | `FlatTorus(d, radii=...)` | `d` | `2 d` | topology-matched (Cor. cor:topo_matched) |
| Sphere | `Sphere(d)` | `d` | `d + 1` | spherical latent |
| Hyperbolic | `Hyperbolic(d, curvature)` | `d` | `d + 1` | hyperbolic latent |
| StereographicProduct | `StereographicProduct([f1, f2, ...])` | sum | sum | mixed-curvature latent |

Each manifold supplies the geodesic distance, the closed-form KL to its
canonical prior, the chart-coordinate reparameterisation, and the
embedding into the decoder's input. Switching manifolds means switching
ONE constructor argument:

```python
model = rieVAE.RiemannianVAE(
    n_features=50, n_latent=2,
    latent_manifold="torus",
    manifold_kwargs={"radii": (2.0, 1.0)},     # Clifford torus radii
    likelihood="gaussian",
)
```

---

## Observation likelihoods

| Likelihood | Constructor | Decoder channels per feature | Use case |
|---|---|---|---|
| Gaussian | `Gaussian(n_features)` | 1 | real-valued (default; recovers MSE) |
| NegativeBinomial | `NegativeBinomial(n_features, dispersion='feature')` | 1 (or 2 for sample-feature dispersion) | overdispersed counts |
| ZeroInflatedNegativeBinomial | `ZeroInflatedNegativeBinomial(n_features, dispersion='feature')` | 2 | counts with extra zeros |
| Poisson | `Poisson(n_features)` | 1 | non-negative integer counts |
| Bernoulli | `Bernoulli(n_features)` | 1 | binary in {0, 1} |

NB / Poisson accept an optional per-sample `scale_factor` (the GLM
exposure / offset) when the feature totals vary across samples. Custom
likelihoods are a one-file class implementing the `Likelihood` protocol
(see `rieVAE/likelihood/_base.py`).

---

## Training plans

| Plan | Objective |
|---|---|
| `IsoTrainingPlan` (default) | `L_rec + beta * L_KL + gamma(t) * L_iso` -- the iso architecture |
| `IsoPlusGlobalOrderTrainingPlan` | the above + RankNet global-ordinal loss |
| `IsoPlusJVPLegacyTrainingPlan` | the above + (Phase-3 stub of) the legacy JVP terms |
| `VanillaTrainingPlan` | `L_rec + beta * L_KL` (no isometry term; baseline) |

Each plan inherits a term-registry training step from `TrainingPlanBase`;
adding a new objective is a 5-line subclass:

```python
from rieVAE.training import Term, IsoTrainingPlan, sigmoid

def my_extra_term(model, outputs, batch):
    return outputs["mu"].pow(2).mean()

class MyPlan(IsoTrainingPlan):
    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        self.add_term(Term("extra", my_extra_term, schedule=sigmoid(0.1)))
```

---

## Multi-GPU + scalability

```python
model.fit(
    x,
    max_steps=50_000,
    plan="iso",
    accelerator="gpu", devices=4, strategy="ddp",
)
```

For `n > 10^5` features:

- Switch the kNN graph builder to FAISS or PyNNDescent
  (`preprocess_kwargs={"graph_builder": "faiss"}`).
- Use the `MmapTensorDataModule` to stream features from disk.
- For `n > 10^6`, the spectral eigensolve becomes the bottleneck;
  swap to LOBPCG-on-GPU or compute eigenpairs on a Nystrom subsample
  (under design; see `op47C.md` C.4.3).

---

## Hydra-driven ablations

```bash
python -m rieVAE manifold=torus likelihood=nb plan=iso

# Multi-run sweep:
python -m rieVAE -m \
    manifold=euclidean,torus,sphere \
    likelihood=gaussian,negative_binomial \
    plan=iso,iso_plus_rank \
    trainer.devices=4 trainer.strategy=ddp
```

The Hydra config tree lives at `rieVAE/configs/`; each sub-group
(`data/`, `manifold/`, `likelihood/`, `model/`, `preprocess/`, `plan/`,
`trainer/`) is one YAML file per option.

---

## Lightning advanced API

For users who want full control over the Lightning lifecycle, every
component is independently importable:

```python
import pytorch_lightning as pl
from rieVAE import (
    RiemannianVAE,
    SpectralPreprocessor, TensorDataModule,
    IsoTrainingPlan,
    CertificateObserverCallback, PostHocCalibrationCallback,
    PEAuxFitCallback, MultiGPUCertificateReducer,
)

artefacts = SpectralPreprocessor(knn_k=16, spectral_truncation=50).fit(x)
dm = TensorDataModule(x, artefacts, anchor_batch_size=512, n_steps_per_epoch=1000)
model = RiemannianVAE(n_features=50, n_latent=2, latent_manifold="torus")
plan  = IsoTrainingPlan(model, max_steps=50_000)
trainer = pl.Trainer(
    max_steps=50_000, accelerator="gpu", devices=4, strategy="ddp",
    callbacks=[
        cert_cb := CertificateObserverCallback(every_n_steps=500),
        PostHocCalibrationCallback(),
        PEAuxFitCallback(),
        MultiGPUCertificateReducer(cert_callback=cert_cb),
    ],
)
trainer.fit(plan, datamodule=dm)
```

---

## Public surface

```python
from rieVAE import (
    # User-facing model
    RiemannianVAE,
    # Manifold registry
    LatentManifold, Euclidean, FlatTorus, Sphere, Hyperbolic,
    StereographicProduct, resolve_manifold,
    # Likelihood registry
    Likelihood, Gaussian, NegativeBinomial,
    ZeroInflatedNegativeBinomial, Poisson, Bernoulli,
    resolve_likelihood,
    # Data
    SpectralPreprocessor, SpectralArtefacts,
    TensorDataModule, DatasetModule, MmapTensorDataModule,
    # Training plans
    TrainingPlanBase, Term,
    IsoTrainingPlan, IsoPlusGlobalOrderTrainingPlan,
    IsoPlusJVPLegacyTrainingPlan, VanillaTrainingPlan,
    # Schedule helpers
    constant, linear_warmup, sigmoid, beta_linear_decay,
    warmup_then_constant,
    # Callbacks
    CertificateObserverCallback, PostHocCalibrationCallback,
    PEAuxFitCallback, MultiGPUCertificateReducer,
    # Certificate
    CertificateReport, CertificateThresholds, compute_certificate,
)
```

---

## Citation

If this code is useful in your work, please cite the accompanying
paper (see `rieVAE_text/main.tex`).
