"""Phase-3 Lightning callbacks for the Certified Riemannian VAE.

Four callbacks (op47C C.3.4):

  - :class:`CertificateObserverCallback`  -- runs the runtime
    certificate every ``every_n_steps`` and at end of training.
  - :class:`PostHocCalibrationCallback`   -- OLS calibration of the
    scalar edge head's softplus(w*) at end of training.
  - :class:`PEAuxFitCallback`             -- fits the post-hoc PE
    auxiliary head A_psi and computes ``delta_pe_aux_sup``.
  - :class:`MultiGPUCertificateReducer`   -- all-reduces certificate
    sup-norm scalars across DDP ranks (no-op single-process).

The standalone preprocessor lives at :mod:`rieVAE.data.preprocessor`
and runs BEFORE ``pl.Trainer.fit()`` (op47C C.3 option (b)).
"""
from rieVAE.callbacks.certificate_observer import CertificateObserverCallback
from rieVAE.callbacks.post_hoc_calibration import PostHocCalibrationCallback
from rieVAE.callbacks.pe_aux_fit import PEAuxFitCallback
from rieVAE.callbacks.multi_gpu_reducer import MultiGPUCertificateReducer

__all__ = [
    "CertificateObserverCallback",
    "PostHocCalibrationCallback",
    "PEAuxFitCallback",
    "MultiGPUCertificateReducer",
]
