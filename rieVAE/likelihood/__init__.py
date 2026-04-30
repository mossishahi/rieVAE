"""Observation-likelihood registry for the Certified Riemannian VAE.

Five concrete likelihoods + the protocol live here:

  - ``Gaussian`` ............. real-valued (the iso-architecture default)
  - ``NegativeBinomial`` ..... non-negative integer counts with overdispersion
  - ``ZeroInflatedNegativeBinomial`` ... counts with extra zeros
  - ``Poisson`` .............. non-negative integer counts
  - ``Bernoulli`` ............ binary in {0, 1}

Construct directly or via the string registry consumed by
``rieVAE.RiemannianVAE``:

    >>> from rieVAE.likelihood import resolve_likelihood
    >>> like = resolve_likelihood("nb", n_features=50)
    >>> like
    NegativeBinomial(n_features=50, dispersion='feature')
"""
from rieVAE.likelihood._base import Likelihood
from rieVAE.likelihood.gaussian import Gaussian
from rieVAE.likelihood.negative_binomial import NegativeBinomial
from rieVAE.likelihood.zinb import ZeroInflatedNegativeBinomial
from rieVAE.likelihood.poisson import Poisson
from rieVAE.likelihood.bernoulli import Bernoulli


_LIKELIHOOD_BY_NAME: dict[str, str] = {
    "gaussian":            "Gaussian",
    "negative_binomial":   "NegativeBinomial",
    "nb":                  "NegativeBinomial",
    "zinb":                "ZeroInflatedNegativeBinomial",
    "zero_inflated_nb":    "ZeroInflatedNegativeBinomial",
    "poisson":             "Poisson",
    "bernoulli":           "Bernoulli",
}


def resolve_likelihood(
    spec: str | Likelihood,
    n_features: int | None = None,
    **kwargs,
) -> Likelihood:
    """Resolve a likelihood spec to a concrete ``Likelihood`` instance.

    Parameters
    ----------
    spec : str or Likelihood
        If an instance is passed, it is returned unchanged. If a
        string, it is looked up in the registry; ``n_features`` and
        any extra ``kwargs`` are forwarded to the constructor.
    n_features : int or None
        Required for string specs.
    **kwargs : forwarded to the likelihood constructor.

    Returns
    -------
    Likelihood instance.
    """
    if isinstance(spec, Likelihood):
        return spec
    if not isinstance(spec, str):
        raise TypeError(
            f"resolve_likelihood(spec): expected str or Likelihood, "
            f"got {type(spec).__name__}."
        )
    key = spec.lower().strip()
    if key not in _LIKELIHOOD_BY_NAME:
        raise ValueError(
            f"Unknown likelihood spec {spec!r}; expected one of "
            f"{sorted(_LIKELIHOOD_BY_NAME)}."
        )
    if n_features is None:
        raise ValueError(
            f"Likelihood {spec!r} requires n_features."
        )
    cls_name = _LIKELIHOOD_BY_NAME[key]
    cls = globals()[cls_name]
    return cls(n_features=int(n_features), **kwargs)


__all__ = [
    "Likelihood",
    "Gaussian",
    "NegativeBinomial",
    "ZeroInflatedNegativeBinomial",
    "Poisson",
    "Bernoulli",
    "resolve_likelihood",
]
