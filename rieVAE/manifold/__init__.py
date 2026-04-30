"""Latent-manifold registry for the Certified Riemannian VAE.

Five concrete manifolds + the protocol live here:

  - ``Euclidean(d)`` ............... R^d (the iso-architecture default)
  - ``FlatTorus(d, radii=...)`` .... T^d = R/(2 pi Z)^d (Cor.~cor:topo_matched)
  - ``Sphere(d)`` .................. round S^d (tangent-at-pole chart)
  - ``Hyperbolic(d, curvature)`` ... H^d (Lorentz hyperboloid)
  - ``StereographicProduct(factors)``  ... product of constant-curvature factors

Construct directly or via the string registry consumed by
``rieVAE.RiemannianVAE``:

    >>> from rieVAE.manifold import resolve_manifold
    >>> M = resolve_manifold("torus", n_latent=2)
    >>> M
    FlatTorus(dim=2, radii=(1.0, 1.0))
"""
from rieVAE.manifold._base import LatentManifold
from rieVAE.manifold.euclidean import Euclidean
from rieVAE.manifold.flat_torus import FlatTorus
from rieVAE.manifold.sphere import Sphere
from rieVAE.manifold.hyperbolic import Hyperbolic
from rieVAE.manifold.stereographic_product import StereographicProduct


_MANIFOLD_BY_NAME: dict[str, str] = {
    "euclidean":              "Euclidean",
    "torus":                  "FlatTorus",
    "flat_torus":             "FlatTorus",
    "sphere":                 "Sphere",
    "hyperbolic":             "Hyperbolic",
    "stereographic_product":  "StereographicProduct",
}


def resolve_manifold(
    spec: str | LatentManifold,
    n_latent: int | None = None,
    **kwargs,
) -> LatentManifold:
    """Resolve a manifold spec to a concrete ``LatentManifold`` instance.

    Parameters
    ----------
    spec : str or LatentManifold
        If an instance is passed, it is returned unchanged. If a
        string, it is looked up in the registry; ``n_latent`` and any
        extra ``kwargs`` are forwarded to the constructor.
    n_latent : int or None
        Required for string specs (the constructor's ``dim`` argument);
        ignored for ``StereographicProduct`` since the product
        dimension is determined by its factors.
    **kwargs : forwarded to the manifold constructor.

    Returns
    -------
    LatentManifold instance.

    Raises
    ------
    ValueError if the spec is not recognised or ``n_latent`` is
    missing where required.
    """
    if isinstance(spec, LatentManifold):
        return spec
    if not isinstance(spec, str):
        raise TypeError(
            f"resolve_manifold(spec): expected str or LatentManifold, "
            f"got {type(spec).__name__}."
        )
    key = spec.lower().strip()
    if key not in _MANIFOLD_BY_NAME:
        raise ValueError(
            f"Unknown manifold spec {spec!r}; expected one of "
            f"{sorted(_MANIFOLD_BY_NAME)}."
        )
    cls_name = _MANIFOLD_BY_NAME[key]
    cls = globals()[cls_name]
    if cls is StereographicProduct:
        if "factors" not in kwargs:
            raise ValueError(
                "StereographicProduct requires a 'factors' kwarg."
            )
        return cls(**kwargs)
    if n_latent is None:
        raise ValueError(
            f"Manifold {spec!r} requires n_latent (the intrinsic "
            "dimension d)."
        )
    return cls(dim=int(n_latent), **kwargs)


__all__ = [
    "LatentManifold",
    "Euclidean",
    "FlatTorus",
    "Sphere",
    "Hyperbolic",
    "StereographicProduct",
    "resolve_manifold",
]
