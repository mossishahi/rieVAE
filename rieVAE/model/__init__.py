"""Neural-network components for the Certified Riemannian VAE.

Phase-2 unification (op47C C.2): the pre-Phase-2 ``SCRVAE`` and
``TopoSCRVAE`` classes are removed. The single model class is
``RiemannianVAE``, parameterised by a ``LatentManifold`` plug-in
(``rieVAE.manifold``) and a ``Likelihood`` plug-in
(``rieVAE.likelihood``).
"""
from rieVAE.model.riemannian_vae import RiemannianVAE
from rieVAE.model.encoder import NodeEncoder
from rieVAE.model.decoder import NodeDecoder
from rieVAE.model.edge import JointEdgeDecoder, ScalarEdgeDecoder
from rieVAE.model._activations import make_activation, supported_activations

__all__ = [
    "RiemannianVAE",
    "NodeEncoder",
    "NodeDecoder",
    "JointEdgeDecoder",
    "ScalarEdgeDecoder",
    "make_activation",
    "supported_activations",
]
