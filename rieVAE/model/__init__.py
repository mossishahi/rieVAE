"""Neural-network components for the Certified Riemannian VAE.

Phase-2 unification (op47C C.2): the pre-Phase-2 ``SCRVAE`` and
``TopoSCRVAE`` classes are removed. The single model class is
``RiemannianVAE``, parameterised by a ``LatentManifold`` plug-in
(``rieVAE.manifolds``) and a ``Likelihood`` plug-in
(``rieVAE.likelihoods``).
"""
from rieVAE.model.riemannian_vae import RiemannianVAE
from rieVAE.modules.encoder import NodeEncoder
from rieVAE.modules.decoder import NodeDecoder
from rieVAE.modules.edge import JointEdgeDecoder, ScalarEdgeDecoder
from rieVAE.modules.activations import make_activation, supported_activations

__all__ = [
    "RiemannianVAE",
    "NodeEncoder",
    "NodeDecoder",
    "JointEdgeDecoder",
    "ScalarEdgeDecoder",
    "make_activation",
    "supported_activations",
]
