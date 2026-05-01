"""Neural-network building blocks for the Certified Riemannian VAE.

Contains the encoder, decoder, edge-decoder, and activation helpers
used by :class:`rieVAE.model.RiemannianVAE`.
"""
from rieVAE.modules.encoder import NodeEncoder
from rieVAE.modules.decoder import NodeDecoder
from rieVAE.modules.edge import JointEdgeDecoder, ScalarEdgeDecoder
from rieVAE.modules.activations import make_activation, supported_activations

__all__ = [
    "NodeEncoder",
    "NodeDecoder",
    "JointEdgeDecoder",
    "ScalarEdgeDecoder",
    "make_activation",
    "supported_activations",
]
