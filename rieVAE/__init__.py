"""
rieVAE: Self-Consistent Riemannian Variational Autoencoder.

Learns an approximately isometric generative model of a data manifold
by alternating between VAE training on a fixed graph and Riemannian KNN
graph reconstruction from the decoder's own pullback metric.

Reference: "Self-Consistent Riemannian VAEs" (see docs/riemannian_vae_paper.tex).
"""
from rieVAE.model.scrvae import SCRVAE

__all__ = ["SCRVAE"]
