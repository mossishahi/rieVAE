"""Activation factory shared by the encoder, decoder, edge head, and
deformation predictor.

The theory in App. ass:kappa requires the decoder to be C^3 with a
bounded Hessian, and the PL^* argument of App. app:sc requires the
activation derivative to span. SiLU, GELU, Tanh, and Softplus all
satisfy both; ReLU is excluded by the C^3 hypothesis. The helper
below makes the choice configurable through a single string key.
"""
from __future__ import annotations

import torch.nn as nn


_ACTIVATIONS = {
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
}


def make_activation(name: str) -> nn.Module:
    """Return a fresh activation module by name.

    Supported names: ``'silu'`` (default), ``'gelu'``, ``'tanh'``,
    ``'softplus'``. ReLU is intentionally not exposed because it
    violates the ``C^3`` hypothesis used in Lemma 1's curvature
    remainder bound.
    """
    key = (name or "silu").lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"unknown activation {name!r}; valid options are "
            f"{sorted(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[key]()


def supported_activations() -> list[str]:
    return sorted(_ACTIVATIONS)
