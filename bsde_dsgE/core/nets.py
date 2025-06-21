# (ResNet, PosEnc, dtype/precision helpers)

"""Minimal neural network utilities used in the examples and tests."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import equinox as eqx


class ResNet(eqx.Module):
    """Simple residual-style network returning ``(y, z)``."""

    mlp: eqx.nn.MLP

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def single(xi):
            return self.mlp(jnp.array([xi]))[0]

        y = jax.vmap(single)(x)
        z = jnp.zeros_like(y)
        return y, z

    @staticmethod
    def make(depth: int, width: int, *, key: jax.random.PRNGKey) -> "ResNet":
        mlp = eqx.nn.MLP(in_size=1, out_size=1, width_size=width, depth=depth, key=key)
        return ResNet(mlp)


__all__ = ["ResNet"]
