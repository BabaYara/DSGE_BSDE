# (ResNet, PosEnc, dtype/precision helpers)

"""Minimal neural network utilities used in the examples and tests."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class ResNet(eqx.Module):  # type: ignore[misc]
    """Simple residual-style network returning ``(y, z)``."""

    mlp: eqx.nn.MLP

    def __call__(self, t: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        def single(xi: jax.Array) -> jax.Array:
            return self.mlp(jnp.array([xi]))[0]

        y = jax.vmap(single)(x)
        z = jnp.zeros_like(y)
        return y, z

    @staticmethod
    def make(depth: int, width: int, *, key: jax.Array) -> "ResNet":
        mlp = eqx.nn.MLP(in_size=1, out_size=1, width_size=width, depth=depth, key=key)
        return ResNet(mlp)


__all__ = ["ResNet"]
