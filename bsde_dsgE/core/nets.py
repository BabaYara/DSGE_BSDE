# (ResNet, PosEnc, dtype/precision helpers)

"""Minimal neural network utilities used in the examples and tests."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class ResNet(eqx.Module):
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


class ResNetND(eqx.Module):
    """Residual-style network for vector inputs returning ``(y, z)``.

    - Input ``x``: array of shape (batch, dim)
    - Output ``y``: array of shape (batch,)  (scalar value per sample)
    - Output ``z``: array of shape (batch, dim) (vector control per sample)

    The implementation uses a single MLP head that outputs ``1 + dim`` units
    per sample, split across ``y`` and ``z``.  Time ``t`` is currently ignored
    (consistent with the scalar ResNet) but passed for API compatibility.
    """

    mlp: eqx.nn.MLP
    dim: int

    def __call__(self, t: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x2 = x if x.ndim == 2 else x[:, None]

        def single(xi: jax.Array) -> jax.Array:
            return self.mlp(xi)

        out = jax.vmap(single)(x2)
        y = out[:, 0]
        z = out[:, 1 : 1 + self.dim]
        return y, z

    @staticmethod
    def make(dim: int, depth: int, width: int, *, key: jax.Array) -> "ResNetND":
        mlp = eqx.nn.MLP(in_size=dim, out_size=1 + dim, width_size=width, depth=depth, key=key)
        return ResNetND(mlp, dim)


__all__ = ["ResNet", "ResNetND"]
