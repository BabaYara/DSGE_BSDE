"""Utility functions for simple PINN residuals."""

import jax.numpy as jnp
from jax import grad, vmap


def poisson_residual(u_fn, x):
    """Return the residual of -u''(x) = f(x)=pi^2*sin(pi*x)."""
    dudx = grad(u_fn)
    d2udx2 = grad(dudx)
    res_fn = lambda xi: -d2udx2(xi) - (jnp.pi**2) * jnp.sin(jnp.pi * xi)
    return vmap(res_fn)(x)
