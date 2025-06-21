"""
Minimal *scalar* Lucas-tree model.

Provides drift, diffusion, generator, terminal.
The drift and diffusion functions now implement a geometric Brownian motion
for the dividend process X_t: dX_t = mu * X_t * dt + sigma * X_t * dW_t.
Generator and terminal functions remain placeholders.
"""

from __future__ import annotations
import jax.numpy as jnp

# Define some default model parameters
DEFAULT_MU = 0.02  # Drift coefficient for the dividend process
DEFAULT_SIGMA = 0.2  # Volatility coefficient for the dividend process

def scalar_lucas(mu: float = DEFAULT_MU, sigma: float = DEFAULT_SIGMA):
    """
    Returns drift, diffusion, generator, and terminal functions
    for a scalar Lucas model where the dividend process follows GBM.

    Args:
        mu: Drift coefficient of the dividend process.
        sigma: Volatility coefficient of the dividend process.

    Returns:
        A tuple (drift, diffusion, generator, terminal).
    """
    drift = lambda x: mu * x
    diffusion  = lambda x: sigma * x  # Note: for dX = mu*X*dt + sigma*X*dW, diffusion is sigma*X
                                      # If it's dX = mu(X)dt + sigma(X)dW, then this is sigma(X)
                                      # Assuming the latter for typical BSDE solver libraries where
                                      # the SDE is dX_t = b(X_t)dt + s(X_t)dW_t,
                                      # then b(x) = mu*x and s(x) = sigma*x.

    # Placeholder generator and terminal functions
    generator = lambda t, x, y, z: jnp.zeros_like(y) # Generator f(t, x, y, z)
    terminal  = lambda x: jnp.zeros_like(x)       # Terminal condition g(x)

    return drift, diffusion, generator, terminal
