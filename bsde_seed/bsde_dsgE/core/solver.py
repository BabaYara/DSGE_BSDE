"""
Minimal BSDE solver stub.

Implements the interface::
    loss = Solver(net, generator).run(x0, key)

The real logic will be filled by later milestones.
"""

from __future__ import annotations
from typing import Callable
import jax, jax.numpy as jnp
import equinox as eqx


class Solver(eqx.Module):
    """
    Single-step dummy solver.
    For this version, it calls the network at t=0, x0 to get (y0, z0),
    then calls the generator with these values, and returns the generator's output.
    """

    net: eqx.Module                     # placeholder neural net
    generator: Callable[..., jnp.ndarray] # driver f(t, x, y, z)

    def run(self, x0: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Calls the network and then the generator.
        Returns the output of the generator function.
        The PRNGKey is currently unused but maintained for API consistency.
        """
        # Assume net takes t (scalar) and x (array) and returns (y_approx, z_approx)
        # t=0.0 is a common convention for the start of the BSDE interval.
        y0_approx, z0_approx = self.net(0.0, x0)

        # Generator f(t, x, y, z)
        # Pass t=0.0, the initial state x0, and network outputs y0_approx, z0_approx
        loss_value = self.generator(0.0, x0, y0_approx, z0_approx)

        return loss_value
