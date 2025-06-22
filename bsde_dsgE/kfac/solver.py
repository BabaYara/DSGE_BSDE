"""A simple solver using the Kronecker-Factored Approximate Curvature.

This submodule defines :class:`KFACPINNSolver`, a minimal training loop for
Physics-informed neural networks.  The solver relies on
:func:`bsde_dsgE.kfac.kfac_update` to perform natural gradient steps.

Examples
--------
>>> import jax
>>> import jax.numpy as jnp
>>> import equinox as eqx
>>> from bsde_dsgE.kfac import KFACPINNSolver

>>> key = jax.random.PRNGKey(0)
>>> net = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, key=key)
>>> def loss_fn(net, x):
...     y = net(x)
...     return jnp.mean(y ** 2)

>>> solver = KFACPINNSolver(net=net, loss_fn=loss_fn, lr=1e-2, num_steps=10)
>>> losses = solver.run(jnp.zeros((1, 1)), key)
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .optimizer import _init_state, kfac_update


class KFACPINNSolver(eqx.Module):  # type: ignore[misc]
    """Trainer for PINNs using a KFAC update.

    Parameters
    ----------
    net : eqx.Module
        Neural network to be optimised.
    loss_fn : Callable[[eqx.Module, jnp.ndarray], jnp.ndarray]
        Function computing the scalar loss given ``net`` and the input data.
    lr : float, default=1e-3
        Learning rate for :func:`bsde_dsgE.kfac.kfac_update`.
    num_steps : int, default=100
        Number of optimisation steps performed in :meth:`run`.
    """

    net: eqx.Module
    loss_fn: Callable[[eqx.Module, jnp.ndarray], jnp.ndarray]
    lr: float = 1e-3
    num_steps: int = 100

    def run(self, x0: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        """Execute the optimisation loop.

        Parameters
        ----------
        x0 : jnp.ndarray
            Input data passed to ``loss_fn`` at every step.
        key : jax.random.PRNGKey
            Random key used for JIT compilation and stochastic layers.

        Notes
        -----
        The model parameters are partitioned into ``params`` (trainable) and
        ``static`` (frozen) arrays. Only ``params`` are updated during
        optimisation while ``static`` is recombined with ``params`` after
        training.

        Returns
        -------
        jnp.ndarray
            Array of loss values with length ``num_steps``.
        """
        params, static = eqx.partition(self.net, eqx.is_array)
        fisher_state = _init_state(params)

        @eqx.filter_jit  # type: ignore[misc]
        def step(
            params: Any,
            fisher_state: Any,
            x: jnp.ndarray,
        ) -> Tuple[Any, Any, jnp.ndarray]:
            net = eqx.combine(params, static)
            loss, grads = eqx.filter_value_and_grad(self.loss_fn)(net, x)
            params, fisher_state = kfac_update(params, grads, fisher_state, self.lr)
            return params, fisher_state, loss

        loss_history = []
        for _ in range(self.num_steps):
            params, fisher_state, loss = step(params, fisher_state, x0)
            loss_history.append(loss)
        object.__setattr__(self, "net", eqx.combine(params, static))
        return jnp.stack(loss_history)


__all__ = ["KFACPINNSolver"]
