import jax
import jax.numpy as jnp
from bsde_seed.bsde_dsgE.core import Solver
from bsde_seed.bsde_dsgE.models import scalar_lucas


def test_solver_runs():
    """
    Tests that the Solver.run method executes with the updated components
    and produces output of the expected shape and value.
    """
    # Define an initial state vector
    x0 = jnp.ones((4,))

    # Get model components from scalar_lucas (using default mu and sigma)
    # The generator 'g' is lambda t, x, y, z: jnp.zeros_like(y)
    _drift, _diff, g, _term = scalar_lucas()

    # Define a dummy neural network
    # It returns (y_approx, z_approx), where y_approx will have the same shape as x
    dummy_net = lambda t, x: (jnp.zeros_like(x), jnp.zeros_like(x))

    # Instantiate the solver
    solver = Solver(net=dummy_net, generator=g)

    # Run the solver
    # Solver.run calls net(0.0, x0) -> (y0, z0)
    # Then calls generator(0.0, x0, y0, z0)
    # Since y0 = jnp.zeros_like(x0), and g returns jnp.zeros_like(y0),
    # the loss should be an array of zeros with the same shape as x0.
    loss = solver.run(x0, jax.random.PRNGKey(0))

    # Assertions
    assert loss.shape == x0.shape, f"Expected shape {x0.shape}, but got {loss.shape}"
    assert jnp.all(jnp.isfinite(loss)), "Loss contains non-finite values"
    assert jnp.all(loss == 0.0), f"Expected all zeros, but got {loss}"
