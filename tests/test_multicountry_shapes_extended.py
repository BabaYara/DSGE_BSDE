import jax
import jax.numpy as jnp

from bsde_dsgE.models.multicountry import multicountry_probab01


def test_multicountry_generator_terminal_shapes():
    dim = 3
    prob = multicountry_probab01(dim=dim)
    batch = 5
    x = jnp.zeros((batch, dim))
    y = jnp.ones((batch,))
    z = jnp.ones((batch, dim))
    g = prob.generator(x, y, z)
    t = prob.terminal(x)
    assert g.shape == (batch,)
    assert t.shape == (batch,)

