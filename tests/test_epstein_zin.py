import numpy as np
import jax
import jax.numpy as jnp

from bsde_dsgE.models.epstein_zin import EZParams, ez_generator, sdf_exposure_from_ez


def test_ez_generator_shapes_and_monotonicity():
    key = jax.random.PRNGKey(0)
    batch, dim = 7, 3
    x = jax.random.normal(key, (batch, dim))
    y = jnp.abs(jax.random.normal(key, (batch,))) + 0.5
    z = jax.random.normal(key, (batch, dim))

    params = EZParams(delta=0.02, gamma=8.0, psi=1.5)

    # Two consumption mappings: c and scaled (higher) consumption
    def c_fn(x):
        x2 = jnp.asarray(x)
        return jnp.sum(jnp.exp(x2), axis=-1)

    def c_fn_hi(x):
        return 1.5 * c_fn(x)

    gen = ez_generator(params, c_fn)
    gen_hi = ez_generator(params, c_fn_hi)

    f = gen(x, y, z)
    f_hi = gen_hi(x, y, z)

    # Shapes
    assert f.shape == (batch,)
    assert f_hi.shape == (batch,)

    # Higher consumption should weakly increase the aggregator
    assert jnp.all(f_hi >= f - 1e-10)


def test_sdf_exposure_matches_formula():
    batch, dim = 5, 2
    y = jnp.linspace(0.5, 1.5, batch)
    z = jnp.arange(batch * dim, dtype=float).reshape(batch, dim)
    params = EZParams(delta=0.01, gamma=12.0, psi=1.2)
    lam = sdf_exposure_from_ez(y, z, params)
    assert lam.shape == (batch, dim)
    # Check definition: -theta * z / y
    theta = params.theta
    expected = -theta * z / y.reshape(-1, 1)
    np.testing.assert_allclose(np.array(lam), np.array(expected), rtol=1e-12, atol=1e-12)

