import jax
import jax.numpy as jnp

from bsde_dsgE.core.nets import ResNetND


def test_resnet_nd_shapes():
    net = ResNetND.make(dim=3, depth=2, width=16, key=jax.random.PRNGKey(0))
    t = jnp.zeros((5,))
    x = jnp.zeros((5, 3))
    y, z = net(t, x)
    assert y.shape == (5,)
    assert z.shape == (5, 3)

