from bsde_dsgE.core.nets import ResNet
import jax.numpy as jnp, jax

def test_resnet_shapes():
    net = ResNet.make(depth=4, width=64, key=jax.random.PRNGKey(0))
    y, z = net(jnp.zeros((8,)), jnp.zeros((8,)))
    assert y.shape == z.shape == (8,)
