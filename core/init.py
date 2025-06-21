from .solver import Solver  # noqa
from .nets import ResNet    # noqa

__all__ = ["load_solver"]


def load_solver(model, *, dt: float = 0.05, depth: int = 8, width: int = 128):
    """Factory: create Solver with ResNet(depth,width) on given model."""
    import jax, equinox as eqx

    key = jax.random.PRNGKey(0)
    net = ResNet.make(depth, width, key)
    return Solver(net, model, dt)
