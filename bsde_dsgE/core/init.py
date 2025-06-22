from .solver import Solver  # noqa
from .nets import ResNet  # noqa

__all__ = ["load_solver"]


from .solver import BSDEProblem


def load_solver(
    model: BSDEProblem,
    *,
    dt: float = 0.05,
    depth: int = 8,
    width: int = 128,
) -> Solver:
    """Factory: create Solver with ResNet(depth,width) on given model."""
    import jax
    import equinox as eqx

    key = jax.random.PRNGKey(0)
    net = ResNet.make(depth, width, key=key)
    return Solver(net, model, dt)
