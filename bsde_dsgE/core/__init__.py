from .nets import ResNet
from .solver import Solver, BSDEProblem

__all__ = ["ResNet", "Solver", "BSDEProblem", "load_solver"]


def load_solver(
    model: BSDEProblem,
    *,
    dt: float = 0.05,
    depth: int = 8,
    width: int = 128,
) -> Solver:
    """Factory: create Solver with ResNet(depth,width) on given model."""
    import jax

    key = jax.random.PRNGKey(0)
    net = ResNet.make(depth, width, key=key)
    return Solver(net, model, dt)
