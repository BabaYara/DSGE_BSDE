from .nets import ResNet, ResNetND
from .solver import Solver, SolverND, BSDEProblem

__all__ = [
    "ResNet",
    "ResNetND",
    "Solver",
    "SolverND",
    "BSDEProblem",
    "load_solver",
    "load_solver_nd",
]


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


def load_solver_nd(
    model: BSDEProblem,
    dim: int,
    *,
    dt: float = 0.05,
    depth: int = 8,
    width: int = 128,
) -> SolverND:
    """Factory: create SolverND with ResNetND(depth,width) on given model."""
    import jax

    key = jax.random.PRNGKey(0)
    net = ResNetND.make(dim, depth, width, key=key)
    return SolverND(net, model, dt)
