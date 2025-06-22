import bsde_dsgE.models.ct_lucas as ct_lucas
from bsde_dsgE.core.solver import BSDEProblem


def test_scalar_lucas_returns_problem():
    assert isinstance(ct_lucas.scalar_lucas(), BSDEProblem)


def test_ct_lucas_no_callable_attribute():
    assert not hasattr(ct_lucas, "Callable")
