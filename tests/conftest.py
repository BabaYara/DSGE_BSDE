# tests/conftest.py
import sys
from pathlib import Path

import pytest
import jax.numpy as jnp

# Ensure the project root is on ``sys.path`` so ``bsde_dsgE`` and
# ``kfac_pinn`` can be imported when running tests via ``pytest``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

@pytest.fixture
def batch_x():
    return jnp.ones((16,)) * 0.5
