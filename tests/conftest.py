# tests/conftest.py
import pytest, jax.numpy as jnp

@pytest.fixture
def batch_x():
    return jnp.ones((16,)) * 0.5
