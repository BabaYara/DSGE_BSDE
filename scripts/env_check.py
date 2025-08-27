"""Environment readiness check for bsde_dsgE.

Runs a few imports and prints versions to help diagnose missing deps before
executing notebooks or tests.
"""

from __future__ import annotations

import importlib
import sys


def _try_import(name: str):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        return True, ver
    except Exception as e:
        return False, str(e)


def main() -> None:
    checks = [
        ("jax",),
        ("equinox",),
        ("optax",),
        ("scipy",),
        ("jupyter",),
        ("pytest",),
    ]
    ok = True
    print({"python": sys.version.split()[0]})
    for (name,) in checks:
        good, info = _try_import(name)
        ok = ok and good
        print({name: info if good else f"MISSING ({info})"})
    if not ok:
        print("One or more required packages are missing.\n"
              "Install dev/docs extras and a JAX wheel for your platform.\n"
              "  pip install -e .[dev,docs]\n"
              "  # and install jax[cpu] or a cuda wheel per your setup\n")


if __name__ == "__main__":
    main()

