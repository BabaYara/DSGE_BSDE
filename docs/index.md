# KFAC PINN Documentation

This site collects minimal documentation for the `bsde_dsgE.kfac` utilities. The tutorial notebooks are rendered directly via ``nbsphinx``.

.. toctree::
   :maxdepth: 1

   ../notebooks/kfac_demo.ipynb
   ../notebooks/kfac_toy_example.ipynb
   ../notebooks/kfac_pinn_quickstart.ipynb
   ../notebooks/kfac_pinn_pkg_quickstart.ipynb
   ../notebooks/kfac_pinn_dirichlet_neumann.ipynb

These notebooks walk through defining a network, setting up ``KFACPINNSolver`` and running the optimisation loop.

The repository includes a small CSV dataset `data/dividend_draws.csv` which is
committed to the repository. To regenerate the file run
``python scripts/generate_dividend_draws.py``.

For a history of notable changes, see the
[`CHANGELOG`](../CHANGELOG.md) in the repository root.
