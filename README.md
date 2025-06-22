# Deep-BSDE -- Continuous-Time DSGE Solver
### *A progressive curriculum & modular JAX library*

---

## 0\tOverview

`bsde_dsgE` provides minimal yet extendable tooling to explore Physics-informed
neural networks (PINNs) and continuous-time DSGE models.  The library focuses on
a Kronecker-Factored Approximate Curvature (KFAC) optimiser with helper
utilities and a collection of worked notebook examples.

## 1\tVision

**Deep-BSDE methods are the new workhorse** for solving high-dimensional
non-linear PDEs.  Continuous-time DSGE models—Lucas trees, two-agent
Epstein–Zin economies, production networks—fit exactly that mould.  Yet
researchers interested in BSDEs face a fragmented landscape of code snippets and
theory papers.

This repository bridges that gap by providing

* a **step-by-step notebook series** ▷ from the one-state Lucas toy model to
  state-of-the-art stochastic-volatility, two-tree, two-agent models, each
  notebook building on the previous;
* a **clean, type-hinted JAX library** (`bsde_dsgE`)—solver classes,
  model primitives, residual nets, control-variates—designed for research
  extension;
* rigorous **testing & CI** so every new contribution (human or agent) preserves
  correctness and style;
* exhaustive inline **commentary** that teaches *why* the method works, not only
  *how* to run it.

Our aim: **from zero to publishable replication** in a weekend.

---

## 2\tQuick-start

### 2.1\tDependencies

* Python 3.11+
* [JAX](https://github.com/google/jax) ≥ 0.4 with CPU **or** CUDA 12 wheel
* [Equinox](https://github.com/patrick-kidger/equinox)
* [Optax](https://github.com/deepmind/optax)
* [SciPy](https://scipy.org)

```bash
# ⬇️ user install
pip install "jax[cpu]"  equinox optax scipy bsde-dsge

# ⬇️ developer clone
git clone https://github.com/your-org/deep-bsde-ct-dsge.git
cd deep-bsde-ct-dsge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
pre-commit install
pytest -q                     # 30 s smoke-tests
```

The distribution on PyPI is named ``bsde-dsge`` while the Python package is
imported as ``bsde_dsgE``.

## 3\tKFAC for PINNs

`KFACPINNSolver` wraps a network and loss in a tiny training loop. Each
iteration calls `kfac_update` to apply a diagonal KFAC natural gradient step.
The notebooks in the [`notebooks`](notebooks/) directory provide hands-on
examples:

* [`kfac_demo.ipynb`](notebooks/kfac_demo.ipynb) – minimal usage
* [`kfac_toy_example.ipynb`](notebooks/kfac_toy_example.ipynb) – quadratic toy
  problem
* [`kfac_pinn_quickstart.ipynb`](notebooks/kfac_pinn_quickstart.ipynb) – Poisson
  example
* [`kfac_pinn_pkg_quickstart.ipynb`](notebooks/kfac_pinn_pkg_quickstart.ipynb)
  – using the `bsde_dsgE.kfac` package
* [`pinn_kfac_quickstart_pkg.ipynb`](notebooks/pinn_kfac_quickstart_pkg.ipynb)
  – integrated module
* [`grid_search.py`](examples/grid_search.py) – sweep risk aversion values

See the generated documentation in [`docs/`](docs/) for a rendered version of
these tutorials.

## 4\tExample notebooks

All tutorial notebooks live in the [`notebooks/`](notebooks/) folder.  Launch
JupyterLab and open any notebook to reproduce the results shown in the
documentation.

```bash
jupyter lab notebooks/
```

## 5\tLibrary overview

The project exposes a single package:

* **`bsde_dsgE`** – library containing KFAC utilities, PDE helpers and
  skeleton continuous-time DSGE solvers.

The package follows standard JAX/Equinox design with optax-style updates and
NumPy-style docstrings.

## 6\tDevelopment setup

After cloning the repository install the development dependencies and activate
pre-commit hooks:

```bash
pip install -e ".[dev,docs]"
pre-commit install
```

The hooks enforce code style via `black`, `ruff` and `mypy`.

## 7\tTesting

Run the full test-suite with `pytest`:

```bash
pytest -q
```

Tests cover the KFAC optimiser, PDE utilities and example integration paths.

## 8\tDocumentation

A minimal Sphinx site is located in [`docs/`](docs/).  Build the HTML pages with

```bash
sphinx-build -b html docs docs/_build
```

The site links directly to the executed notebooks for step-by-step tutorials.

## 9\tContributing

Contributions are welcome!  Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for
coding conventions and the recommended workflow.  Pull requests should reference
the relevant milestone from the table below.

## 10\tRoadmap

The project evolves through small, well-defined milestones.  Features are added
incrementally while keeping the code base easy to understand.

## 11\tMilestones

| ID   | Summary                                   | Status |
|------|-------------------------------------------|:------:|
| M-01 | Initial project scaffold                  |   ✔   |
| M-02 | Clarify package naming                    |   ✔   |
| M-03 | Sobol generator & Pareto root-finding     |   ✔   |
| M-04 | Pre-commit hooks and style guidelines     |   ✔   |
| M-05 | Basic KFAC solver implementation          |   ✔   |
| M-06 | Example PINN notebooks                    |   ✔   |
| M-07 | Continuous-time DSGE solver skeleton      |   ✔   |
| M-08 | Documentation site and tutorials          |   ✔   |
| M-09 | CI with tests for KFAC utilities          |   ✔   |
| M-10 | Public API re-exports                     |   ✔   |
| M-11 | Compatibility tests across packages       |   ✔   |
| M-12 | Future enhancements                       |   ☐   |

## 12\tLicense

This project is licensed under the terms of the MIT license.  See
[`LICENSE`](LICENSE) for details.

