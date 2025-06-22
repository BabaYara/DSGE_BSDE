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
* [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) – required for the
  tutorial notebooks

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

JupyterLab comes with the ``[dev]`` extras so that the tutorial notebooks run
out of the box.

### 2.2\tData

Sample Lucas dividend draws are stored in ``data/dividend_draws.csv`` and
tracked with `dvc`. After cloning the repository, run

```bash
dvc repro fetch-data
```

to fetch the CSV file using the ``fetch-data`` stage defined in
``dvc.yaml``.

## 3\tKFAC for PINNs

`KFACPINNSolver` wraps a network and loss in a tiny training loop. Each
iteration calls `kfac_update` to apply a diagonal KFAC natural gradient step.
`kfac_update` is JIT-compiled with `eqx.filter_jit`, so JAX must be installed
with a working JIT backend.
The notebooks in the [`notebooks`](notebooks/) directory provide hands-on
examples:

* [`kfac_demo.ipynb`](notebooks/kfac_demo.ipynb) – minimal usage
* [`kfac_toy_example.ipynb`](notebooks/kfac_toy_example.ipynb) – quadratic toy
  problem
* [`kfac_pinn_quickstart.ipynb`](notebooks/kfac_pinn_quickstart.ipynb) – Poisson
  example
* [`kfac_pinn_pkg_quickstart.ipynb`](notebooks/kfac_pinn_pkg_quickstart.ipynb)
  – using the `bsde_dsgE.kfac` package
* [`kfac_pinn_dirichlet_neumann.ipynb`](notebooks/kfac_pinn_dirichlet_neumann.ipynb) – mixed Dirichlet/Neumann Poisson example
* [`pinn_kfac_quickstart_pkg.ipynb`](notebooks/pinn_kfac_quickstart_pkg.ipynb)
  – integrated module
* [`grid_search.py`](examples/grid_search.py) – sweep risk aversion values
* `pinn-demo` – command-line Poisson PINN demo
* `pinn-poisson2d` – 2-D Poisson PINN demo

```bash
$ pinn-poisson2d
final loss ...
```

The helper :func:`bsde_dsgE.kfac.pinn_loss` now accepts custom Dirichlet or
Neumann boundary functions.  Pass ``dirichlet_bc`` or ``neumann_bc`` when
constructing the loss to enforce non-zero conditions.

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

### 5.1 Public API

| Import path | Description | Source file |
|-------------|-------------|-------------|
| `bsde_dsgE.core.Solver` | Base BSDE solver | [`bsde_dsgE/core/solver.py`](bsde_dsgE/core/solver.py) |
| `bsde_dsgE.core.ResNet` | Minimal residual network | [`bsde_dsgE/core/nets.py`](bsde_dsgE/core/nets.py) |
| `bsde_dsgE.core.load_solver` | Factory for ``Solver`` and ``ResNet`` | [`bsde_dsgE/core/__init__.py`](bsde_dsgE/core/__init__.py) |
| `bsde_dsgE.kfac.KFACPINNSolver` | KFAC training loop | [`bsde_dsgE/kfac/solver.py`](bsde_dsgE/kfac/solver.py) |
| `bsde_dsgE.kfac.kfac_update` | Single KFAC step | [`bsde_dsgE/kfac/optimizer.py`](bsde_dsgE/kfac/optimizer.py) |
| `bsde_dsgE.kfac.poisson_1d_residual` | 1‑D Poisson residual | [`bsde_dsgE/kfac/pde.py`](bsde_dsgE/kfac/pde.py) |
| `bsde_dsgE.kfac.pinn_loss` | Poisson loss helper | [`bsde_dsgE/kfac/pde.py`](bsde_dsgE/kfac/pde.py) |
| `bsde_dsgE.utils.sobol_brownian` | Sobol Brownian paths | [`bsde_dsgE/utils/sde_tools.py`](bsde_dsgE/utils/sde_tools.py) |
| `bsde_dsgE.models.ct_lucas.scalar_lucas` | Example Lucas model | [`bsde_dsgE/models/ct_lucas.py`](bsde_dsgE/models/ct_lucas.py) |

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

## 13\tDeployment notes

Running the library effectively requires a functional JAX installation. The package works on pure CPU but was designed with GPU or TPU acceleration in mind, especially for high-dimensional PDEs where the memory footprint grows rapidly. The solvers rely on `eqx.filter_jit`, so JAX's JIT compiler must be available. If you plan to run the tutorial notebooks locally, a consumer-grade GPU with at least 8 GB of memory is strongly recommended. CUDA 12 wheels for JAX are available on PyPI and can be installed with `pip`.

For high performance environments, we have experimented with container-based deployments. The project includes a sample `Dockerfile` in the `scripts` folder that installs all dependencies and copies the tutorial notebooks. You can build the image with

```bash
docker build -f scripts/Dockerfile -t bsde-dsge .
```

and run it with

```bash
docker run --rm -it -p 8888:8888 bsde-dsge jupyter lab --no-browser --ip=0.0.0.0
```

This provides a reproducible environment for tutorials and helps avoid version mismatches across machines. On managed clusters, you can use the same container as a base image and add system-specific launch scripts for the scheduler of choice, e.g. Slurm or Torque. The `scripts/` directory includes small templates for interactive versus batch jobs. Note that the container requires a recent version of CUDA and the corresponding driver.

## 14\tDesign rationale

The core library aims to remain small yet expressive. We deliberately avoid hiding the underlying JAX mechanics: users are expected to interact with pure functions and explicit updates. The choice of Equinox over other neural network libraries reflects a preference for minimalism and first-class PyTree support, which simplifies state management when differentiating through solver iterations.

KFAC was selected as the base optimiser because it provides stable updates even for stiff BSDEs. Standard gradient descent often struggles with vanishing or exploding gradients in long time horizons. KFAC uses a Kronecker-factored approximation of the curvature matrix, capturing the geometry of residual networks at a modest computational cost. The modular design means you can replace `kfac_update` with any Optax-compatible optimiser. Inside the solver loop, the residual function is kept separate from the network forward pass, making it straightforward to swap in alternative PDEs or add custom boundary conditions.

A secondary design goal is teaching. Every class and helper function is thoroughly typed and documented. Many functions include extensive inline comments that walk through the mathematical derivation or highlight subtle implementation details. This approach makes the repository a friendly reference for newcomers to continuous-time DSGE models while still offering advanced hooks for researchers.

## 15\tAdvanced API usage

While the quick-start examples cover basic training loops, the API also supports more specialised workflows. For instance, you can inject custom callback functions into `KFACPINNSolver` to log diagnostics or modify the optimisation state on the fly. Simply pass a callable through the `callbacks` argument when constructing the solver:

```python
from bsde_dsgE.kfac import KFACPINNSolver

solver = KFACPINNSolver(
    net, loss, step_size=1e-2,
    callbacks=[my_logging_hook, anneal_step]
)
```

Callbacks receive the current iteration number, parameter tree and auxiliary data returned by the loss function. They can return an updated parameter tree or operate purely for side effects. This mechanism allows for easy integration with experiment tracking tools like Weights & Biases or custom learning rate schedules without modifying the core training loop.

Another advanced feature is partial freezing of network layers. Because the parameters live in a PyTree, you can filter specific subtrees when passing them to `eqx.apply_updates`. The built-in utility `filter_params` demonstrates this pattern and can be extended to implement layer-wise adaptation or two-timescale updates where the last residual block receives a smaller learning rate.

## 16\tExample results & reproducibility

The repository ships with a set of synthetic data in `data/` that reproduces the Lucas tree experiments. For each notebook we provide a fixed random seed so the figures should match the ones in the documentation. To verify the installation, run

```bash
pytest tests/test_pde.py::test_poisson_solution
```

which checks that the Poisson residual network converges to a known analytic solution. The test executes quickly on CPU and serves as a minimal smoke test. More comprehensive integration tests cover the outer loop of the DSGE solver and ensure consistent output across multiple devices.

If you plan to publish results based on this repository, we encourage you to create a new virtual environment or container and start from a tagged release. The changelog tracks API-breaking changes, and the pinned dependencies in `pyproject.toml` guarantee deterministic builds. When possible, open-source your configuration files and note the commit hash of the version you used in your paper or presentation.

## 17\tExtended support

We maintain a small set of community resources beyond the documentation. The `docs/faq.md` file answers frequent questions about JAX installation, while the issue tracker is monitored for bug reports and feature requests. If you encounter difficulties adapting the code to a custom PDE or integrating with other libraries, please open an issue with a minimal reproducer. We cannot promise immediate replies, but we do our best to point you in the right direction or review pull requests that fix a well-defined problem.
