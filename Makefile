PYTHON ?= python3

.PHONY: help setup env-check test docs run-notebooks run-mc run-mc-solver run-primitives table1-check strict-table1 train-macro

help:
	@echo "Targets:"
	@echo "  setup           Install package with dev/docs extras (requires JAX for full tests)"
	@echo "  test            Run test suite (NOTEBOOK_FAST=1)"
	@echo "  docs            Build Sphinx docs"
	@echo "  env-check       Print Python/JAX/Equinox/Optax/Jupyter readiness"
	@echo "  run-mc          Execute notebooks/multicountry_probab01.ipynb (NOTEBOOK_FAST=1)"
	@echo "  run-mc-solver   Execute notebooks/multicountry_probab01_solver.ipynb (NOTEBOOK_FAST=1)"
	@echo "  run-primitives  Execute notebooks/primitives_visuals.ipynb (NOTEBOOK_FAST=1)"
	@echo "  run-notebooks   Execute core notebooks (FAST mode)"
	@echo "  table1-check    Run scripts/check_table1.py"
	@echo "  strict-table1   Enforce STRICT_TABLE1 gating for CI/local"
	@echo "  train-macro     Train MacroFinanceNet (short demo)"

setup:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .[dev,docs]
	@echo "Note: Install an appropriate JAX build for your platform (CPU/GPU)."

test:
	NOTEBOOK_FAST=1 pytest -q

docs:
	sphinx-build -n -b html docs docs/_build/html

env-check:
	$(PYTHON) scripts/env_check.py

run-mc:
	NOTEBOOK_FAST=1 $(PYTHON) -m jupyter nbconvert --to notebook --execute notebooks/multicountry_probab01.ipynb --output notebooks/_mc_exec.ipynb

run-mc-solver:
	NOTEBOOK_FAST=1 $(PYTHON) -m jupyter nbconvert --to notebook --execute notebooks/multicountry_probab01_solver.ipynb --output notebooks/_mc_solver_exec.ipynb

run-primitives:
	NOTEBOOK_FAST=1 $(PYTHON) -m jupyter nbconvert --to notebook --execute notebooks/primitives_visuals.ipynb --output notebooks/_primitives_exec.ipynb

run-notebooks: run-primitives run-mc run-mc-solver

table1-check:
	$(PYTHON) scripts/check_table1.py --calib data/probab01_table1.json --steps 50 --paths 16 --dt 0.02

strict-table1:
	STRICT_TABLE1=1 $(PYTHON) scripts/check_table1.py --calib data/probab01_table1.json --steps 50 --paths 16 --dt 0.02

train-macro:
	$(PYTHON) scripts/train_macro_solver.py --epochs 200 --paths 1024 --dt 0.001 --steps 0 --eval-symmetric
