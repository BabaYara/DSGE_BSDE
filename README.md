# Deep‑BSDE ‑‑ Continuous‑Time DSGE Solver
### *A progressive curriculum & modular JAX library*

---

## 1 Vision

**Deep‑BSDE methods are the new workhorse** for solving high‑dimensional
non‑linear PDEs.  Continuous‑time DSGE models—Lucas trees, two‑agent
Epstein–Zin economies, production networks—fit exactly that mould.  Yet
economists interested in BSDEs face a fragmented landscape of code
snippets and theory papers.

This repository bridges that gap by providing

* a **step‑by‑step notebook series** ▷ from the one‑state Lucas toy model
  to state‑of‑the‑art stochastic‑volatility, two‑tree, two‑agent models,
  each notebook building on the previous;
* a **clean, type‑hinted JAX library** (`bsde_dsgE`)—solver classes,
  model primitives, residual nets, control‑variates—designed for
  research extension;
* rigorous **testing & CI** so every new contribution (human or agent)
  preserves correctness and style;
* exhaustive inline **commentary** that teaches *why* the method works,
  not only *how* to run it.

Our aim: **from zero to publishable replication** in a weekend.

---

## 2 Quick‑start

### 2.1 Dependencies

* Python 3.11+
* [JAX](https://github.com/google/jax) ≥ 0.4 with CPU **or** CUDA 12 wheel
* [Equinox](https://github.com/patrick-kidger/equinox)
* [Optax](https://github.com/deepmind/optax)

```bash
# ⬇️ user install
pip install "jax[cpu]"  equinox optax bsde_dsgE            # PyPI stub

# ⬇️ developer clone
git clone https://github.com/your‑org/deep‑bsde‑ct‑dsge.git
cd deep‑bsde‑ct‑dsge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
pre‑commit install
pytest -q                     # 30 s smoke‑tests
