# BSDE_04 fix plan

Status: Iteration 1 complete

Scope
- Keep the spirit/style of the existing write‑up; only fix correctness and robustness.
- Make the notebook run cleanly under FAST=True with mocked JAX.

Actions
1) Self-check cell: remove undefined names and incorrect signature.
   - Before: simulate_panel(_params, _a, T=..., N=..., seed=...)
   - After:  simulate_panel(params, T=..., N=..., seed=...)
   - Print concise shape/mean confirmation.
2) RNG hygiene in mocks: ensure split() returns independent RandomState objects; fold_in is deterministic.
3) Minor polish: smoke-plot cell uses a tiny figure size and ensures output dir exists.

Verification
- Main training/simulation cell already prints without exceptions in FAST mode.
- Self-check now executes without NameError/signature mismatch; prints R_mkt length and mean.

Notes
- Mocked grad keeps training inert by design; numeric targets in printed table remain illustrative.
- When migrating to real JAX, replace MockJAX/MockRandom with jax/jax.random and remove prints.
# BSDE_04 notebook fix log

Purpose: resolve JAX TypeError on `value_and_grad(bellman)(params)` by making `SpectralParams` a JAX PyTree, then validate end-to-end execution under FAST mode.

## Plan (brief)

- Diagnose error and consult JAX docs (PyTrees, value_and_grad)
- Patch: register `SpectralParams` dataclass as PyTree; clean gradient handling in `sam_step`
- Validate: run main cell in `BSDE_04.ipynb` end-to-end (FAST)
- Record results and lightweight next steps

## Actions taken

- Registered `SpectralParams` as a JAX PyTree via `@functools.partial(jax.tree_util.register_dataclass, data_fields=['W1','b1','W2','b2','Wc','bc'], meta_fields=['mu_logK','r_logK','deg'])`
- Imported `jax` (explicitly) to access `tree_util`
- Fixed gradient handling in `sam_step`:
  - Replace ad-hoc flattening with `gflat = flatten(g)` and `g2flat = flatten(g2)` (mirrors params structure)
  - Keep optimizer utilities (`flatten/unflatten`, AdamW, clipping) unchanged otherwise

Files touched

- `notebooks/BSDE_04.ipynb`: single code cell edited (registration + gradient fix)

## Validation

- Ran the main code cell (FAST mode). Status: PASS (no exceptions)
- Console excerpt:
  - Training runs one outer loop; price-law R2≈1.0
  - Simulation completes; diagnostics printed; plotting skipped (no `zhang_plots` module)

Note: Some reported statistics show `nan` (market/decile aggregates). This is orthogonal to the original JAX error and can be tracked separately.

## References (via Context7 lookups)

- JAX PyTrees: `register_dataclass`, `register_pytree_node_class`, value_and_grad on custom containers

## Next steps (optional)

- Investigate `nan` aggregates (guard weights, finite checks, early-burn-in trimming)
- Add tiny unit cell to assert that `value_and_grad(bellman)` accepts `SpectralParams`
- Optional: gate plotting import or add stub `zhang_plots`
