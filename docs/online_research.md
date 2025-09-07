# Online Research Log (Multicountry BSDE/DSGE)

Purpose

- Capture external references, calibration sources, and related literature to support reproducing Table 1 in Probab_01.pdf.
- Track follow-ups, open questions, and rationale for modeling choices.

Status

- Network access is restricted in this environment. This file outlines the intended references and placeholders to be filled with citations (titles/DOIs/links) during an online pass.
- Actioned offline: transcribed symmetric-state Table 1 blocks (q and sigma_q) into `data/probab01_table1.json`.

Topics & Leads

- Multicountry asset pricing with production or endowment linkages (survey and replication targets): identify comparable calibrations and moments.
- Deep BSDE and FBSDE numerical methods for high‑dimensional PDEs (algorithms, convergence, practice tips): training heuristics and driver regularisation.
- Quasi‑Monte Carlo (Sobol) for SDEs and variance reduction (pairing, scrambling, Brownian bridges): impact on path‑wise variance and estimator stability.

Key references (to fill with DOIs/links)

- Deep BSDE solvers: Han, Jentzen, E (2018); Beck, E, others on BSDE/PDE connections and high‑dimensional problems.
- Four‑step schemes and FBSDEs: Ma, Protter, Yong (1994); Delarue on decoupling fields.
- Macro‑finance applications with heterogeneous agents and international linkages; calibration practices for ρ, γ, κ, σ.
- Quasi‑Monte Carlo for diffusion simulation: Sobol sequences, Owen scrambling, Brownian bridges; variance reduction in path‑dependent options.
- SIREN networks and spectral bias mitigation for smooth signal approximation; relevance for q(Ω) and σ_q(Ω) mappings.

Planned citations (to fill):

- Han, Jentzen, E. (2018) — Deep learning-based numerical methods for high-dimensional BSDEs.
- Beck et al. — Solving PDEs with deep BSDE methods; error analysis and convergence.
- Ma, Yong — Forward-backward SDE methods (four-step scheme); link to semilinear PDEs.
- Owen — Scrambled net variance bounds; practical implications for Sobol in SDEs.
- Sitzmann et al. (SIREN) — Sinusoidal representation networks and initialization.

Implementation notes (to summarise when sources are linked)

- Symmetric‑state analytic: q = (a·ψ + 1)/(ρ·ψ + 1) enables quick sanity checks for `q` at ζ = 1/J (cf. research_notes.md).
- Sign constraints on σ_q stabilize training and enforce the Table 1 sign pattern prior (diag>0, off<0).
- Fast/slow paths: use NOTEBOOK_FAST for figures; full checks require larger paths/steps for stable moment estimates.
- Rolling correlations: windowed correlation heatmaps surface time‑varying co‑movement across countries; use as a diagnostic when matching covariance blocks.
- Error bars: mean ± 2SE plots track sample uncertainty by dimension; treat as descriptive diagnostics (serial correlation ignored).
 - QQ plots: check increment distributional assumptions (Normal(0, dt)); large deviations indicate bugs in Brownian generation or scaling.
- Lagged autocorrelation: near-zero acf at lag 1 for increments; deviations point to stateful RNG misuse or missing key-splitting.

Context7 notes (docs consulted)

- Equinox (ID: /patrick-kidger/equinox): filter_jit and filter_value_and_grad patterns for JIT‑compiled losses and parameter filtering; MLP usage and partition/combine idioms. Our nets/solvers use these in training/eval helpers.
- JAX (ID: /jax-ml/jax): PRNG best practices — always split keys; vectorise random draws; no sequential equivalence; avoid key reuse; batch keys with `random.split` and `vmap`. Diagnostics (QQ/ACF) included to surface RNG misuse.

Visual standards (Aug 2025)

- Matplotlib (ID: /matplotlib/matplotlib): use accessible color cycles (petroff10/8/6) via `plt.style.use`, set major/minor grid rcParams distinctly, and prefer `animation.html='jshtml'` for inline animations with ARIA‑labeled controls. Our notebooks call `apply_notebook_style()` to configure these.
- Accessibility: color‑blind safe palettes and clear contrast; avoid strobing animations and keep autoplay off; ensure readable fonts and adequate figure DPI.

Action Items

- Identify the exact paper or appendix containing Probab_01 Table 1; extract parameters and moment targets.
- Record the stochastic structure: whether Σ is a covariance, correlation, or Cholesky factor; note any state transformations (e.g., logs).
- Cross-check risk aversion γ, discount ρ, persistence κ with standard calibrations in the literature and note any deviations.
- Add DOIs and links for sources used; summarize key equations and any implementation caveats.
- Add a note on terminal condition used in training vs. path sims; quantify impact on moment matching in a small ablation.
- Record training heuristics that improved stability (learning rate schedule, Z‑term weighting, Batch/DT choices, warm‑starts).

Placeholder References (to fill with specific citations)

- Deep BSDE methods for PDEs in finance and economics (survey articles; key algorithms).
- Lucas (1978) Asset Pricing in an Exchange Economy (baseline CRRA intuition).
- Quasi-Monte Carlo for SDEs (Sobol sequences; antithetic pairing).

Notes

- Once citations are added, link this file from AGENTS.md and the main README to document replication provenance.
- Consider adding brief summaries per citation (2–3 lines) covering: model states and shocks; calibration approach; numerical method; reported figures/tables comparable to Table 1.
- The replication checklist in `docs/replication_checklist.md` summarizes the gating steps used locally and in CI when `STRICT_TABLE1=1` is enabled.
