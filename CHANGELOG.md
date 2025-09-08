# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]
### Added
- Tex/BSDE_11.tex: Explicit stationary Master Equation (ME) with equation label and a formal price-externality proposition; new Appendix F with Lean4 micro-proofs (isoelastic identity, algebraic rpow reduction).
- SymPy verification coverage in Appendix E referenced from ME (no new deps).
- Tex/BSDE_11.tex preamble: minted package and new verification box environments `sympycheck` and `leanproof` for inline verification artifacts in Section 7.
### Changed
- Tex/BSDE_11.tex: Clarified ME section and math-to-code linkage; fixed a formatting typo in the common-noise remark; tightened notation around $P'(Y)$ and $Y(m,x)$.
- Section 7: Restored and restructured the Master Equation, introduced transport operator $\mathcal{T}[\delta_m U]$, expanded Proposition~\textit{Price externality} proof, and inserted verification boxes (SymPy and Lean4).
### Removed
-

## [0.0.1a1] - 2025-06-23
### Changed
- Added `scipy` to the core dependencies.

## [0.1.0] - 2025-06-22
### Added
- `bsde_dsgE.kfac` subpackage providing a diagonal KFAC solver for Physics-informed neural networks.
- Basic PDE helpers for the 1D Poisson problem.
- Example notebooks demonstrating the solver.
- Skeleton modules for continuous-time DSGE models.
