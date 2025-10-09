# BSDE_14 plan and rubric

Status: draft, iteration 1 (simple model refactor)

Goals

- Simplify `Tex/BSDE_14.tex` to focus on the core mean-field costly-reversibility model.
- Defer SymPy/Lean4 artifacts to an appendix; keep narrative clean.
- Validate LaTeX compiles cleanly (no shell-escape required).
- Maintain existing section structure and style.

Scope (this iteration)

- Add a simple-mode switch to suppress heavy code boxes in body.
- Insert Appendix A placeholder for verification artifacts.
- Keep equations, HJB–FP, ME, and market mapping; prune extraneous sidebars.
- Leave figures and tables intact where they support the core narrative.

Next

- Tighten the master-equation section to the minimal conditional-on-x form.
- Move any remaining SymPy/Lean snippets into Appendix A.
- Run latexmk/pdflatex and address any compile warnings/errors.

20-line rubric (evolves each edit)

1) Builds without error (pdflatex/latexmk) in safe mode.

2) No shell-escape required for a default build.

3) Core primitives consistent and minimal.

4) HJB stated with clean policy mapping.

5) FP with boundary conditions clearly stated.

6) ME presented conditional on x; no measure diffusion.

7) Lions vs Flat derivative distinction explicit.

8) Price externality simplified; isoelastic identity used.

9) Monotonicity condition stated with sign check.

10) Figures compile; captions concise.

11) Acronyms/notation table compiles.

12) Bibliography loads (biblatex/biber) but tolerates draft runs.

13) SymPy/Lean moved to appendix or suppressed.

14) No unresolved references/labels.

15) No overfull boxes > 2em (microtype + stretch in place).

16) TikZ/pgfplots guarded with compat=1.18.

17) No undefined control sequences.

18) Avoid duplicate package loads and filter noisy warnings.

19) Simple-mode toggle documented.

20) Appendix pointer present and informative.
