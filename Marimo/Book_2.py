import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt 

    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""# BSDE_6 · Book 2 (scaffold)""")
    return


@app.cell
def _(mo):
    import sys
    import platform
    import numpy as np
    # Show environment info and set a deterministic seed for later probes
    py = sys.version.split()[0]
    try:
        np_ver = np.__version__
    except Exception:
        np_ver = "unknown"
    from __main__ import __generated_with as _gen
    mo.callout(mo.md(
        f"- Python {py}\n"
        f"- marimo { _gen }\n"
        f"- NumPy {np_ver}\n"
        f"- Platform {platform.system()}"
    ), kind="info")
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, 3)
    mo.callout(mo.md(f"Deterministic probe (seed=123): {np.round(x, 4).tolist()}"), kind="neutral")
    return


@app.cell
def _(mo):
    mo.callout(mo.md(
        "This is a minimal scaffold for BSDE_6.tex conversion.\n\n"
        "We will add content incrementally with TDD after verifying this opens."
    ), kind="info")
    return


if __name__ == "__main__":
    app.run()
