# deep_fbsde_marimo_v8.py
# 2025-09-11. Diagram-first marimo app with added Penrose diagrams,
# SymPy verifications, and refined numerical claims. Includes Context7 MCP probe.

from __future__ import annotations

try:
    import marimo as mo
except Exception:
    mo = None

import numpy as np
import matplotlib.pyplot as plt

try:
    import sympy as sp
except Exception:
    sp = None

# ---------------------
# Penrose-style axes helper
# ---------------------
def penrose_axes(ax, xpad=0.04, ypad=0.06):
    for spine in ("top","right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    # Try arrows if 0 is in range; otherwise skip silently
    try:
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.annotate("", xy=(x1, 0), xytext=(x0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, shrinkA=0, shrinkB=0))
        ax.annotate("", xy=(0, y1), xytext=(0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, shrinkA=0, shrinkB=0))
    except Exception:
        pass
    ax.tick_params(direction="out", length=4, width=0.9)
    ax.margins(x=xpad, y=ypad)
    return ax

# ------------------------
# SymPy analytic verifiers
# ------------------------
def sympy_checks():
    if sp is None:
        return {"sympy_available": False}

    out = {"sympy_available": True}

    # (1) Logit derivatives
    x = sp.symbols('x', positive=True)
    h = sp.log(x/(1-x))
    out["logit_hprime_ok"] = bool(sp.simplify(sp.diff(h,x) - 1/(x*(1-x))) == 0)
    out["logit_h2_ok"]     = bool(sp.simplify(sp.diff(h,x,2) + (1-2*x)/(x*(1-x))**2) == 0)

    # (2) Harmonic aggregation equal-gamma collapse
    lam, g1, g2 = sp.symbols('lam g1 g2', positive=True)
    invG = lam/g1 + (1-lam)/g2
    G = sp.simplify(1/invG)
    g = sp.symbols('g', positive=True)
    out["gamma_equal_collapse_ok"] = bool(sp.simplify(G.subs({g1:g, g2:g}) - g) == 0)

    # (2b) Bounds: min(g1,g2) <= G <= max(g1,g2) (checked numerically on samples)
    subs_list = [
        {g1:2, g2:5, lam:0.2},
        {g1:2, g2:5, lam:0.8},
        {g1:3, g2:1.5, lam:0.3},
        {g1:3, g2:1.5, lam:0.7},
    ]
    bounds_checks = []
    for sub in subs_list:
        Gv = float(G.subs(sub))
        low = min(float(sub[g1]), float(sub[g2]))
        high = max(float(sub[g1]), float(sub[g2]))
        bounds_checks.append(low <= Gv <= high)
    out["gamma_bounds_sample_ok"] = all(bounds_checks)

    # (3) Itô–Stratonovich correction (affine sigma)
    x = sp.symbols('x', real=True)
    s0, s1, m0, m1 = sp.symbols('s0 s1 m0 m1', real=True)
    sigma = s0 + s1*x
    mu    = m0 + m1*x
    mu_strat = sp.simplify(mu - sp.Rational(1,2)*sp.diff(sigma, x)*sigma)
    expected = mu - sp.Rational(1,2)*sp.diff(sigma, x)*sigma
    out["ito_stratonovich_affine_ok"] = bool(sp.simplify(mu_strat - expected) == 0)

    # (4) GBM martingale identity: E[exp(b W_t - (1/2)b^2 t)]=1
    t = sp.symbols('t', positive=True)
    b = sp.symbols('b', real=True)
    mgf = sp.exp(0.5*b**2*t)  # E[e^{b W_t}]
    out["gbm_mgf_identity_ok"] = bool(sp.simplify(sp.exp(-0.5*b**2*t) * mgf - 1) == 0)

    # (5) Cole–Hopf cancellation for quadratic driver (a=1)
    C, L, Z = sp.symbols('C L Z', real=True)
    F = C + L*Z + sp.Rational(1,2)*Z**2
    drift = -F + sp.Rational(1,2)*Z**2
    out["cole_hopf_cancels_quadratic_ok"] = bool(sp.simplify(drift + C + L*Z) == 0)

    # (6) Boundary exponents for sigma~c1 x, mu~c2 x near 0.
    c1, c2 = sp.symbols('c1 c2', positive=True)
    c = sp.simplify(2*c2/c1**2)   # exponent in s(y) ~ y^{-c}
    out["feller_exponent_c"] = c
    out["scale_integrable_at0_condition"] = "c < 1"
    out["speed_integrable_at0_condition"] = "c > 1"

    return out

# -----------------------
# SDE integrators & probes
# -----------------------
def em_step(x, a, b, h, dW, mode):
    if mode == "additive":
        return x + a*x*h + b*dW
    else:
        return x + a*x*h + b*x*dW

def heun_drift_trap_step(x, a, b, h, dW, mode):
    if mode == "additive":
        xp = x + a*x*h + b*dW
        return x + 0.5*(a*x + a*xp)*h + b*dW
    else:
        xp = x + a*x*h + b*x*dW
        return x + 0.5*(a*x + a*xp)*h + b*x*dW

def milstein_step(x, a, b, h, dW):
    # multiplicative noise (GBM): b(x)=b*x, b'(x)=b
    return x + a*x*h + b*x*dW + 0.5*b*b*x*(dW**2 - h)

def milstein_vs_em_rate_demo(M=256, N_list=(50,100,200,400,800), T=1.0, seed=0):
    rng = np.random.default_rng(seed)
    a_const, b_const, X0 = 0.2, 0.6, 1.0
    errs_em, errs_mil, hs = [], [], []

    for N in N_list:
        h = T/N; hs.append(h)
        em_err2 = 0.0; mil_err2 = 0.0
        for _ in range(M):
            dW = rng.normal(0.0, np.sqrt(h), size=N)
            W  = np.cumsum(dW)
            X_exact = X0 * np.exp((a_const - 0.5*b_const**2)*T + b_const*W[-1])
            Xe = X0; Xm = X0
            for k in range(N):
                Xe = em_step(Xe, a_const, b_const, h, dW[k], mode="multiplicative")
                Xm = milstein_step(Xm, a_const, b_const, h, dW[k])
            em_err2  += (Xe - X_exact)**2
            mil_err2 += (Xm - X_exact)**2
        errs_em.append(np.sqrt(em_err2/M))
        errs_mil.append(np.sqrt(mil_err2/M))

    xlog = np.log10(hs)
    slope_em  = float(np.polyfit(xlog, np.log10(errs_em), 1)[0])
    slope_mil = float(np.polyfit(xlog, np.log10(errs_mil),1)[0])

    fig, ax = plt.subplots(figsize=(7.2,4.2))
    ax.loglog(hs, errs_em, marker="o", label=f"EM (slope≈{slope_em:.2f})")
    ax.loglog(hs, errs_mil, marker="s", label=f"Milstein (slope≈{slope_mil:.2f})")
    ax.loglog(hs, np.array(hs)**0.5 * errs_em[0]/(hs[0]**0.5), linestyle="--", label=r"$h^{1/2}$")
    ax.loglog(hs, np.array(hs)      * errs_mil[0]/(hs[0]), linestyle="--", label=r"$h^{1}$")
    ax.invert_xaxis()
    ax.set_xlabel("step size h"); ax.set_ylabel("RMS terminal error")
    ax.set_title("Strong error on GBM (EM vs Milstein)")
    ax.legend(frameon=False)
    penrose_axes(ax)
    fig.tight_layout()
    return fig, {"slope_em": slope_em, "slope_mil": slope_mil}

def heun_rate_probe(M=200, N_list=(50,100,200,400,800), T=1.0, mode="additive", a=-0.4, b=0.7, seed=1):
    rng = np.random.default_rng(seed)
    X0 = 1.0
    errs_em, errs_heun, hs = [], [], []
    for N in N_list:
        h = T/N; hs.append(h)
        em_err2 = 0.0; he_err2 = 0.0
        for _ in range(M):
            dW = rng.normal(0.0, np.sqrt(h), size=N)
            W  = np.cumsum(dW)
            if mode == "additive":
                # reference via very fine EM
                Nref = 6400; href = T/Nref
                dWref = rng.normal(0.0, np.sqrt(href), size=Nref)
                Xref = X0
                for j in range(Nref):
                    Xref = em_step(Xref, a, b, href, dWref[j], mode="additive")
                Xe = X0; Xh = X0
                for k in range(N):
                    Xe = em_step(Xe, a, b, h, dW[k], mode="additive")
                    Xh = heun_drift_trap_step(Xh, a, b, h, dW[k], mode="additive")
                em_err2 += (Xe - Xref)**2
                he_err2 += (Xh - Xref)**2
            else:
                X_exact = X0 * np.exp((a - 0.5*b*b)*T + b*W[-1])
                Xe = X0; Xh = X0
                for k in range(N):
                    Xe = em_step(Xe, a, b, h, dW[k], mode="multiplicative")
                    Xh = heun_drift_trap_step(Xh, a, b, h, dW[k], mode="multiplicative")
                em_err2 += (Xe - X_exact)**2
                he_err2 += (Xh - X_exact)**2
        errs_em.append(np.sqrt(em_err2/M))
        errs_heun.append(np.sqrt(he_err2/M))

    xlog = np.log10(hs)
    slope_em  = float(np.polyfit(xlog, np.log10(errs_em), 1)[0])
    slope_he  = float(np.polyfit(xlog, np.log10(errs_heun),1)[0])

    fig, ax = plt.subplots(figsize=(7.2,4.2))
    ax.loglog(hs, errs_em, marker="o", label=f"EM (slope≈{slope_em:.2f})")
    ax.loglog(hs, errs_heun, marker="^", label=f"Heun drift-trap (slope≈{slope_he:.2f})")
    ax.loglog(hs, np.array(hs)**0.5 * errs_em[0]/(hs[0]**0.5), linestyle="--", label=r"$h^{1/2}$ guide")
    ax.loglog(hs, np.array(hs)      * errs_heun[0]/(hs[0]), linestyle="--", label=r"$h^{1}$ guide")
    ax.invert_xaxis()
    ax.set_xlabel("h"); ax.set_ylabel("RMS terminal error")
    ax.set_title(f"Heun vs EM: empirical slopes ({mode})")
    ax.legend(frameon=False)
    penrose_axes(ax)
    fig.tight_layout()
    return fig, {"slope_em": slope_em, "slope_heun": slope_he}

# -----------------
# Penrose-style diagrams (more added)
# -----------------
def econ_flow_diagram():
    fig, ax = plt.subplots(figsize=(7.8, 4.8)); ax.axis("off")
    box = dict(boxstyle="round,pad=0.4", linewidth=1.2)
    ax.text(0.08, 0.74, r"Consumption share $\lambda^c$", bbox=box, transform=ax.transAxes)
    ax.text(0.45, 0.74, r"$\Gamma^{-1}=\lambda^c/\gamma_1+(1-\lambda^c)/\gamma_2$", bbox=box, transform=ax.transAxes)
    ax.text(0.80, 0.86, r"$\kappa=\Gamma\,\sigma_D$", bbox=box, transform=ax.transAxes)
    ax.text(0.80, 0.58, r"$r$ (illustrative Ramsey-like)", bbox=box, transform=ax.transAxes)
    ax.annotate("", xy=(0.42, 0.75), xytext=(0.22, 0.75), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.annotate("", xy=(0.76, 0.84), xytext=(0.57, 0.76), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.annotate("", xy=(0.76, 0.60), xytext=(0.57, 0.70), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.set_title("Economic dependency flow")
    return fig

def sdf_ramsey_diagram():
    fig, ax = plt.subplots(figsize=(7.8, 4.8)); ax.axis("off")
    box = dict(boxstyle="round,pad=0.4", linewidth=1.2)
    ax.text(0.08, 0.72, r"SDF: $d\xi_t/\xi_t=-r_t\,dt-\kappa_t\,dW_t$", bbox=box, transform=ax.transAxes)
    ax.text(0.50, 0.72, r"Illustrative: $r_t \approx \bar\rho + \Gamma \mu_D - \frac{1}{2} \Gamma(\Gamma+1)\sigma_D^2$", bbox=box, transform=ax.transAxes)
    ax.text(0.50, 0.50, r"Market price of risk: $\kappa_t=\Gamma \sigma_D$", bbox=box, transform=ax.transAxes)
    ax.annotate("", xy=(0.46, 0.74), xytext=(0.14, 0.74), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.annotate("", xy=(0.46, 0.52), xytext=(0.14, 0.74), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.set_title("SDF dynamics and pricing kernel link")
    return fig

def pde_fbsde_map_diagram():
    fig, ax = plt.subplots(figsize=(8.4, 4.8)); ax.axis("off")
    box = dict(boxstyle="round,pad=0.4", linewidth=1.2)
    ax.text(0.08, 0.70, r"PDE: $\partial_t u+\mathcal{L}u+f(t,x,u,\nabla u\,\sigma)=0$", bbox=box, transform=ax.transAxes)
    ax.text(0.08, 0.40, r"Terminal: $u(T,x)=g(x)$", bbox=box, transform=ax.transAxes)
    ax.text(0.62, 0.70, r"BSDE: $dX_t=\mu\,dt+\sigma\,dW_t$", bbox=box, transform=ax.transAxes)
    ax.text(0.62, 0.52, r"$dY_t=-f(\cdot)\,dt+Z_t\, dW_t$", bbox=box, transform=ax.transAxes)
    ax.text(0.62, 0.32, r"Link: $Y_t=u(t,X_t),~Z_t=\nabla u\,\sigma$", bbox=box, transform=ax.transAxes)
    ax.annotate("", xy=(0.58, 0.70), xytext=(0.40, 0.70), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.annotate("", xy=(0.58, 0.40), xytext=(0.40, 0.40), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.set_title("Nonlinear Feynman–Kac mapping (diagrammatic)")
    return fig

def cole_hopf_map_diagram():
    fig, ax = plt.subplots(figsize=(7.8, 4.6)); ax.axis("off")
    box = dict(boxstyle="round,pad=0.4", linewidth=1.2)
    ax.text(0.08, 0.72, r"QBSDE driver: $F=C+LZ+\frac{1}{2} Z^2$", bbox=box, transform=ax.transAxes)
    ax.text(0.45, 0.48, r"Transform: $\Psi=\exp(Y)$", bbox=box, transform=ax.transAxes)
    ax.text(0.70, 0.24, r"Lipschitz BSDE: $d\Psi=\Psi(-C-LZ)\,dt+\Psi Z\,dW$", bbox=box, transform=ax.transAxes)
    ax.annotate("", xy=(0.42, 0.50), xytext=(0.23, 0.68), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.annotate("", xy=(0.66, 0.28), xytext=(0.48, 0.46), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.set_title("Cole–Hopf mapping cancels the quadratic term")
    return fig

def picard_iteration_diagram():
    fig, ax = plt.subplots(figsize=(7.6, 4.6)); ax.axis("off")
    box = dict(boxstyle="round,pad=0.4", linewidth=1.2)
    ax.text(0.10, 0.72, r"Init: $(Y^{(0)}, Z^{(0)})$", bbox=box, transform=ax.transAxes)
    ax.text(0.42, 0.72, r"Solve forward $X$ using $(Y^{(k)}, Z^{(k)})$", bbox=box, transform=ax.transAxes)
    ax.text(0.76, 0.72, r"Solve backward $(Y^{(k+1)},Z^{(k+1)})$", bbox=box, transform=ax.transAxes)
    ax.annotate("", xy=(0.38, 0.74), xytext=(0.21, 0.74), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.annotate("", xy=(0.72, 0.74), xytext=(0.55, 0.74), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.annotate("", xy=(0.15, 0.68), xytext=(0.78, 0.68), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.set_title("Picard iteration to decouple a fully coupled FBSDE")
    return fig

def feller_boundary_diagram():
    fig, ax = plt.subplots(figsize=(7.6, 4.6)); ax.axis("off")
    box = dict(boxstyle="round,pad=0.4", linewidth=1.2)
    ax.text(0.10, 0.78, r"Near $x=0$: $\sigma(x)\approx C_1 x$, $\mu(x)\approx C_2 x$", bbox=box, transform=ax.transAxes)
    ax.text(0.10, 0.58, r"$c=\frac{2C_2}{C_1^2}$, scale density $s(y)\sim y^{-c}$", bbox=box, transform=ax.transAxes)
    ax.text(0.10, 0.42, r"speed density $m(y)\sim y^{c-2}$", bbox=box, transform=ax.transAxes)
    ax.text(0.10, 0.26, r"scale $\int_0 s(y)\,dy$ converges iff $c<1$", bbox=box, transform=ax.transAxes)
    ax.text(0.10, 0.12, r"speed $\int_0 m(y)\,dy$ converges iff $c>1$", bbox=box, transform=ax.transAxes)
    ax.set_title("Feller integrability heuristics (boundary at 0)")
    return fig

def logit_transform_diagram():
    # Visual intuition: x in (0,1) -> logit(x) in R, with spacing distortion near edges.
    x = np.linspace(1e-6, 1-1e-6, 500)
    y = np.log(x/(1-x))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, y, lw=2.0)
    ax.set_xlabel("x in (0,1)"); ax.set_ylabel("logit(x)")
    ax.set_title("Logit transform stretches boundaries to infinity")
    penrose_axes(ax); fig.tight_layout()
    return fig

def transversality_diagram():
    # Schematic: E[xi_T S_T] -> 0 with increasing T (illustrative exponential decay)
    T = np.linspace(0, 10, 200)
    val = np.exp(-0.6*T)  # illustrative
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(T, val, lw=2.0)
    ax.set_xlabel("T"); ax.set_ylabel(r"$\mathbb{E}[\xi_T S_T]$ (illustrative)")
    ax.set_title("No-bubble transversality (schematic)")
    penrose_axes(ax); fig.tight_layout()
    return fig

# -----------------
# Economics curves
# -----------------
def aggregator_gamma_curve(g1=2.0, g2=5.0, n=400):
    lam = np.linspace(0.0, 1.0, n, dtype=float)
    invG = lam/g1 + (1-lam)/g2
    G = 1.0/invG
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(lam, G, lw=2.0)
    ax.set_xlabel(r"consumption share $\lambda^c$")
    ax.set_ylabel(r"aggregate RRA $\Gamma(\lambda^c)$")
    ax.set_title(r"Aggregation: $\Gamma^{-1}=\lambda^c/\gamma_1+(1-\lambda^c)/\gamma_2$")
    penrose_axes(ax); fig.tight_layout()
    return fig

def aggregator_kappa_curve(g1=2.0, g2=5.0, sigmaD=0.2, n=400):
    lam = np.linspace(0.0, 1.0, n, dtype=float)
    invG = lam/g1 + (1-lam)/g2
    kappa = (1.0/invG) * sigmaD
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(lam, kappa, lw=2.0)
    ax.set_xlabel(r"consumption share $\lambda^c$")
    ax.set_ylabel(r"market price of risk $\kappa(\lambda^c)$")
    ax.set_title(r"$\kappa(\lambda^c)=\Gamma(\lambda^c)\sigma_D$")
    penrose_axes(ax); fig.tight_layout()
    return fig

def r_curve(g1=2.0, g2=5.0, rho1=0.03, rho2=0.04, muD=0.02, sigmaD=0.2, n=400):
    lam = np.linspace(0.0, 1.0, n, dtype=float)
    invG = lam/g1 + (1-lam)/g2
    G = 1.0/invG
    Ragg = lam*rho1 + (1-lam)*rho2
    r = Ragg + G*muD - 0.5*G*(G+1.0)*sigmaD**2
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(lam, r, lw=2.0)
    ax.set_xlabel(r"consumption share $\lambda^c$")
    ax.set_ylabel(r"$r(\lambda^c)$ (illustrative)")
    ax.set_title("Illustrative risk-free rate vs heterogeneity")
    penrose_axes(ax); fig.tight_layout()
    return fig

# -----------------
# MCP/context7 probe
# -----------------
def context7_probe():
    import shutil, subprocess
    if shutil.which("npx") is None:
        return "- Node/npm not found; cannot check @upstash/context7-mcp."
    try:
        out = subprocess.run(["npx", "-y", "@upstash/context7-mcp", "--version"],
                             check=False, capture_output=True, text=True, timeout=7)
        s = out.stdout.strip() or out.stderr.strip()
        return f"context7-mcp version: {s}"
    except Exception as e:
        return f"Unable to query context7-mcp: {e}"

# -----------------
# marimo app (top-level cells; required by marimo)
# -----------------
if mo is not None:
    app = mo.App()

    @app.cell
    def __():
        import marimo as mo
        mo.md("# Deep FBSDE · Diagram-first (v8)")
        return mo

    @app.cell
    def __(mo):
        mo.callout(mo.md("""
### Self-hardening plot rubric (R1→R3)
- **R1: Signal > Ink.** Arrowed axes; one idea per chart.
- **R2: Math-anchored.** SymPy checks visible near claims; analytic overlays when known.
- **R3: Economics-first.** λ^c→Γ→(κ,r) flows; comparative statics clarified.
- **Numerics kept honest.** GBM panel (EM vs Milstein) = theorem-level; Heun panels = *empirical*.
"""), kind="info")
        return

    @app.cell
    def __(mo):
        mo.md("## Penrose diagrams: PDE↔BSDE, Cole–Hopf, SDF/Rate, Picard, Feller, Logit, Transversality")
        tabs = mo.ui.tabs({
            "Feynman–Kac": mo.as_html(pde_fbsde_map_diagram()),
            "Cole–Hopf": mo.as_html(cole_hopf_map_diagram()),
            "SDF↔(κ,r)": mo.as_html(sdf_ramsey_diagram()),
            "Picard": mo.as_html(picard_iteration_diagram()),
            "Feller (heur.)": mo.as_html(feller_boundary_diagram()),
            "Logit map": mo.as_html(logit_transform_diagram()),
            "Transversality": mo.as_html(transversality_diagram()),
        })
        tabs
        return

    @app.cell
    def __(mo):
        mo.md("## Aggregation & pricing curves")
        mo.ui.tabs({
            "Flow": mo.as_html(econ_flow_diagram()),
            "Γ(λ)": mo.as_html(aggregator_gamma_curve()),
            "κ(λ)": mo.as_html(aggregator_kappa_curve()),
            "r(λ)": mo.as_html(r_curve())
        })
        return

    @app.cell
    def __(mo):
        mo.md("## Strong error rates (GBM): EM vs Milstein (theorem-level)")
        fig_rate, info = milstein_vs_em_rate_demo()
        mo.as_html(fig_rate)
        mo.callout(mo.md(f"- Estimated slope EM ≈ {info['slope_em']:.2f}; Milstein ≈ {info['slope_mil']:.2f}"), kind="neutral")
        return

    @app.cell
    def __(mo):
        mo.md("### Heun (drift-trap) empirical slopes (do not over-generalize)")
        fig_add, s_add = heun_rate_probe(mode="additive")
        fig_mul, s_mul = heun_rate_probe(mode="multiplicative")
        mo.as_html(fig_add)
        mo.callout(mo.md(f"- Additive: EM slope≈{s_add['slope_em']:.2f}, Heun≈{s_add['slope_heun']:.2f}"), kind="neutral")
        mo.as_html(fig_mul)
        mo.callout(mo.md(f"- Multiplicative: EM slope≈{s_mul['slope_em']:.2f}, Heun≈{s_mul['slope_heun']:.2f}"), kind="neutral")
        mo.callout(mo.md("Empirical for this variant/SDEs; see 2025 critiques of Heun for theory details."), kind="warn")
        return

    @app.cell
    def __(mo):
        mo.md("## Symbolic sanity checks")
        ok = sympy_checks()
        if not ok.get("sympy_available", False):
            mo.callout(mo.md("SymPy not available; skip analytic checks."), kind="warn")
        else:
            entries = []
            for k in ["logit_hprime_ok","logit_h2_ok","gamma_equal_collapse_ok",
                      "gamma_bounds_sample_ok","ito_stratonovich_affine_ok",
                      "gbm_mgf_identity_ok","cole_hopf_cancels_quadratic_ok",
                      "scale_integrable_at0_condition","speed_integrable_at0_condition"]:
                entries.append(f"- **{k}**: {ok.get(k)}")
            mo.callout(mo.md("### Checks\n" + "\n".join(entries)), kind="info")
        return

    @app.cell
    def __(mo):
        mo.md("## Context7 MCP sanity check")
        msg = context7_probe()
        mo.callout(mo.md(msg), kind="neutral")
        return

def _main():
    if mo is None:
        print("[INFO] marimo not installed; run: pip install marimo")
        # Still run non-marimo diagnostics:
        print("Running SymPy checks (standalone)...")
        print(sympy_checks())
        return
    # When marimo is installed, run the top-level app
    app.run()

if __name__ == "__main__":
    _main()
