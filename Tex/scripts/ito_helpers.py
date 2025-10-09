import sympy as sp

# Minimal Itô engine specialized to share s = D1/(D1+D2)
# Uses generator formula: L f = grad f · μ + 1/2 Tr[(S S^T) Hess f]
# where dD = μ dt + S dW, with aggregated inner products encoded in A = S S^T.

def _ito_pushforward_scalar(f, x, mu_vec, A_mat):
    grad = sp.Matrix([sp.diff(f, xi) for xi in x])
    hess = sp.hessian(f, x)
    drift = (grad.T * sp.Matrix(mu_vec))[0] + sp.Rational(1, 2) * (A_mat * hess).trace()
    # diffusion exposure g satisfies g g^T dt = grad^T A grad dt, so we report the norm^2
    diff_norm_sq = (grad.T * A_mat * grad)[0]
    return sp.simplify(drift), sp.simplify(diff_norm_sq)


def emit_share_identities_tex(outpath: str):
    # Base symbols (neutral names); we apply pretty-print replacements after latexification
    D1, D2 = sp.symbols('D1 D2', positive=True, real=True)
    C = D1 + D2
    s_expr = sp.simplify(D1 / C)           # expression for s in terms of D1, D2

    mu1, mu2 = sp.symbols('mu1 mu2', real=True)
    sig11, sig22, sig12 = sp.symbols('sig11 sig22 sig12', real=True)

    x = sp.Matrix([D1, D2])
    mu_vec = sp.Matrix([mu1 * D1, mu2 * D2])  # geometric dividends
    A = sp.Matrix([[D1 ** 2 * sig11, D1 * D2 * sig12],
                   [D1 * D2 * sig12, D2 ** 2 * sig22]])

    # Raw Itô pushforward in (D1,D2)
    drift_raw, diffnorm_raw = _ito_pushforward_scalar(s_expr, x, mu_vec, A)

    # Stated identities in (s)
    # closed forms recorded via direct LaTeX below; symbolic helpers can be reintroduced if needed for checks

    # Rational forms in (D1,D2)
    drift_num, drift_den = sp.together(drift_raw).as_numer_denom()
    drift_num = sp.factor(drift_num)
    drift_den = sp.factor(drift_den)
    d_terms = sp.Add.make_args(drift_num)
    k = max(1, len(d_terms) // 2)

    def pretty_latex(e: sp.Expr) -> str:
        s = sp.latex(e)
        # Consistent paper notation
        replacements = [
            (r'mu_{1}', r'\\mu_1'),
            (r'mu_{2}', r'\\mu_2'),
            (r'mu1', r'\\mu_1'),
            (r'mu2', r'\\mu_2'),
            (r'sig_{11}', r'\\sigma_1^2'),
            (r'sig_{22}', r'\\sigma_2^2'),
            (r'sig_{12}', r'\\langle\\sigma_1,\\sigma_2\\rangle'),
            (r'sig11', r'\\sigma_1^2'),
            (r'sig22', r'\\sigma_2^2'),
            (r'sig12', r'\\langle\\sigma_1,\\sigma_2\\rangle'),
        ]
        for a, b in replacements:
            s = s.replace(a, b)
        return s

    d_line1 = r" + ".join(pretty_latex(t) for t in d_terms[:k])
    d_line2 = r" + ".join(pretty_latex(t) for t in d_terms[k:])
    drift_num_tex = r"\splitfrac{" + d_line1 + r"}{" + (d_line2 or r"0") + r"}"
    drift_den_tex = pretty_latex(sp.simplify(drift_den))

    diffn_num, diffn_den = sp.together(diffnorm_raw).as_numer_denom()
    diffn_num = sp.factor(diffn_num)
    diffn_den = sp.factor(diffn_den)
    v_terms = sp.Add.make_args(diffn_num)
    k2 = max(1, len(v_terms) // 2)
    v_line1 = r" + ".join(pretty_latex(t) for t in v_terms[:k2])
    v_line2 = r" + ".join(pretty_latex(t) for t in v_terms[k2:])
    diffn_num_tex = r"\splitfrac{" + v_line1 + r"}{" + (v_line2 or r"0") + r"}"
    diffn_den_tex = pretty_latex(sp.simplify(diffn_den))

    # Build compact, professional LaTeX
    stated_drift_tex = r"s(1-s)\big(\mu_1-\mu_2 + (\langle\sigma_C(s),\sigma_2\rangle - \langle\sigma_1,\sigma_C(s)\rangle)\big)"
    stated_diffn_tex = r"s^2(1-s)^2\big(\sigma_1^2 + \sigma_2^2 - 2\langle\sigma_1,\sigma_2\rangle\big)"
    lines = [
        r"\begingroup\small",
        r"\begin{align}",
        r"\mu_s(s) &= " + stated_drift_tex + r"\quad\text{(stated)}\\",
        r"&= \frac{" + drift_num_tex + r"}{" + drift_den_tex + r"}\\[0.35em]",
        r"\|\sigma_s(s)\|^2 &= " + stated_diffn_tex + r"\\",
        r"&= \frac{" + diffn_num_tex + r"}{" + diffn_den_tex + r"}\\,.",
        r"\end{align}",
        r"\endgroup",
    ]

    tex = "\n".join(lines)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(tex)
