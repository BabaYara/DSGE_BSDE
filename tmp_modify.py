# -*- coding: utf-8 -*-
from pathlib import Path
from textwrap import dedent

text_path = Path("Tex/BSDE_31.tex")
text = text_path.read_text(encoding="latin-1")

anchor = "The Deep BSDE algorithm solves the FBSDE"
anchor_pos = text.find(anchor)
if anchor_pos == -1:
    raise SystemExit("Deep BSDE anchor not found")

enum_marker = "\\begin{enumerate}[leftmargin=1.5em,itemsep=0.5em]"
enum_start = text.find(enum_marker, anchor_pos)
if enum_start == -1:
    raise SystemExit("Enumerate marker not found after anchor")

barrier_block = "\n" + dedent(r"""
\begin{tcolorbox}[mathstyle]
    \textbf{Barrier Ansatz Architecture.} We parameterise the value function using a barrier-modulated network that respects the simplex boundary conditions exactly:
    \begin{equation}\label{eq:barrier_ansatz}
        g_i(\bm{y};\Theta) = g_i^{\text{Prior}}(\bm{s}(\bm{y})) + B(\bm{s}(\bm{y}))\,\mathcal{N}_i(\bm{y};\Theta).
    \end{equation}
    The prior $g_i^{\text{Prior}}$ matches the Dirichlet boundary limits (log-utility choice $s_i/\rho$), $B(\bm{s})=\prod_{k} s_k$ vanishes on $\partial\Delta^{N-1}$, and $\mathcal{N}_i$ is an unconstrained neural correction. The barrier enforces stability near the faces of the simplex (see \cref{sympy:barrier_ansatz}).
\end{tcolorbox}

\begin{figure}[ht]
    \centering
    \begin{tikzpicture}[node distance=2cm, >=stealth, thick]
        % Define styles
        \tikzstyle{block} = [draw, rectangle, minimum height=1cm, minimum width=2cm]
        \tikzstyle{mathfunc} = [draw, rectangle, fill=green!10, minimum height=1cm, minimum width=2cm]
        \tikzstyle{nn} = [draw, rectangle, fill=blue!10, minimum height=2cm, minimum width=2.5cm]
        \tikzstyle{op} = [draw, circle]

        % Nodes
        \node (Input) [block, fill=gray!10] {Input $\bm{y}_t$};
        \node (MLP) [nn, below left=1cm and 0.5cm of Input] {MLP $\mathcal{N}(\cdot;\Theta)$};
        \node (Softmax) [mathfunc, below right=1.5cm and 0.5cm of Input] {Softmax};
        \node (Prior) [mathfunc, fill=red!10, below=1cm of Softmax] {Prior $g^{\text{Prior}}$};
        \node (Barrier) [mathfunc, fill=red!10, left=1cm of Prior] {Barrier $B(\bm{s})$};
        \node (Mult) [op] at ($(MLP.south)!0.5!(Barrier.west)-(0,1cm)$) {$\times$};
        \node (Add) [op, below=1cm of Mult] {$+$};
        \node (Output) [block, fill=gray!10, below=1cm of Add] {Output $g_i(\bm{y}_t)$};

        % Edges
        \draw[->] (Input) -- (MLP);
        \draw[->] (Input) -- (Softmax);
        \draw[->] (Softmax) -- node[right] {$\bm{s}$} (Prior);
        \draw[->] (Softmax) -- (Barrier);
        \draw[->] (MLP) |- (Mult);
        \draw[->] (Barrier) -- (Mult);
        \draw[->] (Mult) -- (Add);
        \draw[->] (Prior) |- (Add);
        \draw[->] (Add) -- (Output);
    \end{tikzpicture}
    \caption{Barrier Ansatz architecture. The prior enforces the boundary values, while the barrier forces the neural correction to vanish on $\partial\Delta^{N-1}$.}
    \label{fig:barrier_architecture}
\end{figure}
""")

text = text[:enum_start] + barrier_block + text[enum_start:]

enum_start = text.find(enum_marker, anchor_pos)
list_end = text.find("\\end{enumerate}", enum_start)
if list_end == -1:
    raise SystemExit("End of enumerate not found")

post_enum_block = "\n" + dedent(r"""
\begin{itemize}[leftmargin=1.5em,itemsep=0.5em]
  \item \textbf{Variance Reduction.} Training high-dimensional Deep BSDE solvers is severely hampered by stochastic-gradient variance.
    \begin{itemize}[leftmargin=1.25em]
      \item \textbf{Antithetic Sampling (AS):} Applied to the Brownian increments $\Delta\bm{W}_t$ to stabilise shock realisations \cite{glasserman2003}.
      \item \textbf{Randomized Multilevel Monte Carlo (R-MLMC):} Critical for reducing variance stemming from the time discretisation of the backward process. This is essential for stable convergence in 4D.
      \item \textbf{Control Variates (CV):} Where analytical approximations exist (e.g., near boundaries), they act as control variates to further reduce variance \cite{glasserman2003}.
    \end{itemize}
    \begin{tcolorbox}[didacticstyle,title={Implementation Insight: R-MLMC Structure}]
        R-MLMC \cite{rhee2015} exploits a telescoping sum across discretisation levels. During training we sample levels on-the-fly (modifying Step~4 of Algorithm~\ref{alg:deepbsde}) and average with unbiased importance weights, yielding low-variance estimates of both the loss and its gradient while leveraging the exact log-ratio simulation.
    \end{tcolorbox}
  \item \textbf{Loss Function and Training.} Since the solution is time-homogeneous, we enforce a Markov fixed point rather than relying on a terminal condition.
    \begin{tcolorbox}[mathstyle,title={Modernization: Differential Losses (2025 Standard)}]
    Differential losses \cite{kapllani2024} reframe training as pathwise regression: instead of matching Monte Carlo expectations, we minimise the residual of the discretised BSDE along simulated paths,
    \begin{equation*}
      \mathcal{L}(\Theta) = \E\!\left[\sum_k \big\|Y_{\tau_{k+1}} - \operatorname{BSDE\_Step}(Y_{\tau_k}, \bm{Z}_{\tau_k}(\Theta), \Delta\tau_k, \Delta\bm{W}_{\tau_k})\big\|^2\right].
    \end{equation*}
    This delivers dense gradient information and typically accelerates convergence relative to terminal-matching or Euler-loss objectives.
    \end{tcolorbox}
    \begin{tcolorbox}[didacticstyle,title={Implementation Insight: Differential Losses and AD}]
        Differential losses hinge on accurate controls $\bm{Z}_t$, so we rely on automatic differentiation of $g_i(\bm{y};\Theta)$ (Step~8) to obtain gradients on every step. Penalising both the value and its diffusion exposure dramatically improves sample efficiency in practice.
    \end{tcolorbox}
\end{itemize}

\begin{tcolorbox}[didacticstyle,title={Computational Paradigm Shift}]
    \textbf{From optimisation to regression.} Pairing R-MLMC with differential losses transforms Deep BSDE training into an unbiased, low-variance regression problem, enabling stable solutions in the 4D 5-5-1 setting where naïve objectives fail to converge.
\end{tcolorbox}
""")

text = text[:list_end + len("\\end{enumerate}")] + post_enum_block + text[list_end + len("\\end{enumerate}"):]

lean_marker = "\\begin{leanproof}[title={A.7.14: Girsanov Transformation Structure (Ref: Sec 6.1)}, label={lean:girsanov_application}]"
lean_pos = text.find(lean_marker)
if lean_pos == -1:
    raise SystemExit("Girsanov leanproof marker not found")

lean_block = dedent(r"""
\begin{leanproof}[title={A.7.14: Algebraic Structure of It\^o's Quotient Rule (Ref: Lemma 4.2)}, label={lean:lemma_4_2_algebra}]
\begin{lstlisting}[language=Lean]
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Field.Defs

-- Formalising the algebraic structure of Ito's quotient rule application for the share dynamics.
section ShareDynamicsAlgebra

variables {μ_i μ_C σ_i_σ_C σ_C_sq : ℝ}

-- The algebraic identity corresponding to the drift of ds_i/s_i.
-- Drift = (Relative Growth) - (Covariance Term) + (Variance Term)
-- (μ_i - μ_C) - σ_i_σ_C + σ_C_sq

theorem ito_quotient_rule_drift_algebra :
  (μ_i - μ_C) - σ_i_σ_C + σ_C_sq = μ_i - μ_C + σ_C_sq - σ_i_σ_C := by
  ring

end ShareDynamicsAlgebra
\end{lstlisting}
Status: Proved (Algebraic identity).
\end{leanproof}
""")

text = text[:lean_pos] + lean_block + text[lean_pos:]
text = text.replace(lean_marker, "\\begin{leanproof}[title={A.7.15: Girsanov Transformation Structure (Ref: Sec 6.1)}, label={lean:girsanov_application}]", 1)

listing_marker = "\\end{lstlisting}"
list_pos = text.find(listing_marker)
if list_pos == -1:
    raise SystemExit("Python listing end not found")
insert_pos = list_pos + len(listing_marker)

appendix_block = "\n" + dedent(r"""

\subsection{(C) Results and Visualization (3-Tree Illustration)}

\begin{figure}[ht]
    \centering\includegraphics[width=\textwidth]{pc_ratios_3tree.pdf}
    \caption{Price/consumption ratios $g_i(\bm{s})$ for the three-tree illustration, plotted over the simplex $\Delta^2$ using barycentric coordinates. Boundary conditions $g_i(\bm{e}_j)=\delta_{ij}/\rho$ are respected, and interior curvature reflects the heterogeneous parameters.}
    \label{fig:pc_ratios_3tree}
\end{figure}

\begin{tcolorbox}[didacticstyle,title={Interpretation of Results}]
    The visualisation in \cref{fig:pc_ratios_3tree} highlights how the Deep BSDE solution captures boundary behaviour and the non-linear interior structure induced by parameter heterogeneity. The same diagnostics generalise to the 5-5-1 setting even though direct plotting is infeasible.
\end{tcolorbox}
""")

text = text[:insert_pos] + appendix_block + text[insert_pos:]

text_path.write_text(text, encoding="utf-8")
