import Mathlib.Data.Real.Basic

-- Algebraic rearrangement used in the drift of ds_i/s_i
-- (μ_i - μ_C) - σ_i·σ_C + ∥σ_C∥^2 = μ_i - μ_C + ∥σ_C∥^2 - σ_i·σ_C
theorem ito_quotient_rule_drift_algebra
  (mu_i mu_C sigma_i_sigma_C sigma_C_sq : ℝ) :
  (mu_i - mu_C) - sigma_i_sigma_C + sigma_C_sq
  = mu_i - mu_C + sigma_C_sq - sigma_i_sigma_C := by ring
