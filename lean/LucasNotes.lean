import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

open scoped BigOperators

noncomputable section

namespace LucasNotes

lemma affine_mix (s x₁ x₂ : Real) :
    s * x₁ + (1 - s) * x₂ = x₂ + s * (x₁ - x₂) := by
  ring

-- Algebraic rearrangement used in the drift of ds_i/s_i.
theorem ito_quotient_rule_drift_algebra
    (mu_i mu_C sigma_i_sigma_C sigma_C_sq : Real) :
    (mu_i - mu_C) - sigma_i_sigma_C + sigma_C_sq =
      mu_i - mu_C + sigma_C_sq - sigma_i_sigma_C := by
  ring

/-!
Log-utility SDF drift identity. Requires C ≠ 0 to clear denominators.
-/
theorem sdf_drift_algebra
    (rho muC sigma_C_sq C : Real) (hC : C ≠ 0) :
    (-rho) + (-1 / C) * (C * muC) + (1 / 2) * (2 / (C ^ 2)) * (C ^ 2 * sigma_C_sq)
      = -(rho + muC - sigma_C_sq) := by
  field_simp [hC]
  ring

/-!
Lemma ggregate_consumption_drift represents the algebraic core of Lemma C.
-/
lemma aggregate_consumption_drift
    {ι d : Type*} [Fintype ι] [DecidableEq ι] [Fintype d]
    (dividend : ι → Real) (drift : ι → Real) (diff : ι → d → Real)
    (C : Real) (hC : C = ∑ i, dividend i) (hC_ne : C ≠ 0) :
    (∑ i, dividend i / C = 1) ∧
    (∑ i, (dividend i / C) * drift i = (1 / C) * ∑ i, dividend i * drift i) ∧
    (∀ j, ∑ i, (dividend i / C) * diff i j = (1 / C) * ∑ i, dividend i * diff i j) := by
  have hx : ∑ i, dividend i / C = (∑ i, dividend i) / C := by
    simp [div_eq_mul_inv, Finset.sum_mul, Finset.mul_sum, mul_comm, mul_left_comm, mul_assoc]
  have share_total : ∑ i, dividend i / C = 1 := by
    have hx' : (∑ i, dividend i) / C = 1 := by
      simpa [hC] using (div_self hC_ne)
    exact hx.trans hx'
  have drift_sum :
      ∑ i, (dividend i / C) * drift i = (1 / C) * ∑ i, dividend i * drift i := by
    simp [div_eq_mul_inv, Finset.sum_mul, Finset.mul_sum, mul_comm, mul_left_comm, mul_assoc]
  have diff_sum (j : d) :
      ∑ i, (dividend i / C) * diff i j = (1 / C) * ∑ i, dividend i * diff i j := by
    simp [div_eq_mul_inv, Finset.sum_mul, Finset.mul_sum, mul_comm, mul_left_comm, mul_assoc]
  refine ⟨share_total, drift_sum, ?_⟩
  intro j
  simpa using diff_sum j

end LucasNotes

end
