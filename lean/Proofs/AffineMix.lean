import Mathlib.Data.Real.Basic

lemma affine_mix (s x1 x2 : R) :
    s * x1 + (1 - s) * x2 = x2 + s * (x1 - x2) := by
  ring