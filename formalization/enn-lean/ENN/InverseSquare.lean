import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Data.Real.Basic

/-!
# Inverse-square ENN spacing algebra

Reusable finite-dimensional algebra for Appendix C.1(ii).  The nearest-neighbor
geometry will later instantiate these lemmas with the ordered ENN radii.
-/

namespace ENN

section InverseSquare

/-- The inverse-square variance used by the exact-observation ENN surrogate. -/
noncomputable def InvSq.variance {k : ℕ} (r : Fin k → ℝ) : ℝ :=
  1 / ∑ i, ((r i) ^ 2)⁻¹

/-- The corresponding ENN uncertainty scale. -/
noncomputable def InvSq.sigma {k : ℕ} (r : Fin k → ℝ) : ℝ :=
  Real.sqrt (InvSq.variance r)

theorem InvSq.variance_nonneg {k : ℕ} (r : Fin k → ℝ) :
    0 ≤ InvSq.variance r := by
  unfold InvSq.variance
  exact one_div_nonneg.mpr <| Finset.sum_nonneg fun i _ => by positivity

theorem InvSq.sigma_nonneg {k : ℕ} (r : Fin k → ℝ) :
    0 ≤ InvSq.sigma r :=
  Real.sqrt_nonneg _

/-- If one radius equals `d > 0`, the inverse-square variance is at most `d^2`. -/
theorem InvSq.variance_upper {k : ℕ} (r : Fin k → ℝ) {d : ℝ} (i0 : Fin k)
    (hi0 : r i0 = d) (hd : 0 < d) :
    InvSq.variance r ≤ d ^ 2 := by
  have hsum :
      (d ^ 2)⁻¹ ≤ ∑ i, ((r i) ^ 2)⁻¹ := by
    calc
      (d ^ 2)⁻¹ = ((r i0) ^ 2)⁻¹ := by rw [hi0]
      _ ≤ ∑ i, ((r i) ^ 2)⁻¹ := by
        simpa using
          (Finset.single_le_sum (s := Finset.univ) (f := fun i : Fin k ↦ ((r i) ^ 2)⁻¹)
            (by
              intro i _
              positivity)
            (Finset.mem_univ i0))
  have hpos : 0 < (d ^ 2)⁻¹ := by positivity
  unfold InvSq.variance
  simpa [one_div, hi0, inv_inv] using one_div_le_one_div_of_le hpos hsum

/-- If every radius is at least `d`, the inverse-square variance is at least `d^2 / k`. -/
theorem InvSq.variance_lower {k : ℕ} (hk : 0 < k) (r : Fin k → ℝ) {d : ℝ}
    (hd : 0 < d) (hlower : ∀ i, d ≤ r i) :
    d ^ 2 / (k : ℝ) ≤ InvSq.variance r := by
  have hsle :
      ∑ i, ((r i) ^ 2)⁻¹ ≤ (k : ℝ) * ((d ^ 2)⁻¹) := by
    calc
      ∑ i, ((r i) ^ 2)⁻¹ ≤ ∑ _i : Fin k, (d ^ 2)⁻¹ := by
        apply Finset.sum_le_sum
        intro i _
        have hri : 0 < r i := lt_of_lt_of_le hd (hlower i)
        have hsq : d ^ 2 ≤ (r i) ^ 2 :=
          (sq_le_sq₀ hd.le hri.le).2 (hlower i)
        simpa [one_div] using one_div_le_one_div_of_le (pow_pos hd 2) hsq
      _ = (k : ℝ) * ((d ^ 2)⁻¹) := by simp
  let i0 : Fin k := ⟨0, hk⟩
  have hsum_pos : 0 < ∑ i, ((r i) ^ 2)⁻¹ := by
    have hterm_pos : 0 < ((r i0) ^ 2)⁻¹ := by
      have hri0 : 0 < r i0 := lt_of_lt_of_le hd (hlower i0)
      positivity
    exact lt_of_lt_of_le hterm_pos <|
      by
        simpa using
          (Finset.single_le_sum (s := Finset.univ) (f := fun i : Fin k ↦ ((r i) ^ 2)⁻¹)
            (by
              intro i _
              positivity)
            (Finset.mem_univ i0))
  unfold InvSq.variance
  have hrecip :=
    one_div_le_one_div_of_le hsum_pos hsle
  have hk0 : (k : ℝ) ≠ 0 := by exact_mod_cast hk.ne'
  have hd0 : d ≠ 0 := hd.ne'
  simpa [one_div, InvSq.variance, hk0, hd0, div_eq_mul_inv, mul_comm, mul_left_comm,
    mul_assoc] using hrecip

/-- Appendix C.1(ii): the inverse-square uncertainty is at least `d / sqrt k`. -/
theorem InvSq.sigma_lower {k : ℕ} (hk : 0 < k) (r : Fin k → ℝ) {d : ℝ}
    (hd : 0 < d) (hlower : ∀ i, d ≤ r i) :
    d / Real.sqrt k ≤ InvSq.sigma r := by
  apply Real.le_sqrt_of_sq_le
  calc
    (d / Real.sqrt k) ^ 2 = d ^ 2 / (k : ℝ) := by
      rw [div_pow, Real.sq_sqrt (Nat.cast_nonneg k)]
    _ ≤ InvSq.variance r :=
      InvSq.variance_lower hk r hd hlower

/-- Appendix C.1(ii): if one radius equals `d > 0`, the inverse-square uncertainty is at most `d`. -/
theorem InvSq.sigma_upper {k : ℕ} (r : Fin k → ℝ) {d : ℝ} (i0 : Fin k)
    (hi0 : r i0 = d) (hd : 0 < d) :
    InvSq.sigma r ≤ d := by
  apply le_of_sq_le_sq
  · rw [InvSq.sigma, Real.sq_sqrt (InvSq.variance_nonneg r)]
    exact InvSq.variance_upper r i0 hi0 hd
  · exact hd.le

/-- Combined spacing bounds for a positive finite radius family. -/
theorem InvSq.spacing {k : ℕ} (hk : 0 < k) (r : Fin k → ℝ) {d : ℝ} (i0 : Fin k)
    (hi0 : r i0 = d) (hd : 0 < d) (hlower : ∀ i, d ≤ r i) :
    d / Real.sqrt k ≤ InvSq.sigma r ∧ InvSq.sigma r ≤ d :=
  ⟨InvSq.sigma_lower hk r hd hlower, InvSq.sigma_upper r i0 hi0 hd⟩

end InverseSquare

end ENN
