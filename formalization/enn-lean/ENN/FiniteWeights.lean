import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Real.Basic

/-!
# Finite normalized weights

Reusable algebra for the finite convex combinations that occur throughout the
ENN formalization.  This file deliberately has no nearest-neighbor or
probability dependencies.
-/

namespace ENN

/-- A family of nonnegative real weights on a finite index type whose sum is one. -/
structure Weights (ι : Type*) [Fintype ι] where
  weight : ι → ℝ
  nonnegative : ∀ i, 0 ≤ weight i
  sum_eq_one : ∑ i, weight i = 1

namespace Weights

variable {ι : Type*} [Fintype ι]

/-- Reindex normalized weights along an equivalence of finite index types. -/
noncomputable def reindex {κ : Type*} [Fintype κ] (w : Weights ι) (e : κ ≃ ι) :
    Weights κ where
  weight := fun k ↦ w.weight (e k)
  nonnegative := fun k ↦ w.nonnegative (e k)
  sum_eq_one := by
    calc
      ∑ k, w.weight (e k) = ∑ i, w.weight i := by
        simpa using (Equiv.sum_comp e w.weight)
      _ = 1 := w.sum_eq_one

/-- Taking absolute values inside a finite normalized weighted sum can only increase it. -/
theorem abs_sum (w : Weights ι) (a : ι → ℝ) :
    |∑ i, w.weight i * a i| ≤ ∑ i, w.weight i * |a i| := by
  calc
    |∑ i, w.weight i * a i| ≤ ∑ i, |w.weight i * a i| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, w.weight i * |a i| := by
      apply Finset.sum_congr rfl
      intro i _
      rw [abs_mul, abs_of_nonneg (w.nonnegative i)]

/-- A normalized weighted sum is bounded by any common upper bound on its terms. -/
theorem sum_bound (w : Weights ι) (a : ι → ℝ) (M : ℝ)
    (ha : ∀ i, a i ≤ M) :
    ∑ i, w.weight i * a i ≤ M := by
  calc
    ∑ i, w.weight i * a i ≤ ∑ i, w.weight i * M := by
      apply Finset.sum_le_sum
      intro i _
      exact mul_le_mul_of_nonneg_left (ha i) (w.nonnegative i)
    _ = (∑ i, w.weight i) * M := by
      simpa using (Finset.sum_mul Finset.univ w.weight M).symm
    _ = M := by rw [w.sum_eq_one, one_mul]

/-- A finite normalized weighted sum inherits a common absolute-value bound. -/
theorem abs_bound (w : Weights ι) (a : ι → ℝ) (M : ℝ)
    (ha : ∀ i, |a i| ≤ M) :
    |∑ i, w.weight i * a i| ≤ M :=
  (w.abs_sum a).trans (w.sum_bound (fun i ↦ |a i|) M ha)

/-- Pointwise absolute-value bounds can be pushed through a normalized weighted sum. -/
theorem pointwise_bound (w : Weights ι) (a b : ι → ℝ)
    (hab : ∀ i, |a i| ≤ b i) :
    |∑ i, w.weight i * a i| ≤ ∑ i, w.weight i * b i := by
  refine (w.abs_sum a).trans ?_
  apply Finset.sum_le_sum
  intro i _
  exact mul_le_mul_of_nonneg_left (hab i) (w.nonnegative i)

/-- A monotone pointwise error envelope yields a uniform weighted approximation bound. -/
theorem monotone_bound (w : Weights ι) (a : ι → ℝ) {α : Type*}
    [Preorder α] (r : ι → α) (ω : α → ℝ) (R : α)
    (ha : ∀ i, |a i| ≤ ω (r i)) (hω : Monotone ω) (hr : ∀ i, r i ≤ R) :
    |∑ i, w.weight i * a i| ≤ ω R :=
  (w.pointwise_bound a (fun i ↦ ω (r i)) ha).trans <|
    w.sum_bound (fun i ↦ ω (r i)) (ω R) fun i ↦ hω (hr i)

/-- A Lipschitz pointwise envelope yields the weighted-distance error bound used in Appendix C. -/
theorem lipschitz_bound (w : Weights ι) (a r : ι → ℝ) (L : ℝ)
    (_hL : 0 ≤ L) (ha : ∀ i, |a i| ≤ L * r i) :
    |∑ i, w.weight i * a i| ≤ ∑ i, w.weight i * (L * r i) :=
  w.pointwise_bound a (fun i ↦ L * r i) ha

end Weights

section Normalize

variable {ι : Type*} [Fintype ι]

/-- Normalize finite raw weights by their sum. -/
noncomputable def normalize (raw : ι → ℝ) (i : ι) : ℝ :=
  raw i / ∑ j, raw j

/-- Normalization preserves nonnegativity when the denominator is positive. -/
theorem normalize_nonnegative (raw : ι → ℝ) (hraw : ∀ i, 0 ≤ raw i)
    (hdenom : 0 < ∑ i, raw i) (i : ι) :
    0 ≤ normalize raw i :=
  div_nonneg (hraw i) hdenom.le

/-- Normalized weights sum to one when their denominator is positive. -/
theorem sum_normalize (raw : ι → ℝ) (hdenom : 0 < ∑ i, raw i) :
    ∑ i, normalize raw i = 1 := by
  rw [show (∑ i, normalize raw i) = (∑ i, raw i) / ∑ i, raw i by
    simpa [normalize] using
      (Finset.sum_div Finset.univ raw (∑ i, raw i)).symm]
  exact div_self hdenom.ne'

/-- Positive raw weights over a nonempty finite type have a positive denominator. -/
theorem positive_sum [Nonempty ι] (raw : ι → ℝ) (hraw : ∀ i, 0 < raw i) :
    0 < ∑ i, raw i := by
  classical
  exact Finset.sum_pos (fun i _ ↦ hraw i) Finset.univ_nonempty

/-- Strictly positive raw weights canonically produce normalized weights. -/
noncomputable def ofPositive [Nonempty ι] (raw : ι → ℝ) (hraw : ∀ i, 0 < raw i) :
    Weights ι where
  weight := normalize raw
  nonnegative := normalize_nonnegative raw (fun i ↦ (hraw i).le) (positive_sum raw hraw)
  sum_eq_one := sum_normalize raw (positive_sum raw hraw)

end Normalize

end ENN
