import Mathlib.Topology.MetricSpace.Basic
import Mathlib.MeasureTheory.Function.ConditionalExpectation.PullOut
import Mathlib.Probability.Moments.Variance
import Mathlib.Tactic
import ENN.FiniteWeights

/-!
# Finite estimators and variance weights
-/

namespace ENN

open Filter MeasureTheory

open scoped ProbabilityTheory

section WeightedEstimator

variable {ι X : Type*} [Fintype ι] [MetricSpace X]

/-- A finite weighted estimator. -/
noncomputable def Estimator.mean (w : Weights ι) (y : ι → ℝ) : ℝ :=
  ∑ i, w.weight i * y i

/-- The deterministic bias term in Appendix C.3. -/
noncomputable def Estimator.bias (w : Weights ι) (f : X → ℝ) (z : ι → X) (x : X) : ℝ :=
  ∑ i, w.weight i * (f (z i) - f x)

/-- The noise term in Appendix C.3. -/
noncomputable def Estimator.noise (w : Weights ι) (ε : ι → ℝ) : ℝ :=
  ∑ i, w.weight i * ε i

omit [MetricSpace X] in
/-- Algebraic decomposition of the weighted estimator error into bias plus noise. -/
theorem Estimator.decompose
    (w : Weights ι) (f : X → ℝ) (z : ι → X) (ε : ι → ℝ) (x : X) :
    Estimator.mean w (fun i ↦ f (z i) + ε i) - f x =
      Estimator.bias w f z x + Estimator.noise w ε := by
  unfold Estimator.mean Estimator.bias Estimator.noise
  have hconst : ∑ i, w.weight i * f x = f x := by
    calc
      ∑ i, w.weight i * f x = (∑ i, w.weight i) * f x := by
        simpa using (Finset.sum_mul Finset.univ w.weight (f x)).symm
      _ = f x := by rw [w.sum_eq_one, one_mul]
  calc
    (∑ i, w.weight i * (f (z i) + ε i)) - f x
      = (∑ i, w.weight i * (f (z i) + ε i)) - ∑ i, w.weight i * f x := by rw [hconst]
    _ = ∑ i, (w.weight i * (f (z i) + ε i) - w.weight i * f x) := by
          rw [Finset.sum_sub_distrib]
    _ = ∑ i, (w.weight i * (f (z i) - f x) + w.weight i * ε i) := by
          apply Finset.sum_congr rfl
          intro i _
          ring_nf
    _ = (∑ i, w.weight i * (f (z i) - f x)) + ∑ i, w.weight i * ε i := by
          rw [Finset.sum_add_distrib]

/-- The deterministic weighted bias bound used in the proof of Appendix C.3 Theorem 2. -/
theorem Noise.bias
    (w : Weights ι) (f : X → ℝ) (z : ι → X) (x : X) {L ρ : ℝ}
    (hL : 0 ≤ L)
    (hLip : ∀ i, |f (z i) - f x| ≤ L * dist (z i) x)
    (hρ : ∀ i, dist (z i) x ≤ ρ) :
    |Estimator.bias w f z x| ≤ L * ρ := by
  unfold Estimator.bias
  refine (w.pointwise_bound
    (fun i ↦ f (z i) - f x) (fun i ↦ L * dist (z i) x) hLip).trans ?_
  refine w.sum_bound (fun i ↦ L * dist (z i) x) (L * ρ) ?_
  intro i
  exact mul_le_mul_of_nonneg_left (hρ i) hL

/-- Reindexing the observation index type preserves the weighted estimator. -/
theorem Estimator.mean_reindex {κ : Type*} [Fintype κ]
    (w : Weights ι) (y : ι → ℝ) (e : κ ≃ ι) :
    Estimator.mean (w.reindex e) (y ∘ e) = Estimator.mean w y := by
  unfold Estimator.mean Weights.reindex
  simpa using (Equiv.sum_comp e (fun i ↦ w.weight i * y i))

omit [MetricSpace X] in
/-- Reindexing the observation index type preserves the deterministic bias term. -/
theorem Estimator.bias_reindex {κ : Type*} [Fintype κ]
    (w : Weights ι) (f : X → ℝ) (z : ι → X) (x : X) (e : κ ≃ ι) :
    Estimator.bias (w.reindex e) f (z ∘ e) x = Estimator.bias w f z x := by
  unfold Estimator.bias Weights.reindex
  simpa using (Equiv.sum_comp e (fun i ↦ w.weight i * (f (z i) - f x)))

/-- Reindexing the observation index type preserves the noise term. -/
theorem Estimator.noise_reindex {κ : Type*} [Fintype κ]
    (w : Weights ι) (ε : ι → ℝ) (e : κ ≃ ι) :
    Estimator.noise (w.reindex e) (ε ∘ e) = Estimator.noise w ε := by
  unfold Estimator.noise Weights.reindex
  simpa using (Equiv.sum_comp e (fun i ↦ w.weight i * ε i))

/-- Reindexing preserves the sum of squared normalized weights. -/
theorem Estimator.squares_reindex {κ : Type*} [Fintype κ]
    (w : Weights ι) (e : κ ≃ ι) :
    ∑ k, ((w.reindex e).weight k) ^ 2 = ∑ i, (w.weight i) ^ 2 := by
  unfold Weights.reindex
  simpa using (Equiv.sum_comp e (fun i ↦ (w.weight i) ^ 2))

end WeightedEstimator

section VarianceScaleWeights

variable {ι : Type*} [Fintype ι] [Nonempty ι]

/-- The Appendix C.3 variance scale `v_i = s0^2 + s_i^2 + c_e r_i^2`. -/
def Variance.scale (α γ : ℝ) (s r : ι → ℝ) (i : ι) : ℝ :=
  α + (s i) ^ 2 + γ * (r i) ^ 2

omit [Fintype ι] [Nonempty ι] in
theorem Variance.nonneg (α γ : ℝ) (s r : ι → ℝ)
    (hα : 0 ≤ α) (hγ : 0 ≤ γ) (i : ι) :
    0 ≤ Variance.scale α γ s r i := by
  unfold Variance.scale
  positivity

omit [Fintype ι] [Nonempty ι] in
theorem Variance.positive (α γ : ℝ) (s r : ι → ℝ)
    (hα : 0 < α) (hγ : 0 ≤ γ) (i : ι) :
    0 < Variance.scale α γ s r i := by
  unfold Variance.scale
  positivity

/-- Raw inverse-variance weights from Appendix C.3. -/
noncomputable def Variance.raw (α γ : ℝ) (s r : ι → ℝ) (i : ι) : ℝ :=
  (Variance.scale α γ s r i)⁻¹

omit [Fintype ι] [Nonempty ι] in
theorem Variance.raw_positive (α γ : ℝ) (s r : ι → ℝ)
    (hα : 0 < α) (hγ : 0 ≤ γ) (i : ι) :
    0 < Variance.raw α γ s r i := by
  unfold Variance.raw
  exact inv_pos.mpr (Variance.positive α γ s r hα hγ i)

/-- Normalized inverse-variance weights from Appendix C.3. -/
noncomputable def Variance.weights (α γ : ℝ) (s r : ι → ℝ)
    (hα : 0 < α) (hγ : 0 ≤ γ) :
    Weights ι :=
  ofPositive
    (Variance.raw α γ s r)
    (Variance.raw_positive α γ s r hα hγ)

/-- Reindexing commutes with the inverse-variance weight construction. -/
theorem Variance.reindex {κ : Type*} [Fintype κ] [Nonempty κ]
    (α γ : ℝ) (s r : ι → ℝ)
    (hα : 0 < α) (hγ : 0 ≤ γ) (e : κ ≃ ι) :
    (Variance.weights α γ (s ∘ e) (r ∘ e) hα hγ).weight
      = ((Variance.weights α γ s r hα hγ).reindex e).weight := by
  funext k
  simp [Variance.weights, Weights.reindex, ofPositive, normalize,
    Variance.raw, Variance.scale]
  have hsum :
      ∑ j, (α + s (e j) ^ 2 + γ * r (e j) ^ 2)⁻¹
        = ∑ j, (α + s j ^ 2 + γ * r j ^ 2)⁻¹ := by
    simpa using (Equiv.sum_comp e (fun i ↦ (α + s i ^ 2 + γ * r i ^ 2)⁻¹))
  rw [hsum]

end VarianceScaleWeights

end ENN
