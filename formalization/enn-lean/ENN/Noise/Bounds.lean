import ENN.Noise.Moments

/-!
# Conditional prediction-error bounds
-/

namespace ENN

open Filter MeasureTheory

variable {ι Ω : Type*} [Fintype ι]
variable {m mΩ : MeasurableSpace Ω} {μ : @Measure Ω mΩ}
variable [IsFiniteMeasure μ]

/-- The conditional square decomposition with an `L²` conditioning-measurable
bias.  This version supports the random neighbor radii in the paper. -/
lemma Noise.sq_add_l2
    (hm : m ≤ mΩ)
    (b N : Ω → ℝ)
    (hb_meas : StronglyMeasurable[m] b)
    (hb2 : MemLp b 2 μ)
    (hN2 : MemLp N 2 μ)
    (hN_zero : μ[N | m] =ᵐ[μ] 0) :
    μ[fun ω ↦ (b ω + N ω) ^ 2 | m]
      =ᵐ[μ] fun ω ↦ (b ω) ^ 2 + μ[fun ω' ↦ (N ω') ^ 2 | m] ω := by
  have hb2_int : Integrable (fun ω ↦ (b ω) ^ 2) μ := by
    convert MemLp.integrable_mul hb2 hb2 using 1
    ext ω
    simp [pow_two]
  have hprod_int : Integrable (fun ω ↦ b ω * N ω) μ := by
    exact MemLp.integrable_mul hb2 hN2
  have hcross_int : Integrable (fun ω ↦ 2 * (b ω * N ω)) μ :=
    hprod_int.const_mul 2
  have hN2_int : Integrable (fun ω ↦ (N ω) ^ 2) μ := by
    convert MemLp.integrable_mul hN2 hN2 using 1
    ext ω
    simp [pow_two]
  have hb_cond :
      μ[fun ω ↦ (b ω) ^ 2 | m] = fun ω ↦ (b ω) ^ 2 := by
    rw [condExp_of_stronglyMeasurable (m := m) (m₀ := mΩ) (μ := μ) hm]
    · exact hb_meas.pow 2
    · exact hb2_int
  have hprod_zero : μ[fun ω ↦ b ω * N ω | m] =ᵐ[μ] 0 := by
    calc
      μ[fun ω ↦ b ω * N ω | m]
          =ᵐ[μ] b * μ[N | m] :=
            condExp_mul_of_stronglyMeasurable_left hb_meas hprod_int
              (hN2.integrable one_le_two)
      _ =ᵐ[μ] b * 0 := Filter.EventuallyEq.mul Filter.EventuallyEq.rfl hN_zero
      _ =ᵐ[μ] 0 := by simp
  have hcross_zero : μ[fun ω ↦ 2 * (b ω * N ω) | m] =ᵐ[μ] 0 := by
    calc
      μ[fun ω ↦ 2 * (b ω * N ω) | m]
          =ᵐ[μ] fun ω ↦ 2 * μ[fun ω ↦ b ω * N ω | m] ω :=
            condExp_mul_of_stronglyMeasurable_left stronglyMeasurable_const
              hcross_int hprod_int
      _ =ᵐ[μ] fun _ ↦ 2 * (0 : ℝ) :=
        Filter.EventuallyEq.mul Filter.EventuallyEq.rfl hprod_zero
      _ =ᵐ[μ] 0 := Filter.Eventually.of_forall fun _ ↦ by simp
  have hsquare :
      (fun ω ↦ (b ω + N ω) ^ 2) =ᵐ[μ]
        fun ω ↦ (b ω) ^ 2 + (2 * (b ω * N ω) + (N ω) ^ 2) :=
    Filter.Eventually.of_forall fun ω ↦ by ring
  calc
    μ[fun ω ↦ (b ω + N ω) ^ 2 | m]
        =ᵐ[μ] μ[fun ω ↦ (b ω) ^ 2 + (2 * (b ω * N ω) + (N ω) ^ 2) | m] :=
          condExp_congr_ae hsquare
    _ =ᵐ[μ] μ[(fun ω ↦ (b ω) ^ 2) +
        ((fun ω ↦ 2 * (b ω * N ω)) + fun ω ↦ (N ω) ^ 2) | m] :=
          condExp_congr_ae (Filter.Eventually.of_forall fun _ ↦ rfl)
    _ =ᵐ[μ] μ[fun ω ↦ (b ω) ^ 2 | m] +
        μ[(fun ω ↦ 2 * (b ω * N ω)) + fun ω ↦ (N ω) ^ 2 | m] := by
          simpa using condExp_add hb2_int (hcross_int.add hN2_int) m
    _ =ᵐ[μ] (fun ω ↦ (b ω) ^ 2) +
        μ[(fun ω ↦ 2 * (b ω * N ω)) + fun ω ↦ (N ω) ^ 2 | m] := by
          rw [hb_cond]
    _ =ᵐ[μ] (fun ω ↦ (b ω) ^ 2) +
        (μ[fun ω ↦ 2 * (b ω * N ω) | m] + μ[fun ω ↦ (N ω) ^ 2 | m]) :=
          Filter.EventuallyEq.add Filter.EventuallyEq.rfl
            (condExp_add hcross_int hN2_int m)
    _ =ᵐ[μ] (fun ω ↦ (b ω) ^ 2) +
        (0 + μ[fun ω ↦ (N ω) ^ 2 | m]) :=
          Filter.EventuallyEq.add Filter.EventuallyEq.rfl
            (hcross_zero.add Filter.EventuallyEq.rfl)
    _ =ᵐ[μ] fun ω ↦ (b ω) ^ 2 + μ[fun ω' ↦ (N ω') ^ 2 | m] ω :=
      Filter.Eventually.of_forall fun _ ↦ by simp

/-- Conditional square decomposition for a measurable bias term plus a
conditionally centered noise term. -/
lemma Noise.sq_add_zero
    (hm : m ≤ mΩ)
    (b N : Ω → ℝ) {B : ℝ}
    (hb_meas : StronglyMeasurable[m] b)
    (hb_bound : ∀ᵐ ω ∂μ, |b ω| ≤ B)
    (hN2 : MemLp N 2 μ)
    (hN_zero : μ[N | m] =ᵐ[μ] 0) :
    μ[fun ω ↦ (b ω + N ω) ^ 2 | m]
      =ᵐ[μ] fun ω ↦ (b ω) ^ 2 + μ[fun ω' ↦ (N ω') ^ 2 | m] ω := by
  have hb2 : MemLp b 2 μ :=
    MemLp.of_bound (hb_meas.mono hm).aestronglyMeasurable B
      (by simpa only [Real.norm_eq_abs] using hb_bound)
  exact Noise.sq_add_l2 hm b N hb_meas hb2 hN2 hN_zero

/-- A conditional weighted-kNN MSE bound for random normalized weights,
conditioned on a sub-σ-algebra with respect to which the weights are
measurable. -/
theorem Noise.mse [DecidableEq ι] {X : Type*} [MetricSpace X]
    (hm : m ≤ mΩ)
    (w : ι → Ω → ℝ) (f : X → ℝ) (z : ι → X) (ε : ι → Ω → ℝ) (x : X)
    {L ρ σ2 : ℝ}
    (hw_meas : ∀ i, StronglyMeasurable[m] (w i))
    (hw0 : ∀ i, ∀ᵐ ω ∂μ, 0 ≤ w i ω)
    (hw_sum : ∀ᵐ ω ∂μ, ∑ i, w i ω = 1)
    (hL : 0 ≤ L)
    (hLip : ∀ i, |f (z i) - f x| ≤ L * dist (z i) x)
    (hρ : ∀ i, dist (z i) x ≤ ρ)
    (hε2 : ∀ i, MemLp (ε i) 2 μ)
    (hε_zero : ∀ i, μ[ε i | m] =ᵐ[μ] 0)
    (hε_orth : ∀ i j, i ≠ j → μ[fun ω ↦ ε i ω * ε j ω | m] =ᵐ[μ] 0)
    (hε_sq : ∀ i, μ[fun ω ↦ (ε i ω) ^ 2 | m] ≤ᵐ[μ] fun _ ↦ σ2) :
    μ[fun ω ↦ (∑ i, w i ω * (f (z i) + ε i ω) - f x) ^ 2 | m]
      ≤ᵐ[μ] fun ω ↦ (L * ρ) ^ 2 + σ2 * ∑ i, (w i ω) ^ 2 := by
  let b : Ω → ℝ := ∑ i, fun ω ↦ w i ω * (f (z i) - f x)
  let N : Ω → ℝ := ∑ i, fun ω ↦ w i ω * ε i ω
  have hw0_all : ∀ᵐ ω ∂μ, ∀ i, 0 ≤ w i ω := by
    rw [ae_all_iff]
    intro i
    exact hw0 i
  have hw_bound : ∀ i, ∀ᵐ ω ∂μ, |w i ω| ≤ 1 := by
    intro i
    filter_upwards [hw0_all, hw_sum] with ω hwω0 hω_sum
    have hrest0 : 0 ≤ Finset.sum (Finset.univ.erase i) (fun j ↦ w j ω) := by
      exact Finset.sum_nonneg fun j _ ↦ hwω0 j
    have hsum_decomp :
        Finset.sum (Finset.univ.erase i) (fun j ↦ w j ω) + w i ω = 1 := by
      simp [hω_sum]
    have hi_le : w i ω ≤ 1 := by
      nlinarith
    simpa [abs_of_nonneg (hwω0 i)] using hi_le
  have hb_meas : StronglyMeasurable[m] b := by
    change StronglyMeasurable[m] (∑ i, fun ω ↦ w i ω * (f (z i) - f x))
    refine Finset.stronglyMeasurable_sum _ ?_
    intro i _
    simpa [mul_comm] using (hw_meas i).const_mul (f (z i) - f x)
  have hb_bound : ∀ᵐ ω ∂μ, |b ω| ≤ L * ρ := by
    filter_upwards [hw0_all, hw_sum] with ω hwω0 hω_sum
    let wω : Weights ι := {
      weight := fun i ↦ w i ω
      nonnegative := hwω0
      sum_eq_one := hω_sum }
    simpa [b, wω, Estimator.bias] using Noise.bias
      (w := wω) f z x hL hLip hρ
  have hN_zero :
      μ[N | m] =ᵐ[μ] 0 := by
    simpa [N] using
      Noise.noise_mean hm
        w ε hw_meas hw_bound hε2 hε_zero
  have hN_sq :
      μ[fun ω ↦ (N ω) ^ 2 | m]
        ≤ᵐ[μ] fun ω ↦ σ2 * ∑ i, (w i ω) ^ 2 := by
    simpa [N] using
      Noise.noise_var hm
        w ε hw_meas hw_bound hε2 hε_orth hε_sq
  have hN2 : MemLp N 2 μ := by
    change MemLp (∑ i, fun ω ↦ w i ω * ε i ω) 2 μ
    simpa using
      (memLp_finsetSum' (μ := μ) Finset.univ
        (fun i _ ↦ Noise.product_l2 (μ := μ)
          (w i) (ε i) ((hw_meas i).mono hm).aestronglyMeasurable
          (hw_bound i) (hε2 i)))
  have hE_eq :
      (fun ω ↦ ∑ i, w i ω * (f (z i) + ε i ω) - f x)
        =ᵐ[μ] fun ω ↦ b ω + N ω := by
    filter_upwards [hw_sum] with ω hω_sum
    have hconst : ∑ i, w i ω * f x = f x := by
      calc
        ∑ i, w i ω * f x = (∑ i, w i ω) * f x := by
          simpa using (Finset.sum_mul Finset.univ (fun i ↦ w i ω) (f x)).symm
        _ = f x := by rw [hω_sum, one_mul]
    calc
      (∑ i, w i ω * (f (z i) + ε i ω)) - f x
        = (∑ i, w i ω * (f (z i) + ε i ω)) - ∑ i, w i ω * f x := by
            rw [hconst]
      _ = ∑ i, (w i ω * (f (z i) + ε i ω) - w i ω * f x) := by
            rw [Finset.sum_sub_distrib]
      _ = ∑ i, (w i ω * (f (z i) - f x) + w i ω * ε i ω) := by
            refine Finset.sum_congr rfl ?_
            intro i _
            ring_nf
      _ = (∑ i, w i ω * (f (z i) - f x)) + ∑ i, w i ω * ε i ω := by
            rw [Finset.sum_add_distrib]
      _ = b ω + N ω := by simp [b, N]
  have hsq_decomp :
      μ[fun ω ↦ (b ω + N ω) ^ 2 | m]
        =ᵐ[μ] fun ω ↦ (b ω) ^ 2 + μ[fun ω' ↦ (N ω') ^ 2 | m] ω := by
    exact Noise.sq_add_zero hm b N hb_meas hb_bound hN2 hN_zero
  have hb_sq : ∀ᵐ ω ∂μ, (b ω) ^ 2 ≤ (L * ρ) ^ 2 := by
    filter_upwards [hb_bound] with ω hω
    have hω' := abs_le.mp hω
    nlinarith
  calc
    μ[fun ω ↦ (∑ i, w i ω * (f (z i) + ε i ω) - f x) ^ 2 | m]
      =ᵐ[μ] μ[fun ω ↦ (b ω + N ω) ^ 2 | m] := by
          exact condExp_congr_ae <|
            Filter.EventuallyEq.fun_comp hE_eq (fun t : ℝ ↦ t ^ 2)
    _ =ᵐ[μ] fun ω ↦ (b ω) ^ 2 + μ[fun ω' ↦ (N ω') ^ 2 | m] ω := hsq_decomp
    _ ≤ᵐ[μ] fun ω ↦ (L * ρ) ^ 2 + σ2 * ∑ i, (w i ω) ^ 2 := by
          filter_upwards [hb_sq, hN_sq] with ω hωb hωN
          linarith

/-- The inverse-variance ENN weights are measurable with respect to any
conditioning σ-algebra that already makes the radius and noise-scale inputs
measurable. -/
lemma Variance.measurable [Nonempty ι]
    (s r : ι → Ω → ℝ) {α γ : ℝ}
    (hs_meas : ∀ i, StronglyMeasurable[m] (s i))
    (hr_meas : ∀ i, StronglyMeasurable[m] (r i))
    (hα : 0 < α) (hγ : 0 ≤ γ) (i : ι) :
    StronglyMeasurable[m] (fun ω ↦
      (Variance.weights α γ (fun j ↦ s j ω) (fun j ↦ r j ω) hα hγ).weight i) := by
  let raw : ι → Ω → ℝ := fun j ω ↦
    Variance.raw α γ (fun k ↦ s k ω) (fun k ↦ r k ω) j
  have hraw_meas :
      ∀ j, StronglyMeasurable[m] (raw j) := by
    intro j
    have hbase : StronglyMeasurable[m] (fun ω ↦ α + (s j ω) ^ 2 + γ * (r j ω) ^ 2) := by
      fun_prop
    have hraw_expr : StronglyMeasurable[m] (fun ω ↦ ((α + (s j ω) ^ 2 + γ * (r j ω) ^ 2 : ℝ))⁻¹) :=
      hbase.inv₀
    simpa [raw, Variance.raw, Variance.scale] using hraw_expr
  have hsum_meas : StronglyMeasurable[m] (fun ω ↦ ∑ j, raw j ω) := by
    refine Finset.stronglyMeasurable_fun_sum _ ?_
    intro j _
    exact hraw_meas j
  have hdiv : StronglyMeasurable[m] (fun ω ↦ raw i ω / ∑ j, raw j ω) := by
    exact (hraw_meas i).div hsum_meas
  simpa [raw, Variance.weights, ofPositive, normalize] using hdiv

/-- Concrete Appendix C.3 conditional MSE theorem for the ENN inverse-variance
weight construction. This instantiates the generic random-weight theorem with
weights built from the variance-scale formula. -/
theorem Noise.mse_weights [DecidableEq ι]
    [Nonempty ι] {X : Type*} [MetricSpace X]
    (hm : m ≤ mΩ)
    (s r : ι → Ω → ℝ) (f : X → ℝ) (z : ι → X) (ε : ι → Ω → ℝ) (x : X)
    {α γ L ρ σ2 : ℝ}
    (hs_meas : ∀ i, StronglyMeasurable[m] (s i))
    (hr_meas : ∀ i, StronglyMeasurable[m] (r i))
    (hα : 0 < α) (hγ : 0 ≤ γ)
    (hL : 0 ≤ L)
    (hLip : ∀ i, |f (z i) - f x| ≤ L * dist (z i) x)
    (hρ : ∀ i, dist (z i) x ≤ ρ)
    (hε2 : ∀ i, MemLp (ε i) 2 μ)
    (hε_zero : ∀ i, μ[ε i | m] =ᵐ[μ] 0)
    (hε_orth : ∀ i j, i ≠ j → μ[fun ω ↦ ε i ω * ε j ω | m] =ᵐ[μ] 0)
    (hε_sq : ∀ i, μ[fun ω ↦ (ε i ω) ^ 2 | m] ≤ᵐ[μ] fun _ ↦ σ2) :
    μ[fun ω ↦
      (∑ i,
        (Variance.weights α γ (fun j ↦ s j ω) (fun j ↦ r j ω) hα hγ).weight i *
          (f (z i) + ε i ω) - f x) ^ 2 | m]
      ≤ᵐ[μ] fun ω ↦
        (L * ρ) ^ 2 + σ2 * ∑ i,
          ((Variance.weights α γ (fun j ↦ s j ω) (fun j ↦ r j ω) hα hγ).weight i) ^ 2 := by
  let w : ι → Ω → ℝ := fun i ω ↦
    (Variance.weights α γ (fun j ↦ s j ω) (fun j ↦ r j ω) hα hγ).weight i
  have hw_meas : ∀ i, StronglyMeasurable[m] (w i) := by
    intro i
    simpa [w] using
      Variance.measurable s r hs_meas hr_meas hα hγ i
  have hw0 : ∀ i, ∀ᵐ ω ∂μ, 0 ≤ w i ω := by
    intro i
    exact Filter.Eventually.of_forall fun ω ↦
      (Variance.weights α γ (fun j ↦ s j ω) (fun j ↦ r j ω) hα hγ).nonnegative i
  have hw_sum : ∀ᵐ ω ∂μ, ∑ i, w i ω = 1 := by
    exact Filter.Eventually.of_forall fun ω ↦
      (Variance.weights α γ (fun j ↦ s j ω) (fun j ↦ r j ω) hα hγ).sum_eq_one
  simpa [w] using
    Noise.mse hm
      w f z ε x hw_meas hw0 hw_sum
      hL hLip hρ hε2 hε_zero hε_orth hε_sq

end ENN
