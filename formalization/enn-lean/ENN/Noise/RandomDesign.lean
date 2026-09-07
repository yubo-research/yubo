import ENN.Noise.Independence
import ENN.OrderedWeightedKNN

/-!
# Conditional bounds for random designs
-/

namespace ENN

open Filter MeasureTheory

open ProbabilityTheory Topology

section ConditionalRandomDesign

variable {ι Ω X : Type*} [Fintype ι] [Nonempty ι] [MetricSpace X]
variable {m mΩ : MeasurableSpace Ω} {μ : @Measure Ω mΩ}
variable [IsFiniteMeasure μ]

/-- Appendix C, Theorem 2's conditional MSE inequality with random,
conditioning-measurable reordered design points and a random neighbor radius.
The conditional-orthogonality premise is precisely the consequence of
conditional independence used by the paper's proof. -/
theorem Noise.mse_random [DecidableEq ι]
    (hm : m ≤ mΩ)
    (w : ι → Ω → ℝ) (f : X → ℝ) (z : ι → Ω → X)
    (ε : ι → Ω → ℝ) (x : X) (radius : Ω → ℝ)
    {L σ2 : ℝ}
    (hw_meas : ∀ i, StronglyMeasurable[m] (w i))
    (hw0 : ∀ i, ∀ᵐ ω ∂μ, 0 ≤ w i ω)
    (hw_sum : ∀ᵐ ω ∂μ, ∑ i, w i ω = 1)
    (hfz_meas : ∀ i, StronglyMeasurable[m] (fun ω ↦ f (z i ω)))
    (hρ2 : MemLp radius 2 μ)
    (hL : 0 ≤ L)
    (hLip : ∀ i, ∀ᵐ ω ∂μ,
      |f (z i ω) - f x| ≤ L * dist (z i ω) x)
    (hradius : ∀ i, ∀ᵐ ω ∂μ, dist (z i ω) x ≤ radius ω)
    (hε2 : ∀ i, MemLp (ε i) 2 μ)
    (hε_zero : ∀ i, μ[ε i | m] =ᵐ[μ] 0)
    (hε_orth : ∀ i j, i ≠ j → μ[fun ω ↦ ε i ω * ε j ω | m] =ᵐ[μ] 0)
    (hε_sq : ∀ i, μ[fun ω ↦ (ε i ω) ^ 2 | m] ≤ᵐ[μ] fun _ ↦ σ2) :
    μ[fun ω ↦
      (∑ i, w i ω * (f (z i ω) + ε i ω) - f x) ^ 2 | m]
      ≤ᵐ[μ] fun ω ↦
        (L * radius ω) ^ 2 + σ2 * ∑ i, (w i ω) ^ 2 := by
  let b : Ω → ℝ := fun ω ↦ ∑ i, w i ω * (f (z i ω) - f x)
  let N : Ω → ℝ := fun ω ↦ ∑ i, w i ω * ε i ω
  have hw0_all : ∀ᵐ ω ∂μ, ∀ i, 0 ≤ w i ω := by
    rw [ae_all_iff]
    exact hw0
  have hw_bound : ∀ i, ∀ᵐ ω ∂μ, |w i ω| ≤ 1 := by
    intro i
    filter_upwards [hw0_all, hw_sum] with ω hwω0 hω_sum
    have hrest : 0 ≤ Finset.sum (Finset.univ.erase i) (fun j ↦ w j ω) :=
      Finset.sum_nonneg fun j _ ↦ hwω0 j
    have hsum : Finset.sum (Finset.univ.erase i) (fun j ↦ w j ω) + w i ω = 1 := by
      rw [Finset.sum_erase_add (s := Finset.univ) (f := fun j ↦ w j ω)
        (a := i) (h := Finset.mem_univ i), hω_sum]
    rw [abs_of_nonneg (hwω0 i)]
    linarith
  have hb_meas : StronglyMeasurable[m] b := by
    apply Finset.stronglyMeasurable_fun_sum
    intro i _
    exact (hw_meas i).mul ((hfz_meas i).sub stronglyMeasurable_const)
  have hLip_all : ∀ᵐ ω ∂μ, ∀ i,
      |f (z i ω) - f x| ≤ L * dist (z i ω) x := by
    rw [ae_all_iff]
    exact hLip
  have hρ_all : ∀ᵐ ω ∂μ, ∀ i, dist (z i ω) x ≤ radius ω := by
    rw [ae_all_iff]
    exact hradius
  have hb_bound : ∀ᵐ ω ∂μ, |b ω| ≤ L * radius ω := by
    filter_upwards [hw0_all, hw_sum, hLip_all, hρ_all] with
      ω hwω0 hω_sum hωLip hωradius
    let wω : Weights ι := {
      weight := fun i ↦ w i ω
      nonnegative := hwω0
      sum_eq_one := hω_sum }
    exact Noise.bias wω f (fun i ↦ z i ω) x
      hL hωLip hωradius
  have hρ0 : ∀ᵐ ω ∂μ, 0 ≤ radius ω := by
    let i0 : ι := Classical.choice inferInstance
    filter_upwards [hradius i0] with ω hω
    exact (dist_nonneg.trans hω)
  have hb2 : MemLp b 2 μ := by
    refine hρ2.of_le_mul (c := L) (hb_meas.mono hm).aestronglyMeasurable ?_
    filter_upwards [hb_bound, hρ0] with ω hb hr
    change |b ω| ≤ L * |radius ω|
    rw [abs_of_nonneg hr]
    exact hb
  have hN_eq : N = ∑ i, fun ω ↦ w i ω * ε i ω := by
    funext ω
    simp [N, Finset.sum_apply]
  have hN_zero : μ[N | m] =ᵐ[μ] 0 := by
    rw [hN_eq]
    exact Noise.noise_mean hm w ε hw_meas hw_bound hε2 hε_zero
  have hN_sq :
      μ[fun ω ↦ (N ω) ^ 2 | m]
        ≤ᵐ[μ] fun ω ↦ σ2 * ∑ i, (w i ω) ^ 2 := by
    exact Noise.noise_var hm
      w ε hw_meas hw_bound hε2 hε_orth hε_sq
  have hN2 : MemLp N 2 μ := by
    rw [hN_eq]
    exact memLp_finsetSum' Finset.univ
      (fun i _ ↦ Noise.product_l2 (μ := μ)
        (w i) (ε i) ((hw_meas i).mono hm).aestronglyMeasurable
        (hw_bound i) (hε2 i))
  have herror :
      (fun ω ↦ ∑ i, w i ω * (f (z i ω) + ε i ω) - f x)
        =ᵐ[μ] fun ω ↦ b ω + N ω := by
    filter_upwards [hw_sum] with ω hω_sum
    dsimp [b, N]
    calc
      (∑ i, w i ω * (f (z i ω) + ε i ω)) - f x
          = (∑ i, w i ω * (f (z i ω) + ε i ω)) -
              ∑ i, w i ω * f x := by
                congr 1
                rw [← Finset.sum_mul, hω_sum, one_mul]
      _ = ∑ i, (w i ω * (f (z i ω) + ε i ω) - w i ω * f x) := by
        rw [Finset.sum_sub_distrib]
      _ = ∑ i, (w i ω * (f (z i ω) - f x) + w i ω * ε i ω) := by
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = (∑ i, w i ω * (f (z i ω) - f x)) + ∑ i, w i ω * ε i ω := by
        rw [Finset.sum_add_distrib]
  have hdecomp :
      μ[fun ω ↦ (b ω + N ω) ^ 2 | m]
        =ᵐ[μ] fun ω ↦ (b ω) ^ 2 + μ[fun ω' ↦ (N ω') ^ 2 | m] ω :=
    Noise.sq_add_l2 hm b N hb_meas hb2 hN2 hN_zero
  have hb_sq : ∀ᵐ ω ∂μ, (b ω) ^ 2 ≤ (L * radius ω) ^ 2 := by
    filter_upwards [hb_bound, hρ0] with ω hω hr
    apply sq_le_sq.mpr
    rw [abs_of_nonneg (mul_nonneg hL hr)]
    exact hω
  calc
    μ[fun ω ↦ (∑ i, w i ω * (f (z i ω) + ε i ω) - f x) ^ 2 | m]
        =ᵐ[μ] μ[fun ω ↦ (b ω + N ω) ^ 2 | m] :=
          condExp_congr_ae (herror.fun_comp fun t : ℝ ↦ t ^ 2)
    _ =ᵐ[μ] fun ω ↦ (b ω) ^ 2 + μ[fun ω' ↦ (N ω') ^ 2 | m] ω := hdecomp
    _ ≤ᵐ[μ] fun ω ↦ (L * radius ω) ^ 2 + σ2 * ∑ i, (w i ω) ^ 2 := by
      filter_upwards [hb_sq, hN_sq] with ω hb hN
      linarith

/-- The random-design conditional MSE theorem specialized to the paper's ENN
inverse-variance weights.  The functions `z`, `s`, and `r` represent the
measurably reordered neighbors and their attached scales and radii. -/
theorem Noise.mse_random_weights
    [DecidableEq ι]
    (hm : m ≤ mΩ)
    (s r : ι → Ω → ℝ) (f : X → ℝ) (z : ι → Ω → X)
    (ε : ι → Ω → ℝ) (x : X) (radius : Ω → ℝ)
    {α γ L σ2 : ℝ}
    (hs_meas : ∀ i, StronglyMeasurable[m] (s i))
    (hr_meas : ∀ i, StronglyMeasurable[m] (r i))
    (hfz_meas : ∀ i, StronglyMeasurable[m] (fun ω ↦ f (z i ω)))
    (hρ2 : MemLp radius 2 μ)
    (hr_eq : ∀ i ω, r i ω = dist (z i ω) x)
    (hα : 0 < α) (hγ : 0 ≤ γ) (hL : 0 ≤ L)
    (hLip : ∀ i, ∀ᵐ ω ∂μ, |f (z i ω) - f x| ≤ L * r i ω)
    (hradius : ∀ i, ∀ᵐ ω ∂μ, r i ω ≤ radius ω)
    (hε2 : ∀ i, MemLp (ε i) 2 μ)
    (hε_zero : ∀ i, μ[ε i | m] =ᵐ[μ] 0)
    (hε_orth : ∀ i j, i ≠ j → μ[fun ω ↦ ε i ω * ε j ω | m] =ᵐ[μ] 0)
    (hε_sq : ∀ i, μ[fun ω ↦ (ε i ω) ^ 2 | m] ≤ᵐ[μ] fun _ ↦ σ2) :
    μ[fun ω ↦
      (∑ i,
          (Variance.weights α γ (fun j ↦ s j ω) (fun j ↦ r j ω) hα hγ).weight i *
            (f (z i ω) + ε i ω) - f x) ^ 2 | m]
      ≤ᵐ[μ] fun ω ↦
        (L * radius ω) ^ 2 + σ2 * ∑ i,
          ((Variance.weights α γ (fun j ↦ s j ω) (fun j ↦ r j ω) hα hγ).weight i) ^ 2 := by
  let w : ι → Ω → ℝ := fun i ω ↦
    (Variance.weights α γ (fun j ↦ s j ω) (fun j ↦ r j ω) hα hγ).weight i
  have hw_meas : ∀ i, StronglyMeasurable[m] (w i) := by
    intro i
    exact Variance.measurable
      s r hs_meas hr_meas hα hγ i
  have hw0 : ∀ i, ∀ᵐ ω ∂μ, 0 ≤ w i ω := by
    intro i
    exact Filter.Eventually.of_forall fun ω ↦
      (Variance.weights α γ (fun j ↦ s j ω) (fun j ↦ r j ω) hα hγ).nonnegative i
  have hw_sum : ∀ᵐ ω ∂μ, ∑ i, w i ω = 1 :=
    Filter.Eventually.of_forall fun ω ↦
      (Variance.weights α γ (fun j ↦ s j ω) (fun j ↦ r j ω) hα hγ).sum_eq_one
  have hLip' : ∀ i, ∀ᵐ ω ∂μ,
      |f (z i ω) - f x| ≤ L * dist (z i ω) x := by
    intro i
    filter_upwards [hLip i] with ω hω
    rwa [← hr_eq i ω]
  have hradius' : ∀ i, ∀ᵐ ω ∂μ, dist (z i ω) x ≤ radius ω := by
    intro i
    filter_upwards [hradius i] with ω hω
    rwa [← hr_eq i ω]
  exact Noise.mse_random hm
    w f z ε x radius hw_meas hw0 hw_sum hfz_meas hρ2
      hL hLip' hradius' hε2 hε_zero hε_orth hε_sq

omit [Fintype ι] [Nonempty ι] in
/-- The paper's conditional-independence and conditional-centering assumptions
imply the conditional orthogonality used by the weighted-noise calculation. -/
theorem Noise.orthogonal
    [StandardBorelSpace Ω]
    (hm : m ≤ mΩ) (ε : ι → Ω → ℝ)
    (hε_meas : ∀ i, @Measurable Ω ℝ mΩ _ (ε i))
    (hε2 : ∀ i, MemLp (ε i) 2 μ)
    (hε_zero : ∀ i, μ[ε i | m] =ᵐ[μ] 0)
    (hε_indep : iCondIndepFun m hm ε μ) :
    ∀ i j, i ≠ j → μ[fun ω ↦ ε i ω * ε j ω | m] =ᵐ[μ] 0 := by
  intro i j hij
  exact Noise.condExp_zero
    (mΩ := mΩ) hm (hε_meas i) (hε_meas j)
    ((hε2 i).integrable one_le_two)
    ((hε2 j).integrable one_le_two)
    (MemLp.integrable_mul (hε2 i) (hε2 j))
    (hε_indep.condIndepFun hij) (hε_zero i) (hε_zero j)

/-- Appendix C, Theorem 2's random-design conditional MSE bound under the
paper's stated conditional-independence hypothesis, specialized to the ENN
inverse-variance weights. -/
theorem Noise.mse_indep
    [DecidableEq ι] [StandardBorelSpace Ω]
    (hm : m ≤ mΩ)
    (s r : ι → Ω → ℝ) (f : X → ℝ) (z : ι → Ω → X)
    (ε : ι → Ω → ℝ) (x : X) (radius : Ω → ℝ)
    {α γ L σ2 : ℝ}
    (hs_meas : ∀ i, StronglyMeasurable[m] (s i))
    (hr_meas : ∀ i, StronglyMeasurable[m] (r i))
    (hfz_meas : ∀ i, StronglyMeasurable[m] (fun ω ↦ f (z i ω)))
    (hρ2 : MemLp radius 2 μ)
    (hr_eq : ∀ i ω, r i ω = dist (z i ω) x)
    (hα : 0 < α) (hγ : 0 ≤ γ) (hL : 0 ≤ L)
    (hLip : ∀ i, ∀ᵐ ω ∂μ, |f (z i ω) - f x| ≤ L * r i ω)
    (hradius : ∀ i, ∀ᵐ ω ∂μ, r i ω ≤ radius ω)
    (hε_meas : ∀ i, @Measurable Ω ℝ mΩ _ (ε i))
    (hε2 : ∀ i, MemLp (ε i) 2 μ)
    (hε_zero : ∀ i, μ[ε i | m] =ᵐ[μ] 0)
    (hε_indep : iCondIndepFun m hm ε μ)
    (hε_sq : ∀ i, μ[fun ω ↦ (ε i ω) ^ 2 | m] ≤ᵐ[μ] fun _ ↦ σ2) :
    μ[fun ω ↦
      (∑ i,
          (Variance.weights α γ (fun j ↦ s j ω) (fun j ↦ r j ω) hα hγ).weight i *
            (f (z i ω) + ε i ω) - f x) ^ 2 | m]
      ≤ᵐ[μ] fun ω ↦
        (L * radius ω) ^ 2 + σ2 * ∑ i,
          ((Variance.weights α γ (fun j ↦ s j ω) (fun j ↦ r j ω) hα hγ).weight i) ^ 2 := by
  apply Noise.mse_random_weights hm
    s r f z ε x radius hs_meas hr_meas hfz_meas hρ2 hr_eq
    hα hγ hL hLip hradius hε2 hε_zero _ hε_sq
  exact Noise.orthogonal
    hm ε hε_meas hε2 hε_zero hε_indep

end ConditionalRandomDesign

end ENN
