import ENN.Noise.Estimator

/-!
# Conditional noise moments
-/

namespace ENN

open Filter MeasureTheory

variable {ι Ω : Type*} [Fintype ι]
variable {m mΩ : MeasurableSpace Ω} {μ : @Measure Ω mΩ}
variable [IsFiniteMeasure μ]

/-- A generic conditional mean-zero theorem for weighted finite noise sums. The
weights are allowed to depend on the conditioning sigma-algebra, provided they
are measurable with respect to it and uniformly bounded. -/
theorem Noise.noise_mean
    (hm : m ≤ mΩ)
    (w : ι → Ω → ℝ) (ε : ι → Ω → ℝ)
    (hw_meas : ∀ i, StronglyMeasurable[m] (w i))
    (hw_bound : ∀ i, ∀ᵐ ω ∂μ, |w i ω| ≤ 1)
    (hε2 : ∀ i, MemLp (ε i) 2 μ)
    (hε_zero : ∀ i, μ[ε i | m] =ᵐ[μ] 0) :
    μ[∑ i, fun ω ↦ w i ω * ε i ω | m] =ᵐ[μ] 0 := by
  have hterm_int : ∀ i, Integrable (fun ω ↦ w i ω * ε i ω) μ := by
    intro i
    exact Integrable.bdd_mul ((hε2 i).integrable one_le_two)
      ((hw_meas i).mono hm).aestronglyMeasurable (hw_bound i)
  calc
    μ[∑ i, fun ω ↦ w i ω * ε i ω | m]
      =ᵐ[μ] ∑ i, μ[(fun ω ↦ w i ω * ε i ω) | m] := by
          simpa using
            condExp_finsetSum
              (s := Finset.univ)
              (f := fun i ω ↦ w i ω * ε i ω)
              (hf := fun i _ ↦ hterm_int i) m
    _ =ᵐ[μ] ∑ i, w i * μ[ε i | m] := by
          refine eventuallyEq_sum fun i _ ↦ ?_
          exact condExp_mul_of_stronglyMeasurable_left
            (hw_meas i) (hterm_int i) ((hε2 i).integrable one_le_two)
    _ =ᵐ[μ] ∑ i, w i * 0 := by
          refine eventuallyEq_sum fun i _ ↦ ?_
          exact Filter.EventuallyEq.mul Filter.EventuallyEq.rfl (hε_zero i)
    _ =ᵐ[μ] 0 := by simp

omit [IsFiniteMeasure μ] in
/-- Multiplying an `Lᵖ` scalar function by an a.e. bounded scalar factor
preserves `Lᵖ`. This is a reusable helper for the conditional C.3 weighted-noise
analysis. -/
lemma Noise.product_l2 {p : ENNReal} (w ε : Ω → ℝ)
    (hw_meas : AEStronglyMeasurable w μ) (hw_bound : ∀ᵐ ω ∂μ, |w ω| ≤ 1)
    (hε2 : MemLp ε p μ) :
    MemLp (fun ω ↦ w ω * ε ω) p μ := by
  refine hε2.of_le_mul (c := 1) (hw_meas.mul hε2.aestronglyMeasurable) ?_
  filter_upwards [hw_bound] with ω hω
  have hmul : |w ω| * |ε ω| ≤ 1 * |ε ω| := by
    exact mul_le_mul_of_nonneg_right hω (abs_nonneg _)
  simpa [Real.norm_eq_abs, abs_mul] using hmul

omit [IsFiniteMeasure μ] in
/-- If two scalar factors are a.e. bounded by `1` and the remaining factors are
in `L²`, then their product is integrable. This is the finite-sample product
integrability fact needed for the conditional C.3 second-moment expansion. -/
lemma Noise.product_integrable (w₁ w₂ ε₁ ε₂ : Ω → ℝ)
    (hw₁_meas : AEStronglyMeasurable w₁ μ) (hw₂_meas : AEStronglyMeasurable w₂ μ)
    (hw₁_bound : ∀ᵐ ω ∂μ, |w₁ ω| ≤ 1) (hw₂_bound : ∀ᵐ ω ∂μ, |w₂ ω| ≤ 1)
    (hε₁2 : MemLp ε₁ 2 μ) (hε₂2 : MemLp ε₂ 2 μ) :
    Integrable (fun ω ↦ (w₁ ω * ε₁ ω) * (w₂ ω * ε₂ ω)) μ := by
  have hξ₁ : MemLp (fun ω ↦ w₁ ω * ε₁ ω) 2 μ :=
    Noise.product_l2 w₁ ε₁ hw₁_meas hw₁_bound hε₁2
  have hξ₂ : MemLp (fun ω ↦ w₂ ω * ε₂ ω) 2 μ :=
    Noise.product_l2 w₂ ε₂ hw₂_meas hw₂_bound hε₂2
  exact MemLp.integrable_mul hξ₁ hξ₂

omit [IsFiniteMeasure μ] in
/-- A diagonal weighted-noise square pulls out of conditional expectation when
the weight is measurable with respect to the conditioning sigma-algebra. -/
lemma Noise.diag_term
    (hm : m ≤ mΩ)
    (w ε : Ω → ℝ)
    (hw_meas : StronglyMeasurable[m] w)
    (hw_bound : ∀ᵐ ω ∂μ, |w ω| ≤ 1)
    (hε2 : MemLp ε 2 μ) :
    μ[(fun ω ↦ (w ω * ε ω) ^ 2) | m]
      =ᵐ[μ] fun ω ↦ (w ω) ^ 2 * μ[fun ω' ↦ (ε ω') ^ 2 | m] ω := by
  have hprod2 : MemLp (fun ω ↦ w ω * ε ω) 2 μ :=
    Noise.product_l2 (μ := μ) w ε (hw_meas.mono hm).aestronglyMeasurable
      hw_bound hε2
  have hprod_int : Integrable (fun ω ↦ (w ω * ε ω) * (w ω * ε ω)) μ :=
    MemLp.integrable_mul hprod2 hprod2
  have hmid_int : Integrable (fun ω ↦ ε ω * (w ω * ε ω)) μ :=
    MemLp.integrable_mul hε2 hprod2
  have hsq_int : Integrable (fun ω ↦ ε ω * ε ω) μ :=
    MemLp.integrable_mul hε2 hε2
  have hrew₁ :
      (fun ω ↦ (w ω * ε ω) ^ 2) =ᵐ[μ] fun ω ↦ w ω * (ε ω * (w ω * ε ω)) := by
    exact Filter.Eventually.of_forall fun ω => by ring_nf
  have hrew₂ :
      (fun ω ↦ ε ω * (w ω * ε ω)) =ᵐ[μ] fun ω ↦ w ω * (ε ω * ε ω) := by
    exact Filter.Eventually.of_forall fun ω => by ring_nf
  calc
    μ[(fun ω ↦ (w ω * ε ω) ^ 2) | m]
      =ᵐ[μ] μ[fun ω ↦ w ω * (ε ω * (w ω * ε ω)) | m] := condExp_congr_ae hrew₁
    _ =ᵐ[μ] fun ω ↦ w ω * μ[fun ω ↦ ε ω * (w ω * ε ω) | m] ω := by
          exact condExp_mul_of_stronglyMeasurable_left hw_meas
            (by
              convert hprod_int using 1
              ext ω
              simp [Pi.mul_apply, mul_left_comm, mul_comm])
            hmid_int
    _ =ᵐ[μ] fun ω ↦ w ω * μ[fun ω ↦ w ω * (ε ω * ε ω) | m] ω := by
          exact Filter.EventuallyEq.mul Filter.EventuallyEq.rfl (condExp_congr_ae hrew₂)
    _ =ᵐ[μ] fun ω ↦ w ω * (w ω * μ[fun ω ↦ ε ω * ε ω | m] ω) := by
          exact Filter.EventuallyEq.mul Filter.EventuallyEq.rfl <|
            condExp_mul_of_stronglyMeasurable_left hw_meas
              (by
                convert hmid_int using 1
                ext ω
                simp [Pi.mul_apply, mul_left_comm])
              hsq_int
    _ =ᵐ[μ] fun ω ↦ (w ω) ^ 2 * μ[fun ω' ↦ (ε ω') ^ 2 | m] ω := by
          have hsq_rew :
              μ[fun ω ↦ ε ω * ε ω | m] =ᵐ[μ] μ[fun ω ↦ (ε ω) ^ 2 | m] := by
            exact condExp_congr_ae (Filter.Eventually.of_forall fun ω => by ring_nf)
          filter_upwards [hsq_rew] with ω hω
          simp [pow_two, mul_assoc, hω]

omit [IsFiniteMeasure μ] in
/-- An off-diagonal weighted-noise product has zero conditional expectation once
the underlying noise product is conditionally centered. -/
lemma Noise.cross_zero
    (hm : m ≤ mΩ)
    (w₁ w₂ ε₁ ε₂ : Ω → ℝ)
    (hw₁_meas : StronglyMeasurable[m] w₁)
    (hw₂_meas : StronglyMeasurable[m] w₂)
    (hw₁_bound : ∀ᵐ ω ∂μ, |w₁ ω| ≤ 1)
    (hw₂_bound : ∀ᵐ ω ∂μ, |w₂ ω| ≤ 1)
    (hε₁2 : MemLp ε₁ 2 μ)
    (hε₂2 : MemLp ε₂ 2 μ)
    (hε_cross : μ[fun ω ↦ ε₁ ω * ε₂ ω | m] =ᵐ[μ] 0) :
    μ[(fun ω ↦ (w₁ ω * ε₁ ω) * (w₂ ω * ε₂ ω)) | m] =ᵐ[μ] 0 := by
  have hprod₂2 : MemLp (fun ω ↦ w₂ ω * ε₂ ω) 2 μ :=
    Noise.product_l2 (μ := μ) w₂ ε₂ (hw₂_meas.mono hm).aestronglyMeasurable
      hw₂_bound hε₂2
  have hprod_int : Integrable (fun ω ↦ (w₁ ω * ε₁ ω) * (w₂ ω * ε₂ ω)) μ :=
    Noise.product_integrable (μ := μ) w₁ w₂ ε₁ ε₂
      (hw₁_meas.mono hm).aestronglyMeasurable (hw₂_meas.mono hm).aestronglyMeasurable
      hw₁_bound hw₂_bound hε₁2 hε₂2
  have hmid_int : Integrable (fun ω ↦ ε₁ ω * (w₂ ω * ε₂ ω)) μ :=
    MemLp.integrable_mul hε₁2 hprod₂2
  have hcross_int : Integrable (fun ω ↦ ε₁ ω * ε₂ ω) μ :=
    MemLp.integrable_mul hε₁2 hε₂2
  have hrew₁ :
      (fun ω ↦ (w₁ ω * ε₁ ω) * (w₂ ω * ε₂ ω))
        =ᵐ[μ] fun ω ↦ w₁ ω * (ε₁ ω * (w₂ ω * ε₂ ω)) := by
    exact Filter.Eventually.of_forall fun ω => by ring_nf
  have hrew₂ :
      (fun ω ↦ ε₁ ω * (w₂ ω * ε₂ ω))
        =ᵐ[μ] fun ω ↦ w₂ ω * (ε₁ ω * ε₂ ω) := by
    exact Filter.Eventually.of_forall fun ω => by ring_nf
  calc
    μ[(fun ω ↦ (w₁ ω * ε₁ ω) * (w₂ ω * ε₂ ω)) | m]
      =ᵐ[μ] μ[fun ω ↦ w₁ ω * (ε₁ ω * (w₂ ω * ε₂ ω)) | m] := condExp_congr_ae hrew₁
    _ =ᵐ[μ] fun ω ↦ w₁ ω * μ[fun ω ↦ ε₁ ω * (w₂ ω * ε₂ ω) | m] ω := by
          exact condExp_mul_of_stronglyMeasurable_left hw₁_meas
            (by
              convert hprod_int using 1
              ext ω
              simp [Pi.mul_apply, mul_left_comm, mul_comm])
            hmid_int
    _ =ᵐ[μ] fun ω ↦ w₁ ω * μ[fun ω ↦ w₂ ω * (ε₁ ω * ε₂ ω) | m] ω := by
          exact Filter.EventuallyEq.mul Filter.EventuallyEq.rfl (condExp_congr_ae hrew₂)
    _ =ᵐ[μ] fun ω ↦ w₁ ω * (w₂ ω * μ[fun ω ↦ ε₁ ω * ε₂ ω | m] ω) := by
          exact Filter.EventuallyEq.mul Filter.EventuallyEq.rfl <|
            condExp_mul_of_stronglyMeasurable_left hw₂_meas
              (by
                convert hmid_int using 1
                ext ω
                simp [Pi.mul_apply, mul_left_comm])
              hcross_int
    _ =ᵐ[μ] 0 := by
          filter_upwards [hε_cross] with ω hω
          simp [hω]

/-- Deterministic square expansion into diagonal and off-diagonal terms. This
is the finite combinatorial identity underlying the conditional second-moment
expansion. -/
lemma Noise.sq_split [DecidableEq ι] (a : ι → ℝ) :
    (∑ i, a i) ^ 2
      = (∑ i, (a i) ^ 2) +
          ∑ i, Finset.sum (Finset.univ.erase i) (fun j ↦ a i * a j) := by
  calc
    (∑ i, a i) ^ 2 = (∑ i, a i) * (∑ j, a j) := by rw [pow_two]
    _ = ∑ i, a i * (∑ j, a j) := by rw [Finset.sum_mul]
    _ = ∑ i, ((a i) ^ 2 + Finset.sum (Finset.univ.erase i) (fun j ↦ a i * a j)) := by
          refine Finset.sum_congr rfl ?_
          intro i _
          rw [Finset.mul_sum]
          rw [← Finset.sum_erase_add (s := Finset.univ) (f := fun j ↦ a i * a j)
            (a := i) (h := Finset.mem_univ i)]
          ring
    _ = (∑ i, (a i) ^ 2) + ∑ i, Finset.sum (Finset.univ.erase i) (fun j ↦ a i * a j) := by
          rw [Finset.sum_add_distrib]

omit [IsFiniteMeasure μ] in
/-- Conditional expectation of the diagonal part of the square expansion. -/
lemma Noise.diag_sum [DecidableEq ι]
    (hm : m ≤ mΩ)
    (w : ι → Ω → ℝ) (ε : ι → Ω → ℝ)
    (hw_meas : ∀ i, StronglyMeasurable[m] (w i))
    (hw_bound : ∀ i, ∀ᵐ ω ∂μ, |w i ω| ≤ 1)
    (hε2 : ∀ i, MemLp (ε i) 2 μ) :
    μ[∑ i, fun ω ↦ (w i ω * ε i ω) ^ 2 | m]
      =ᵐ[μ] ∑ i, fun ω ↦ (w i ω) ^ 2 * μ[fun ω' ↦ (ε i ω') ^ 2 | m] ω := by
  have hdiag_int : ∀ i, Integrable (fun ω ↦ (w i ω * ε i ω) ^ 2) μ := by
    intro i
    simpa [pow_two] using
      Noise.product_integrable (μ := μ) (w i) (w i) (ε i) (ε i)
        ((hw_meas i).mono hm).aestronglyMeasurable
        ((hw_meas i).mono hm).aestronglyMeasurable
        (hw_bound i) (hw_bound i) (hε2 i) (hε2 i)
  calc
    μ[∑ i, fun ω ↦ (w i ω * ε i ω) ^ 2 | m]
      =ᵐ[μ] ∑ i, μ[(fun ω ↦ (w i ω * ε i ω) ^ 2) | m] := by
          simpa using
            condExp_finsetSum
              (s := Finset.univ)
              (f := fun i ω ↦ (w i ω * ε i ω) ^ 2)
              (hf := fun i _ ↦ hdiag_int i) m
    _ =ᵐ[μ] ∑ i, (fun ω ↦ (w i ω) ^ 2 * μ[fun ω' ↦ (ε i ω') ^ 2 | m] ω) := by
          refine eventuallyEq_sum fun i _ ↦ ?_
          simpa using Noise.diag_term hm (w i) (ε i) (hw_meas i) (hw_bound i)
            (hε2 i)

omit [IsFiniteMeasure μ] in
/-- Conditional expectation of the off-diagonal part of the square expansion
vanishes under conditional orthogonality. -/
lemma Noise.offdiag_zero [DecidableEq ι]
    (hm : m ≤ mΩ)
    (w : ι → Ω → ℝ) (ε : ι → Ω → ℝ)
    (hw_meas : ∀ i, StronglyMeasurable[m] (w i))
    (hw_bound : ∀ i, ∀ᵐ ω ∂μ, |w i ω| ≤ 1)
    (hε2 : ∀ i, MemLp (ε i) 2 μ)
    (hε_orth : ∀ i j, i ≠ j → μ[fun ω ↦ ε i ω * ε j ω | m] =ᵐ[μ] 0) :
    μ[∑ i, fun ω ↦ Finset.sum (Finset.univ.erase i)
      (fun j ↦ (w i ω * ε i ω) * (w j ω * ε j ω)) | m]
      =ᵐ[μ] 0 := by
  have hinner_int :
      ∀ i, Integrable (fun ω ↦ Finset.sum (Finset.univ.erase i)
        (fun j ↦ (w i ω * ε i ω) * (w j ω * ε j ω))) μ := by
    intro i
    simpa using
      integrable_finsetSum
        (s := Finset.univ.erase i)
        (f := fun j ω ↦ (w i ω * ε i ω) * (w j ω * ε j ω))
        (hf := fun j _ ↦
          Noise.product_integrable (μ := μ) (w i) (w j) (ε i) (ε j)
            ((hw_meas i).mono hm).aestronglyMeasurable
            ((hw_meas j).mono hm).aestronglyMeasurable
            (hw_bound i) (hw_bound j) (hε2 i) (hε2 j))
  calc
    μ[∑ i, fun ω ↦ Finset.sum (Finset.univ.erase i)
        (fun j ↦ (w i ω * ε i ω) * (w j ω * ε j ω)) | m]
      =ᵐ[μ] ∑ i, μ[(fun ω ↦ Finset.sum (Finset.univ.erase i)
          (fun j ↦ (w i ω * ε i ω) * (w j ω * ε j ω))) | m] := by
            simpa using
              condExp_finsetSum
                (s := Finset.univ)
                (f := fun i ω ↦ Finset.sum (Finset.univ.erase i)
                  (fun j ↦ (w i ω * ε i ω) * (w j ω * ε j ω)))
                (hf := fun i _ ↦ hinner_int i) m
    _ =ᵐ[μ] ∑ i, 0 := by
          refine eventuallyEq_sum fun i _ ↦ ?_
          let cross : ι → Ω → ℝ := fun j ω ↦ (w i ω * ε i ω) * (w j ω * ε j ω)
          have hsum_int : Integrable (fun ω ↦ ∑ j, cross j ω) μ := by
            simpa using
              integrable_finsetSum
                (s := Finset.univ)
                (f := cross)
                (hf := fun j _ ↦
                  Noise.product_integrable (μ := μ) (w i) (w j) (ε i) (ε j)
                    ((hw_meas i).mono hm).aestronglyMeasurable
                    ((hw_meas j).mono hm).aestronglyMeasurable
                    (hw_bound i) (hw_bound j) (hε2 i) (hε2 j))
          have hdiag_int : Integrable (cross i) μ := by
            simpa [cross] using
              Noise.product_integrable (μ := μ) (w i) (w i) (ε i) (ε i)
                ((hw_meas i).mono hm).aestronglyMeasurable
                ((hw_meas i).mono hm).aestronglyMeasurable
                (hw_bound i) (hw_bound i) (hε2 i) (hε2 i)
          have hsum_ae : (fun ω ↦ ∑ j, cross j ω) =ᵐ[μ] (∑ j, cross j) := by
            exact Filter.Eventually.of_forall fun ω ↦ by simp [cross]
          have herase_rew :
              (fun ω ↦ Finset.sum (Finset.univ.erase i)
                (fun j ↦ (w i ω * ε i ω) * (w j ω * ε j ω)))
                = fun ω ↦ (∑ j, cross j ω) - cross i ω := by
            funext ω
            have hsum :
                Finset.sum (Finset.univ.erase i) (fun j ↦ cross j ω) + cross i ω
                  = ∑ j, cross j ω := by
              simp [cross]
            linarith
          calc
            μ[(fun ω ↦ Finset.sum (Finset.univ.erase i)
                (fun j ↦ (w i ω * ε i ω) * (w j ω * ε j ω))) | m]
              =ᵐ[μ] μ[(fun ω ↦ (∑ j, cross j ω) - cross i ω) | m] := by
                    exact condExp_congr_ae <|
                      Filter.Eventually.of_forall fun ω ↦ congrFun herase_rew ω
            _ =ᵐ[μ] μ[(fun ω ↦ ∑ j, cross j ω) - cross i | m] := by
                    exact condExp_congr_ae <|
                      Filter.Eventually.of_forall fun ω ↦ by rfl
            _ =ᵐ[μ] μ[fun ω ↦ ∑ j, cross j ω | m] - μ[cross i | m] := by
                    have hsub :
                        μ[(fun ω ↦ ∑ j, cross j ω) - cross i | m]
                          =ᵐ[μ] μ[fun ω ↦ ∑ j, cross j ω | m] - μ[cross i | m] := by
                      simpa using condExp_sub hsum_int hdiag_int m
                    exact hsub
            _ =ᵐ[μ] μ[∑ j, cross j | m] - μ[cross i | m] := by
                    exact (condExp_congr_ae hsum_ae).sub Filter.EventuallyEq.rfl
            _ =ᵐ[μ] (∑ j, μ[cross j | m]) - μ[cross i | m] := by
                    exact (condExp_finsetSum
                      (s := Finset.univ)
                      (f := cross)
                      (hf := fun j _ ↦
                        Noise.product_integrable (μ := μ) (w i) (w j) (ε i) (ε j)
                          ((hw_meas i).mono hm).aestronglyMeasurable
                          ((hw_meas j).mono hm).aestronglyMeasurable
                          (hw_bound i) (hw_bound j) (hε2 i) (hε2 j)) m).sub
                            Filter.EventuallyEq.rfl
            _ =ᵐ[μ] Finset.sum (Finset.univ.erase i)
                (fun j ↦ μ[(cross j) | m]) := by
                    exact Filter.Eventually.of_forall fun ω ↦ by
                      have hsum :
                          Finset.sum (Finset.univ.erase i) (fun j ↦ μ[cross j | m] ω) +
                            μ[cross i | m] ω
                            = ∑ j, μ[cross j | m] ω := by
                        simp
                      have hsum' := congrArg (fun t : ℝ ↦ t - μ[cross i | m] ω) hsum
                      simp [sub_eq_add_neg]
            _ =ᵐ[μ] Finset.sum (Finset.univ.erase i) (fun _ ↦ (0 : Ω → ℝ)) := by
                    refine eventuallyEq_sum fun j hj ↦ ?_
                    exact Noise.cross_zero hm (w i) (w j) (ε i) (ε j)
                      (hw_meas i) (hw_meas j) (hw_bound i) (hw_bound j)
                      (hε2 i) (hε2 j)
                      (hε_orth i j ((Finset.mem_erase.mp hj).1).symm)
            _ =ᵐ[μ] 0 := by simp
    _ =ᵐ[μ] 0 := by simp

omit [IsFiniteMeasure μ] in
/-- Conditional second-moment identity for finite weighted noise sums under
conditional orthogonality of the off-diagonal noise products. -/
theorem Noise.noise_var_eq [DecidableEq ι]
    (hm : m ≤ mΩ)
    (w : ι → Ω → ℝ) (ε : ι → Ω → ℝ)
    (hw_meas : ∀ i, StronglyMeasurable[m] (w i))
    (hw_bound : ∀ i, ∀ᵐ ω ∂μ, |w i ω| ≤ 1)
    (hε2 : ∀ i, MemLp (ε i) 2 μ)
    (hε_orth : ∀ i j, i ≠ j → μ[fun ω ↦ ε i ω * ε j ω | m] =ᵐ[μ] 0) :
    μ[fun ω ↦ (∑ i, w i ω * ε i ω) ^ 2 | m]
      =ᵐ[μ] ∑ i, fun ω ↦ (w i ω) ^ 2 * μ[fun ω' ↦ (ε i ω') ^ 2 | m] ω := by
  let wε : ι → Ω → ℝ := fun i ω ↦ w i ω * ε i ω
  have hdiag_int : ∀ i, Integrable (fun ω ↦ (wε i ω) ^ 2) μ := by
    intro i
    simpa [wε, pow_two] using
      Noise.product_integrable (μ := μ) (w i) (w i) (ε i) (ε i)
        ((hw_meas i).mono hm).aestronglyMeasurable
        ((hw_meas i).mono hm).aestronglyMeasurable
        (hw_bound i) (hw_bound i) (hε2 i) (hε2 i)
  have hdiag_int :
      Integrable (fun ω ↦ ∑ i, (wε i ω) ^ 2) μ := by
    simpa using
      integrable_finsetSum
        (s := Finset.univ)
        (f := fun i ω ↦ (wε i ω) ^ 2)
        (hf := fun i _ ↦ hdiag_int i)
  have hinner_int :
      ∀ i, Integrable (fun ω ↦ Finset.sum (Finset.univ.erase i)
        (fun j ↦ (wε i ω) * (wε j ω))) μ := by
    intro i
    simpa using
      integrable_finsetSum
        (s := Finset.univ.erase i)
        (f := fun j ω ↦ (wε i ω) * (wε j ω))
        (hf := fun j _ ↦
          Noise.product_integrable (μ := μ) (w i) (w j) (ε i) (ε j)
            ((hw_meas i).mono hm).aestronglyMeasurable
            ((hw_meas j).mono hm).aestronglyMeasurable
            (hw_bound i) (hw_bound j) (hε2 i) (hε2 j))
  have hoff_int :
      Integrable (fun ω ↦ ∑ i, Finset.sum (Finset.univ.erase i)
        (fun j ↦ (wε i ω) * (wε j ω))) μ := by
    simpa using
      integrable_finsetSum
        (s := Finset.univ)
        (f := fun i ω ↦ Finset.sum (Finset.univ.erase i)
          (fun j ↦ (wε i ω) * (wε j ω)))
        (hf := fun i _ ↦ hinner_int i)
  let sqPart : Ω → ℝ := fun ω ↦ ∑ i, (wε i ω) ^ 2
  let offDiagPart : Ω → ℝ := fun ω ↦
    ∑ i, Finset.sum (Finset.univ.erase i) (fun j ↦ (wε i ω) * (wε j ω))
  have hsquare :
      (fun ω ↦ (∑ i, wε i ω) ^ 2) =ᵐ[μ]
        fun ω ↦ (∑ i, (wε i ω) ^ 2) + offDiagPart ω := by
    exact Filter.Eventually.of_forall fun ω ↦ by
      simpa [offDiagPart, pow_two] using
        (Noise.sq_split (a := fun i ↦ wε i ω))
  have hsplit_raw :
      μ[fun ω ↦ (∑ i, wε i ω) ^ 2 | m]
        =ᵐ[μ] μ[fun ω ↦ ∑ i, (wε i ω) ^ 2 | m] + μ[offDiagPart | m] := by
    calc
      μ[fun ω ↦ (∑ i, wε i ω) ^ 2 | m]
          =ᵐ[μ] μ[fun ω ↦ (∑ i, (wε i ω) ^ 2) + offDiagPart ω | m] := by
              exact condExp_congr_ae hsquare
      _ =ᵐ[μ] μ[fun ω ↦ ∑ i, (wε i ω) ^ 2 | m] +
            μ[offDiagPart | m] := by
              exact condExp_add hdiag_int hoff_int m
  have hsplit :
      μ[fun ω ↦ (∑ i, wε i ω) ^ 2 | m]
        =ᵐ[μ] μ[sqPart | m] + μ[offDiagPart | m] := by
    simpa [sqPart, offDiagPart] using hsplit_raw
  have hdiag_raw :
      μ[∑ i, fun ω ↦ (wε i ω) ^ 2 | m]
        =ᵐ[μ] ∑ i, fun ω ↦ (w i ω) ^ 2 * μ[fun ω' ↦ (ε i ω') ^ 2 | m] ω := by
    exact Noise.diag_sum hm w ε hw_meas hw_bound hε2
  have hdiag :
      μ[sqPart | m]
        =ᵐ[μ] ∑ i, fun ω ↦ (w i ω) ^ 2 * μ[fun ω' ↦ (ε i ω') ^ 2 | m] ω := by
    have hdiag_fun :
        (fun ω ↦ ∑ i, (wε i ω) ^ 2) =ᵐ[μ]
          ∑ i, fun ω ↦ (wε i ω) ^ 2 := by
      exact Filter.Eventually.of_forall fun ω ↦ by
        simp [Finset.sum_apply]
    have hdiag_congr :
        μ[fun ω ↦ ∑ i, (wε i ω) ^ 2 | m] =ᵐ[μ]
          μ[∑ i, fun ω ↦ (wε i ω) ^ 2 | m] := by
      exact condExp_congr_ae hdiag_fun
    have hdiag' :
        μ[fun ω ↦ ∑ i, (wε i ω) ^ 2 | m] =ᵐ[μ]
          ∑ i, fun ω ↦ (w i ω) ^ 2 * μ[fun ω' ↦ (ε i ω') ^ 2 | m] ω := by
      exact hdiag_congr.trans hdiag_raw
    simpa [sqPart, wε] using hdiag'
  have hoff_raw :
      μ[∑ i, fun ω ↦ Finset.sum (Finset.univ.erase i)
        (fun j ↦ (wε i ω) * (wε j ω)) | m] =ᵐ[μ] 0 := by
    exact Noise.offdiag_zero hm w ε hw_meas hw_bound hε2 hε_orth
  have hoff :
      μ[offDiagPart | m] =ᵐ[μ] 0 := by
    have hoff_fun :
        (fun ω ↦ ∑ i, Finset.sum (Finset.univ.erase i)
          (fun j ↦ (wε i ω) * (wε j ω))) =ᵐ[μ]
          ∑ i, fun ω ↦ Finset.sum (Finset.univ.erase i)
            (fun j ↦ (wε i ω) * (wε j ω)) := by
      exact Filter.Eventually.of_forall fun ω ↦ by
        simp [Finset.sum_apply]
    have hoff_congr :
        μ[fun ω ↦ ∑ i, Finset.sum (Finset.univ.erase i)
          (fun j ↦ (wε i ω) * (wε j ω)) | m] =ᵐ[μ]
          μ[∑ i, fun ω ↦ Finset.sum (Finset.univ.erase i)
            (fun j ↦ (wε i ω) * (wε j ω)) | m] := by
      exact condExp_congr_ae hoff_fun
    have hoff' :
        μ[fun ω ↦ ∑ i, Finset.sum (Finset.univ.erase i)
          (fun j ↦ (wε i ω) * (wε j ω)) | m] =ᵐ[μ] 0 := by
      exact hoff_congr.trans hoff_raw
    simpa [offDiagPart, wε] using hoff'
  calc
    μ[fun ω ↦ (∑ i, w i ω * ε i ω) ^ 2 | m]
      =ᵐ[μ] μ[fun ω ↦ (∑ i, wε i ω) ^ 2 | m] := by
        exact condExp_congr_ae <| Filter.Eventually.of_forall <| by simp [wε]
    _ =ᵐ[μ] μ[sqPart | m] + μ[offDiagPart | m] := hsplit
    _ =ᵐ[μ] (∑ i, fun ω ↦ (w i ω) ^ 2 * μ[fun ω' ↦ (ε i ω') ^ 2 | m] ω) + 0 := by
      exact hdiag.add hoff
    _ =ᵐ[μ] ∑ i, fun ω ↦ (w i ω) ^ 2 * μ[fun ω' ↦ (ε i ω') ^ 2 | m] ω := by
      simp

omit [IsFiniteMeasure μ] in
/-- Conditional second-moment bound for a finite weighted noise sum, assuming
uniform a.e. control of the conditional component second moments. -/
theorem Noise.noise_var [DecidableEq ι]
    (hm : m ≤ mΩ)
    (w : ι → Ω → ℝ) (ε : ι → Ω → ℝ) {σ2 : ℝ}
    (hw_meas : ∀ i, StronglyMeasurable[m] (w i))
    (hw_bound : ∀ i, ∀ᵐ ω ∂μ, |w i ω| ≤ 1)
    (hε2 : ∀ i, MemLp (ε i) 2 μ)
    (hε_orth : ∀ i j, i ≠ j → μ[fun ω ↦ ε i ω * ε j ω | m] =ᵐ[μ] 0)
    (hε_sq : ∀ i, μ[fun ω ↦ (ε i ω) ^ 2 | m] ≤ᵐ[μ] fun _ ↦ σ2) :
    μ[fun ω ↦ (∑ i, w i ω * ε i ω) ^ 2 | m]
      ≤ᵐ[μ] fun ω ↦ σ2 * ∑ i, (w i ω) ^ 2 := by
  refine (Noise.noise_var_eq hm w ε hw_meas hw_bound
    hε2 hε_orth).trans_le ?_
  have hsq_all : ∀ᵐ ω ∂μ, ∀ i, μ[fun ω' ↦ (ε i ω') ^ 2 | m] ω ≤ σ2 := by
    rw [ae_all_iff]
    intro i
    simpa [Filter.EventuallyLE] using hε_sq i
  filter_upwards [hsq_all] with ω hω
  have hsum :
      ∑ i, (w i ω) ^ 2 * μ[fun ω' ↦ (ε i ω') ^ 2 | m] ω
        ≤ ∑ i, (w i ω) ^ 2 * σ2 := by
          refine Finset.sum_le_sum ?_
          intro i _
          gcongr
          exact hω i
  calc
    (∑ i, fun ω ↦ (w i ω) ^ 2 * μ[fun ω' ↦ (ε i ω') ^ 2 | m] ω) ω
      ≤ ∑ i, (w i ω) ^ 2 * σ2 := by
        simpa using hsum
    _ = ∑ i, σ2 * (w i ω) ^ 2 := by
      refine Finset.sum_congr rfl ?_
      intro i _
      ring
    _ = σ2 * ∑ i, (w i ω) ^ 2 := by
      simpa using
        (Finset.mul_sum (s := (Finset.univ : Finset ι)) (a := σ2)
          (f := fun i => (w i ω) ^ 2)).symm

end ENN
