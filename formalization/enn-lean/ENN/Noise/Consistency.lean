import ENN.Noise.RandomDesign

/-!
# Mean-square consistency
-/

namespace ENN

open Filter MeasureTheory

open ProbabilityTheory Topology

section MeanSquare

variable {Ω : Type*} [m₀ : MeasurableSpace Ω] {μ : Measure Ω}

/-- Mean-square convergence of a sequence of real random variables to a
possibly random target. -/
def MeanSquare (X : ℕ → Ω → ℝ) (target : Ω → ℝ) (μ : Measure Ω) : Prop :=
  Filter.Tendsto
    (fun n ↦ ∫ ω, (X n ω - target ω) ^ 2 ∂μ)
    Filter.atTop (nhds 0)

omit [MeasurableSpace Ω] in
/-- Integrating an a.e. conditional-MSE bound yields its unconditional-MSE
bound. -/
theorem Noise.integrate_bound
    {m m₀ : MeasurableSpace Ω} {ν : @Measure Ω m₀} [IsFiniteMeasure ν]
    (hm : m ≤ m₀)
    {error rhs : Ω → ℝ}
    (hrhs_int : Integrable rhs ν)
    (hcond : ν[fun ω ↦ (error ω) ^ 2 | m] ≤ᵐ[ν] rhs) :
    (∫ ω, (error ω) ^ 2 ∂ν) ≤ ∫ ω, rhs ω ∂ν := by
  calc
    (∫ ω, (error ω) ^ 2 ∂ν) = ∫ ω, ν[fun ω ↦ (error ω) ^ 2 | m] ω ∂ν := by
      exact (integral_condExp hm).symm
    _ ≤ ∫ ω, rhs ω ∂ν :=
      integral_mono_ae integrable_condExp hrhs_int hcond

/-- The scalar squeeze argument behind Appendix C, Theorem 2(i): if the MSE is
bounded by bias and variance budgets that both vanish, then the MSE vanishes. -/
theorem Noise.mse_zero
    {mse R W : ℕ → ℝ} {L σ2 : ℝ}
    (hmse0 : ∀ n, 0 ≤ mse n)
    (hmse_le : ∀ n, mse n ≤ L ^ 2 * R n + σ2 * W n)
    (hradius : Filter.Tendsto R Filter.atTop (nhds 0))
    (hweight : Filter.Tendsto W Filter.atTop (nhds 0)) :
    Filter.Tendsto mse Filter.atTop (nhds 0) := by
  have hbias :
      Filter.Tendsto (fun n ↦ L ^ 2 * R n) Filter.atTop (nhds 0) := by
    simpa using tendsto_const_nhds.mul hradius
  have hvariance :
      Filter.Tendsto (fun n ↦ σ2 * W n) Filter.atTop (nhds 0) := by
    simpa using tendsto_const_nhds.mul hweight
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds
    (by simpa using hbias.add hvariance) hmse0 hmse_le

/-- Mean-square convergence implies convergence in probability.  This is the
Chebyshev step stated explicitly in Appendix C, Theorem 2(i). -/
theorem Noise.l2_probability
    [IsFiniteMeasure μ] {X : ℕ → Ω → ℝ} {target : Ω → ℝ}
    (hsq_int : ∀ n, Integrable (fun ω ↦ (X n ω - target ω) ^ 2) μ)
    (hmse : MeanSquare X target μ) :
    TendstoInMeasure μ X Filter.atTop target := by
  rw [tendstoInMeasure_iff_measureReal_dist]
  intro δ hδ
  have hδsq : 0 < δ ^ 2 := sq_pos_of_pos hδ
  have hupper :
      Filter.Tendsto
        (fun n ↦ (∫ ω, (X n ω - target ω) ^ 2 ∂μ) / δ ^ 2)
        Filter.atTop (nhds 0) := by
    simpa using hmse.div_const (δ ^ 2)
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hupper
    (fun _ ↦ by positivity) ?_
  intro n
  have hmarkov := mul_meas_ge_le_integral_of_nonneg
    (μ := μ)
    (f := fun ω ↦ (X n ω - target ω) ^ 2)
    (Filter.Eventually.of_forall fun _ ↦ sq_nonneg _)
    (hsq_int n) (δ ^ 2)
  have hevent :
      {ω | δ ≤ dist (X n ω) (target ω)} =
        {ω | δ ^ 2 ≤ (X n ω - target ω) ^ 2} := by
    ext ω
    simp only [Set.mem_setOf_eq, Real.dist_eq]
    constructor
    · intro h
      calc
        δ ^ 2 ≤ |X n ω - target ω| ^ 2 :=
          (sq_le_sq₀ hδ.le (abs_nonneg _)).2 h
        _ = (X n ω - target ω) ^ 2 := sq_abs _
    · intro h
      apply (sq_le_sq₀ hδ.le (abs_nonneg _)).1
      rwa [sq_abs]
  change μ.real {ω | δ ≤ dist (X n ω) (target ω)} ≤
    (∫ ω, (X n ω - target ω) ^ 2 ∂μ) / δ ^ 2
  rw [hevent]
  exact (le_div_iff₀ hδsq).2 (by simpa [mul_comm] using hmarkov)

/-- Appendix C, Theorem 2(i), including both mean-square consistency and its
convergence-in-probability consequence. -/
theorem Noise.consistency
    [IsFiniteMeasure μ]
    (X : ℕ → Ω → ℝ) (target : Ω → ℝ)
    (R W : ℕ → ℝ) {L σ2 : ℝ}
    (hsq_int : ∀ n, Integrable (fun ω ↦ (X n ω - target ω) ^ 2) μ)
    (hmse_le : ∀ n,
      (∫ ω, (X n ω - target ω) ^ 2 ∂μ) ≤
        L ^ 2 * R n + σ2 * W n)
    (hradius : Filter.Tendsto R Filter.atTop (nhds 0))
    (hweight : Filter.Tendsto W Filter.atTop (nhds 0)) :
    MeanSquare X target μ ∧
      TendstoInMeasure μ X Filter.atTop target := by
  have hmse : MeanSquare X target μ := by
    exact Noise.mse_zero
      (fun n ↦ integral_nonneg_of_ae
        (Filter.Eventually.of_forall fun _ ↦ sq_nonneg _))
      hmse_le hradius hweight
  exact ⟨hmse, Noise.l2_probability hsq_int hmse⟩

/-- Appendix C, Theorem 2(i), starting directly from the conditional MSE
inequality.  This theorem formalizes the paper's "taking expectations" step
before applying the bias--variance squeeze and Chebyshev's inequality. -/
theorem Noise.consistency_cond
    [IsFiniteMeasure μ]
    {m : MeasurableSpace Ω} (hm : m ≤ m₀)
    (X : ℕ → Ω → ℝ) (target : Ω → ℝ) (rhs : ℕ → Ω → ℝ)
    (R W : ℕ → ℝ) {L σ2 : ℝ}
    (hsq_int : ∀ n, Integrable (fun ω ↦ (X n ω - target ω) ^ 2) μ)
    (hrhs_int : ∀ n, Integrable (rhs n) μ)
    (hcond : ∀ n,
      @condExp Ω ℝ m m₀ _ _ μ (fun ω ↦ (X n ω - target ω) ^ 2)
        ≤ᵐ[μ] rhs n)
    (hrhs_le : ∀ n, ∫ ω, rhs n ω ∂μ ≤
      L ^ 2 * R n + σ2 * W n)
    (hradius : Filter.Tendsto R Filter.atTop (nhds 0))
    (hweight : Filter.Tendsto W Filter.atTop (nhds 0)) :
    @MeanSquare Ω m₀ X target μ ∧
      @TendstoInMeasure Ω ℕ ℝ _ m₀ μ X Filter.atTop target := by
  apply @Noise.consistency Ω m₀ μ _ X target
    R W L σ2 hsq_int _ hradius hweight
  intro n
  exact (Noise.integrate_bound hm (hrhs_int n) (hcond n)).trans
    (hrhs_le n)

end MeanSquare

end ENN
