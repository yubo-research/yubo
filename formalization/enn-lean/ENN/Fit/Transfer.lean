import ENN.Noise.Consistency
import ENN.Fit.Argmax

/-!
# Fitted prediction transfer
-/

namespace ENN

open Filter MeasureTheory

open ProbabilityTheory Topology

section PredictorTransfer

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- Convergence in probability for two triangular arrays of real random
variables.  The fixed-target case is obtained by taking the second array to be
constant in `n`. -/
def ProbLimit
    (X target : ℕ → Ω → ℝ) (μ : Measure Ω) : Prop :=
  ∀ δ, 0 < δ →
    Filter.Tendsto
      (fun n ↦ μ {ω | δ ≤ |X n ω - target n ω|})
      Filter.atTop (nhds 0)

/-- Nonnegative stochastic errors converging to zero in probability. -/
def ProbZero
    (error : ℕ → Ω → ℝ) (μ : Measure Ω) : Prop :=
  ∀ δ, 0 < δ →
    Filter.Tendsto (fun n ↦ μ {ω | δ ≤ error n ω}) Filter.atTop (nhds 0)

/-- The event-tail definition of `Oₚ(1)` used in Appendix C, Theorem 3. -/
def ProbBounded
    (scale : ℕ → Ω → ℝ) (μ : Measure Ω) : Prop :=
  ∀ ε : ENNReal, 0 < ε →
    ∃ M : ℝ, 0 < M ∧ ∀ᶠ n in Filter.atTop, μ {ω | M < scale n ω} ≤ ε

/-- The standard `Oₚ(1) · oₚ(1) = oₚ(1)` lemma, written for the nonnegative
scale and error quantities used by the fitted-parameter theorem. -/
theorem Fit.Op_mul_oP
    [IsFiniteMeasure μ]
    {scale error : ℕ → Ω → ℝ}
    (he0 : ∀ n ω, 0 ≤ error n ω)
    (hscale : ProbBounded scale μ)
    (herror : ProbZero error μ) :
    ProbZero (fun n ω ↦ scale n ω * error n ω) μ := by
  intro δ hδ
  rw [ENNReal.tendsto_nhds_zero]
  intro ε hε
  have hε2 : 0 < ε / 2 := ENNReal.div_pos hε.ne' (by simp)
  obtain ⟨M, hM, hs_event⟩ := hscale (ε / 2) hε2
  have hδM : 0 < δ / M := div_pos hδ hM
  have he_lim := herror (δ / M) hδM
  rw [ENNReal.tendsto_nhds_zero] at he_lim
  have he_event := he_lim (ε / 2) hε2
  filter_upwards [hs_event, he_event] with n hsn hen
  calc
    μ {ω | δ ≤ scale n ω * error n ω}
      ≤ μ ({ω | M < scale n ω} ∪ {ω | δ / M ≤ error n ω}) := by
          apply measure_mono
          intro ω hω
          by_cases hs : M < scale n ω
          · exact Or.inl hs
          · right
            have hsM : scale n ω ≤ M := le_of_not_gt hs
            have hprod : scale n ω * error n ω ≤ M * error n ω :=
              mul_le_mul_of_nonneg_right hsM (he0 n ω)
            apply (div_le_iff₀ hM).2
            exact hω.trans (by simpa [mul_comm] using hprod)
    _ ≤ μ {ω | M < scale n ω} + μ {ω | δ / M ≤ error n ω} := measure_union_le _ _
    _ ≤ ε / 2 + ε / 2 := add_le_add hsn hen
    _ = ε := ENNReal.add_halves ε

/-- A pointwise nonnegative error envelope that vanishes in probability also
forces the enclosed difference to vanish in probability. -/
theorem Fit.error_transfer
    {X target : ℕ → Ω → ℝ} {error : ℕ → Ω → ℝ}
    (hbound : ∀ n ω, |X n ω - target n ω| ≤ error n ω)
    (herror : ProbZero error μ) :
    ProbLimit X target μ := by
  intro δ hδ
  have ht := herror δ hδ
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds ht
    (fun _ ↦ bot_le) ?_
  intro n
  apply measure_mono
  intro ω hω
  exact hω.trans (hbound n ω)

/-- Convergence in probability is closed under the triangle inequality. -/
theorem ProbLimit.trans
    {X middle target : ℕ → Ω → ℝ}
    (hXm : ProbLimit X middle μ)
    (hmt : ProbLimit middle target μ) :
    ProbLimit X target μ := by
  intro δ hδ
  have hδ2 : 0 < δ / 2 := half_pos hδ
  have hleft := hXm (δ / 2) hδ2
  have hright := hmt (δ / 2) hδ2
  have hsum :
      Filter.Tendsto
        (fun n ↦
          μ {ω | δ / 2 ≤ |X n ω - middle n ω|} +
          μ {ω | δ / 2 ≤ |middle n ω - target n ω|})
        Filter.atTop (nhds 0) := by
    simpa using hleft.add hright
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hsum
    (fun _ ↦ bot_le) ?_
  intro n
  calc
    μ {ω | δ ≤ |X n ω - target n ω|}
      ≤ μ ({ω | δ / 2 ≤ |X n ω - middle n ω|} ∪
          {ω | δ / 2 ≤ |middle n ω - target n ω|}) := by
            apply measure_mono
            intro ω hω
            change δ ≤ |X n ω - target n ω| at hω
            by_contra hnot
            simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_le] at hnot
            have htriangle :
                |X n ω - target n ω| ≤
                  |X n ω - middle n ω| + |middle n ω - target n ω| := by
              calc
                |X n ω - target n ω| =
                    |(X n ω - middle n ω) + (middle n ω - target n ω)| := by ring_nf
                _ ≤ |X n ω - middle n ω| + |middle n ω - target n ω| :=
                  abs_add_le _ _
            linarith
    _ ≤ μ {ω | δ / 2 ≤ |X n ω - middle n ω|} +
          μ {ω | δ / 2 ≤ |middle n ω - target n ω|} := measure_union_le _ _

/-- Appendix C, Theorem 3's probability-transfer conclusion. -/
theorem Fit.fitted_prob
    [IsFiniteMeasure μ]
    (fitted oracle target : ℕ → Ω → ℝ)
    (scale err : ℕ → Ω → ℝ)
    (he0 : ∀ n ω, 0 ≤ err n ω)
    (hscale : ProbBounded scale μ)
    (hparam : ProbZero err μ)
    (hstab : ∀ n ω,
      |fitted n ω - oracle n ω| ≤ scale n ω * err n ω)
    (horacle : ProbLimit oracle target μ) :
    ProbLimit fitted target μ := by
  have hproduct := Fit.Op_mul_oP
    he0 hscale hparam
  exact (Fit.error_transfer hstab hproduct).trans horacle

/-- Appendix C, Theorem 3's `L²` transfer conclusion.  The hypothesis on the
weighted parameter error is exactly the paper's expectation condition. -/
theorem Fit.fitted_l2
    (fitted oracle : ℕ → Ω → ℝ) (target : Ω → ℝ)
    (scale err : ℕ → Ω → ℝ)
    (hfit_int : ∀ n, Integrable (fun ω ↦ (fitted n ω - target ω) ^ 2) μ)
    (href_int : ∀ n, Integrable (fun ω ↦ (oracle n ω - target ω) ^ 2) μ)
    (herr_int : ∀ n,
      Integrable (fun ω ↦ (scale n ω * err n ω) ^ 2) μ)
    (hstab : ∀ n ω,
      |fitted n ω - oracle n ω| ≤ |scale n ω * err n ω|)
    (horacle : MeanSquare oracle target μ)
    (hweighted : Filter.Tendsto
      (fun n ↦ ∫ ω, (scale n ω * err n ω) ^ 2 ∂μ)
      Filter.atTop (nhds 0)) :
    MeanSquare fitted target μ := by
  have hupper : Filter.Tendsto
      (fun n ↦
        2 * (∫ ω, (scale n ω * err n ω) ^ 2 ∂μ) +
        2 * (∫ ω, (oracle n ω - target ω) ^ 2 ∂μ))
      Filter.atTop (nhds 0) := by
    have hw : Filter.Tendsto
        (fun n ↦ 2 * (∫ ω, (scale n ω * err n ω) ^ 2 ∂μ))
        Filter.atTop (nhds 0) := by
      simpa using (tendsto_const_nhds.mul hweighted :
        Filter.Tendsto
          (fun n ↦ 2 * (∫ ω, (scale n ω * err n ω) ^ 2 ∂μ))
          Filter.atTop (nhds (2 * 0)))
    have ho : Filter.Tendsto
        (fun n ↦ 2 * (∫ ω, (oracle n ω - target ω) ^ 2 ∂μ))
        Filter.atTop (nhds 0) := by
      simpa using (tendsto_const_nhds.mul horacle :
        Filter.Tendsto
          (fun n ↦ 2 * (∫ ω, (oracle n ω - target ω) ^ 2 ∂μ))
          Filter.atTop (nhds (2 * 0)))
    simpa using hw.add ho
  unfold MeanSquare
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hupper
    (fun n ↦ integral_nonneg_of_ae (Filter.Eventually.of_forall fun _ ↦ sq_nonneg _)) ?_
  intro n
  have hrhs_int : Integrable (fun ω ↦
      2 * (scale n ω * err n ω) ^ 2 +
      2 * (oracle n ω - target ω) ^ 2) μ :=
    (herr_int n).const_mul 2 |>.add ((href_int n).const_mul 2)
  calc
    ∫ ω, (fitted n ω - target ω) ^ 2 ∂μ
      ≤ ∫ ω,
        (2 * (scale n ω * err n ω) ^ 2 +
          2 * (oracle n ω - target ω) ^ 2) ∂μ := by
          apply integral_mono (hfit_int n) hrhs_int
          intro ω
          have hstab_sq :
              (fitted n ω - oracle n ω) ^ 2 ≤
                (scale n ω * err n ω) ^ 2 := by
            exact sq_le_sq.mpr (hstab n ω)
          have hdecomp :
              fitted n ω - target ω =
                (fitted n ω - oracle n ω) + (oracle n ω - target ω) := by ring
          change (fitted n ω - target ω) ^ 2 ≤
            2 * (scale n ω * err n ω) ^ 2 +
              2 * (oracle n ω - target ω) ^ 2
          rw [hdecomp]
          calc
            ((fitted n ω - oracle n ω) + (oracle n ω - target ω)) ^ 2
                ≤ 2 * (fitted n ω - oracle n ω) ^ 2 +
                  2 * (oracle n ω - target ω) ^ 2 := by
                    nlinarith [sq_nonneg
                      ((fitted n ω - oracle n ω) - (oracle n ω - target ω))]
            _ ≤ 2 * (scale n ω * err n ω) ^ 2 +
                  2 * (oracle n ω - target ω) ^ 2 := by linarith
    _ = 2 * (∫ ω, (scale n ω * err n ω) ^ 2 ∂μ) +
          2 * (∫ ω, (oracle n ω - target ω) ^ 2 ∂μ) := by
          rw [integral_add ((herr_int n).const_mul 2)
            ((href_int n).const_mul 2), integral_const_mul, integral_const_mul]

end PredictorTransfer

end ENN
