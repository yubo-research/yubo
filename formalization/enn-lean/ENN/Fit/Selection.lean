import ENN.Fit.Stability
import ENN.Select.Limit
import ENN.Select.Approx

/-!
# Measurable maximizers and selected predictions
-/

namespace ENN

open Filter MeasureTheory

section SelectedPredictorBox

variable {ι Ω : Type*} [Fintype ι] [Nonempty ι]

/-- Compose the concrete box-restricted predictor with a parameter selector. -/
noncomputable def Fit.selected
    (B : Fit.Box) (s r y : ι → ℝ) (θ : Ω → Fit.region B) :
    Ω → ℝ := fun ω ↦ Fit.box_predict B s r y (θ ω)

/-- Pointwise predictor errors are controlled by the parameter-selector error
through the deterministic box Lipschitz constant. -/
theorem Fit.selected_bound
    (B : Fit.Box) (s r y : ι → ℝ) {S2 R2 Y : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2)
    (hY0 : 0 ≤ Y) (hY : ∀ i, |y i| ≤ Y)
    (θ η : Ω → Fit.region B) (ω : Ω) :
    |Fit.selected B s r y θ ω -
        Fit.selected B s r y η ω|
      ≤ (2 * (Y * (2 * (B.alpha_hi + S2 + B.gamma_hi * R2) *
          ((B.alpha_lo⁻¹) ^ 2 * (1 + R2))))) * dist (θ ω) (η ω) := by
  simpa [Fit.selected, Real.dist_eq] using
    (Fit.predict_lipschitz B s r y hS2 hR2 hY0 hY (θ ω) (η ω))

end SelectedPredictorBox

section MeasurableSelectionTransfer

variable {ι Ω κ : Type*} [Fintype ι] [Nonempty ι]
variable [MeasurableSpace Ω] {μ : Measure Ω}

/-- An explicit assumption package for a measurable maximizer selector on the
compact C.4 parameter box. This remains useful as an interface for downstream
results, even though the current file now also constructs such selectors
directly from jointly measurable continuous objective families. -/
structure Select.Argmax
    (Ω : Type*) [MeasurableSpace Ω]
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ) where
  toFun : Ω → Fit.region B
  measurable_fn : Measurable toFun
  maximizes : ∀ ω, IsMaxOn (Q ω) Set.univ (toFun ω)

instance {B : Fit.Box} {Q : Ω → Fit.region B → ℝ} :
    CoeFun (Select.Argmax Ω B Q) (fun _ => Ω → Fit.region B) where
  coe θ := θ.toFun

theorem Select.Argmax.measurable
    {B : Fit.Box} {Q : Ω → Fit.region B → ℝ}
    (θ : Select.Argmax Ω B Q) :
    Measurable θ :=
  θ.measurable_fn

/-- A jointly measurable objective that is continuous in the parameter admits
an exact measurable argmax selector on the compact parameter box. The selector
is constructed as the limit of the measurable center sequence of nested exact-
maximizing quadtree cells. -/
noncomputable def Select.argmax
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    Select.Argmax Ω B Q where
  toFun := Select.limit B Q hQ hcont
  measurable_fn := Select.limit_measurable B Q hQ hcont
  maximizes := Select.limit_max B Q hQ hcont

/-- Under pointwise uniqueness of the maximizer, the measurable argmax-selector
assumption package can be constructed directly from the objective family. -/
noncomputable def Select.unique_argmax
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (hunique : ∀ ω, ∃! p : Fit.region B, IsMaxOn (Q ω) Set.univ p) :
    Select.Argmax Ω B Q where
  toFun := Select.unique B Q hunique
  measurable_fn := Select.unique_measurable B Q hQ hcont hunique
  maximizes := Select.unique_spec B Q hunique

/-- A measurable selector yields a measurable selected predictor. -/
theorem Fit.selected_measurable
    (B : Fit.Box) (s r y : ι → ℝ) {S2 R2 Y : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2)
    (hY0 : 0 ≤ Y) (hY : ∀ i, |y i| ≤ Y)
    {θ : Ω → Fit.region B} (hθ : Measurable θ) :
    Measurable (Fit.selected B s r y θ) :=
  (Fit.predict_measurable B s r y hS2 hR2 hY0 hY).comp hθ

/-- Hence any almost-everywhere measurable selector yields an almost-everywhere
measurable selected predictor. -/
theorem Fit.selected_aemeasurable
    (B : Fit.Box) (s r y : ι → ℝ) {S2 R2 Y : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2)
    (hY0 : 0 ≤ Y) (hY : ∀ i, |y i| ≤ Y)
    {θ : Ω → Fit.region B} (hθ : AEMeasurable θ μ) :
    AEMeasurable (Fit.selected B s r y θ) μ :=
  Measurable.comp_aemeasurable
    (Fit.predict_measurable B s r y hS2 hR2 hY0 hY) hθ

/-- Specialized to an assumed measurable argmax selector, the concrete selected
predictor is measurable. -/
theorem Fit.selector_measurable
    {B : Fit.Box} {Q : Ω → Fit.region B → ℝ}
    (θ : Select.Argmax Ω B Q) (s r y : ι → ℝ) {S2 R2 Y : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2)
    (hY0 : 0 ≤ Y) (hY : ∀ i, |y i| ≤ Y) :
    Measurable (Fit.selected B s r y θ) :=
  Fit.selected_measurable B s r y hS2 hR2 hY0 hY θ.measurable

/-- Specializing the exact measurable selector constructed above, the concrete
selected predictor is measurable without any extra selector assumption. -/
theorem Fit.argmax_measurable
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (s r y : ι → ℝ) {S2 R2 Y : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2)
    (hY0 : 0 ≤ Y) (hY : ∀ i, |y i| ≤ Y) :
    Measurable
      (Fit.selected B s r y
        (Select.argmax B Q hQ hcont)) :=
  Fit.selector_measurable
    (Select.argmax B Q hQ hcont) s r y hS2 hR2 hY0 hY

/-- In the pointwise unique-maximizer case, the concrete selected predictor is
measurable without separately assuming a selector package. -/
theorem Fit.unique_measurable
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (hunique : ∀ ω, ∃! p : Fit.region B, IsMaxOn (Q ω) Set.univ p)
    (s r y : ι → ℝ) {S2 R2 Y : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2)
    (hY0 : 0 ≤ Y) (hY : ∀ i, |y i| ≤ Y) :
    Measurable
      (Fit.selected B s r y
        (Select.unique B Q hunique)) :=
  Fit.selected_measurable B s r y hS2 hR2 hY0 hY
    (Select.unique_measurable B Q hQ hcont hunique)

/-- If the selector error belongs to `L²`, then so does the induced predictor
error. This is the explicit `L²` transfer theorem available without a general
measurable-selection theorem. -/
theorem Fit.selected_l2
    (B : Fit.Box) (s r y : ι → ℝ) {S2 R2 Y : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2)
    (hY0 : 0 ≤ Y) (hY : ∀ i, |y i| ≤ Y)
    (θ η : Ω → Fit.region B)
    (hθ : AEMeasurable θ μ) (hη : AEMeasurable η μ)
    (hdist : MemLp (fun ω ↦ dist (θ ω) (η ω)) 2 μ) :
    MemLp
      (fun ω ↦ Fit.selected B s r y θ ω -
        Fit.selected B s r y η ω) 2 μ := by
  refine hdist.of_le_mul
    (c := 2 * (Y * (2 * (B.alpha_hi + S2 + B.gamma_hi * R2) *
      ((B.alpha_lo⁻¹) ^ 2 * (1 + R2))))) ?_ ?_
  · exact ((Fit.selected_aemeasurable B s r y hS2 hR2 hY0 hY hθ).sub
      (Fit.selected_aemeasurable B s r y hS2 hR2 hY0 hY hη)).aestronglyMeasurable
  · refine Filter.Eventually.of_forall ?_
    intro ω
    simpa [Real.norm_eq_abs, abs_of_nonneg (dist_nonneg : 0 ≤ dist (θ ω) (η ω))] using
      Fit.selected_bound B s r y hS2 hR2 hY0 hY θ η ω

/-- The preceding `L²` transfer theorem gives integrability of the squared
selected-predictor error. -/
theorem Fit.selected_integrable
    (B : Fit.Box) (s r y : ι → ℝ) {S2 R2 Y : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2)
    (hY0 : 0 ≤ Y) (hY : ∀ i, |y i| ≤ Y)
    (θ η : Ω → Fit.region B)
    (hθ : AEMeasurable θ μ) (hη : AEMeasurable η μ)
    (hdist : MemLp (fun ω ↦ dist (θ ω) (η ω)) 2 μ) :
    Integrable
      (fun ω ↦
        (Fit.selected B s r y θ ω -
          Fit.selected B s r y η ω) ^ 2) μ :=
  (Fit.selected_l2
    B s r y hS2 hR2 hY0 hY θ η hθ hη hdist).integrable_sq

/-- Convergence in measure of parameter selectors transfers to convergence in
measure of the selected predictors, by the deterministic Lipschitz bound. -/
theorem Fit.selected_limit
    (B : Fit.Box) (s r y : ι → ℝ) {S2 R2 Y : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2)
    (hY0 : 0 ≤ Y) (hY : ∀ i, |y i| ≤ Y)
    {l : Filter κ} (θ : κ → Ω → Fit.region B) (η : Ω → Fit.region B)
    (hparam : TendstoInMeasure μ θ l η) :
    TendstoInMeasure μ
      (fun n ω ↦ Fit.selected B s r y (θ n) ω) l
      (Fit.selected B s r y η) := by
  rw [tendstoInMeasure_iff_dist] at hparam ⊢
  intro ε hε
  let L : ℝ := 2 * (Y * (2 * (B.alpha_hi + S2 + B.gamma_hi * R2) *
    ((B.alpha_lo⁻¹) ^ 2 * (1 + R2))))
  have hR20 : 0 ≤ R2 := by
    let i0 : ι := Classical.choice ‹Nonempty ι›
    exact le_trans (sq_nonneg (r i0)) (hR2 i0)
  have hS20 : 0 ≤ S2 := by
    let i0 : ι := Classical.choice ‹Nonempty ι›
    exact le_trans (sq_nonneg (s i0)) (hS2 i0)
  have hγhi0 : 0 ≤ B.gamma_hi := le_trans B.gamma_nonneg B.gamma_bounds
  have hV0 : 0 ≤ B.alpha_hi + S2 + B.gamma_hi * R2 := by
    have hαhi0 : 0 ≤ B.alpha_hi := le_of_lt (lt_of_lt_of_le B.alpha_pos B.alpha_bounds)
    have hmul : 0 ≤ B.gamma_hi * R2 := mul_nonneg hγhi0 hR20
    linarith
  have hL0 : 0 ≤ L := by
    have hR1 : 0 ≤ 1 + R2 := by linarith
    have hinner : 0 ≤ (B.alpha_lo⁻¹) ^ 2 * (1 + R2) := mul_nonneg (sq_nonneg _) hR1
    have hC0 : 0 ≤ 2 * (B.alpha_hi + S2 + B.gamma_hi * R2) * ((B.alpha_lo⁻¹) ^ 2 * (1 + R2)) :=
      mul_nonneg (mul_nonneg (by positivity) hV0) hinner
    dsimp [L]
    exact mul_nonneg (by positivity) (mul_nonneg hY0 hC0)
  have hzero_lim : Filter.Tendsto (fun _ : κ ↦ (0 : ENNReal)) l (nhds (0 : ENNReal)) :=
    tendsto_const_nhds
  by_cases hL : L = 0
  · have hzero :
        ∀ n ω,
          dist (Fit.selected B s r y (θ n) ω)
            (Fit.selected B s r y η ω) = 0 := by
      intro n ω
      have hdist :=
        Fit.predict_lipschitz B s r y hS2 hR2 hY0 hY (θ n ω) (η ω)
      have hle' :
          dist (Fit.selected B s r y (θ n) ω)
            (Fit.selected B s r y η ω) ≤ L * dist (θ n ω) (η ω) := by
        simpa [L, Fit.selected] using hdist
      have hle :
          dist (Fit.selected B s r y (θ n) ω)
            (Fit.selected B s r y η ω) ≤ 0 := by
        rw [hL] at hle'
        simpa using hle'
      exact le_antisymm hle dist_nonneg
    have hset :
        ∀ n,
          {ω | ε ≤ dist (Fit.selected B s r y (θ n) ω)
            (Fit.selected B s r y η ω)} = ∅ := by
      intro n
      have hε0 : ¬ ε ≤ 0 := not_le_of_gt hε
      ext ω
      simp [hzero n ω, hε0]
    simpa [hset] using hzero_lim
  · have hLne : 0 ≠ L := by
      intro h0
      exact hL h0.symm
    have hLpos : 0 < L := lt_of_le_of_ne hL0 hLne
    have hparam' := hparam (ε / L) (div_pos hε hLpos)
    have hlower :
        (fun _ : κ => (0 : ENNReal)) ≤
          fun i =>
            μ {x | ε ≤ dist (Fit.selected B s r y (θ i) x)
              (Fit.selected B s r y η x)} := by
      intro i
      positivity
    have hupper :
        (fun i =>
          μ {x | ε ≤ dist (Fit.selected B s r y (θ i) x)
            (Fit.selected B s r y η x)}) ≤
          fun i => μ {x | ε / L ≤ dist (θ i x) (η x)} := by
      intro n
      exact measure_mono <| by
        intro ω hω
        have hdist :=
          Fit.predict_lipschitz B s r y hS2 hR2 hY0 hY (θ n ω) (η ω)
        have hle : ε ≤ L * dist (θ n ω) (η ω) := by
          exact le_trans hω (by simpa [L, Fit.selected] using hdist)
        have hle' : ε ≤ dist (θ n ω) (η ω) * L := by
          simpa [mul_comm] using hle
        exact (div_le_iff₀ hLpos).2 hle'
    exact tendsto_of_tendsto_of_tendsto_of_le_of_le hzero_lim hparam' hlower hupper

end MeasurableSelectionTransfer

end ENN
