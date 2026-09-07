import ENN.Fit.Box

/-!
# Argmax consistency
-/

namespace ENN

open Filter MeasureTheory

open ProbabilityTheory Topology

section ArgmaxConsistency

variable {P Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- Separation of a target maximizer at parameter distance `δ` by objective
gap `η`. -/
def Separated (distance : P → P → ℝ) (Q : P → ℝ) (star : P)
    (δ η : ℝ) : Prop :=
  ∀ p, δ ≤ distance p star → η ≤ Q star - Q p

/-- Deterministic argmax lemma used pointwise in Appendix C, Theorem 3. -/
theorem Fit.argmax_dist
    (distance : P → P → ℝ) (L Q : P → ℝ) (hat star : P)
    {δ η Δ : ℝ}
    (hhat : ∀ p, L p ≤ L hat)
    (hclose : ∀ p, |L p - Q p| ≤ Δ)
    (hsep : Separated distance Q star δ η)
    (hΔ : Δ < η / 3) :
    distance hat star < δ := by
  by_contra hnot
  have hfar : δ ≤ distance hat star := le_of_not_gt hnot
  have hsep_hat : η ≤ Q star - Q hat := hsep hat hfar
  have hclose_hat := abs_le.mp (hclose hat)
  have hclose_star := abs_le.mp (hclose star)
  have hmax := hhat star
  linarith

/-- Paper-level uniform convergence plus separation implies convergence in
probability of empirical maximizers to target maximizers.  `Δ n ω` is any
measurable envelope for the uniform objective error; using the supremum itself
is the paper's specialization. -/
theorem Fit.argmax_limit
    [IsFiniteMeasure μ]
    (distance : P → P → ℝ)
    (L Q : ℕ → Ω → P → ℝ) (hat star : ℕ → Ω → P) (Δ : ℕ → Ω → ℝ)
    (hhat : ∀ n ω p, L n ω p ≤ L n ω (hat n ω))
    (hclose : ∀ n ω p, |L n ω p - Q n ω p| ≤ Δ n ω)
    (huniform : ∀ ε, 0 < ε →
      Filter.Tendsto (fun n ↦ μ {ω | ε ≤ Δ n ω}) Filter.atTop (nhds 0))
    (hsep_prob : ∀ δ, 0 < δ → ∃ η, 0 < η ∧
      Filter.Tendsto
        (fun n ↦ μ {ω | ¬ Separated distance (Q n ω) (star n ω) δ η})
        Filter.atTop (nhds 0)) :
    ∀ δ, 0 < δ →
      Filter.Tendsto
        (fun n ↦ μ {ω | δ ≤ distance (hat n ω) (star n ω)})
        Filter.atTop (nhds 0) := by
  intro δ hδ
  obtain ⟨η, hη, hsep_lim⟩ := hsep_prob δ hδ
  have hη3 : 0 < η / 3 := by positivity
  have hΔ_prob := huniform (η / 3) hη3
  have hunion :
      Filter.Tendsto
        (fun n ↦
          μ {ω | ¬ Separated distance (Q n ω) (star n ω) δ η} +
          μ {ω | η / 3 ≤ Δ n ω})
        Filter.atTop (nhds 0) := by
    simpa using hsep_lim.add hΔ_prob
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hunion
    (fun _ ↦ bot_le) ?_
  intro n
  calc
    μ {ω | δ ≤ distance (hat n ω) (star n ω)}
      ≤ μ ({ω | ¬ Separated distance (Q n ω) (star n ω) δ η} ∪
          {ω | η / 3 ≤ Δ n ω}) := by
            apply measure_mono
            intro ω hω
            by_cases hsep : Separated distance (Q n ω) (star n ω) δ η
            · right
              by_contra hsmall
              have hlt : Δ n ω < η / 3 := lt_of_not_ge hsmall
              have := Fit.argmax_dist
                distance (L n ω) (Q n ω) (hat n ω) (star n ω)
                (hhat n ω) (hclose n ω) hsep hlt
              exact (not_lt_of_ge hω) this
            · exact Or.inl hsep
    _ ≤ μ {ω | ¬ Separated distance (Q n ω) (star n ω) δ η} +
          μ {ω | η / 3 ≤ Δ n ω} := measure_union_le _ _

/-- Appendix C, Theorem 3 specialized to the paper's variance-scale parameter
pair and its `ℓ¹` distance. -/
theorem Fit.param_argmax
    [IsFiniteMeasure μ]
    (L Q : ℕ → Ω → Fit.Params → ℝ)
    (hat star : ℕ → Ω → Fit.Params) (Δ : ℕ → Ω → ℝ)
    (hhat : ∀ n ω p, L n ω p ≤ L n ω (hat n ω))
    (hclose : ∀ n ω p, |L n ω p - Q n ω p| ≤ Δ n ω)
    (huniform : ∀ ε, 0 < ε →
      Filter.Tendsto (fun n ↦ μ {ω | ε ≤ Δ n ω}) Filter.atTop (nhds 0))
    (hsep_prob : ∀ δ, 0 < δ → ∃ η, 0 < η ∧
      Filter.Tendsto
        (fun n ↦ μ {ω |
          ¬ Separated Fit.distance (Q n ω) (star n ω) δ η})
        Filter.atTop (nhds 0)) :
    ∀ δ, 0 < δ →
      Filter.Tendsto
        (fun n ↦ μ {ω | δ ≤ Fit.distance (hat n ω) (star n ω)})
        Filter.atTop (nhds 0) :=
  Fit.argmax_limit
    Fit.distance L Q hat star Δ hhat hclose huniform hsep_prob

end ArgmaxConsistency

end ENN
