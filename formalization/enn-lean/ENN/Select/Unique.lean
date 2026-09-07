import ENN.Fit.Box

/-!
# Unique maximizers
-/

namespace ENN

open Filter MeasureTheory

section UniqueArgmax

variable {Ω : Type*}

/-- A pointwise unique maximizer on the compact parameter box, chosen
noncomputably from the unique-existence hypothesis. -/
noncomputable def Select.unique
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hunique : ∀ ω, ∃! p : Fit.region B, IsMaxOn (Q ω) Set.univ p) :
    Ω → Fit.region B := fun ω ↦
  Classical.choose (ExistsUnique.exists (hunique ω))

theorem Select.unique_spec
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hunique : ∀ ω, ∃! p : Fit.region B, IsMaxOn (Q ω) Set.univ p) :
    ∀ ω, IsMaxOn (Q ω) Set.univ (Select.unique B Q hunique ω) := by
  intro ω
  exact Classical.choose_spec (ExistsUnique.exists (hunique ω))

theorem Select.unique_eq
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hunique : ∀ ω, ∃! p : Fit.region B, IsMaxOn (Q ω) Set.univ p)
    {ω : Ω} {p : Fit.region B} (hp : IsMaxOn (Q ω) Set.univ p) :
    p = Select.unique B Q hunique ω := by
  exact (hunique ω).unique hp (Select.unique_spec B Q hunique ω)

theorem Select.unique_value
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hcont : ∀ ω, Continuous (Q ω))
    (hunique : ∀ ω, ∃! p : Fit.region B, IsMaxOn (Q ω) Set.univ p) :
    ∀ ω,
      Q ω (Select.unique B Q hunique ω) =
        sSup (Q ω '' Set.univ) := by
  intro ω
  obtain ⟨pmax, -, hmax_eq, hmax_ge⟩ :=
    isCompact_univ.exists_sSup_image_eq_and_ge Set.univ_nonempty (hcont ω).continuousOn
  have hpmax : IsMaxOn (Q ω) Set.univ pmax := by
    rw [isMaxOn_iff]
    intro q _
    exact hmax_ge q trivial
  have hpuniq : pmax = Select.unique B Q hunique ω :=
    Select.unique_eq B Q hunique hpmax
  simpa [hpuniq] using hmax_eq.symm

end UniqueArgmax

end ENN
