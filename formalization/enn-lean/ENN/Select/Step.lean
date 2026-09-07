import ENN.Select.Cover

/-!
# One maximizing quadtree step
-/

namespace ENN

open Filter MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω]

/-- For a fixed parent quadtree cell, choose measurably the child digit whose
local maximum still equals the parent-cell maximum. -/
noncomputable def Select.digit
    (B : Fit.Box) (n : ℕ)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (i : Fin (4 ^ n)) :
    Ω → ℕ :=
  Select.child B (Fit.Cell.tree B n i)
    Q hQ hcont

theorem Select.digit_measurable
    (B : Fit.Box) (n : ℕ)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (i : Fin (4 ^ n)) :
    Measurable
      (Select.digit B n Q hQ hcont i) := by
  simpa [Select.digit] using
    Select.child_measurable B
      (Fit.Cell.tree B n i)
      Q hQ hcont

theorem Select.digit_lt
    (B : Fit.Box) (n : ℕ)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (i : Fin (4 ^ n)) :
    ∀ ω, Select.digit B n Q hQ hcont i ω < 4 :=
  fun ω ↦ Classical.choose
    (Select.child_spec
      B (Fit.Cell.tree B n i)
      Q hQ hcont ω)

theorem Select.digit_spec
    (B : Fit.Box) (n : ℕ)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (i : Fin (4 ^ n)) :
    ∀ ω,
      sSup
          (Q ω ''
            ((Fit.Cell.tree B n i).child
              ⟨Select.digit B n Q hQ hcont i ω,
                Select.digit_lt
                  B n Q hQ hcont i ω⟩).toSet) =
        sSup (Q ω '' (Fit.Cell.tree B n i).toSet) := by
  intro ω
  simpa [Select.digit] using
    (Classical.choose_spec
      (Select.child_spec
        B (Fit.Cell.tree B n i)
        Q hQ hcont ω))

theorem Select.step_lt
    (B : Fit.Box) (n : ℕ)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (i : Fin (4 ^ n)) :
    ∀ ω,
      Select.digit B n Q hQ hcont i ω + 4 * i.1 <
        4 ^ (n + 1) := by
  intro ω
  have hd :
      Select.digit B n Q hQ hcont i ω < 4 :=
    Select.digit_lt B n Q hQ hcont i ω
  have hi : i.1 < 4 ^ n := i.2
  rw [pow_succ, Nat.mul_comm]
  omega

/-- From a fixed parent quadtree cell, move to the chosen exact-maximizing
child cell one level deeper in the quadtree. -/
noncomputable def Select.step
    (B : Fit.Box) (n : ℕ)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (i : Fin (4 ^ n)) :
    Ω → Fin (4 ^ (n + 1)) := fun ω ↦
  ⟨Select.digit B n Q hQ hcont i ω + 4 * i.1,
    Select.step_lt B n Q hQ hcont i ω⟩

theorem Select.step_measurable
    (B : Fit.Box) (n : ℕ)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (i : Fin (4 ^ n)) :
    Measurable (Select.step B n Q hQ hcont i) := by
  let digitFin : Ω → Fin 4 := fun ω ↦
    ⟨Select.digit B n Q hQ hcont i ω,
      Select.digit_lt B n Q hQ hcont i ω⟩
  have hdigitFin : Measurable digitFin := by
    apply measurable_to_countable'
    intro d
    have hpre :
        digitFin ⁻¹' ({d} : Set (Fin 4)) =
          {ω | Select.digit B n Q hQ hcont i ω = d.1} := by
      ext ω
      constructor
      · intro hω
        exact congrArg Fin.val hω
      · intro hω
        apply Fin.ext
        exact hω
    rw [hpre]
    exact
      (Select.digit_measurable
        B n Q hQ hcont i) (measurableSet_singleton d.1)
  let stepMap : Fin 4 → Fin (4 ^ (n + 1)) := fun d ↦
    ⟨d.1 + 4 * i.1, by
      have hd : d.1 < 4 := d.2
      have hi : i.1 < 4 ^ n := i.2
      rw [pow_succ, Nat.mul_comm]
      omega⟩
  have hstepMap : Measurable stepMap := measurable_of_finite stepMap
  have hEq :
      Select.step B n Q hQ hcont i = stepMap ∘ digitFin := by
    funext ω
    apply Fin.ext
    rfl
  rw [hEq]
  exact hstepMap.comp hdigitFin

theorem Select.step_spec
    (B : Fit.Box) (n : ℕ)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (i : Fin (4 ^ n)) :
    ∀ ω,
      sSup
          (Q ω ''
            (Fit.Cell.tree B (n + 1)
              (Select.step B n Q hQ hcont i ω)).toSet) =
        sSup (Q ω '' (Fit.Cell.tree B n i).toSet) := by
  intro ω
  let d : Fin 4 := ⟨Select.digit B n Q hQ hcont i ω,
    Select.digit_lt B n Q hQ hcont i ω⟩
  have hbox :
      Fit.Cell.tree B (n + 1)
        (Select.step B n Q hQ hcont i ω) =
      (Fit.Cell.tree B n i).child d := by
    simpa [Select.step, d] using
      (Fit.Cell.tree_child B n i d)
  calc
    sSup
        (Q ω ''
          (Fit.Cell.tree B (n + 1)
            (Select.step B n Q hQ hcont i ω)).toSet) =
      sSup (Q ω '' ((Fit.Cell.tree B n i).child d).toSet) := by
        simp [hbox]
    _ =
      sSup (Q ω '' (Fit.Cell.tree B n i).toSet) := by
        simpa [d] using
          (Select.digit_spec B n Q hQ hcont i ω)

theorem Select.step_nested
    (B : Fit.Box) (n : ℕ)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (i : Fin (4 ^ n)) :
    ∀ ω,
      (Fit.Cell.tree B (n + 1)
        (Select.step B n Q hQ hcont i ω)).toSet ⊆
        (Fit.Cell.tree B n i).toSet := by
  intro ω
  let d : Fin 4 := ⟨Select.digit B n Q hQ hcont i ω,
    Select.digit_lt B n Q hQ hcont i ω⟩
  have hbox :
      Fit.Cell.tree B (n + 1)
        (Select.step B n Q hQ hcont i ω) =
      (Fit.Cell.tree B n i).child d := by
    simpa [Select.step, d] using
      (Fit.Cell.tree_child B n i d)
  intro p hp
  have hp' : p ∈ ((Fit.Cell.tree B n i).child d).toSet := by
    simpa [hbox] using hp
  exact (Fit.Cell.tree B n i).child_subset d hp'

end ENN
