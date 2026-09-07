import ENN.Fit.Box

/-!
# Measurable maximum values
-/

namespace ENN

open Filter MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω]

/-- For a jointly measurable objective that is continuous in the parameter, the
maximum value attained on any nonempty compact subset of the parameter box is
measurable in the outer parameter. This is the local measurable-value theorem
used to analyze exact selectors through compact subcells of the box. -/
theorem Select.subset_measurable
    (B : Fit.Box) (K : Set (Fit.region B))
    (_hK : IsCompact K) (hKn : K.Nonempty)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    Measurable fun ω ↦ sSup (Q ω '' K) := by
  letI : Nonempty K := hKn.to_subtype
  letI : CompactSpace K := isCompact_iff_compactSpace.mp _hK
  let seq : ℕ → K := TopologicalSpace.denseSeq K
  have hseq_dense : DenseRange seq := by
    simp [seq]
  have hseq_m : ∀ n, Measurable fun ω ↦ Q ω (seq n) := by
    intro n
    have hpair : Measurable fun ω : Ω ↦ (ω, ((seq n : K) : Fit.region B)) :=
      measurable_id.prodMk measurable_const
    change Measurable
      (Function.uncurry Q ∘ fun ω : Ω ↦ (ω, ((seq n : K) : Fit.region B)))
    exact hQ.comp hpair
  have hsup_meas : Measurable fun ω ↦ ⨆ n, Q ω (seq n) :=
    Measurable.iSup hseq_m
  convert hsup_meas using 1
  ext ω
  have hdense : Dense (Set.range seq) := hseq_dense
  have hciSup :
      (⨆ p : Set.range seq, Q ω ((p : K) : Fit.region B)) =
        ⨆ p : K, Q ω (p : Fit.region B) := by
    simpa using
      hdense.ciSup'
        (f := fun p : K ↦ Q ω (p : Fit.region B))
        (hf := (hcont ω).comp continuous_subtype_val)
  have hrange :
      Set.range (fun p : Set.range seq ↦ Q ω ((p : K) : Fit.region B)) =
        Set.range (fun n : ℕ ↦ Q ω (((seq n : K) : Fit.region B))) := by
    ext z
    constructor
    · rintro ⟨p, hp⟩
      rcases p with ⟨p, ⟨n, hn⟩⟩
      refine ⟨n, ?_⟩
      simpa [hn] using hp
    · rintro ⟨n, rfl⟩
      exact ⟨⟨seq n, ⟨n, rfl⟩⟩, rfl⟩
  have himage :
      Set.range (fun p : K ↦ Q ω (p : Fit.region B)) =
        Q ω '' K := by
    ext z
    constructor
    · rintro ⟨p, rfl⟩
      exact ⟨p, p.2, rfl⟩
    · rintro ⟨p, hpK, rfl⟩
      exact ⟨⟨p, hpK⟩, rfl⟩
  calc
    sSup (Q ω '' K)
      = sSup (Set.range fun p : K ↦ Q ω (p : Fit.region B)) := by
          rw [himage]
    _ = ⨆ p : K, Q ω (p : Fit.region B) := by
          rw [sSup_range]
    _ = ⨆ p : Set.range seq, Q ω ((p : K) : Fit.region B) := by
          symm
          exact hciSup
    _ = sSup (Set.range fun p : Set.range seq ↦ Q ω ((p : K) : Fit.region B)) := by
          rw [sSup_range]
    _ = sSup (Set.range fun n : ℕ ↦ Q ω (((seq n : K) : Fit.region B))) := by
          rw [hrange]
    _ = ⨆ n, Q ω (((seq n : K) : Fit.region B)) := by
          rw [sSup_range]

/-- For a jointly measurable objective that is continuous in the parameter, the
maximum value attained on the compact parameter box is measurable in the outer
parameter. This isolates the value-function part of the C.4 measurable-
selection problem without assuming a selector. -/
theorem Select.max_measurable
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    Measurable fun ω ↦ sSup (Q ω '' Set.univ) := by
  simpa [Set.image_univ] using
    Select.subset_measurable
      B Set.univ isCompact_univ Set.univ_nonempty Q hQ hcont

end ENN
