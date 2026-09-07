import ENN.Select.Value
import ENN.Fit.Cell

/-!
# Measurable choices from finite covers
-/

namespace ENN

open Filter MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω]

/-- For a finite compact cover of the parameter box, the lexicographically
first cell whose local maximum equals the global maximum can be chosen
measurably. This is the discrete-selection primitive needed for recursive exact
selector constructions based on box partitions. -/
theorem Select.exists_subcover
    (B : Fit.Box) (S : Set (Fit.region B))
    (hSc : IsCompact S) (hSn : S.Nonempty)
    {m : ℕ} (K : Fin m → Set (Fit.region B))
    (hKS : ∀ i, K i ⊆ S)
    (hKc : ∀ i, IsCompact (K i)) (hKn : ∀ i, (K i).Nonempty)
    (hcover : S ⊆ ⋃ i, K i)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    ∃ idx : Ω → ℕ, Measurable idx ∧
      ∀ ω, ∃ hidx : idx ω < m,
        sSup (Q ω '' K ⟨idx ω, hidx⟩) = sSup (Q ω '' S) := by
  let maxval : Ω → ℝ := fun ω ↦ sSup (Q ω '' S)
  have hmax : Measurable maxval :=
    Select.subset_measurable
      B S hSc hSn Q hQ hcont
  let cellval : ℕ → Ω → ℝ := fun n ω ↦
    if h : n < m then sSup (Q ω '' K ⟨n, h⟩) else 0
  have hj_max : ∀ n, Measurable (cellval n) := by
    intro n
    dsimp [cellval]
    by_cases h : n < m
    · simpa [h] using
        Select.subset_measurable
          B (K ⟨n, h⟩) (hKc ⟨n, h⟩) (hKn ⟨n, h⟩)
          Q hQ hcont
    · simp [h]
  let good : ℕ → Ω → Prop := fun n ω ↦ n < m ∧ cellval n ω = maxval ω
  have hgood_m : ∀ n, MeasurableSet {ω | good n ω} := by
    intro n
    dsimp [good]
    by_cases h : n < m
    · simpa [h] using measurableSet_eq_fun (hj_max n) hmax
    · simp [h]
  have hex : ∀ ω, ∃ n, good n ω := by
    intro ω
    obtain ⟨pmax, hpmaxS, hmax_eq, hmax_ge⟩ :=
      hSc.exists_sSup_image_eq_and_ge hSn (hcont ω).continuousOn
    have hp_cover : pmax ∈ ⋃ i, K i := hcover hpmaxS
    rcases Set.mem_iUnion.mp hp_cover with ⟨i, hpmaxK⟩
    obtain ⟨q, hqK, hq_eq, hq_ge⟩ :=
      (hKc i).exists_sSup_image_eq_and_ge (hKn i) (hcont ω).continuousOn
    have hq_upper : Q ω q ≤ Q ω pmax := hmax_ge q (hKS i hqK)
    have hq_lower : Q ω pmax ≤ Q ω q := hq_ge pmax hpmaxK
    have hcell_max : sSup (Q ω '' K i) = sSup (Q ω '' S) := by
      rw [hq_eq, hmax_eq]
      linarith
    refine ⟨i.1, ?_⟩
    constructor
    · exact i.2
    · simpa [good, cellval, maxval, i.2] using hcell_max
  let sel : ℕ → Ω → ℕ := fun n _ ↦ n
  refine ⟨fun ω ↦ Nat.find (hex ω), ?_, ?_⟩
  · have hfind_m :
        Measurable fun ω ↦ sel (Nat.find (hex ω)) ω :=
      Measurable.find
        (f := sel)
        (p := fun n ω ↦ good n ω)
        (fun _ ↦ measurable_const) hgood_m hex
    simpa [sel] using hfind_m
  · intro ω
    refine ⟨(Nat.find_spec (hex ω)).1, ?_⟩
    simpa [cellval, maxval, (Nat.find_spec (hex ω)).1] using
      (Nat.find_spec (hex ω)).2

/-- For a finite compact cover of the parameter box, the lexicographically
first cell whose local maximum equals the global maximum can be chosen
measurably. This is the discrete-selection primitive needed for recursive exact
selector constructions based on box partitions. -/
theorem Select.exists_cover
    (B : Fit.Box) {m : ℕ} (K : Fin m → Set (Fit.region B))
    (hKc : ∀ i, IsCompact (K i)) (hKn : ∀ i, (K i).Nonempty)
    (hcover : Set.univ ⊆ ⋃ i, K i)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    ∃ idx : Ω → ℕ, Measurable idx ∧
      ∀ ω, ∃ hidx : idx ω < m,
        sSup (Q ω '' K ⟨idx ω, hidx⟩) = sSup (Q ω '' Set.univ) := by
  simpa [Set.image_univ] using
    Select.exists_subcover
      B Set.univ isCompact_univ Set.univ_nonempty K
      (by
        intro i x hx
        trivial) hKc hKn hcover
      Q hQ hcont

/-- A noncomputable measurable choice of the first max-attaining cell index on
an arbitrary compact subset of the parameter box. -/
noncomputable def Select.subcover
    (B : Fit.Box) (S : Set (Fit.region B))
    (hSc : IsCompact S) (hSn : S.Nonempty)
    {m : ℕ} (K : Fin m → Set (Fit.region B))
    (hKS : ∀ i, K i ⊆ S)
    (hKc : ∀ i, IsCompact (K i)) (hKn : ∀ i, (K i).Nonempty)
    (hcover : S ⊆ ⋃ i, K i)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    Ω → ℕ :=
  Classical.choose
    (Select.exists_subcover
      B S hSc hSn K hKS hKc hKn hcover
      Q hQ hcont)

theorem Select.subcover_measurable
    (B : Fit.Box) (S : Set (Fit.region B))
    (hSc : IsCompact S) (hSn : S.Nonempty)
    {m : ℕ} (K : Fin m → Set (Fit.region B))
    (hKS : ∀ i, K i ⊆ S)
    (hKc : ∀ i, IsCompact (K i)) (hKn : ∀ i, (K i).Nonempty)
    (hcover : S ⊆ ⋃ i, K i)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    Measurable
      (Select.subcover B S hSc hSn
        K hKS hKc hKn hcover Q hQ hcont) :=
  (Classical.choose_spec
    (Select.exists_subcover
      B S hSc hSn K hKS hKc hKn hcover
      Q hQ hcont)).1

theorem Select.subcover_spec
    (B : Fit.Box) (S : Set (Fit.region B))
    (hSc : IsCompact S) (hSn : S.Nonempty)
    {m : ℕ} (K : Fin m → Set (Fit.region B))
    (hKS : ∀ i, K i ⊆ S)
    (hKc : ∀ i, IsCompact (K i)) (hKn : ∀ i, (K i).Nonempty)
    (hcover : S ⊆ ⋃ i, K i)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    ∀ ω, ∃ hidx :
        Select.subcover B S hSc hSn
          K hKS hKc hKn hcover Q hQ hcont ω < m,
      sSup
          (Q ω ''
            K ⟨Select.subcover B S hSc hSn
                K hKS hKc hKn hcover Q hQ hcont ω,
              hidx⟩) =
        sSup (Q ω '' S) :=
  (Classical.choose_spec
    (Select.exists_subcover
      B S hSc hSn K hKS hKc hKn hcover
      Q hQ hcont)).2

/-- For any fixed compact subbox, the lexicographically first child whose local
maximum equals the parent-box maximum can be chosen measurably. This is the
one-step measurable branching primitive used in recursive quadtree argmax
constructions. -/
theorem Select.exists_child
    (B : Fit.Box) (C : Fit.Cell B)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    ∃ idx : Ω → ℕ, Measurable idx ∧
      ∀ ω, ∃ hidx : idx ω < 4,
        sSup (Q ω '' (C.child ⟨idx ω, hidx⟩).toSet) =
          sSup (Q ω '' C.toSet) := by
  simpa using
    Select.exists_subcover
      B C.toSet C.isCompact_toSet C.toSet_nonempty
      (fun j => (C.child j).toSet)
      (fun j => C.child_subset j)
      (fun j => (C.child j).isCompact_toSet)
      (fun j => (C.child j).toSet_nonempty)
      C.cover
      Q hQ hcont

/-- A noncomputable measurable choice of the first maximizing child of a fixed
compact subbox. -/
noncomputable def Select.child
    (B : Fit.Box) (C : Fit.Cell B)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    Ω → ℕ :=
  Classical.choose
    (Select.exists_child
      B C Q hQ hcont)

theorem Select.child_measurable
    (B : Fit.Box) (C : Fit.Cell B)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    Measurable
      (Select.child B C Q hQ hcont) :=
  (Classical.choose_spec
    (Select.exists_child
      B C Q hQ hcont)).1

theorem Select.child_spec
    (B : Fit.Box) (C : Fit.Cell B)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    ∀ ω, ∃ hidx :
        Select.child B C Q hQ hcont ω < 4,
      sSup
          (Q ω ''
            (C.child
              ⟨Select.child B C Q hQ hcont ω,
                hidx⟩).toSet) =
        sSup (Q ω '' C.toSet) :=
  (Classical.choose_spec
    (Select.exists_child
      B C Q hQ hcont)).2

/-- At every quadtree depth, the lexicographically first cell whose local
maximum equals the global box maximum can be chosen measurably. This provides a
measurable sequence of exact maximizing cells at successively finer finite
partitions of the compact parameter box. -/
theorem Select.exists_level
    (B : Fit.Box) (n : ℕ)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    ∃ idx : Ω → ℕ, Measurable idx ∧
      ∀ ω, ∃ hidx : idx ω < 4 ^ n,
        sSup (Q ω ''
          (Fit.Cell.tree B n ⟨idx ω, hidx⟩).toSet) =
          sSup (Q ω '' Set.univ) := by
  simpa using
    Select.exists_cover
      B (fun j => (Fit.Cell.tree B n j).toSet)
      (fun j => Fit.Cell.tree_compact B n j)
      (fun j => Fit.Cell.tree_nonempty B n j)
      (Fit.Cell.tree_cover B n)
      Q hQ hcont

/-- A noncomputable measurable choice of the first globally maximizing quadtree
cell index at depth `n`. -/
noncomputable def Select.level
    (B : Fit.Box) (n : ℕ)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    Ω → ℕ :=
  Classical.choose
    (Select.exists_level
      B n Q hQ hcont)

theorem Select.level_measurable
    (B : Fit.Box) (n : ℕ)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    Measurable
      (Select.level B n Q hQ hcont) :=
  (Classical.choose_spec
    (Select.exists_level
      B n Q hQ hcont)).1

theorem Select.level_spec
    (B : Fit.Box) (n : ℕ)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    ∀ ω, ∃ hidx :
        Select.level B n Q hQ hcont ω < 4 ^ n,
      sSup
          (Q ω ''
            (Fit.Cell.tree B n
              ⟨Select.level B n Q hQ hcont ω,
                hidx⟩).toSet) =
        sSup (Q ω '' Set.univ) :=
  (Classical.choose_spec
    (Select.exists_level
      B n Q hQ hcont)).2

/-- A measurable finite-valued index can be used to select from a measurable
finite family of measurable functions. -/
theorem Select.apply_measurable
    {m : ℕ} (hm : 0 < m) {β : Type*} [MeasurableSpace β]
    (g : Ω → Fin m) (hg : Measurable g)
    (f : Fin m → Ω → β) (hf : ∀ i, Measurable (f i)) :
    Measurable (fun ω ↦ f (g ω) ω) := by
  classical
  let t : Fin m → Set Ω := fun i => {ω | g ω = i}
  have ht : ∀ i, MeasurableSet (t i) := by
    intro i
    exact hg (measurableSet_singleton i)
  letI : Nonempty (Fin m) := ⟨⟨0, hm⟩⟩
  obtain ⟨h, hh_meas, hh_eq⟩ :=
    exists_measurable_piecewise t ht f hf (fun i j hij ω hω => by
      exfalso
      exact hij (hω.1.symm.trans hω.2))
  have hh : h = fun ω ↦ f (g ω) ω := by
    funext ω
    exact hh_eq (g ω) (by simp [t])
  simpa [hh] using hh_meas

end ENN
