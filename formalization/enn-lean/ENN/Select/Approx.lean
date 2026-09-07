import ENN.Select.Cover
import ENN.Select.Unique

/-!
# Approximate selectors and convergence to a unique maximizer
-/

namespace ENN

open Filter MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω]

/-- A noncomputable measurable choice of the first max-attaining cell index in a
finite compact cover of the parameter box. -/
noncomputable def Select.cover
    (B : Fit.Box) {m : ℕ} (K : Fin m → Set (Fit.region B))
    (hKc : ∀ i, IsCompact (K i)) (hKn : ∀ i, (K i).Nonempty)
    (hcover : Set.univ ⊆ ⋃ i, K i)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    Ω → ℕ :=
  Classical.choose
    (Select.exists_cover
      B K hKc hKn hcover Q hQ hcont)

theorem Select.cover_measurable
    (B : Fit.Box) {m : ℕ} (K : Fin m → Set (Fit.region B))
    (hKc : ∀ i, IsCompact (K i)) (hKn : ∀ i, (K i).Nonempty)
    (hcover : Set.univ ⊆ ⋃ i, K i)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    Measurable
      (Select.cover B K hKc hKn hcover
        Q hQ hcont) :=
  (Classical.choose_spec
    (Select.exists_cover
      B K hKc hKn hcover Q hQ hcont)).1

theorem Select.cover_spec
    (B : Fit.Box) {m : ℕ} (K : Fin m → Set (Fit.region B))
    (hKc : ∀ i, IsCompact (K i)) (hKn : ∀ i, (K i).Nonempty)
    (hcover : Set.univ ⊆ ⋃ i, K i)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    ∀ ω, ∃ hidx :
        Select.cover B K hKc hKn hcover
          Q hQ hcont ω < m,
      sSup
          (Q ω ''
            K ⟨Select.cover B K hKc hKn hcover
                Q hQ hcont ω, hidx⟩) =
        sSup (Q ω '' Set.univ) :=
  (Classical.choose_spec
    (Select.exists_cover
      B K hKc hKn hcover Q hQ hcont)).2

/-- A jointly measurable objective that is continuous in the parameter admits
measurable `ε`-argmax selectors on the compact parameter box. The selector is
constructed from a fixed dense sequence by taking the first point whose value is
within `ε` of the maximum. -/
theorem Select.exists_approx
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) {ε : ℝ} (hε : 0 < ε) :
    ∃ θ : Ω → Fit.region B, Measurable θ ∧
      ∀ ω, sSup (Q ω '' Set.univ) < Q ω (θ ω) + ε := by
  let seq : ℕ → Fit.region B := TopologicalSpace.denseSeq (Fit.region B)
  have hseq_dense : DenseRange seq := by
    simp [seq]
  have hseq_m : ∀ n, Measurable fun ω ↦ Q ω (seq n) := by
    intro n
    have hpair : Measurable fun ω : Ω ↦ (ω, seq n) :=
      measurable_id.prodMk measurable_const
    change Measurable (Function.uncurry Q ∘ fun ω : Ω ↦ (ω, seq n))
    exact hQ.comp hpair
  let maxValue : Ω → ℝ := fun ω ↦ sSup (Q ω '' Set.univ)
  have hmax_val : Measurable maxValue :=
    Select.max_measurable B Q hQ hcont
  let good : ℕ → Ω → Prop := fun n ω ↦ maxValue ω < Q ω (seq n) + ε
  have hgood_m : ∀ n, MeasurableSet {ω | good n ω} := by
    intro n
    exact measurableSet_lt hmax_val ((hseq_m n).add measurable_const)
  have hex : ∀ ω, ∃ n, good n ω := by
    intro ω
    obtain ⟨pmax, -, hmax_eq, _⟩ :=
      isCompact_univ.exists_sSup_image_eq_and_ge Set.univ_nonempty (hcont ω).continuousOn
    have hmax_eq : maxValue ω = Q ω pmax := by
      simpa [maxValue] using hmax_eq
    let U : Set (Fit.region B) := {p | Q ω pmax - ε < Q ω p}
    have hU_open : IsOpen U := by
      dsimp [U]
      exact isOpen_lt continuous_const (hcont ω)
    have hp_mem : pmax ∈ U := by
      dsimp [U]
      linarith
    obtain ⟨n, hn⟩ := hseq_dense.exists_mem_open hU_open ⟨pmax, hp_mem⟩
    refine ⟨n, ?_⟩
    have hlt : Q ω pmax < Q ω (seq n) + ε := by
      dsimp [U] at hn
      linarith
    show maxValue ω < Q ω (seq n) + ε
    rwa [hmax_eq]
  let sel : ℕ → Ω → Fit.region B := fun n _ ↦ seq n
  refine ⟨fun ω ↦ seq (Nat.find (hex ω)), ?_, ?_⟩
  · have hfind_m :
        Measurable fun ω ↦ sel (Nat.find (hex ω)) ω :=
      Measurable.find
        (f := sel)
        (p := fun n ω ↦ good n ω)
        (fun _ ↦ measurable_const) hgood_m hex
    simpa [sel] using hfind_m
  · intro ω
    exact Nat.find_spec (hex ω)

/-- A noncomputable measurable `ε`-argmax selector on the compact parameter
box, obtained from the preceding existence theorem. -/
noncomputable def Select.approx
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) {ε : ℝ} (hε : 0 < ε) :
    Ω → Fit.region B :=
  Classical.choose
    (Select.exists_approx B Q hQ hcont hε)

theorem Select.approx_measurable
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) {ε : ℝ} (hε : 0 < ε) :
    Measurable (Select.approx B Q hQ hcont hε) :=
  (Classical.choose_spec
    (Select.exists_approx B Q hQ hcont hε)).1

theorem Select.approx_spec
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) {ε : ℝ} (hε : 0 < ε) :
    ∀ ω,
      sSup (Q ω '' Set.univ) <
        Q ω (Select.approx B Q hQ hcont hε ω) + ε :=
  (Classical.choose_spec
    (Select.exists_approx B Q hQ hcont hε)).2

/-- If the approximation error tends to zero, then the corresponding measurable
`ε`-argmax selector values converge pointwise to the measurable maximum-value
map. This is the main value-convergence ingredient for upgrading approximate
measurable selections toward exact ones when additional identifiability
information is available. -/
theorem Select.value_limit
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) {ε : ℕ → ℝ}
    (hε_pos : ∀ n, 0 < ε n) (hε_lim : Filter.Tendsto ε atTop (nhds 0)) :
    ∀ ω,
      Filter.Tendsto
        (fun n : ℕ ↦ Q ω
          (Select.approx B Q hQ hcont (hε_pos n) ω))
        atTop (nhds (sSup (Q ω '' Set.univ))) := by
  intro ω
  obtain ⟨pmax, -, hmax_eq, hmax_ge⟩ :=
    isCompact_univ.exists_sSup_image_eq_and_ge Set.univ_nonempty (hcont ω).continuousOn
  rw [tendsto_iff_dist_tendsto_zero]
  refine squeeze_zero (fun n => dist_nonneg) ?_ hε_lim
  intro n
  have hle :
      Q ω (Select.approx B Q hQ hcont (hε_pos n) ω) ≤
        sSup (Q ω '' Set.univ) := by
    rw [hmax_eq]
    exact hmax_ge _ trivial
  have happrox :
      sSup (Q ω '' Set.univ) <
        Q ω (Select.approx B Q hQ hcont (hε_pos n) ω) +
          ε n :=
    Select.approx_spec B Q hQ hcont (hε_pos n) ω
  rw [Real.dist_eq]
  have habs :
      |Q ω (Select.approx B Q hQ hcont (hε_pos n) ω) -
          sSup (Q ω '' Set.univ)| =
        sSup (Q ω '' Set.univ) -
          Q ω (Select.approx B Q hQ hcont (hε_pos n) ω) := by
    rw [abs_of_nonpos (sub_nonpos.mpr hle), neg_sub]
  rw [habs]
  linarith

/-- Specializing the previous theorem to the canonical error schedule
`εₙ = 1 / (n + 1)`, the measurable approximate selector values converge
pointwise to the maximum-value map. -/
theorem Select.schedule_limit
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    ∀ ω,
      Filter.Tendsto
        (fun n : ℕ ↦ Q ω
          (Select.approx B Q hQ hcont
            (show 0 < (1 : ℝ) / (n + 1) by positivity) ω))
        atTop (nhds (sSup (Q ω '' Set.univ))) := by
  exact
    Select.value_limit
      B Q hQ hcont
      (fun n ↦ by positivity)
      tendsto_one_div_add_atTop_nhds_zero_nat

/-- Under pointwise uniqueness of the maximizer, the canonical measurable
`εₙ = 1 / (n + 1)`-argmax selectors converge pointwise to the exact maximizer. -/
theorem Select.unique_limit
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (hunique : ∀ ω, ∃! p : Fit.region B, IsMaxOn (Q ω) Set.univ p) :
    ∀ ω,
      Filter.Tendsto
        (fun n : ℕ ↦
          Select.approx B Q hQ hcont
            (show 0 < (1 : ℝ) / (n + 1) by positivity) ω)
        atTop (nhds (Select.unique B Q hunique ω)) := by
  intro ω
  let pseq : ℕ → Fit.region B := fun n ↦
    Select.approx B Q hQ hcont
      (show 0 < (1 : ℝ) / (n + 1) by positivity) ω
  let maxValue : ℝ := sSup (Q ω '' Set.univ)
  have hθmax : IsMaxOn (Q ω) Set.univ (Select.unique B Q hunique ω) :=
    Select.unique_spec B Q hunique ω
  have hθeq :
      Q ω (Select.unique B Q hunique ω) = maxValue := by
    simpa [maxValue] using
      Select.unique_value B Q hcont hunique ω
  have hval_lim :
      Filter.Tendsto (fun n : ℕ ↦ Q ω (pseq n)) atTop (nhds maxValue) := by
    simpa [pseq, maxValue] using
      Select.schedule_limit B Q hQ hcont ω
  have hcompact : IsCompact (Set.univ : Set (Fit.region B)) := isCompact_univ
  refine hcompact.tendsto_nhds_of_unique_mapClusterPt
    (Eventually.of_forall fun _ ↦ trivial) ?_
  intro x _ hxcluster
  have hxle : Q ω x ≤ maxValue := by
    calc
      Q ω x ≤ Q ω (Select.unique B Q hunique ω) :=
        (isMaxOn_iff.mp hθmax) x trivial
      _ = maxValue := hθeq
  have hxge : maxValue ≤ Q ω x := by
    by_contra hlt
    have hgap : 0 < (maxValue - Q ω x) / 2 := by
      linarith
    let s : Set (Fit.region B) := {p | maxValue - Q ω p ≤ (maxValue - Q ω x) / 2}
    have hsclosed : IsClosed s := by
      dsimp [s]
      exact isClosed_le (continuous_const.sub (hcont ω)) continuous_const
    have hmem : ∀ᶠ n in atTop, pseq n ∈ s := by
      rcases (Metric.tendsto_atTop.1 hval_lim) ((maxValue - Q ω x) / 2) hgap with
        ⟨N, hN⟩
      exact Filter.eventually_atTop.2 ⟨N, fun n hn ↦ by
        dsimp [s]
        have hle_n : Q ω (pseq n) ≤ maxValue := by
          calc
            Q ω (pseq n) ≤ Q ω (Select.unique B Q hunique ω) :=
              (isMaxOn_iff.mp hθmax) (pseq n) trivial
            _ = maxValue := hθeq
        have hdist : dist (Q ω (pseq n)) maxValue < (maxValue - Q ω x) / 2 :=
          hN n hn
        rw [Real.dist_eq] at hdist
        have habs :
            |Q ω (pseq n) - maxValue| = maxValue - Q ω (pseq n) := by
          rw [abs_of_nonpos (sub_nonpos.mpr hle_n), neg_sub]
        rw [habs] at hdist
        linarith⟩
    have hxmem : x ∈ s := hsclosed.mem_of_mapClusterPt hxcluster hmem
    have : maxValue - Q ω x ≤ (maxValue - Q ω x) / 2 := by
      simpa [s] using hxmem
    linarith
  have hxismax : IsMaxOn (Q ω) Set.univ x := by
    rw [isMaxOn_iff]
    intro y _
    calc
      Q ω y ≤ Q ω (Select.unique B Q hunique ω) :=
        (isMaxOn_iff.mp hθmax) y trivial
      _ = maxValue := hθeq
      _ ≤ Q ω x := hxge
  exact Select.unique_eq B Q hunique hxismax

/-- Under pointwise uniqueness of the maximizer, the exact selector is
measurable. -/
theorem Select.unique_measurable
    (B : Fit.Box) (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (hunique : ∀ ω, ∃! p : Fit.region B, IsMaxOn (Q ω) Set.univ p) :
    Measurable (Select.unique B Q hunique) := by
  let f : ℕ → Ω → Fit.region B := fun n ω ↦
    Select.approx B Q hQ hcont
      (show 0 < (1 : ℝ) / (n + 1) by positivity) ω
  have hf : ∀ n, Measurable (f n) := by
    intro n
    change Measurable
      (Select.approx B Q hQ hcont
        (show 0 < (1 : ℝ) / (n + 1) by positivity))
    exact Select.approx_measurable B Q hQ hcont
      (show 0 < (1 : ℝ) / (n + 1) by positivity)
  have hlim : Filter.Tendsto f atTop (nhds (Select.unique B Q hunique)) := by
    rw [tendsto_pi_nhds]
    intro ω
    simpa [f] using
      Select.unique_limit B Q hQ hcont hunique ω
  exact measurable_of_tendsto_metrizable hf hlim

end ENN
