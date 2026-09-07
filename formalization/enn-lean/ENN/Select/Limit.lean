import ENN.Select.Path

/-!
# The exact maximizing limit
-/

namespace ENN

open Filter MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω]

/-- The limit point of the measurable center sequence associated to the nested
exact-maximizing quadtree cells. -/
noncomputable def Select.limit
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    Ω → Fit.region B := fun ω ↦
  Classical.choose
    (Select.exists_limit B Q hQ hcont ω)

theorem Select.center_limit
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    ∀ ω,
      Filter.Tendsto
        (Select.center B Q hQ hcont · ω)
        atTop (nhds (Select.limit B Q hQ hcont ω)) :=
  fun ω ↦ Classical.choose_spec
    (Select.exists_limit B Q hQ hcont ω)

theorem Select.limit_measurable
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    Measurable (Select.limit B Q hQ hcont) := by
  let f : ℕ → Ω → Fit.region B := Select.center B Q hQ hcont
  have hf : ∀ n, Measurable (f n) := by
    intro n
    simpa [f] using
      Select.center_measurable B Q hQ hcont n
  have hlim : Filter.Tendsto f atTop (nhds (Select.limit B Q hQ hcont)) := by
    rw [tendsto_pi_nhds]
    intro ω
    simpa [f] using
      Select.center_limit B Q hQ hcont ω
  exact measurable_of_tendsto_metrizable hf hlim

theorem Select.limit_in_cell
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (n : ℕ) :
    ∀ ω,
      Select.limit B Q hQ hcont ω ∈
        (Fit.Cell.tree B n
          (Select.path B Q hQ hcont n ω)).toSet := by
  intro ω
  let C :=
    Fit.Cell.tree B n
      (Select.path B Q hQ hcont n ω)
  apply C.isClosed_toSet.mem_of_tendsto
    (Select.center_limit B Q hQ hcont ω)
  filter_upwards [Filter.Ici_mem_atTop n] with m hm
  simpa [C] using
    Select.center_in_cell B Q hQ hcont hm ω

theorem Select.limit_value
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    ∀ ω,
      Q ω (Select.limit B Q hQ hcont ω) =
        sSup (Q ω '' Set.univ) := by
  intro ω
  let center : ℕ → Fit.region B := fun n ↦
    Select.center B Q hQ hcont n ω
  let limit : Fit.region B :=
    Select.limit B Q hQ hcont ω
  let cell : ℕ → Fit.Cell B := fun n ↦
    Fit.Cell.tree B n
      (Select.path B Q hQ hcont n ω)
  have hc_mem : ∀ n, center n ∈ (cell n).toSet := by
    intro n
    simpa [center, cell] using
      Select.center_mem B Q hQ hcont n ω
  have hq_ex :
      ∀ n, ∃ q : Fit.region B,
        q ∈ (cell n).toSet ∧
          sSup (Q ω '' (cell n).toSet) = Q ω q := by
    intro n
    exact (cell n).isCompact_toSet.exists_sSup_image_eq (cell n).toSet_nonempty
      (hcont ω).continuousOn
  let q : ℕ → Fit.region B := fun n ↦ Classical.choose (hq_ex n)
  have hq_mem : ∀ n, q n ∈ (cell n).toSet := by
    intro n
    exact (Classical.choose_spec (hq_ex n)).1
  have hq_eq : ∀ n, sSup (Q ω '' (cell n).toSet) = Q ω (q n) := by
    intro n
    exact (Classical.choose_spec (hq_ex n)).2
  have hdist_qc : Filter.Tendsto (fun n ↦ dist (q n) (center n)) atTop (nhds 0) := by
    refine squeeze_zero (fun n ↦ dist_nonneg) ?_ (Select.width_limit B)
    intro n
    calc
      dist (q n) (center n) ≤ (cell n).alphaWidth + (cell n).gammaWidth := by
        exact Fit.Cell.diameter_bound
          (cell n) (hq_mem n) (hc_mem n)
      _ =
          (B.alpha_hi - B.alpha_lo) / (2 : ℝ) ^ n +
            (B.gamma_hi - B.gamma_lo) / (2 : ℝ) ^ n := by
          simp [cell, Fit.Cell.tree_alpha,
            Fit.Cell.tree_gamma]
  have hc_lim : Filter.Tendsto center atTop (nhds limit) := by
    simpa [center, limit] using
      Select.center_limit B Q hQ hcont ω
  have hc_dist : Filter.Tendsto (fun n ↦ dist (center n) limit) atTop (nhds 0) := by
    exact (tendsto_iff_dist_tendsto_zero.1 hc_lim)
  have hq_lim : Filter.Tendsto q atTop (nhds limit) := by
    have hsum0 :
        Filter.Tendsto
          (fun n ↦ dist (q n) (center n) + dist (center n) limit)
          atTop (nhds 0) := by
      simpa using (hdist_qc.add hc_dist)
    rw [tendsto_iff_dist_tendsto_zero]
    exact squeeze_zero
      (fun n ↦ dist_nonneg)
      (fun n ↦ dist_triangle (q n) (center n) limit)
      hsum0
  let M : Set (Fit.region B) := {p | Q ω p = sSup (Q ω '' Set.univ)}
  have hMc : IsClosed M := by
    dsimp [M]
    exact isClosed_eq (hcont ω) continuous_const
  have hq_mem : ∀ n, q n ∈ M := by
    intro n
    dsimp [M]
    calc
      Q ω (q n) = sSup (Q ω '' (cell n).toSet) := (hq_eq n).symm
      _ = sSup (Q ω '' Set.univ) := by
          simpa [cell] using
            Select.path_spec B Q hQ hcont n ω
  have hlim_mem : limit ∈ M :=
    hMc.mem_of_tendsto hq_lim (Filter.Eventually.of_forall hq_mem)
  simpa [limit, M] using hlim_mem

theorem Select.limit_max
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    ∀ ω,
      IsMaxOn (Q ω) Set.univ
        (Select.limit B Q hQ hcont ω) := by
  intro ω
  obtain ⟨pmax, -, hmax_eq, hmax_ge⟩ :=
    isCompact_univ.exists_sSup_image_eq_and_ge Set.univ_nonempty (hcont ω).continuousOn
  rw [isMaxOn_iff]
  intro p _
  calc
    Q ω p ≤ Q ω pmax := hmax_ge p trivial
    _ = sSup (Q ω '' Set.univ) := hmax_eq.symm
    _ = Q ω (Select.limit B Q hQ hcont ω) := by
        symm
        exact Select.limit_value B Q hQ hcont ω

end ENN
