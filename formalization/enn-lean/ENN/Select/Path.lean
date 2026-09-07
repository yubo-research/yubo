import ENN.Select.Step

/-!
# Nested maximizing paths and shrinking centers
-/

namespace ENN

open Filter MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω]

/-- Recursively choose a nested sequence of exact-maximizing quadtree cells,
one at each finite depth. -/
noncomputable def Select.path
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    (n : ℕ) → Ω → Fin (4 ^ n)
  | 0 => fun _ ↦ 0
  | n + 1 => fun ω ↦
      Select.step B n Q hQ hcont
        (Select.path B Q hQ hcont n ω) ω

theorem Select.path_measurable
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    ∀ n, Measurable (Select.path B Q hQ hcont n)
  | 0 => measurable_const
  | n + 1 => by
      let f : Fin (4 ^ n) → Ω → Fin (4 ^ (n + 1)) :=
        Select.step B n Q hQ hcont
      have hf : ∀ i, Measurable (f i) := by
        intro i
        exact Select.step_measurable B n Q hQ hcont i
      simpa [Select.path, f] using
        (Select.apply_measurable (m := 4 ^ n) (β := Fin (4 ^ (n + 1)))
          (pow_pos (by norm_num) n)
          (Select.path B Q hQ hcont n)
          (Select.path_measurable B Q hQ hcont n)
          f hf)

theorem Select.path_spec
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    ∀ n ω,
      sSup
          (Q ω ''
            (Fit.Cell.tree B n
              (Select.path B Q hQ hcont n ω)).toSet) =
        sSup (Q ω '' Set.univ)
  | 0, ω => by
      simp [Fit.Cell.tree,
        Fit.Cell.whole_toSet]
  | n + 1, ω => by
      let i := Select.path B Q hQ hcont n ω
      calc
        sSup
            (Q ω ''
              (Fit.Cell.tree B (n + 1)
                (Select.path B Q hQ hcont (n + 1) ω)).toSet) =
          sSup (Q ω '' (Fit.Cell.tree B n i).toSet) := by
            simpa [Select.path, i] using
              (Select.step_spec B n Q hQ hcont i ω)
        _ = sSup (Q ω '' Set.univ) := by
            simpa [i] using
              (Select.path_spec B Q hQ hcont n ω)

theorem Select.path_nested
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (n : ℕ) :
    ∀ ω,
      (Fit.Cell.tree B (n + 1)
        (Select.path B Q hQ hcont (n + 1) ω)).toSet ⊆
        (Fit.Cell.tree B n
          (Select.path B Q hQ hcont n ω)).toSet := by
  intro ω
  let i := Select.path B Q hQ hcont n ω
  simpa [Select.path, i] using
    (Select.step_nested B n Q hQ hcont i ω)

/-- The center point of the recursively chosen exact-maximizing quadtree cell at
depth `n`. -/
noncomputable def Select.center
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (n : ℕ) :
    Ω → Fit.region B := fun ω ↦
  (Fit.Cell.tree B n
    (Select.path B Q hQ hcont n ω)).center

theorem Select.center_measurable
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (n : ℕ) :
    Measurable (Select.center B Q hQ hcont n) := by
  change Measurable
    (fun ω ↦
      (Fit.Cell.tree B n
        (Select.path B Q hQ hcont n ω)).center)
  let f : Fin (4 ^ n) → Ω → Fit.region B := fun i _ ↦
    (Fit.Cell.tree B n i).center
  have hf : ∀ i, Measurable (f i) := by
    intro i
    exact measurable_const
  simpa [f] using
    (Select.apply_measurable (m := 4 ^ n) (β := Fit.region B)
      (pow_pos (by norm_num) n)
      (Select.path B Q hQ hcont n)
      (Select.path_measurable B Q hQ hcont n)
      f hf)

theorem Select.center_mem
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    (n : ℕ) :
    ∀ ω,
      Select.center B Q hQ hcont n ω ∈
        (Fit.Cell.tree B n
          (Select.path B Q hQ hcont n ω)).toSet := by
  intro ω
  change
    (Fit.Cell.tree B n
      (Select.path B Q hQ hcont n ω)).center ∈
      (Fit.Cell.tree B n
        (Select.path B Q hQ hcont n ω)).toSet
  exact Fit.Cell.center_mem _

theorem Select.path_subset
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    {m n : ℕ} (hmn : m ≤ n) :
    ∀ ω,
      (Fit.Cell.tree B n
        (Select.path B Q hQ hcont n ω)).toSet ⊆
        (Fit.Cell.tree B m
          (Select.path B Q hQ hcont m ω)).toSet := by
  rcases Nat.exists_eq_add_of_le hmn with ⟨k, rfl⟩
  suffices haux :
      ∀ k ω,
        (Fit.Cell.tree B (m + k)
          (Select.path B Q hQ hcont (m + k) ω)).toSet ⊆
          (Fit.Cell.tree B m
            (Select.path B Q hQ hcont m ω)).toSet by
    simpa using haux k
  intro k
  induction k with
  | zero =>
      intro ω
      simp
  | succ k ih =>
      intro ω
      have hstep :
          (Fit.Cell.tree B (m + k + 1)
            (Select.path B Q hQ hcont (m + k + 1) ω)).toSet ⊆
            (Fit.Cell.tree B (m + k)
              (Select.path B Q hQ hcont (m + k) ω)).toSet := by
        simpa [Nat.add_assoc] using
          Select.path_nested B Q hQ hcont (m + k) ω
      exact Set.Subset.trans hstep (ih ω)

theorem Select.center_in_cell
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    {m n : ℕ} (hmn : m ≤ n) :
    ∀ ω,
      Select.center B Q hQ hcont n ω ∈
        (Fit.Cell.tree B m
          (Select.path B Q hQ hcont m ω)).toSet := by
  intro ω
  exact
    Select.path_subset B Q hQ hcont hmn ω
      (Select.center_mem B Q hQ hcont n ω)

theorem Select.center_distance
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω))
    {N n m : ℕ} (hNn : N ≤ n) (hNm : N ≤ m) :
    ∀ ω,
      dist (Select.center B Q hQ hcont n ω)
        (Select.center B Q hQ hcont m ω) ≤
          (B.alpha_hi - B.alpha_lo) / (2 : ℝ) ^ N +
            (B.gamma_hi - B.gamma_lo) / (2 : ℝ) ^ N := by
  intro ω
  let C :=
    Fit.Cell.tree B N
      (Select.path B Q hQ hcont N ω)
  have hp :
      Select.center B Q hQ hcont n ω ∈ C.toSet := by
    simpa [C] using
      Select.center_in_cell B Q hQ hcont hNn ω
  have hq :
      Select.center B Q hQ hcont m ω ∈ C.toSet := by
    simpa [C] using
      Select.center_in_cell B Q hQ hcont hNm ω
  calc
    dist (Select.center B Q hQ hcont n ω)
        (Select.center B Q hQ hcont m ω)
      ≤ C.alphaWidth + C.gammaWidth := by
          exact Fit.Cell.diameter_bound C hp hq
    _ =
        (B.alpha_hi - B.alpha_lo) / (2 : ℝ) ^ N +
          (B.gamma_hi - B.gamma_lo) / (2 : ℝ) ^ N := by
          simp [C, Fit.Cell.tree_alpha,
            Fit.Cell.tree_gamma]

theorem Select.width_limit (B : Fit.Box) :
    Filter.Tendsto
      (fun n : ℕ ↦
        (B.alpha_hi - B.alpha_lo) / (2 : ℝ) ^ n +
          (B.gamma_hi - B.gamma_lo) / (2 : ℝ) ^ n)
      atTop (nhds 0) := by
  have hpow : Filter.Tendsto (fun n : ℕ ↦ ((2 : ℝ)⁻¹) ^ n) atTop (nhds 0) := by
    exact tendsto_pow_atTop_nhds_zero_of_lt_one (by positivity) (by norm_num)
  have hα :
      Filter.Tendsto
        (fun n : ℕ ↦ (B.alpha_hi - B.alpha_lo) / (2 : ℝ) ^ n)
        atTop (nhds 0) := by
    simpa [div_eq_mul_inv, inv_pow] using
      (tendsto_const_nhds.mul hpow)
  have hγ :
      Filter.Tendsto
        (fun n : ℕ ↦ (B.gamma_hi - B.gamma_lo) / (2 : ℝ) ^ n)
        atTop (nhds 0) := by
    simpa [div_eq_mul_inv, inv_pow] using
      (tendsto_const_nhds.mul hpow)
  simpa using hα.add hγ

theorem Select.center_cauchy
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    ∀ ω, CauchySeq (Select.center B Q hQ hcont · ω) := by
  intro ω
  refine cauchySeq_of_le_tendsto_0
    (fun N : ℕ ↦
      (B.alpha_hi - B.alpha_lo) / (2 : ℝ) ^ N +
        (B.gamma_hi - B.gamma_lo) / (2 : ℝ) ^ N) ?_
    (Select.width_limit B)
  intro n m N hNn hNm
  exact Select.center_distance B Q hQ hcont hNn hNm ω

theorem Select.exists_limit
    (B : Fit.Box)
    (Q : Ω → Fit.region B → ℝ)
    (hQ : Measurable (Function.uncurry Q))
    (hcont : ∀ ω, Continuous (Q ω)) :
    ∀ ω, ∃ p : Fit.region B,
      Filter.Tendsto
        (Select.center B Q hQ hcont · ω)
        atTop (nhds p) := by
  intro ω
  exact cauchySeq_tendsto_of_complete
    (Select.center_cauchy B Q hQ hcont ω)

end ENN
