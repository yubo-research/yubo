import Mathlib.Data.Finset.Sort
import Mathlib.Data.Prod.Lex
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Topology.MetricSpace.Cauchy
import ENN.FiniteWeights
import ENN.InverseSquare

/-!
# Deterministic nearest-neighbor order

This file packages the deterministic tie-breaking discipline required by
Appendix C: sample indices are ordered by the lexicographic key
`(dist query (xᵢ), i)`.
-/

namespace ENN

abbrev Dataset (X : Type*) (n : ℕ) := Fin n → X

section NearestNeighbor

variable {X : Type*} [MetricSpace X] {n : ℕ}

/-- Exact-observation datasets record only sampled inputs. -/
abbrev ExactDataset := Dataset X n

/-- A query is already sampled if it appears among the observed inputs. -/
def Sampled (data : ExactDataset (X := X) (n := n)) (query : X) : Prop :=
  ∃ i, data i = query

/-- Lexicographic nearest-neighbor key: distance first, original index second. -/
def Neighbor.key (data : ExactDataset (X := X) (n := n)) (query : X) (i : Fin n) : ℝ ×ₗ Fin n :=
  toLex (dist query (data i), i)

theorem Neighbor.key_injective (data : ExactDataset (X := X) (n := n)) (query : X) :
    Function.Injective (Neighbor.key data query) := by
  intro i j hij
  exact congrArg (fun p => (ofLex p).2) hij

/-- The finite set of all lexicographic nearest-neighbor keys. -/
noncomputable def Neighbor.keys (data : ExactDataset (X := X) (n := n)) (query : X) :
    Finset (ℝ ×ₗ Fin n) :=
  Finset.univ.image (Neighbor.key data query)

theorem Neighbor.keys_card (data : ExactDataset (X := X) (n := n)) (query : X) :
    (Neighbor.keys data query).card = n := by
  classical
  simpa [Neighbor.keys] using
    (Finset.card_image_of_injective (s := Finset.univ) (f := Neighbor.key data query)
      (Neighbor.key_injective data query))

/-- The `j`-th nearest-neighbor key in deterministic lexicographic order. -/
noncomputable def Neighbor.ordered_key (data : ExactDataset (X := X) (n := n)) (query : X) (j : Fin n) :
    ℝ ×ₗ Fin n :=
  (Neighbor.keys data query).orderEmbOfFin (Neighbor.keys_card data query) j

/-- The `j`-th nearest sample index under deterministic lexicographic order. -/
noncomputable def Neighbor.index (data : ExactDataset (X := X) (n := n)) (query : X) (j : Fin n) :
    Fin n :=
  (ofLex (Neighbor.ordered_key data query j)).2

/-- The `j`-th nearest sample radius under deterministic lexicographic order. -/
noncomputable def Neighbor.radius (data : ExactDataset (X := X) (n := n)) (query : X) (j : Fin n) :
    ℝ :=
  dist query (data (Neighbor.index data query j))

theorem Neighbor.key_index
    (data : ExactDataset (X := X) (n := n)) (query : X) (j : Fin n) :
    Neighbor.ordered_key data query j = Neighbor.key data query (Neighbor.index data query j) := by
  classical
  have hmem : Neighbor.ordered_key data query j ∈ Neighbor.keys data query := by
    simp [Neighbor.ordered_key]
  rw [Neighbor.keys, Finset.mem_image] at hmem
  rcases hmem with ⟨i, _hi, hkey⟩
  have hidx : i = Neighbor.index data query j := by
    simpa [Neighbor.key, Neighbor.index] using congrArg (fun p => (ofLex p).2) hkey
  have hkey' : Neighbor.ordered_key data query j = Neighbor.key data query i := hkey.symm
  simpa [hidx] using hkey'

theorem Neighbor.index_injective (data : ExactDataset (X := X) (n := n)) (query : X) :
    Function.Injective (Neighbor.index data query) := by
  intro i j hij
  have hkeyi := Neighbor.key_index data query i
  have hkeyj := Neighbor.key_index data query j
  rw [hij] at hkeyi
  exact ((Neighbor.keys data query).orderEmbOfFin (Neighbor.keys_card data query)).injective <|
    hkeyi.trans hkeyj.symm

theorem Neighbor.key_radius
    (data : ExactDataset (X := X) (n := n)) (query : X) (j : Fin n) :
    (ofLex (Neighbor.ordered_key data query j)).1 = Neighbor.radius data query j := by
  have hkey := Neighbor.key_index data query j
  simpa [Neighbor.key, Neighbor.radius, Neighbor.index] using congrArg (fun p => (ofLex p).1) hkey

theorem Neighbor.key_monotone (data : ExactDataset (X := X) (n := n)) (query : X) :
    Monotone (Neighbor.ordered_key data query) := by
  classical
  intro i j hij
  exact ((Neighbor.keys data query).orderEmbOfFin (Neighbor.keys_card data query)).monotone hij

theorem Neighbor.radius_monotone (data : ExactDataset (X := X) (n := n)) (query : X) :
    Monotone (Neighbor.radius data query) := by
  intro i j hij
  have hkey : Neighbor.ordered_key data query i ≤ Neighbor.ordered_key data query j :=
    Neighbor.key_monotone data query hij
  have hfst : (ofLex (Neighbor.ordered_key data query i)).1 ≤ (ofLex (Neighbor.ordered_key data query j)).1 :=
    Prod.Lex.monotone_fst _ _ hkey
  rwa [Neighbor.key_radius, Neighbor.key_radius] at hfst

/-- The nearest-sample distance `d(query, Xₙ)`, represented as the first ordered radius. -/
noncomputable def Neighbor.distance (data : ExactDataset (X := X) (n := n)) (query : X)
    (hn : 0 < n) : ℝ :=
  Neighbor.radius data query ⟨0, hn⟩

theorem Neighbor.distance_nonneg (data : ExactDataset (X := X) (n := n)) (query : X) (hn : 0 < n) :
    0 ≤ Neighbor.distance data query hn := by
  unfold Neighbor.distance Neighbor.radius
  exact dist_nonneg

theorem Neighbor.distance_radius (data : ExactDataset (X := X) (n := n)) (query : X)
    (hn : 0 < n) (j : Fin n) :
    Neighbor.distance data query hn ≤ Neighbor.radius data query j :=
  Neighbor.radius_monotone data query (show (⟨0, hn⟩ : Fin n) ≤ j by exact Nat.zero_le _)

theorem Neighbor.distance_le (data : ExactDataset (X := X) (n := n)) (query : X)
    (hn : 0 < n) (i : Fin n) :
    Neighbor.distance data query hn ≤ dist query (data i) := by
  classical
  have hmem : Neighbor.key data query i ∈ (Neighbor.keys data query : Set (ℝ ×ₗ Fin n)) := by
    exact Finset.mem_image_of_mem _ (Finset.mem_univ i)
  have hrange :
      Set.range ((Neighbor.keys data query).orderEmbOfFin (Neighbor.keys_card data query)) =
        (Neighbor.keys data query : Set (ℝ ×ₗ Fin n)) :=
    Finset.range_orderEmbOfFin (s := Neighbor.keys data query) (h := Neighbor.keys_card data query)
  have hmemRange : Neighbor.key data query i ∈ Set.range (Neighbor.ordered_key data query) := by
    change Neighbor.key data query i ∈
      Set.range ((Neighbor.keys data query).orderEmbOfFin (Neighbor.keys_card data query))
    rw [hrange]
    exact hmem
  rcases hmemRange with ⟨j, hj⟩
  have hkey : Neighbor.ordered_key data query ⟨0, hn⟩ ≤ Neighbor.ordered_key data query j :=
    Neighbor.key_monotone data query (Nat.zero_le j)
  have hfst : (ofLex (Neighbor.ordered_key data query ⟨0, hn⟩)).1 ≤ (ofLex (Neighbor.key data query i)).1 := by
    simpa [hj] using Prod.Lex.monotone_fst _ _ hkey
  calc
    Neighbor.distance data query hn = (ofLex (Neighbor.ordered_key data query ⟨0, hn⟩)).1 := by
      simpa [Neighbor.distance] using (Neighbor.key_radius data query ⟨0, hn⟩).symm
    _ ≤ (ofLex (Neighbor.key data query i)).1 := hfst
    _ = dist query (data i) := by rfl

theorem Neighbor.dist_positive (data : ExactDataset (X := X) (n := n)) (query : X)
    (hquery : ¬ Sampled data query) (i : Fin n) :
    0 < dist query (data i) := by
  apply dist_pos.mpr
  intro hEq
  apply hquery
  exact ⟨i, hEq.symm⟩

/-- The first `k` ordered radii, viewed as a `Fin k → ℝ` family. -/
noncomputable def Neighbor.prefix_index (data : ExactDataset (X := X) (n := n)) (query : X) {k : ℕ}
    (hk : k ≤ n) : Fin k → Fin n :=
  fun i ↦ Neighbor.index data query ⟨i.1, lt_of_lt_of_le i.2 hk⟩

/-- The first `k` ordered radii, viewed as a `Fin k → ℝ` family. -/
noncomputable def Neighbor.prefix_radius (data : ExactDataset (X := X) (n := n)) (query : X) {k : ℕ}
    (hk : k ≤ n) : Fin k → ℝ :=
  fun i ↦ dist query (data (Neighbor.prefix_index data query hk i))

theorem Neighbor.prefix_injective (data : ExactDataset (X := X) (n := n)) (query : X) {k : ℕ}
    (hk : k ≤ n) : Function.Injective (Neighbor.prefix_index data query hk) := by
  intro i j hij
  have hlift :
      (⟨i.1, lt_of_lt_of_le i.2 hk⟩ : Fin n) = ⟨j.1, lt_of_lt_of_le j.2 hk⟩ := by
    exact Neighbor.index_injective data query <| by simpa [Neighbor.prefix_index] using hij
  cases i
  cases j
  cases hlift
  rfl

theorem Neighbor.prefix_positive (data : ExactDataset (X := X) (n := n)) (query : X) {k : ℕ}
    (hk : k ≤ n) (hquery : ¬ Sampled data query) (i : Fin k) :
    0 < Neighbor.prefix_radius data query hk i :=
  Neighbor.dist_positive data query hquery (Neighbor.prefix_index data query hk i)

theorem Neighbor.prefix_monotone (data : ExactDataset (X := X) (n := n)) (query : X) {k : ℕ}
    (hk : k ≤ n) : Monotone (Neighbor.prefix_radius data query hk) := by
  intro i j hij
  exact Neighbor.radius_monotone data query (show (⟨i.1, lt_of_lt_of_le i.2 hk⟩ : Fin n) ≤
      ⟨j.1, lt_of_lt_of_le j.2 hk⟩ by exact hij)

/-- The inverse-square ENN uncertainty built from the first `k` ordered radii. -/
noncomputable def Exact.prefix_sigma (data : ExactDataset (X := X) (n := n)) (query : X) {k : ℕ}
    (hk : k ≤ n) : ℝ :=
  InvSq.sigma (Neighbor.prefix_radius data query hk)

/-- The normalized inverse-square weights on the first `k` ordered neighbors. -/
noncomputable def Exact.weights (data : ExactDataset (X := X) (n := n)) (query : X) {k : ℕ}
    (hk : 0 < k) (hkn : k ≤ n) (hquery : ¬ Sampled data query) :
    Weights (Fin k) := by
  let _ : Nonempty (Fin k) := ⟨⟨0, hk⟩⟩
  exact
    ofPositive (fun i ↦ ((Neighbor.prefix_radius data query hkn i) ^ 2)⁻¹)
      (fun i ↦ by
        have hpos : 0 < Neighbor.prefix_radius data query hkn i := Neighbor.prefix_positive data query hkn hquery i
        positivity)

theorem Exact.weight_eq (data : ExactDataset (X := X) (n := n)) (query : X) {k : ℕ}
    (hk : 0 < k) (hkn : k ≤ n) (hquery : ¬ Sampled data query) (i : Fin k) :
    (Exact.weights data query hk hkn hquery).weight i =
      ((Neighbor.prefix_radius data query hkn i) ^ 2)⁻¹ /
        ∑ j : Fin k, ((Neighbor.prefix_radius data query hkn j) ^ 2)⁻¹ := by
  simp [Exact.weights, ofPositive, normalize]

theorem Exact.weight_bound
    (data : ExactDataset (X := X) (n := n)) (query : X) {k : ℕ}
    (hn : 0 < n) (hk : 0 < k) (hkn : k ≤ n) (hquery : ¬ Sampled data query)
    {δ : ℝ} (hδ : 0 < δ) (hdδ : Neighbor.distance data query hn ≤ δ) (i : Fin k)
    (hfar : δ ≤ Neighbor.prefix_radius data query hkn i) :
    (Exact.weights data query hk hkn hquery).weight i ≤ Neighbor.distance data query hn / δ := by
  let w := Exact.weights data query hk hkn hquery
  let d := Neighbor.distance data query hn
  let r : Fin k → ℝ := Neighbor.prefix_radius data query hkn
  have hd : 0 < d := by
    change 0 < r ⟨0, hk⟩
    exact Neighbor.prefix_positive data query hkn hquery ⟨0, hk⟩
  have hvar :
      InvSq.variance r ≤ d ^ 2 := by
    have hfirst : r ⟨0, hk⟩ = d := by
      simp [r, d, Neighbor.prefix_radius, Neighbor.prefix_index, Neighbor.distance, Neighbor.radius]
    exact InvSq.variance_upper r ⟨0, hk⟩ hfirst hd
  have hri_pos : 0 < r i := lt_of_lt_of_le hδ hfar
  have hnum :
      ((r i) ^ 2)⁻¹ ≤ (δ ^ 2)⁻¹ := by
    have hsq : δ ^ 2 ≤ (r i) ^ 2 :=
      (sq_le_sq₀ hδ.le hri_pos.le).2 hfar
    simpa [one_div] using one_div_le_one_div_of_le (pow_pos hδ 2) hsq
  have hweight :
      w.weight i = ((r i) ^ 2)⁻¹ * InvSq.variance r := by
    rw [Exact.weight_eq]
    simp [InvSq.variance, div_eq_mul_inv, r]
  calc
    w.weight i = ((r i) ^ 2)⁻¹ * InvSq.variance r := hweight
    _ ≤ ((r i) ^ 2)⁻¹ * (d ^ 2) := by
      exact mul_le_mul_of_nonneg_left hvar (by positivity)
    _ ≤ (δ ^ 2)⁻¹ * (d ^ 2) := by
      exact mul_le_mul_of_nonneg_right hnum (sq_nonneg d)
    _ = d ^ 2 / δ ^ 2 := by
      rw [div_eq_mul_inv]
      ring
    _ ≤ d / δ := by
      have hsq : d ^ 2 ≤ d * δ := by nlinarith
      calc
        d ^ 2 / δ ^ 2 ≤ (d * δ) / δ ^ 2 := by
          exact div_le_div_of_nonneg_right hsq (sq_nonneg δ)
        _ = d / δ := by
          field_simp [hδ.ne']

theorem Exact.far_mass
    (data : ExactDataset (X := X) (n := n)) (query : X) {k : ℕ}
    (hn : 0 < n) (hk : 0 < k) (hkn : k ≤ n) (hquery : ¬ Sampled data query)
    {δ : ℝ} (hδ : 0 < δ) (hdδ : Neighbor.distance data query hn ≤ δ) :
    ∑ i : Fin k, (Exact.weights data query hk hkn hquery).weight i *
      (if δ ≤ Neighbor.prefix_radius data query hkn i then (1 : ℝ) else 0) ≤
      (k : ℝ) * (Neighbor.distance data query hn / δ) := by
  have hterm :
      ∀ i : Fin k,
        (Exact.weights data query hk hkn hquery).weight i *
          (if δ ≤ Neighbor.prefix_radius data query hkn i then (1 : ℝ) else 0)
            ≤ Neighbor.distance data query hn / δ := by
    intro i
    by_cases hfar : δ ≤ Neighbor.prefix_radius data query hkn i
    · simp [hfar]
      exact Exact.weight_bound
        data query hn hk hkn hquery hδ hdδ i hfar
    · simp [hfar]
      exact div_nonneg (Neighbor.distance_nonneg data query hn) hδ.le
  calc
    ∑ i : Fin k, (Exact.weights data query hk hkn hquery).weight i *
        (if δ ≤ Neighbor.prefix_radius data query hkn i then (1 : ℝ) else 0)
      ≤ ∑ _i : Fin k, Neighbor.distance data query hn / δ := by
        exact Finset.sum_le_sum fun i _ ↦ hterm i
    _ = (k : ℝ) * (Neighbor.distance data query hn / δ) := by simp

/-- The exact-observation ENN mean on the first `k` ordered neighbors. -/
noncomputable def Exact.prefix_mu (data : ExactDataset (X := X) (n := n)) (f : X → ℝ) (query : X)
    {k : ℕ} (hk : 0 < k) (hkn : k ≤ n) (hquery : ¬ Sampled data query) : ℝ :=
  ∑ i, (Exact.weights data query hk hkn hquery).weight i * f (data (Neighbor.prefix_index data query hkn i))

theorem Exact.error_sum (data : ExactDataset (X := X) (n := n)) (f : X → ℝ)
    (query : X) {k : ℕ} (hk : 0 < k) (hkn : k ≤ n) (hquery : ¬ Sampled data query) :
    Exact.prefix_mu data f query hk hkn hquery - f query =
      ∑ i, (Exact.weights data query hk hkn hquery).weight i *
        (f (data (Neighbor.prefix_index data query hkn i)) - f query) := by
  let w := Exact.weights data query hk hkn hquery
  calc
    Exact.prefix_mu data f query hk hkn hquery - f query
      = ∑ i, w.weight i * f (data (Neighbor.prefix_index data query hkn i)) -
          ((∑ i, w.weight i) * f query) := by
          rw [Exact.prefix_mu]
          simp [w, w.sum_eq_one]
    _ = ∑ i, w.weight i * f (data (Neighbor.prefix_index data query hkn i)) - ∑ i, w.weight i * f query := by
          rw [Finset.sum_mul]
    _ = ∑ i, w.weight i * (f (data (Neighbor.prefix_index data query hkn i)) - f query) := by
          rw [← Finset.sum_sub_distrib]
          apply Finset.sum_congr rfl
          intro i _
          rw [mul_sub]

theorem Exact.prefix_envelope (data : ExactDataset (X := X) (n := n)) (f : X → ℝ)
    (query : X) {k : ℕ} (hk : 0 < k) (hkn : k ≤ n) (hquery : ¬ Sampled data query)
    (ω : ℝ → ℝ)
    (herr : ∀ i, |f (data (Neighbor.prefix_index data query hkn i)) - f query| ≤ ω (Neighbor.prefix_radius data query hkn i)) :
    |Exact.prefix_mu data f query hk hkn hquery - f query| ≤
      ∑ i, (Exact.weights data query hk hkn hquery).weight i * ω (Neighbor.prefix_radius data query hkn i) := by
  let w := Exact.weights data query hk hkn hquery
  rw [Exact.error_sum data f query hk hkn hquery]
  simpa [w] using
    w.pointwise_bound
      (fun i ↦ f (data (Neighbor.prefix_index data query hkn i)) - f query)
      (fun i ↦ ω (Neighbor.prefix_radius data query hkn i))
      herr

theorem Exact.envelope_last (data : ExactDataset (X := X) (n := n)) (query : X) {k : ℕ}
    (hk : 0 < k) (hkn : k ≤ n) (hquery : ¬ Sampled data query) (ω : ℝ → ℝ) (hω : Monotone ω) :
    ∑ i, (Exact.weights data query hk hkn hquery).weight i * ω (Neighbor.prefix_radius data query hkn i) ≤
      ω (Neighbor.prefix_radius data query hkn ⟨k - 1, Nat.sub_lt hk Nat.zero_lt_one⟩) := by
  let w := Exact.weights data query hk hkn hquery
  refine w.sum_bound (fun i ↦ ω (Neighbor.prefix_radius data query hkn i))
    (ω (Neighbor.prefix_radius data query hkn ⟨k - 1, Nat.sub_lt hk Nat.zero_lt_one⟩)) ?_
  intro i
  have hi_last : i ≤ ⟨k - 1, Nat.sub_lt hk Nat.zero_lt_one⟩ := by
    exact Fin.le_iff_val_le_val.mpr (Nat.le_pred_of_lt i.isLt)
  exact hω <| Neighbor.prefix_monotone data query hkn hi_last

theorem Exact.prefix_bound (data : ExactDataset (X := X) (n := n)) (f : X → ℝ)
    (query : X) {k : ℕ} (hk : 0 < k) (hkn : k ≤ n) (hquery : ¬ Sampled data query)
    (ω : ℝ → ℝ) (herr : ∀ i, |f (data (Neighbor.prefix_index data query hkn i)) - f query| ≤ ω (Neighbor.prefix_radius data query hkn i))
    (hω : Monotone ω) :
    |Exact.prefix_mu data f query hk hkn hquery - f query| ≤
      ω (Neighbor.prefix_radius data query hkn ⟨k - 1, Nat.sub_lt hk Nat.zero_lt_one⟩) := by
  exact (Exact.prefix_envelope data f query hk hkn hquery ω herr).trans <|
    Exact.envelope_last data query hk hkn hquery ω hω

/-- `Kₙ = min K n`, the paper's truncated neighbor count. -/
abbrev Neighbor.count (K n : ℕ) : ℕ := min K n

/-- The exact-observation ENN uncertainty with the sampled-point interpolation branch. -/
noncomputable def Exact.sigma (data : ExactDataset (X := X) (n := n)) (query : X) (K : ℕ) : ℝ :=
  by
    classical
    let k := Neighbor.count K n
    exact
      if Sampled data query then 0
      else if hk : 0 < k then
        Exact.prefix_sigma data query (show k ≤ n from Nat.min_le_right K n)
      else 0

theorem Exact.sigma_prefix (data : ExactDataset (X := X) (n := n)) (query : X)
    (K : ℕ) (hk : 0 < Neighbor.count K n) (hquery : ¬ Sampled data query) :
    Exact.sigma data query K = Exact.prefix_sigma data query (Nat.min_le_right K n) := by
  classical
  unfold Exact.sigma
  simp [hquery, hk]

theorem Exact.sigma_nonneg (data : ExactDataset (X := X) (n := n)) (query : X) (K : ℕ) :
    0 ≤ Exact.sigma data query K := by
  classical
  by_cases hquery : Sampled data query
  · simp [Exact.sigma, hquery]
  · by_cases hk : 0 < Neighbor.count K n
    · rw [Exact.sigma_prefix data query K hk hquery]
      unfold Exact.prefix_sigma
      exact InvSq.sigma_nonneg _
    · simp [Exact.sigma, hquery, hk]

theorem Exact.prefix_spacing (data : ExactDataset (X := X) (n := n)) (query : X)
    {k : ℕ} (hn : 0 < n) (hk : 0 < k) (hkn : k ≤ n) (hquery : ¬ Sampled data query) :
    Neighbor.distance data query hn / Real.sqrt k ≤ Exact.prefix_sigma data query hkn ∧
      Exact.prefix_sigma data query hkn ≤ Neighbor.distance data query hn := by
  have hd : 0 < Neighbor.distance data query hn := by
    unfold Neighbor.distance
    exact Neighbor.dist_positive data query hquery (Neighbor.index data query ⟨0, hn⟩)
  have hfirst : Neighbor.prefix_radius data query hkn ⟨0, hk⟩ = Neighbor.distance data query hn := by
    simp [Neighbor.prefix_radius, Neighbor.prefix_index, Neighbor.distance, Neighbor.radius]
  have hlower : ∀ i : Fin k, Neighbor.distance data query hn ≤ Neighbor.prefix_radius data query hkn i := by
    intro i
    exact Neighbor.distance_radius data query hn ⟨i.1, lt_of_lt_of_le i.2 hkn⟩
  simpa [Exact.prefix_sigma] using
    InvSq.spacing hk (Neighbor.prefix_radius data query hkn) ⟨0, hk⟩ hfirst hd hlower

theorem Exact.spacing (data : ExactDataset (X := X) (n := n)) (query : X)
    (K : ℕ) (hn : 0 < n) (hk : 0 < Neighbor.count K n) (hquery : ¬ Sampled data query) :
    Neighbor.distance data query hn / Real.sqrt (Neighbor.count K n) ≤
        Exact.prefix_sigma data query (Nat.min_le_right K n) ∧
      Exact.prefix_sigma data query (Nat.min_le_right K n) ≤ Neighbor.distance data query hn :=
  Exact.prefix_spacing data query hn hk (Nat.min_le_right K n) hquery

theorem Exact.sigma_nearest (data : ExactDataset (X := X) (n := n)) (query : X)
    (K : ℕ) (hk : 0 < Neighbor.count K n) (hquery : ¬ Sampled data query) :
    Exact.sigma data query K ≤ Neighbor.distance data query (lt_of_lt_of_le hk (Nat.min_le_right K n)) := by
  let hn : 0 < n := lt_of_lt_of_le hk (Nat.min_le_right K n)
  rw [Exact.sigma_prefix data query K hk hquery]
  exact (Exact.spacing data query K hn hk hquery).2

theorem Exact.sigma_distance (data : ExactDataset (X := X) (n := n)) (query : X)
    (K : ℕ) (hk : 0 < Neighbor.count K n) (i : Fin n) :
    Exact.sigma data query K ≤ dist query (data i) := by
  classical
  by_cases hquery : Sampled data query
  · unfold Exact.sigma
    simp [hquery, dist_nonneg]
  · let hn : 0 < n := lt_of_lt_of_le hk (Nat.min_le_right K n)
    exact (Exact.sigma_nearest data query K hk hquery).trans (Neighbor.distance_le data query hn i)

theorem Exact.empty_ball (data : ExactDataset (X := X) (n := n)) (query : X)
    (K : ℕ) {r : ℝ} (hk : 0 < Neighbor.count K n) (hr : 0 < r)
    (hball : ∀ i : Fin n, r ≤ dist query (data i)) :
    r / Real.sqrt (Neighbor.count K n) ≤ Exact.sigma data query K := by
  let hn : 0 < n := lt_of_lt_of_le hk (Nat.min_le_right K n)
  have hquery : ¬ Sampled data query := by
    intro hs
    rcases hs with ⟨i, rfl⟩
    have : r ≤ 0 := by simpa using hball i
    exact (not_le_of_gt hr) this
  have hnear : r ≤ Neighbor.distance data query hn := by
    simpa [Neighbor.distance, Neighbor.radius] using hball (Neighbor.index data query ⟨0, hn⟩)
  have hdiv :
      r / Real.sqrt (Neighbor.count K n) ≤
        Neighbor.distance data query hn / Real.sqrt (Neighbor.count K n) :=
    div_le_div_of_nonneg_right hnear (Real.sqrt_nonneg _)
  rw [Exact.sigma_prefix data query K hk hquery]
  exact hdiv.trans (Exact.spacing data query K hn hk hquery).1

theorem Exact.sigma_limit {nSeq : ℕ → ℕ}
    (data : ∀ m, ExactDataset (X := X) (n := nSeq m)) (query : X) (K : ℕ)
    (hk : ∀ m, 0 < Neighbor.count K (nSeq m))
    (i : ∀ m, Fin (nSeq m))
    (hi : Filter.Tendsto (fun m ↦ data m (i m)) Filter.atTop (nhds query)) :
    Filter.Tendsto (fun m ↦ Exact.sigma (data m) query K) Filter.atTop (nhds 0) := by
  have hdist : Filter.Tendsto (fun m ↦ dist query (data m (i m))) Filter.atTop (nhds 0) := by
    simpa [dist_comm] using (tendsto_iff_dist_tendsto_zero.1 hi)
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hdist
    (fun m ↦ Exact.sigma_nonneg (data m) query K) ?_
  intro m
  exact Exact.sigma_distance (data := data m) (query := query) (K := K) (hk := hk m) (i := i m)

theorem dist_shift {u : ℕ → X} {x : X}
    (hu : Filter.Tendsto u Filter.atTop (nhds x)) :
    Filter.Tendsto (fun n ↦ dist (u (n + 1)) (u n)) Filter.atTop (nhds 0) := by
  rw [Metric.tendsto_atTop]
  intro ε hε
  rcases Metric.cauchySeq_iff.1 hu.cauchySeq ε hε with ⟨N, hN⟩
  refine ⟨N, ?_⟩
  intro n hn
  simpa [Real.dist_eq, abs_of_nonneg (dist_nonneg)] using
    hN (n + 1) (le_trans hn (Nat.le_succ n)) n hn

theorem limit_shift {u : ℕ → X} {x : X}
    (hu : Filter.Tendsto u Filter.atTop (nhds x)) :
    Filter.Tendsto (fun n ↦ u (n + 1)) Filter.atTop (nhds x) := by
  rw [Metric.tendsto_atTop]
  intro ε hε
  rcases Metric.tendsto_atTop.1 hu ε hε with ⟨N, hN⟩
  refine ⟨N, ?_⟩
  intro n hn
  exact hN (n + 1) (le_trans hn (Nat.le_succ n))

theorem Exact.moving_sneb {nSeq : ℕ → ℕ}
    (data : ∀ m, ExactDataset (X := X) (n := nSeq m)) (query : ℕ → X) (K : ℕ)
    (hK : 0 < K) (i : ∀ m, Fin (nSeq m))
    (hd : Filter.Tendsto (fun m ↦ dist (query m) (data m (i m))) Filter.atTop (nhds 0)) :
    Filter.Tendsto (fun m ↦ Exact.sigma (data m) (query m) K) Filter.atTop (nhds 0) := by
  have hk : ∀ m, 0 < Neighbor.count K (nSeq m) := by
    intro m
    have hn : 0 < nSeq m := by
      exact Nat.pos_of_ne_zero (fun h0 => by simpa [h0] using (i m).2)
    simpa [Neighbor.count] using lt_min hK hn
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hd
    (fun m ↦ Exact.sigma_nonneg (data m) (query m) K) ?_
  intro m
  exact Exact.sigma_distance (data := data m) (query := query m) (K := K) (hk := hk m) (i := i m)

theorem Exact.previous_sneb {nSeq : ℕ → ℕ}
    (data : ∀ m, ExactDataset (X := X) (n := nSeq m)) (query : ℕ → X) {x' : X}
    (K : ℕ) (hK : 0 < K) (hq : Filter.Tendsto query Filter.atTop (nhds x'))
    (i : ∀ m, Fin (nSeq (m + 1)))
    (his : ∀ m, data (m + 1) (i m) = query m) :
    Filter.Tendsto (fun m ↦ Exact.sigma (data (m + 1)) (query (m + 1)) K) Filter.atTop (nhds 0) := by
  have hsucc : Filter.Tendsto (fun m ↦ dist (query (m + 1)) (query m)) Filter.atTop (nhds 0) :=
    dist_shift hq
  have hd :
      Filter.Tendsto (fun m ↦ dist (query (m + 1)) (data (m + 1) (i m))) Filter.atTop (nhds 0) := by
    refine Filter.Tendsto.congr' ?_ hsucc
    filter_upwards with m
    simp [his m]
  simpa using Exact.moving_sneb (data := fun m ↦ data (m + 1)) (query := fun m ↦ query (m + 1))
    (K := K) hK i hd

end NearestNeighbor

end ENN
