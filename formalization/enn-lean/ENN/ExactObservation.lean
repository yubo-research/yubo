import Mathlib.Topology.Compactness.Compact
import Mathlib.Topology.MetricSpace.Bounded
import ENN.NearestNeighbor

/-!
# Exact-observation ENN theorems

This file turns the deterministic nearest-neighbor layer into an actual
exact-observation ENN predictor with the sampled-point interpolation branch,
then states the reusable deterministic approximation bounds from Appendix C.1(i).
-/

namespace ENN

section ExactObservation

variable {X : Type*} [MetricSpace X] {n : ℕ}

/-- The exact-observation ENN mean with the sampled-point interpolation branch. -/
noncomputable def Exact.mu (data : ExactDataset (X := X) (n := n)) (f : X → ℝ) (query : X)
    (K : ℕ) : ℝ := by
  classical
  let k := Neighbor.count K n
  exact
    if hquery : Sampled data query then
      f query
    else if hk : 0 < k then
      Exact.prefix_mu data f query hk (Nat.min_le_right K n) hquery
    else 0

theorem Exact.interpolate (data : ExactDataset (X := X) (n := n)) (f : X → ℝ) (query : X)
    (K : ℕ) (hquery : Sampled data query) :
    Exact.mu data f query K = f query := by
  classical
  unfold Exact.mu
  simp [hquery]

theorem Exact.mu_prefix (data : ExactDataset (X := X) (n := n)) (f : X → ℝ)
    (query : X) (K : ℕ) (hk : 0 < Neighbor.count K n) (hquery : ¬ Sampled data query) :
    Exact.mu data f query K = Exact.prefix_mu data f query hk (Nat.min_le_right K n) hquery := by
  classical
  unfold Exact.mu
  simp [hquery, hk]

/-- The largest radius among the first `Kₙ = min K n` ordered neighbors. -/
noncomputable def Neighbor.last_radius (data : ExactDataset (X := X) (n := n)) (query : X)
    (K : ℕ) (hk : 0 < Neighbor.count K n) : ℝ :=
  Neighbor.prefix_radius data query (Nat.min_le_right K n) ⟨Neighbor.count K n - 1, Nat.sub_lt hk Nat.zero_lt_one⟩

/-- Appendix C.1(i), first inequality: exact-observation ENN error is bounded by the weighted
neighbor envelope. -/
theorem Exact.error_envelope (data : ExactDataset (X := X) (n := n)) (f : X → ℝ)
    (query : X) (K : ℕ) (hk : 0 < Neighbor.count K n) (hquery : ¬ Sampled data query)
    (ω : ℝ → ℝ)
    (herr : ∀ i : Fin (Neighbor.count K n),
      |f (data (Neighbor.prefix_index data query (Nat.min_le_right K n) i)) - f query|
        ≤ ω (Neighbor.prefix_radius data query (Nat.min_le_right K n) i)) :
    |Exact.mu data f query K - f query| ≤
      ∑ i, (Exact.weights data query hk (Nat.min_le_right K n) hquery).weight i *
        ω (Neighbor.prefix_radius data query (Nat.min_le_right K n) i) := by
  rw [Exact.mu_prefix data f query K hk hquery]
  exact Exact.prefix_envelope data f query hk (Nat.min_le_right K n) hquery ω herr

/-- Appendix C.1(i), second inequality: a monotone envelope can be evaluated at the largest
neighbor radius. -/
theorem Exact.error_bound (data : ExactDataset (X := X) (n := n)) (f : X → ℝ)
    (query : X) (K : ℕ) (hk : 0 < Neighbor.count K n) (hquery : ¬ Sampled data query)
    (ω : ℝ → ℝ)
    (herr : ∀ i : Fin (Neighbor.count K n),
      |f (data (Neighbor.prefix_index data query (Nat.min_le_right K n) i)) - f query|
        ≤ ω (Neighbor.prefix_radius data query (Nat.min_le_right K n) i))
    (hω : Monotone ω) :
    |Exact.mu data f query K - f query| ≤
      ω (Neighbor.last_radius data query K hk) := by
  rw [Exact.mu_prefix data f query K hk hquery]
  simpa [Neighbor.last_radius] using
    Exact.prefix_bound data f query hk (Nat.min_le_right K n) hquery ω herr hω

/-- Appendix C.1(i), Lipschitz corollary in exact-observation form. -/
theorem Exact.lipschitz_sum (data : ExactDataset (X := X) (n := n)) (f : X → ℝ)
    (query : X) (K : ℕ) (hk : 0 < Neighbor.count K n) (hquery : ¬ Sampled data query)
    {L : ℝ}
    (herr : ∀ i : Fin (Neighbor.count K n),
      |f (data (Neighbor.prefix_index data query (Nat.min_le_right K n) i)) - f query|
        ≤ L * Neighbor.prefix_radius data query (Nat.min_le_right K n) i) :
    |Exact.mu data f query K - f query| ≤
      ∑ i, (Exact.weights data query hk (Nat.min_le_right K n) hquery).weight i *
        (L * Neighbor.prefix_radius data query (Nat.min_le_right K n) i) := by
  rw [Exact.mu_prefix data f query K hk hquery]
  exact Exact.prefix_envelope data f query hk (Nat.min_le_right K n) hquery
    (fun r ↦ L * r) herr

/-- Rearranged Lipschitz corollary with the Lipschitz constant factored out. -/
theorem Exact.lipschitz_bound (data : ExactDataset (X := X) (n := n)) (f : X → ℝ)
    (query : X) (K : ℕ) (hk : 0 < Neighbor.count K n) (hquery : ¬ Sampled data query)
    {L : ℝ}
    (herr : ∀ i : Fin (Neighbor.count K n),
      |f (data (Neighbor.prefix_index data query (Nat.min_le_right K n) i)) - f query|
        ≤ L * Neighbor.prefix_radius data query (Nat.min_le_right K n) i) :
    |Exact.mu data f query K - f query| ≤
      L * ∑ i, (Exact.weights data query hk (Nat.min_le_right K n) hquery).weight i *
        Neighbor.prefix_radius data query (Nat.min_le_right K n) i := by
  calc
    |Exact.mu data f query K - f query| ≤
        ∑ i, (Exact.weights data query hk (Nat.min_le_right K n) hquery).weight i *
          (L * Neighbor.prefix_radius data query (Nat.min_le_right K n) i) :=
      Exact.lipschitz_sum data f query K hk hquery herr
    _ = ∑ i, L * ((Exact.weights data query hk (Nat.min_le_right K n) hquery).weight i *
          Neighbor.prefix_radius data query (Nat.min_le_right K n) i) := by
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = L * ∑ i, (Exact.weights data query hk (Nat.min_le_right K n) hquery).weight i *
          Neighbor.prefix_radius data query (Nat.min_le_right K n) i := by
      rw [Finset.mul_sum]

theorem Exact.bounded_consistency
    {nSeq : ℕ → ℕ} (data : ∀ m, ExactDataset (X := X) (n := nSeq m)) (f : X → ℝ)
    (query : X) (K : ℕ) (hK : 0 < K) (i : ∀ m, Fin (nSeq m))
    (hi : Filter.Tendsto (fun m ↦ data m (i m)) Filter.atTop (nhds query))
    (hf : ContinuousAt f query) {C : ℝ} (hC : ∀ x, |f x - f query| ≤ C) :
    Filter.Tendsto (fun m ↦ Exact.mu (data m) f query K) Filter.atTop (nhds (f query)) := by
  rw [Metric.tendsto_atTop]
  intro ε hε
  have hε2 : 0 < ε / 2 := half_pos hε
  obtain ⟨δ, hδ, hcont⟩ := Metric.continuousAt_iff.mp hf (ε / 2) hε2
  let B : ℝ := max C 1
  have hBpos : 0 < B := by
    dsimp [B]
    exact lt_of_lt_of_le zero_lt_one (le_max_right _ _)
  have hBnonneg : 0 ≤ B := hBpos.le
  have hBbound : ∀ x, |f x - f query| ≤ B := by
    intro x
    exact (hC x).trans (le_max_left _ _)
  have hKreal : (0 : ℝ) < K := by exact_mod_cast hK
  have hηpos : 0 < min δ (ε * δ / (2 * B * K)) := by
    refine lt_min hδ ?_
    positivity
  obtain ⟨N, hN⟩ := Metric.tendsto_atTop.1 hi (min δ (ε * δ / (2 * B * K))) hηpos
  refine ⟨N, fun m hm => ?_⟩
  by_cases hquery : Sampled (data m) query
  · simpa [Exact.interpolate (data := data m) (f := f) (query := query) (K := K) hquery, Real.dist_eq]
      using hε
  · have hn : 0 < nSeq m := by
      exact Nat.pos_of_ne_zero fun h0 => by simpa [h0] using (i m).2
    have hk : 0 < Neighbor.count K (nSeq m) := by
      simpa [Neighbor.count] using lt_min hK hn
    let hkn : Neighbor.count K (nSeq m) ≤ nSeq m := Nat.min_le_right K (nSeq m)
    let d : ℝ := Neighbor.distance (data m) query hn
    have hsample : dist query (data m (i m)) < min δ (ε * δ / (2 * B * K)) := by
      simpa [dist_comm] using hN m hm
    have hdsmall : d < min δ (ε * δ / (2 * B * K)) := by
      exact lt_of_le_of_lt (Neighbor.distance_le (data := data m) (query := query) hn (i m)) hsample
    have hdδ : d ≤ δ := by
      exact (le_of_lt hdsmall).trans (min_le_left _ _)
    have herr :
        ∀ j : Fin (Neighbor.count K (nSeq m)),
          |f (data m (Neighbor.prefix_index (data m) query hkn j)) - f query| ≤
            (if Neighbor.prefix_radius (data m) query hkn j < δ then ε / 2 else B) := by
      intro j
      by_cases hj : Neighbor.prefix_radius (data m) query hkn j < δ
      · have : dist (f (data m (Neighbor.prefix_index (data m) query hkn j))) (f query) < ε / 2 := by
          exact hcont (x := data m (Neighbor.prefix_index (data m) query hkn j)) <|
            by simpa [Neighbor.prefix_radius, dist_comm] using hj
        simpa [Real.dist_eq, hj] using this.le
      · simp [hj, hBbound]
    have hμ :
        |Exact.mu (data m) f query K - f query| ≤
          ∑ j : Fin (Neighbor.count K (nSeq m)),
            (Exact.weights (data m) query hk hkn hquery).weight j *
              (if Neighbor.prefix_radius (data m) query hkn j < δ then ε / 2 else B) := by
      rw [Exact.mu_prefix (data := data m) (f := f) (query := query) (K := K) hk hquery]
      exact Exact.prefix_envelope (data := data m) (f := f) (query := query)
        (hk := hk) (hkn := hkn) (hquery := hquery)
        (ω := fun r ↦ if r < δ then ε / 2 else B) herr
    let w := Exact.weights (data m) query hk hkn hquery
    have hsplit :
        ∑ j : Fin (Neighbor.count K (nSeq m)),
            w.weight j * (if Neighbor.prefix_radius (data m) query hkn j < δ then ε / 2 else B)
          ≤ ∑ j : Fin (Neighbor.count K (nSeq m)), w.weight j * (ε / 2) +
            ∑ j : Fin (Neighbor.count K (nSeq m)),
              w.weight j * (if δ ≤ Neighbor.prefix_radius (data m) query hkn j then B else 0) := by
      calc
        ∑ j : Fin (Neighbor.count K (nSeq m)),
            w.weight j * (if Neighbor.prefix_radius (data m) query hkn j < δ then ε / 2 else B)
          ≤ ∑ j : Fin (Neighbor.count K (nSeq m)),
              (w.weight j * (ε / 2) +
                w.weight j * (if δ ≤ Neighbor.prefix_radius (data m) query hkn j then B else 0)) := by
              refine Finset.sum_le_sum ?_
              intro j _
              by_cases hj : Neighbor.prefix_radius (data m) query hkn j < δ
              · have hnot : ¬ δ ≤ Neighbor.prefix_radius (data m) query hkn j := not_le_of_gt hj
                simp [hj, hnot]
              · have hfar : δ ≤ Neighbor.prefix_radius (data m) query hkn j := le_of_not_gt hj
                have hw : 0 ≤ w.weight j := w.nonnegative j
                simp [hj, hfar]
                positivity
        _ = ∑ j : Fin (Neighbor.count K (nSeq m)), w.weight j * (ε / 2) +
              ∑ j : Fin (Neighbor.count K (nSeq m)),
                w.weight j * (if δ ≤ Neighbor.prefix_radius (data m) query hkn j then B else 0) := by
              rw [Finset.sum_add_distrib]
    have htail_mass :
        ∑ j : Fin (Neighbor.count K (nSeq m)),
            w.weight j * (if δ ≤ Neighbor.prefix_radius (data m) query hkn j then (1 : ℝ) else 0)
          ≤ (Neighbor.count K (nSeq m) : ℝ) * (d / δ) := by
      simpa [w, d] using
        Exact.far_mass (data := data m) (query := query) (hn := hn)
          (hk := hk) (hkn := hkn) (hquery := hquery) hδ hdδ
    have htail :
        ∑ j : Fin (Neighbor.count K (nSeq m)),
            w.weight j * (if δ ≤ Neighbor.prefix_radius (data m) query hkn j then B else 0)
          ≤ B * ((Neighbor.count K (nSeq m) : ℝ) * (d / δ)) := by
      calc
        ∑ j : Fin (Neighbor.count K (nSeq m)),
            w.weight j * (if δ ≤ Neighbor.prefix_radius (data m) query hkn j then B else 0)
          = ∑ j : Fin (Neighbor.count K (nSeq m)),
              B * (w.weight j * (if δ ≤ Neighbor.prefix_radius (data m) query hkn j then (1 : ℝ) else 0)) := by
                apply Finset.sum_congr rfl
                intro j _
                by_cases hfar : δ ≤ Neighbor.prefix_radius (data m) query hkn j <;> simp [hfar, mul_comm]
        _ = B * ∑ j : Fin (Neighbor.count K (nSeq m)),
              w.weight j * (if δ ≤ Neighbor.prefix_radius (data m) query hkn j then (1 : ℝ) else 0) := by
                rw [Finset.mul_sum]
        _ ≤ B * ((Neighbor.count K (nSeq m) : ℝ) * (d / δ)) := by
          exact mul_le_mul_of_nonneg_left htail_mass hBnonneg
    have hmain :
        ∑ j : Fin (Neighbor.count K (nSeq m)),
            w.weight j * (if Neighbor.prefix_radius (data m) query hkn j < δ then ε / 2 else B)
          < ε := by
      have hsumHalf :
          ∑ j : Fin (Neighbor.count K (nSeq m)), w.weight j * (ε / 2) = ε / 2 := by
        calc
          ∑ j : Fin (Neighbor.count K (nSeq m)), w.weight j * (ε / 2)
            = (∑ j : Fin (Neighbor.count K (nSeq m)), w.weight j) * (ε / 2) := by
                simpa using (Finset.sum_mul Finset.univ w.weight (ε / 2)).symm
          _ = ε / 2 := by rw [w.sum_eq_one, one_mul]
      have hk_le : ((Neighbor.count K (nSeq m) : ℕ) : ℝ) ≤ K := by
        exact_mod_cast Nat.min_le_left K (nSeq m)
      have hd_over : d / δ < ε / (2 * B * K) := by
        have : d < ε * δ / (2 * B * K) := lt_of_lt_of_le hdsmall (min_le_right _ _)
        have hrewrite : ε * δ / (2 * B * K) = (ε / (2 * B * K)) * δ := by
          ring
        rw [hrewrite] at this
        exact (div_lt_iff₀ hδ).2 this
      have hK_small : (K : ℝ) * (d / δ) < ε / (2 * B) := by
        calc
          (K : ℝ) * (d / δ) < (K : ℝ) * (ε / (2 * B * K)) := by
            exact mul_lt_mul_of_pos_left hd_over hKreal
          _ = ε / (2 * B) := by
            field_simp [hK.ne', hBpos.ne']
      have hk_small : (Neighbor.count K (nSeq m) : ℝ) * (d / δ) < ε / (2 * B) := by
        exact lt_of_le_of_lt (mul_le_mul_of_nonneg_right hk_le (div_nonneg (Neighbor.distance_nonneg (data m) query hn) hδ.le))
          hK_small
      have htail_small : B * ((Neighbor.count K (nSeq m) : ℝ) * (d / δ)) < ε / 2 := by
        have := mul_lt_mul_of_pos_left hk_small hBpos
        have hrewrite : B * (ε / (2 * B)) = ε / 2 := by
          field_simp [hBpos.ne']
        simpa [mul_assoc, hrewrite] using this
      have hsumTail :
          ∑ j : Fin (Neighbor.count K (nSeq m)),
              w.weight j * (if Neighbor.prefix_radius (data m) query hkn j < δ then ε / 2 else B)
            ≤ ε / 2 + B * ((Neighbor.count K (nSeq m) : ℝ) * (d / δ)) := by
        calc
          ∑ j : Fin (Neighbor.count K (nSeq m)),
              w.weight j * (if Neighbor.prefix_radius (data m) query hkn j < δ then ε / 2 else B)
            ≤ ∑ j : Fin (Neighbor.count K (nSeq m)), w.weight j * (ε / 2) +
                ∑ j : Fin (Neighbor.count K (nSeq m)),
                  w.weight j * (if δ ≤ Neighbor.prefix_radius (data m) query hkn j then B else 0) := hsplit
          _ ≤ ε / 2 + B * ((Neighbor.count K (nSeq m) : ℝ) * (d / δ)) := by
            rw [hsumHalf]
            gcongr
      exact lt_of_le_of_lt hsumTail <| by nlinarith
    simpa [Real.dist_eq] using (lt_of_le_of_lt hμ hmain)

theorem Exact.consistency [CompactSpace X]
    {nSeq : ℕ → ℕ} (data : ∀ m, ExactDataset (X := X) (n := nSeq m)) (f : X → ℝ)
    (query : X) (K : ℕ) (hK : 0 < K) (i : ∀ m, Fin (nSeq m))
    (hi : Filter.Tendsto (fun m ↦ data m (i m)) Filter.atTop (nhds query))
    (hf : Continuous f) :
    Filter.Tendsto (fun m ↦ Exact.mu (data m) f query K) Filter.atTop (nhds (f query)) := by
  rcases Metric.isBounded_range_iff.mp ((isCompact_range hf).isBounded) with ⟨C, hC⟩
  refine Exact.bounded_consistency (C := C)
    (data := data) (f := f) (query := query) (K := K) hK i hi hf.continuousAt ?_
  intro x
  simpa [Real.dist_eq] using hC x query

theorem Exact.local_consistency [CompactSpace X]
    {nSeq : ℕ → ℕ} (data : ∀ m, ExactDataset (X := X) (n := nSeq m)) (f : X → ℝ)
    (query : ℕ → X) {x' : X} (K : ℕ) (hK : 0 < K)
    (hq : Filter.Tendsto query Filter.atTop (nhds x'))
    (i : ∀ m, Fin (nSeq m))
    (hd : Filter.Tendsto (fun m ↦ dist (query m) (data m (i m))) Filter.atTop (nhds 0))
    (hf : Continuous f) :
    Filter.Tendsto (fun m ↦ Exact.mu (data m) f (query m) K) Filter.atTop (nhds (f x')) := by
  rcases Metric.isBounded_range_iff.mp ((isCompact_range hf).isBounded) with ⟨C, hC⟩
  rw [Metric.tendsto_atTop]
  intro ε hε
  have hε4 : 0 < ε / 4 := by positivity
  obtain ⟨δ, hδ, hcont⟩ := Metric.continuousAt_iff.mp hf.continuousAt (ε / 4) hε4
  let r : ℝ := δ / 2
  have hr : 0 < r := by
    dsimp [r]
    positivity
  have hrδ : r ≤ δ := by
    dsimp [r]
    linarith
  let B : ℝ := max C 1
  have hBpos : 0 < B := by
    dsimp [B]
    exact lt_of_lt_of_le zero_lt_one (le_max_right _ _)
  have hBnonneg : 0 ≤ B := hBpos.le
  have hBbound : ∀ x y, dist (f x) (f y) ≤ B := by
    intro x y
    exact (hC x y).trans (le_max_left _ _)
  have hKreal : (0 : ℝ) < K := by
    exact_mod_cast hK
  have hηpos : 0 < min r (ε * r / (4 * B * K)) := by
    refine lt_min hr ?_
    positivity
  obtain ⟨Nq, hNq⟩ := Metric.tendsto_atTop.1 hq r hr
  obtain ⟨Nd, hNd⟩ := Metric.tendsto_atTop.1 hd (min r (ε * r / (4 * B * K))) hηpos
  refine ⟨max Nq Nd, fun m hm => ?_⟩
  have hmq : Nq ≤ m := le_trans (Nat.le_max_left _ _) hm
  have hmd : Nd ≤ m := le_trans (Nat.le_max_right _ _) hm
  have hx_near : dist (query m) x' < r := hNq m hmq
  have hqueryδ : dist (query m) x' < δ := lt_of_lt_of_le hx_near hrδ
  have hx_val : dist (f (query m)) (f x') < ε / 4 := hcont hqueryδ
  by_cases hquery : Sampled (data m) (query m)
  · have hz_val : dist (Exact.mu (data m) f (query m) K) (f x') < ε / 4 := by
      simpa [Exact.interpolate (data := data m) (f := f) (query := query m) (K := K) hquery] using
        hx_val
    have hquarter : ε / 4 < ε := by
      nlinarith
    exact lt_trans hz_val hquarter
  · have hn : 0 < nSeq m := by
      exact Nat.pos_of_ne_zero fun h0 => by simpa [h0] using (i m).2
    have hk : 0 < Neighbor.count K (nSeq m) := by
      simpa [Neighbor.count] using lt_min hK hn
    let hkn : Neighbor.count K (nSeq m) ≤ nSeq m := Nat.min_le_right K (nSeq m)
    let d : ℝ := Neighbor.distance (data m) (query m) hn
    have hsample :
        dist (query m) (data m (i m)) < min r (ε * r / (4 * B * K)) := by
      simpa using hNd m hmd
    have hdsmall : d < min r (ε * r / (4 * B * K)) := by
      exact lt_of_le_of_lt
        (Neighbor.distance_le (data := data m) (query := query m) hn (i m))
        hsample
    have hdr : d ≤ r := by
      exact (le_of_lt hdsmall).trans (min_le_left _ _)
    have herr :
        ∀ j : Fin (Neighbor.count K (nSeq m)),
          |f (data m (Neighbor.prefix_index (data m) (query m) hkn j)) - f (query m)| ≤
            (if Neighbor.prefix_radius (data m) (query m) hkn j < r then ε / 2 else B) := by
      intro j
      by_cases hj : Neighbor.prefix_radius (data m) (query m) hkn j < r
      · have hz_near :
            dist (data m (Neighbor.prefix_index (data m) (query m) hkn j)) x' < δ := by
          have hprefix :
              dist (data m (Neighbor.prefix_index (data m) (query m) hkn j)) (query m) < r := by
            simpa [Neighbor.prefix_radius, dist_comm] using hj
          calc
            dist (data m (Neighbor.prefix_index (data m) (query m) hkn j)) x'
              ≤ dist (data m (Neighbor.prefix_index (data m) (query m) hkn j)) (query m) +
                  dist (query m) x' :=
                    dist_triangle (data m (Neighbor.prefix_index (data m) (query m) hkn j)) (query m) x'
            _ < r + r := add_lt_add hprefix hx_near
            _ = δ := by
              dsimp [r]
              ring
        have hz_val :
            dist (f (data m (Neighbor.prefix_index (data m) (query m) hkn j))) (f x') < ε / 4 :=
          hcont hz_near
        have hpair :
            dist (f (data m (Neighbor.prefix_index (data m) (query m) hkn j))) (f (query m)) < ε / 2 := by
          calc
            dist (f (data m (Neighbor.prefix_index (data m) (query m) hkn j))) (f (query m))
              ≤ dist (f (data m (Neighbor.prefix_index (data m) (query m) hkn j))) (f x') +
                  dist (f (query m)) (f x') :=
                    by
                      simpa [dist_comm] using
                        (dist_triangle (f (data m (Neighbor.prefix_index (data m) (query m) hkn j))) (f x')
                          (f (query m)))
            _ < ε / 4 + ε / 4 := add_lt_add hz_val hx_val
            _ = ε / 2 := by ring
        simpa [Real.dist_eq, hj] using hpair.le
      · have hfar :
            dist (f (data m (Neighbor.prefix_index (data m) (query m) hkn j))) (f (query m)) ≤ B :=
          hBbound _ _
        simpa [Real.dist_eq, hj] using hfar
    have hμ :
        |Exact.mu (data m) f (query m) K - f (query m)| ≤
          ∑ j : Fin (Neighbor.count K (nSeq m)),
            (Exact.weights (data m) (query m) hk hkn hquery).weight j *
              (if Neighbor.prefix_radius (data m) (query m) hkn j < r then ε / 2 else B) := by
      rw [Exact.mu_prefix (data := data m) (f := f) (query := query m) (K := K)
        hk hquery]
      exact Exact.prefix_envelope (data := data m) (f := f) (query := query m)
        (hk := hk) (hkn := hkn) (hquery := hquery)
        (ω := fun s ↦ if s < r then ε / 2 else B) herr
    let w := Exact.weights (data m) (query m) hk hkn hquery
    have hsplit :
        ∑ j : Fin (Neighbor.count K (nSeq m)),
            w.weight j * (if Neighbor.prefix_radius (data m) (query m) hkn j < r then ε / 2 else B)
          ≤ ∑ j : Fin (Neighbor.count K (nSeq m)), w.weight j * (ε / 2) +
            ∑ j : Fin (Neighbor.count K (nSeq m)),
              w.weight j * (if r ≤ Neighbor.prefix_radius (data m) (query m) hkn j then B else 0) := by
      calc
        ∑ j : Fin (Neighbor.count K (nSeq m)),
            w.weight j * (if Neighbor.prefix_radius (data m) (query m) hkn j < r then ε / 2 else B)
          ≤ ∑ j : Fin (Neighbor.count K (nSeq m)),
              (w.weight j * (ε / 2) +
                w.weight j * (if r ≤ Neighbor.prefix_radius (data m) (query m) hkn j then B else 0)) := by
              refine Finset.sum_le_sum ?_
              intro j _
              by_cases hj : Neighbor.prefix_radius (data m) (query m) hkn j < r
              · have hnot : ¬ r ≤ Neighbor.prefix_radius (data m) (query m) hkn j := not_le_of_gt hj
                simp [hj, hnot]
              · have hfar : r ≤ Neighbor.prefix_radius (data m) (query m) hkn j := le_of_not_gt hj
                have hw : 0 ≤ w.weight j := w.nonnegative j
                simp [hj, hfar]
                positivity
        _ = ∑ j : Fin (Neighbor.count K (nSeq m)), w.weight j * (ε / 2) +
              ∑ j : Fin (Neighbor.count K (nSeq m)),
                w.weight j * (if r ≤ Neighbor.prefix_radius (data m) (query m) hkn j then B else 0) := by
              rw [Finset.sum_add_distrib]
    have htail_mass :
        ∑ j : Fin (Neighbor.count K (nSeq m)),
            w.weight j * (if r ≤ Neighbor.prefix_radius (data m) (query m) hkn j then (1 : ℝ) else 0)
          ≤ (Neighbor.count K (nSeq m) : ℝ) * (d / r) := by
      simpa [w, d] using
        Exact.far_mass (data := data m) (query := query m) (hn := hn)
          (hk := hk) (hkn := hkn) (hquery := hquery) hr hdr
    have htail :
        ∑ j : Fin (Neighbor.count K (nSeq m)),
            w.weight j * (if r ≤ Neighbor.prefix_radius (data m) (query m) hkn j then B else 0)
          ≤ B * ((Neighbor.count K (nSeq m) : ℝ) * (d / r)) := by
      calc
        ∑ j : Fin (Neighbor.count K (nSeq m)),
            w.weight j * (if r ≤ Neighbor.prefix_radius (data m) (query m) hkn j then B else 0)
          = ∑ j : Fin (Neighbor.count K (nSeq m)),
              B * (w.weight j * (if r ≤ Neighbor.prefix_radius (data m) (query m) hkn j then (1 : ℝ) else 0)) := by
                apply Finset.sum_congr rfl
                intro j _
                by_cases hfar : r ≤ Neighbor.prefix_radius (data m) (query m) hkn j <;> simp [hfar, mul_comm]
        _ = B * ∑ j : Fin (Neighbor.count K (nSeq m)),
              w.weight j * (if r ≤ Neighbor.prefix_radius (data m) (query m) hkn j then (1 : ℝ) else 0) := by
                rw [Finset.mul_sum]
        _ ≤ B * ((Neighbor.count K (nSeq m) : ℝ) * (d / r)) := by
          exact mul_le_mul_of_nonneg_left htail_mass hBnonneg
    have hmain :
        ∑ j : Fin (Neighbor.count K (nSeq m)),
            w.weight j * (if Neighbor.prefix_radius (data m) (query m) hkn j < r then ε / 2 else B)
          < 3 * ε / 4 := by
      have hsumHalf :
          ∑ j : Fin (Neighbor.count K (nSeq m)), w.weight j * (ε / 2) = ε / 2 := by
        calc
          ∑ j : Fin (Neighbor.count K (nSeq m)), w.weight j * (ε / 2)
            = (∑ j : Fin (Neighbor.count K (nSeq m)), w.weight j) * (ε / 2) := by
                simpa using (Finset.sum_mul Finset.univ w.weight (ε / 2)).symm
          _ = ε / 2 := by rw [w.sum_eq_one, one_mul]
      have hk_le : ((Neighbor.count K (nSeq m) : ℕ) : ℝ) ≤ K := by
        exact_mod_cast Nat.min_le_left K (nSeq m)
      have hd_over : d / r < ε / (4 * B * K) := by
        have : d < ε * r / (4 * B * K) := lt_of_lt_of_le hdsmall (min_le_right _ _)
        have hrewrite : ε * r / (4 * B * K) = (ε / (4 * B * K)) * r := by
          ring
        rw [hrewrite] at this
        exact (div_lt_iff₀ hr).2 this
      have hK_small : (K : ℝ) * (d / r) < ε / (4 * B) := by
        calc
          (K : ℝ) * (d / r) < (K : ℝ) * (ε / (4 * B * K)) := by
            exact mul_lt_mul_of_pos_left hd_over hKreal
          _ = ε / (4 * B) := by
            field_simp [hK.ne', hBpos.ne']
      have hk_small : (Neighbor.count K (nSeq m) : ℝ) * (d / r) < ε / (4 * B) := by
        exact lt_of_le_of_lt
          (mul_le_mul_of_nonneg_right hk_le
            (div_nonneg (Neighbor.distance_nonneg (data m) (query m) hn) hr.le))
          hK_small
      have htail_small : B * ((Neighbor.count K (nSeq m) : ℝ) * (d / r)) < ε / 4 := by
        have := mul_lt_mul_of_pos_left hk_small hBpos
        have hrewrite : B * (ε / (4 * B)) = ε / 4 := by
          field_simp [hBpos.ne']
        simpa [mul_assoc, hrewrite] using this
      have hsumTail :
          ∑ j : Fin (Neighbor.count K (nSeq m)),
              w.weight j * (if Neighbor.prefix_radius (data m) (query m) hkn j < r then ε / 2 else B)
            ≤ ε / 2 + B * ((Neighbor.count K (nSeq m) : ℝ) * (d / r)) := by
        calc
          ∑ j : Fin (Neighbor.count K (nSeq m)),
              w.weight j * (if Neighbor.prefix_radius (data m) (query m) hkn j < r then ε / 2 else B)
            ≤ ∑ j : Fin (Neighbor.count K (nSeq m)), w.weight j * (ε / 2) +
                ∑ j : Fin (Neighbor.count K (nSeq m)),
                  w.weight j * (if r ≤ Neighbor.prefix_radius (data m) (query m) hkn j then B else 0) := hsplit
          _ ≤ ε / 2 + B * ((Neighbor.count K (nSeq m) : ℝ) * (d / r)) := by
            rw [hsumHalf]
            gcongr
      exact lt_of_le_of_lt hsumTail <| by nlinarith
    have hmuQuery : dist (Exact.mu (data m) f (query m) K) (f (query m)) < 3 * ε / 4 := by
      simpa [Real.dist_eq] using (lt_of_le_of_lt hμ hmain)
    calc
      dist (Exact.mu (data m) f (query m) K) (f x')
        ≤ dist (Exact.mu (data m) f (query m) K) (f (query m)) + dist (f (query m)) (f x') :=
          dist_triangle (Exact.mu (data m) f (query m) K) (f (query m)) (f x')
      _ < 3 * ε / 4 + ε / 4 := add_lt_add hmuQuery hx_val
      _ = ε := by ring

theorem Exact.previous_consistency [CompactSpace X]
    {nSeq : ℕ → ℕ} (data : ∀ m, ExactDataset (X := X) (n := nSeq m)) (f : X → ℝ)
    (query : ℕ → X) {x' : X} (K : ℕ) (hK : 0 < K)
    (hq : Filter.Tendsto query Filter.atTop (nhds x'))
    (i : ∀ m, Fin (nSeq (m + 1)))
    (his : ∀ m, data (m + 1) (i m) = query m)
    (hf : Continuous f) :
    Filter.Tendsto (fun m ↦ Exact.mu (data (m + 1)) f (query (m + 1)) K) Filter.atTop (nhds (f x')) := by
  have hqsucc : Filter.Tendsto (fun m ↦ query (m + 1)) Filter.atTop (nhds x') :=
    limit_shift hq
  have hd :
      Filter.Tendsto (fun m ↦ dist (query (m + 1)) (data (m + 1) (i m))) Filter.atTop (nhds 0) := by
    refine Filter.Tendsto.congr' ?_ (dist_shift hq)
    filter_upwards with m
    simp [his m]
  simpa using Exact.local_consistency (data := fun m ↦ data (m + 1)) (f := f)
    (query := fun m ↦ query (m + 1)) (x' := x') (K := K) hK hqsucc i hd hf

end ExactObservation

end ENN
