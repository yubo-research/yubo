import ENN.ExactObservation
import ENN.WeightedKNN

/-!
# Ordered weighted-kNN bridge for Appendix C.3

This file connects the deterministic ordered-neighbor indexing from
`ENN.NearestNeighbor` / `ENN.ExactObservation` to the permutation-invariant
conditional weighted-kNN theorems in `ENN.WeightedKNN`.
-/

namespace ENN

open MeasureTheory

section OrderedWeightedKNN

variable {X Ω : Type*} [MetricSpace X] {n : ℕ}
variable {m mΩ : MeasurableSpace Ω} {μ : @Measure Ω mΩ}
variable [IsFiniteMeasure μ]

/-- The concrete ENN inverse-variance weights on the first `Kₙ = min K n`
ordered neighbors. The noise-scale input may be random, while the ordered
radius family is deterministic. -/
noncomputable def Neighbor.noisy_weights
    (data : ExactDataset (X := X) (n := n)) (query : X) (K : ℕ)
    (hk : 0 < Neighbor.count K n) (s : Fin n → Ω → ℝ)
    {α γ : ℝ} (hα : 0 < α) (hγ : 0 ≤ γ) (ω : Ω) :
    Weights (Fin (Neighbor.count K n)) :=
  by
    let _ : Nonempty (Fin (Neighbor.count K n)) := ⟨⟨0, hk⟩⟩
    exact Variance.weights α γ
      (fun j ↦ s (Neighbor.prefix_index data query (Nat.min_le_right K n) j) ω)
      (fun j ↦ Neighbor.prefix_radius data query (Nat.min_le_right K n) j)
      hα hγ

/-- The corresponding weighted-kNN estimator on the first `Kₙ = min K n`
ordered neighbors. -/
noncomputable def Neighbor.noisy_mean
    (data : ExactDataset (X := X) (n := n)) (query : X) (K : ℕ)
    (hk : 0 < Neighbor.count K n) (s : Fin n → Ω → ℝ) (y : Fin n → Ω → ℝ)
    {α γ : ℝ} (hα : 0 < α) (hγ : 0 ≤ γ) : Ω → ℝ :=
  fun ω ↦
    Estimator.mean
      (Neighbor.noisy_weights data query K hk s hα hγ ω)
      (fun i ↦ y (Neighbor.prefix_index data query (Nat.min_le_right K n) i) ω)

/-- Appendix C.3 in the paper's ordered-neighbor indexing style: the conditional
weighted-kNN MSE bound for the first `Kₙ = min K n` ordered neighbors, using
the concrete ENN inverse-variance weight construction. -/
theorem Noise.mse_ordered
    (hm : m ≤ mΩ)
    (data : ExactDataset (X := X) (n := n)) (f : X → ℝ) (query : X) (K : ℕ)
    (hk : 0 < Neighbor.count K n) (s ε : Fin n → Ω → ℝ)
    {α γ L σ2 : ℝ}
    (hs_meas : ∀ i, StronglyMeasurable[m] (s i))
    (hα : 0 < α) (hγ : 0 ≤ γ)
    (hL : 0 ≤ L)
    (hLip : ∀ i : Fin (Neighbor.count K n),
      |f (data (Neighbor.prefix_index data query (Nat.min_le_right K n) i)) - f query|
        ≤ L * Neighbor.prefix_radius data query (Nat.min_le_right K n) i)
    (hε2 : ∀ i, MemLp (ε i) 2 μ)
    (hε_zero : ∀ i, μ[ε i | m] =ᵐ[μ] 0)
    (hε_orth : ∀ i j, i ≠ j → μ[fun ω ↦ ε i ω * ε j ω | m] =ᵐ[μ] 0)
    (hε_sq : ∀ i, μ[fun ω ↦ (ε i ω) ^ 2 | m] ≤ᵐ[μ] fun _ ↦ σ2) :
    μ[fun ω ↦
      (Neighbor.noisy_mean data query K hk s
        (fun i ω ↦ f (data i) + ε i ω) hα hγ ω - f query) ^ 2 | m]
      ≤ᵐ[μ] fun ω ↦
        (L * Neighbor.last_radius data query K hk) ^ 2 +
          σ2 * ∑ i,
            ((Neighbor.noisy_weights data query K hk s hα hγ ω).weight i) ^ 2 := by
  let k := Neighbor.count K n
  let hkn : k ≤ n := Nat.min_le_right K n
  let z : Fin k → X := data ∘ Neighbor.prefix_index data query hkn
  let sp : Fin k → Ω → ℝ := fun i ω ↦ s (Neighbor.prefix_index data query hkn i) ω
  let rp : Fin k → Ω → ℝ := fun i _ ↦ Neighbor.prefix_radius data query hkn i
  let εp : Fin k → Ω → ℝ := fun i ω ↦ ε (Neighbor.prefix_index data query hkn i) ω
  let last : Fin k := ⟨k - 1, Nat.sub_lt hk Nat.zero_lt_one⟩
  let _ : Nonempty (Fin k) := ⟨⟨0, hk⟩⟩
  have hsp_meas : ∀ i, StronglyMeasurable[m] (sp i) := by
    intro i
    simpa [sp] using hs_meas (Neighbor.prefix_index data query hkn i)
  have hrp_meas : ∀ i, StronglyMeasurable[m] (rp i) := by
    intro i
    simpa [rp] using (stronglyMeasurable_const : StronglyMeasurable[m] fun _ : Ω ↦ Neighbor.prefix_radius data query hkn i)
  have hLip' : ∀ i, |f (z i) - f query| ≤ L * dist (z i) query := by
    intro i
    simpa [z, hkn, Neighbor.prefix_radius, dist_comm] using hLip i
  have hρ : ∀ i, dist (z i) query ≤ Neighbor.prefix_radius data query hkn last := by
    intro i
    have hi_last : i ≤ last := by
      exact Fin.le_iff_val_le_val.mpr (Nat.le_pred_of_lt i.isLt)
    simpa [z, hkn, Neighbor.prefix_radius, dist_comm, last] using
      (Neighbor.prefix_monotone data query hkn hi_last)
  have hεp2 : ∀ i, MemLp (εp i) 2 μ := by
    intro i
    simpa [εp] using hε2 (Neighbor.prefix_index data query hkn i)
  have hεp_zero : ∀ i, μ[εp i | m] =ᵐ[μ] 0 := by
    intro i
    simpa [εp] using hε_zero (Neighbor.prefix_index data query hkn i)
  have hεp_orth : ∀ i j, i ≠ j → μ[fun ω ↦ εp i ω * εp j ω | m] =ᵐ[μ] 0 := by
    intro i j hij
    exact hε_orth _ _ <| fun h => hij <| Neighbor.prefix_injective data query hkn h
  have hεp_sq : ∀ i, μ[fun ω ↦ (εp i ω) ^ 2 | m] ≤ᵐ[μ] fun _ ↦ σ2 := by
    intro i
    simpa [εp] using hε_sq (Neighbor.prefix_index data query hkn i)
  simpa [Neighbor.noisy_mean, Neighbor.noisy_weights, Estimator.mean,
    z, sp, rp, εp, Neighbor.last_radius, hkn, last] using
    (Noise.mse_weights
      (hm := hm) (s := sp) (r := rp) (f := f) (z := z) (ε := εp) (x := query)
      (hs_meas := hsp_meas) (hr_meas := hrp_meas)
      (hα := hα) (hγ := hγ)
      (hL := hL) (hLip := hLip') (hρ := hρ)
      (hε2 := hεp2) (hε_zero := hεp_zero)
      (hε_orth := hεp_orth) (hε_sq := hεp_sq))

end OrderedWeightedKNN

end ENN
