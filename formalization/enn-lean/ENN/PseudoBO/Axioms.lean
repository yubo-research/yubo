import ENN.ExactObservation
import ENN.AcquisitionGeometry

/-!
# PseudoBO-style axioms and the current ENN bridge

This file packages the moving-query properties proved for exact-observation ENN
into reusable predicates mirroring the PseudoBO assumptions discussed in the
Appendix C audit.
-/

namespace ENN
namespace PseudoBO

section Core

variable {X : Type*} [MetricSpace X]

/-- A moving query sequence approaches the current dataset if it has a sampled
point in each stage whose distance to the query tends to zero. This is the
common side condition behind the moving-query local-consistency and SNEB
interfaces. -/
def NearData {nSeq : ℕ → ℕ}
    (data : ∀ n, ExactDataset (X := X) (n := nSeq n)) (query : ℕ → X) : Prop :=
  ∃ i : ∀ n, Fin (nSeq n),
    Filter.Tendsto (fun n ↦ dist (query n) (data n (i n))) Filter.atTop (nhds 0)

/-- PBO Assumption 3.3 style local consistency, phrased on moving queries that
approach the current exact dataset. -/
def Consistent {nSeq : ℕ → ℕ}
    (data : ∀ n, ExactDataset (X := X) (n := nSeq n))
    (μ : ℕ → X → ℝ) (f : X → ℝ) : Prop :=
  ∀ ⦃query : ℕ → X⦄ ⦃x' : X⦄,
    Filter.Tendsto query Filter.atTop (nhds x') →
    NearData data query →
    Filter.Tendsto (fun n ↦ μ n (query n)) Filter.atTop (nhds (f x'))

/-- PBO Assumption 3.4 style SNEB, phrased on moving queries that approach the
current exact dataset. -/
def SNEB {nSeq : ℕ → ℕ}
    (data : ∀ n, ExactDataset (X := X) (n := nSeq n))
    (σ : ℕ → X → ℝ) : Prop :=
  ∀ ⦃query : ℕ → X⦄,
    NearData data query →
    Filter.Tendsto (fun n ↦ σ n (query n)) Filter.atTop (nhds 0)

/-- PBO Assumption 3.1 style evaluation worthiness: the acquisition vanishes on
moving queries that approach the current exact dataset. -/
def Worthwhile {nSeq : ℕ → ℕ}
    (data : ∀ n, ExactDataset (X := X) (n := nSeq n))
    (W : ℕ → X → ℝ) : Prop :=
  ∀ ⦃query : ℕ → X⦄,
    NearData data query →
    Filter.Tendsto (fun n ↦ W n (query n)) Filter.atTop (nhds 0)

/-- Appendix C Proposition 1 / PBO Assumption 3.5 style improvement property,
formalized for a real-valued score `A lam g u`.

Part (1) is packaged in the eventual-positive form that the current bridge uses.
Part (2) is the `→ 0` statement from Proposition 1. -/
def Improvement (A : ℝ → ℝ → ℝ → ℝ) : Prop :=
  (∀ {lam : ℝ} {g u : ℕ → ℝ},
      0 < lam → lam < 1 →
      (∃ b > 0, ∀ᶠ n in Filter.atTop, b ≤ u n) →
      ∃ c > 0, ∀ᶠ n in Filter.atTop, c ≤ A lam (g n) (u n)) ∧
  (∀ {lam : ℝ} {g u : ℕ → ℝ},
      0 < lam → lam < 1 →
      (∀ n, 0 ≤ u n) →
      Filter.IsBoundedUnder LE.le Filter.atTop g →
      Filter.limsup g Filter.atTop ≤ 0 →
      Filter.Tendsto u Filter.atTop (nhds 0) →
      Filter.Tendsto (fun n ↦ A lam (g n) (u n)) Filter.atTop (nhds 0))

/-- A gain term vanishes along explored sequences if, along every moving query
that approaches the dataset, it has nonpositive limsup. The bounded-above side
condition is included explicitly because real-valued `limsup` in mathlib needs
it. -/
def VanishingGain {nSeq : ℕ → ℕ}
    (data : ∀ n, ExactDataset (X := X) (n := nSeq n))
    (g : ℕ → X → ℝ) : Prop :=
  ∀ ⦃query : ℕ → X⦄, NearData data query →
    Filter.IsBoundedUnder LE.le Filter.atTop (fun n ↦ g n (query n)) ∧
      Filter.limsup (fun n ↦ g n (query n)) Filter.atTop ≤ 0

/-- If a score satisfies the improvement property, the uncertainty term is SNEB,
and the gain term has nonpositive limsup along explored sequences, then the
resulting acquisition is evaluation-worthy. This is the exact bridge point where
the Appendix C baseline issue appears. -/
theorem worthy_of_gain
    {nSeq : ℕ → ℕ} {data : ∀ n, ExactDataset (X := X) (n := nSeq n)}
    {A : ℝ → ℝ → ℝ → ℝ} {g σ : ℕ → X → ℝ} {lam : ℝ}
    (hA : Improvement A) (hlam0 : 0 < lam) (hlam1 : lam < 1)
    (hσ_nonneg : ∀ n x, 0 ≤ σ n x) (hσ : SNEB data σ)
    (hg : VanishingGain data g) :
    Worthwhile data (fun n x ↦ A lam (g n x) (σ n x)) := by
  intro query hnear
  rcases hg hnear with ⟨hg_bdd, hg_lim⟩
  exact hA.2 hlam0 hlam1 (fun n ↦ hσ_nonneg n (query n)) hg_bdd hg_lim (hσ hnear)

end Core

section ENNBridge

variable {X : Type*} [MetricSpace X]

/-- Exact-observation ENN satisfies the moving-query local-consistency interface
from the current PseudoBO axiom layer. This is stronger than the fixed-query
Appendix C.1 statement. -/
theorem local_consistency [CompactSpace X]
    {nSeq : ℕ → ℕ} (data : ∀ n, ExactDataset (X := X) (n := nSeq n))
    (f : X → ℝ) (K : ℕ) (hK : 0 < K) (hf : Continuous f) :
    Consistent data (fun n x ↦ Exact.mu (data n) f x K) f := by
  intro query x' hq hnear
  rcases hnear with ⟨i, hi⟩
  exact Exact.local_consistency (data := data) (f := f) (query := query)
    (x' := x') (K := K) hK hq i hi hf

/-- Exact-observation ENN satisfies the moving-query SNEB interface from the
current PseudoBO axiom layer. -/
theorem sneb
    {nSeq : ℕ → ℕ} (data : ∀ n, ExactDataset (X := X) (n := nSeq n))
    (K : ℕ) (hK : 0 < K) :
    SNEB data (fun n x ↦ Exact.sigma (data n) x K) := by
  intro query hnear
  rcases hnear with ⟨i, hi⟩
  exact Exact.moving_sneb (data := data) (query := query) (K := K) hK i hi

/-- The Appendix C / Proposition 1 acquisition score satisfies the current
improvement-property interface. -/
theorem score_improves :
    Improvement Acq.score :=
  ⟨
    fun hlam0 hlam1 hu =>
      Acq.eventually_positive hlam0 hlam1 hu,
    fun hlam0 hlam1 hu_nonneg hg_bdd hg hu =>
      Acq.vanishes hlam0 hlam1 hu_nonneg hg_bdd hg hu
  ⟩

end ENNBridge

end PseudoBO
end ENN
