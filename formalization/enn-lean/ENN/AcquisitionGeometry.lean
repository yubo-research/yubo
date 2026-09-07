import Mathlib.Data.Real.Basic
import Mathlib.Order.LiminfLimsup
import Mathlib.Topology.Algebra.Order.LiminfLimsup
import Mathlib.Tactic

/-!
# Acquisition geometry for Appendix C.2

This file separates the positive-part scalarization algebra from ENN-specific
nearest-neighbor definitions.
-/

namespace ENN

section AcquisitionGeometry

variable {α : Type*}

/-- Positive part on `ℝ`. -/
def Acq.positive (x : ℝ) : ℝ := max x 0

/-- Positive linear scalarization of mean and uncertainty. -/
def Acq.linear (μ σ : α → ℝ) (lam : ℝ) (x : α) : ℝ :=
  lam * μ x + (1 - lam) * σ x

/-- Appendix C auxiliary acquisition with a generic lower bound `m ≤ μ(x)`. -/
def Acq.auxiliary (μ σ : α → ℝ) (m lam : ℝ) (x : α) : ℝ :=
  lam * Acq.positive (μ x - m) + (1 - lam) * σ x

/-- Proposition 1's positive-part score `A_λ(g,u) = λ g⁺ + (1-λ)u`. -/
def Acq.score (lam g u : ℝ) : ℝ :=
  lam * Acq.positive g + (1 - lam) * u

theorem Acq.positive_eq {x : ℝ} (hx : 0 ≤ x) : Acq.positive x = x := by
  unfold Acq.positive
  exact max_eq_left hx

theorem Acq.positive_zero {x : ℝ} (hx : x ≤ 0) : Acq.positive x = 0 := by
  unfold Acq.positive
  exact max_eq_right hx

theorem Acq.positive_nonneg (x : ℝ) : 0 ≤ Acq.positive x := by
  unfold Acq.positive
  exact le_max_right _ _

theorem Acq.positive_lt {x ε : ℝ} (hε : 0 < ε) (hx : x < ε) : Acq.positive x < ε := by
  by_cases hx0 : x ≤ 0
  · rw [Acq.positive_zero hx0]
    exact hε
  · rw [Acq.positive_eq (le_of_not_ge hx0)]
    exact hx

theorem Acq.linear_offset
    (μ σ : α → ℝ) {m lam : ℝ} {x : α} (hx : m ≤ μ x) :
    Acq.auxiliary μ σ m lam x = Acq.linear μ σ lam x - lam * m := by
  have hnonneg : 0 ≤ μ x - m := sub_nonneg.mpr hx
  unfold Acq.auxiliary Acq.linear
  rw [Acq.positive_eq hnonneg]
  ring

theorem Acq.order_equiv
    (μ σ : α → ℝ) {m lam : ℝ} {x y : α} (hx : m ≤ μ x) (hy : m ≤ μ y) :
    Acq.auxiliary μ σ m lam x ≤ Acq.auxiliary μ σ m lam y ↔
      Acq.linear μ σ lam x ≤ Acq.linear μ σ lam y := by
  rw [Acq.linear_offset μ σ hx,
    Acq.linear_offset μ σ hy]
  exact
    (sub_le_sub_iff_right (lam * m) :
      Acq.linear μ σ lam x - lam * m ≤ Acq.linear μ σ lam y - lam * m ↔
        Acq.linear μ σ lam x ≤ Acq.linear μ σ lam y)

theorem Acq.mean_strict
    (μ σ : α → ℝ) {lam : ℝ} {x y : α} (hlam0 : 0 < lam) (hlam1 : lam < 1)
    (hμ : μ x < μ y) (hσ : σ x ≤ σ y) :
    Acq.linear μ σ lam x < Acq.linear μ σ lam y := by
  unfold Acq.linear
  exact add_lt_add_of_lt_of_le
    (mul_lt_mul_of_pos_left hμ hlam0)
    (mul_le_mul_of_nonneg_left hσ (sub_nonneg.mpr hlam1.le))

theorem Acq.sigma_strict
    (μ σ : α → ℝ) {lam : ℝ} {x y : α} (hlam0 : 0 < lam) (hlam1 : lam < 1)
    (hμ : μ x ≤ μ y) (hσ : σ x < σ y) :
    Acq.linear μ σ lam x < Acq.linear μ σ lam y := by
  unfold Acq.linear
  exact add_lt_add_of_le_of_lt
    (mul_le_mul_of_nonneg_left hμ hlam0.le)
    (mul_lt_mul_of_pos_left hσ (sub_pos.mpr hlam1))

theorem Acq.dominated_lt
    (μ σ : α → ℝ) {lam : ℝ} {x y : α} (hlam0 : 0 < lam) (hlam1 : lam < 1)
    (hμ : μ x ≤ μ y) (hσ : σ x ≤ σ y) (hstrict : μ x < μ y ∨ σ x < σ y) :
    Acq.linear μ σ lam x < Acq.linear μ σ lam y := by
  rcases hstrict with hμlt | hσlt
  · exact Acq.mean_strict μ σ hlam0 hlam1 hμlt hσ
  · exact Acq.sigma_strict μ σ hlam0 hlam1 hμ hσlt

/-- A point is nondominated on `s` if no other point in `s` improves one
coordinate without worsening the other. -/
def Acq.Nondominated (s : Set α) (μ σ : α → ℝ) (x : α) : Prop :=
  x ∈ s ∧ ∀ y ∈ s, μ x ≤ μ y → σ x ≤ σ y → μ y ≤ μ x ∧ σ y ≤ σ x

theorem Acq.linear_nondominated
    (s : Set α) (μ σ : α → ℝ) {lam : ℝ} {x : α}
    (hx : x ∈ s)
    (hmax : ∀ y ∈ s, Acq.linear μ σ lam y ≤ Acq.linear μ σ lam x)
    (hlam0 : 0 < lam) (hlam1 : lam < 1) :
    Acq.Nondominated s μ σ x := by
  refine ⟨hx, ?_⟩
  intro y hy hμ hσ
  constructor
  · by_contra hxy
    have hμlt : μ x < μ y := lt_of_not_ge hxy
    have hscore :
        Acq.linear μ σ lam x < Acq.linear μ σ lam y :=
      Acq.mean_strict μ σ hlam0 hlam1 hμlt hσ
    exact not_lt_of_ge (hmax y hy) hscore
  · by_contra hxy
    have hσlt : σ x < σ y := lt_of_not_ge hxy
    have hscore :
        Acq.linear μ σ lam x < Acq.linear μ σ lam y :=
      Acq.sigma_strict μ σ hlam0 hlam1 hμ hσlt
    exact not_lt_of_ge (hmax y hy) hscore

theorem Acq.auxiliary_nondominated
    (s : Set α) (μ σ : α → ℝ) {m lam : ℝ} {x : α}
    (hm : ∀ y ∈ s, m ≤ μ y)
    (hx : x ∈ s)
    (hmax : ∀ y ∈ s, Acq.auxiliary μ σ m lam y ≤ Acq.auxiliary μ σ m lam x)
    (hlam0 : 0 < lam) (hlam1 : lam < 1) :
    Acq.Nondominated s μ σ x := by
  refine Acq.linear_nondominated s μ σ hx ?_ hlam0 hlam1
  intro y hy
  have hxy := (Acq.order_equiv μ σ (hm y hy) (hm x hx)).1 (hmax y hy)
  simpa using hxy

/-- Appendix C Proposition 1(1), in the eventual form used by the later PBO bridge:
an eventual strictly positive lower bound on the uncertainty term yields an eventual strictly
positive lower bound on the positive-part score. For real-valued `g`, the paper's extra
assumption `liminf g_n > -∞` is unnecessary because `g ↦ g⁺` is nonnegative. -/
theorem Acq.eventually_positive
    {g u : ℕ → ℝ} {lam : ℝ} (hlam0 : 0 < lam) (hlam1 : lam < 1)
    (hu : ∃ b > 0, ∀ᶠ n in Filter.atTop, b ≤ u n) :
    ∃ c > 0, ∀ᶠ n in Filter.atTop, c ≤ Acq.score lam (g n) (u n) := by
  rcases hu with ⟨b, hb0, hu_event⟩
  have hlam_pos : 0 < 1 - lam := sub_pos.mpr hlam1
  refine ⟨(1 - lam) * b, mul_pos hlam_pos hb0, ?_⟩
  filter_upwards [hu_event] with n hn
  have hleft : 0 ≤ lam * Acq.positive (g n) := mul_nonneg hlam0.le (Acq.positive_nonneg _)
  have hright : (1 - lam) * b ≤ (1 - lam) * u n :=
    mul_le_mul_of_nonneg_left hn hlam_pos.le
  unfold Acq.score
  linarith

/-- Appendix C Proposition 1(1), `liminf` corollary under the standard coboundedness
side condition required by mathlib's real-valued `liminf`. -/
theorem Acq.positive_liminf
    {g u : ℕ → ℝ} {lam : ℝ} (hlam0 : 0 < lam) (hlam1 : lam < 1)
    (hu : ∃ b > 0, ∀ᶠ n in Filter.atTop, b ≤ u n)
    (hA_cobdd :
      Filter.IsCoboundedUnder (· ≥ ·) Filter.atTop
        (fun n ↦ Acq.score lam (g n) (u n))) :
    0 < Filter.liminf (fun n ↦ Acq.score lam (g n) (u n)) Filter.atTop := by
  rcases Acq.eventually_positive hlam0 hlam1 hu with ⟨c, hc0, hγ⟩
  have hlim :
      c ≤ Filter.liminf (fun n ↦ Acq.score lam (g n) (u n)) Filter.atTop :=
    Filter.le_liminf_of_le hA_cobdd hγ
  exact hc0.trans_le hlim

/-- Appendix C Proposition 1(2): if the gain term has nonpositive limsup and the uncertainty
term tends to zero, then the positive-part score tends to zero. -/
theorem Acq.vanishes
    {g u : ℕ → ℝ} {lam : ℝ} (hlam0 : 0 < lam) (hlam1 : lam < 1)
    (hu_nonneg : ∀ n, 0 ≤ u n)
    (hg_bdd : Filter.IsBoundedUnder LE.le Filter.atTop g)
    (hg : Filter.limsup g Filter.atTop ≤ 0)
    (hu : Filter.Tendsto u Filter.atTop (nhds 0)) :
    Filter.Tendsto (fun n ↦ Acq.score lam (g n) (u n)) Filter.atTop (nhds 0) := by
  rw [Metric.tendsto_atTop]
  intro ε hε
  have hε2 : 0 < ε / 2 := half_pos hε
  have hg_event : ∀ᶠ n in Filter.atTop, g n < ε / 2 :=
    by
      simpa [zero_add] using
        (Filter.eventually_lt_add_pos_of_limsup_le (x := 0) (u := g)
          hg_bdd (by simpa using hg) hε2)
  have hu_event : ∀ᶠ n in Filter.atTop, |u n| < ε / 2 := by
    simpa [Real.dist_eq] using Metric.tendsto_atTop.1 hu (ε / 2) hε2
  simp only [Filter.eventually_atTop] at hg_event hu_event ⊢
  obtain ⟨Ng, hNg⟩ := hg_event
  obtain ⟨Nu, hNu⟩ := hu_event
  refine ⟨max Ng Nu, ?_⟩
  intro n hn
  have hgn : g n < ε / 2 := hNg n (le_trans (Nat.le_max_left _ _) hn)
  have hun_abs : |u n| < ε / 2 := hNu n (le_trans (Nat.le_max_right _ _) hn)
  have hun : u n < ε / 2 := by
    simpa [abs_of_nonneg (hu_nonneg n)] using hun_abs
  have hpos : Acq.positive (g n) < ε / 2 := Acq.positive_lt hε2 hgn
  have hlam_pos : 0 ≤ lam := hlam0.le
  have hone_pos : 0 ≤ 1 - lam := sub_nonneg.mpr hlam1.le
  have h1 : lam * Acq.positive (g n) ≤ lam * (ε / 2) :=
    mul_le_mul_of_nonneg_left hpos.le hlam_pos
  have h2 : (1 - lam) * u n ≤ (1 - lam) * (ε / 2) :=
    mul_le_mul_of_nonneg_left hun.le hone_pos
  have hA0 : 0 ≤ Acq.score lam (g n) (u n) := by
    unfold Acq.score
    have : 0 ≤ lam * Acq.positive (g n) := mul_nonneg hlam0.le (Acq.positive_nonneg _)
    nlinarith [hu_nonneg n]
  have hsum :
      Acq.score lam (g n) (u n) ≤
        lam * (ε / 2) + (1 - lam) * (ε / 2) := by
    unfold Acq.score
    exact add_le_add h1 h2
  have hright : lam * (ε / 2) + (1 - lam) * (ε / 2) = ε / 2 := by ring
  have hfinal : Acq.score lam (g n) (u n) ≤ ε / 2 := by simpa [hright] using hsum
  have hlt : Acq.score lam (g n) (u n) < ε := by
    exact lt_of_le_of_lt hfinal (by linarith)
  simpa [Real.dist_eq, abs_of_nonneg hA0] using hlt

end AcquisitionGeometry

end ENN
