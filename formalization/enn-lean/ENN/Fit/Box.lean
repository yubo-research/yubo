import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Topology.Order.Compact
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.MeasureTheory.Function.ConvergenceInMeasure
import Mathlib.MeasureTheory.Function.L2Space
import ENN.WeightedKNN

/-!
# Variance parameters and their compact box
-/

namespace ENN

open Filter MeasureTheory

section Parameters

variable {ι : Type*}

/-- The Appendix C.4 variance-scale parameter pair `φ = (α, γ) = (s₀², cₑ)`. -/
structure Fit.Params where
  alpha : ℝ
  gamma : ℝ

/-- The `ℓ¹` distance on variance-scale parameters. -/
def Fit.distance (φ ψ : Fit.Params) : ℝ :=
  |φ.alpha - ψ.alpha| + |φ.gamma - ψ.gamma|

theorem Fit.distance_nonneg (φ ψ : Fit.Params) :
    0 ≤ Fit.distance φ ψ := by
  unfold Fit.distance
  positivity

/-- A compact axis-aligned parameter box for the Appendix C.4 variance-scale
coordinates. -/
structure Fit.Box where
  alpha_lo : ℝ
  alpha_hi : ℝ
  gamma_lo : ℝ
  gamma_hi : ℝ
  alpha_pos : 0 < alpha_lo
  alpha_bounds : alpha_lo ≤ alpha_hi
  gamma_nonneg : 0 ≤ gamma_lo
  gamma_bounds : gamma_lo ≤ gamma_hi

/-- Membership in a variance-scale parameter box. -/
def Fit.InBox (B : Fit.Box) (φ : Fit.Params) : Prop :=
  B.alpha_lo ≤ φ.alpha ∧ φ.alpha ≤ B.alpha_hi ∧
    B.gamma_lo ≤ φ.gamma ∧ φ.gamma ≤ B.gamma_hi

theorem Fit.InBox.alpha_lower {B : Fit.Box} {φ : Fit.Params}
    (h : Fit.InBox B φ) :
    B.alpha_lo ≤ φ.alpha :=
  h.1

theorem Fit.InBox.alpha_upper {B : Fit.Box} {φ : Fit.Params}
    (h : Fit.InBox B φ) :
    φ.alpha ≤ B.alpha_hi :=
  h.2.1

theorem Fit.InBox.gamma_lower {B : Fit.Box} {φ : Fit.Params}
    (h : Fit.InBox B φ) :
    B.gamma_lo ≤ φ.gamma :=
  h.2.2.1

theorem Fit.InBox.gamma_upper {B : Fit.Box} {φ : Fit.Params}
    (h : Fit.InBox B φ) :
    φ.gamma ≤ B.gamma_hi :=
  h.2.2.2

theorem Fit.InBox.gamma_nonneg {B : Fit.Box} {φ : Fit.Params}
    (h : Fit.InBox B φ) :
    0 ≤ φ.gamma :=
  le_trans B.gamma_nonneg h.gamma_lower

end Parameters

section ParameterBoxArgmax

/-- View a variance-parameter pair as a point in `ℝ × ℝ`. -/
def Fit.Params.toProd (φ : Fit.Params) : ℝ × ℝ :=
  (φ.alpha, φ.gamma)

/-- Build a variance-parameter pair from a point in `ℝ × ℝ`. -/
def Fit.Params.ofProd (p : ℝ × ℝ) : Fit.Params where
  alpha := p.1
  gamma := p.2

@[simp] theorem Fit.Params.ofProd_fst (p : ℝ × ℝ) :
    (Fit.Params.ofProd p).alpha = p.1 :=
  rfl

@[simp] theorem Fit.Params.ofProd_snd (p : ℝ × ℝ) :
    (Fit.Params.ofProd p).gamma = p.2 :=
  rfl

@[simp] theorem Fit.Params.toProd_ofProd (p : ℝ × ℝ) :
    (Fit.Params.ofProd p).toProd = p := by
  cases p
  rfl

@[simp] theorem Fit.Params.ofProd_toProd (φ : Fit.Params) :
    Fit.Params.ofProd φ.toProd = φ := by
  cases φ
  rfl

/-- The rectangular parameter box, viewed as a subset of `ℝ × ℝ`. -/
def Fit.region (B : Fit.Box) : Set (ℝ × ℝ) :=
  Set.Icc (B.alpha_lo, B.gamma_lo) (B.alpha_hi, B.gamma_hi)

/-- The parameter box, viewed as a subset of `Fit.Params`. -/
def Fit.domain (B : Fit.Box) : Set Fit.Params :=
  {φ | Fit.InBox B φ}

theorem Fit.mem_domain (B : Fit.Box) (φ : Fit.Params) :
    φ ∈ Fit.domain B ↔ Fit.InBox B φ :=
  Iff.rfl

theorem Fit.mem_region (B : Fit.Box) (φ : Fit.Params) :
    Fit.InBox B φ ↔ φ.toProd ∈ Fit.region B := by
  constructor
  · intro h
    exact ⟨⟨h.1, h.2.2.1⟩, h.2.1, h.2.2.2⟩
  · intro h
    exact ⟨h.1.1, h.2.1, h.1.2, h.2.2⟩

theorem Fit.region_nonempty (B : Fit.Box) :
    (Fit.region B).Nonempty := by
  refine ⟨(B.alpha_lo, B.gamma_lo), ?_⟩
  simp [Fit.region, B.alpha_bounds, B.gamma_bounds]

theorem Fit.domain_nonempty (B : Fit.Box) :
    (Fit.domain B).Nonempty := by
  refine ⟨⟨B.alpha_lo, B.gamma_lo⟩, ?_⟩
  exact ⟨le_rfl, B.alpha_bounds, le_rfl, B.gamma_bounds⟩

/-- The parameter box is compact when viewed as a closed rectangle in `ℝ × ℝ`. -/
theorem Fit.region_compact (B : Fit.Box) :
    IsCompact (Fit.region B) := by
  simpa [Fit.region] using
    (isCompact_Icc : IsCompact (Set.Icc (B.alpha_lo, B.gamma_lo) (B.alpha_hi, B.gamma_hi)))

/-- A closed axis-aligned subbox inside a fixed parameter box. -/
structure Fit.Cell (B : Fit.Box) where
  alpha_lo : ℝ
  alpha_hi : ℝ
  gamma_lo : ℝ
  gamma_hi : ℝ
  alpha_lower : B.alpha_lo ≤ alpha_lo
  alpha_bounds : alpha_lo ≤ alpha_hi
  alpha_upper : alpha_hi ≤ B.alpha_hi
  gamma_lower : B.gamma_lo ≤ gamma_lo
  gamma_bounds : gamma_lo ≤ gamma_hi
  gamma_upper : gamma_hi ≤ B.gamma_hi


/-- Any continuous real-valued objective on the parameter box achieves a
maximum. The continuity is stated on the transported function on `ℝ × ℝ` so no
topological structure is needed on `Fit.Params` itself. -/
theorem Fit.exists_max
    (B : Fit.Box) {f : Fit.Params → ℝ}
    (hf : Continuous fun p : ℝ × ℝ ↦ f (Fit.Params.ofProd p)) :
    ∃ φ ∈ Fit.domain B, IsMaxOn f (Fit.domain B) φ := by
  let g : ℝ × ℝ → ℝ := fun p ↦ f (Fit.Params.ofProd p)
  obtain ⟨p, hp, hpmax⟩ :=
    (Fit.region_compact B).exists_isMaxOn (Fit.region_nonempty B) hf.continuousOn
  refine ⟨Fit.Params.ofProd p, ?_, ?_⟩
  · exact (Fit.mem_region B _).2 hp
  · rw [isMaxOn_iff]
    intro ψ hψ
    have hψ' : ψ.toProd ∈ Fit.region B :=
      (Fit.mem_region B ψ).1 hψ
    simpa [g, Fit.Params.ofProd_toProd] using
      (isMaxOn_iff.mp hpmax ψ.toProd hψ')

/-- The lower corner supplies a canonical point of the parameter box. -/
instance Fit.region_inhabited (B : Fit.Box) : Inhabited (Fit.region B) :=
  ⟨⟨(B.alpha_lo, B.gamma_lo), by
    simp [Fit.region, B.alpha_bounds, B.gamma_bounds]⟩⟩

/-- The closed parameter rectangle is compact. -/
instance Fit.compact_space (B : Fit.Box) : CompactSpace (Fit.region B) :=
  isCompact_iff_compactSpace.mp (Fit.region_compact B)

/-- A noncomputable maximizer selected from a continuous objective on the
parameter box. -/
noncomputable def Fit.argmax
    (B : Fit.Box) (f : Fit.Params → ℝ)
    (hf : Continuous fun p : ℝ × ℝ ↦ f (Fit.Params.ofProd p)) :
    Fit.Params :=
  Classical.choose (Fit.exists_max B hf)

theorem Fit.argmax_mem
    (B : Fit.Box) (f : Fit.Params → ℝ)
    (hf : Continuous fun p : ℝ × ℝ ↦ f (Fit.Params.ofProd p)) :
    Fit.argmax B f hf ∈ Fit.domain B :=
  (Classical.choose_spec (Fit.exists_max B hf)).1

theorem Fit.argmax_spec
    (B : Fit.Box) (f : Fit.Params → ℝ)
    (hf : Continuous fun p : ℝ × ℝ ↦ f (Fit.Params.ofProd p)) :
    IsMaxOn f (Fit.domain B) (Fit.argmax B f hf) :=
  (Classical.choose_spec (Fit.exists_max B hf)).2

/-- If two objectives are uniformly `ε`-close on the parameter box, then a
maximizer of one objective is `2ε`-optimal for the other. This is the standard
deterministic argmax-stability inequality used in consistency arguments. -/
theorem Fit.max_stability
    (B : Fit.Box) {f g : Fit.Params → ℝ}
    {φf φg : Fit.Params} {ε : ℝ}
    (hφf : φf ∈ Fit.domain B) (_hmaxf : IsMaxOn f (Fit.domain B) φf)
    (hφg : φg ∈ Fit.domain B) (hmaxg : IsMaxOn g (Fit.domain B) φg)
    (_hε : 0 ≤ ε)
    (hclose : ∀ φ ∈ Fit.domain B, |f φ - g φ| ≤ ε) :
    f φf ≤ f φg + 2 * ε := by
  have hfg : ∀ φ ∈ Fit.domain B, f φ ≤ g φ + ε := by
    intro φ hφ
    have h := abs_sub_le_iff.mp (hclose φ hφ)
    linarith
  have hgf : ∀ φ ∈ Fit.domain B, g φ ≤ f φ + ε := by
    intro φ hφ
    have h := abs_sub_le_iff.mp (hclose φ hφ)
    linarith
  calc
    f φf ≤ g φf + ε := hfg φf hφf
    _ ≤ g φg + ε := by
          simpa [add_comm, add_left_comm, add_assoc] using
            add_le_add_right (isMaxOn_iff.mp hmaxg φf hφf) ε
    _ ≤ (f φg + ε) + ε := by gcongr; exact hgf φg hφg
    _ = f φg + 2 * ε := by ring

/-- Applied to the selected maximizers, uniform objective control yields the
standard deterministic `2ε` argmax gap on the parameter box. -/
theorem Fit.argmax_stability
    (B : Fit.Box) {f g : Fit.Params → ℝ}
    (hf : Continuous fun p : ℝ × ℝ ↦ f (Fit.Params.ofProd p))
    (hg : Continuous fun p : ℝ × ℝ ↦ g (Fit.Params.ofProd p))
    {ε : ℝ} (hε : 0 ≤ ε)
    (hclose : ∀ φ ∈ Fit.domain B, |f φ - g φ| ≤ ε) :
    f (Fit.argmax B f hf) ≤ f (Fit.argmax B g hg) + 2 * ε := by
  exact Fit.max_stability B
    (Fit.argmax_mem B f hf) (Fit.argmax_spec B f hf)
    (Fit.argmax_mem B g hg) (Fit.argmax_spec B g hg)
    hε hclose

/-- The parameter `ℓ¹` distance is controlled by twice the product metric on
`ℝ × ℝ`, whose distance is the maximum of the coordinate distances. -/
theorem Fit.distance_bound (p q : ℝ × ℝ) :
    Fit.distance (Fit.Params.ofProd p) (Fit.Params.ofProd q) ≤ 2 * dist p q := by
  simp only [Fit.distance, Fit.Params.ofProd_fst, Fit.Params.ofProd_snd,
    Prod.dist_eq, Real.dist_eq]
  have h1 : |p.1 - q.1| ≤ max |p.1 - q.1| |p.2 - q.2| := le_max_left _ _
  have h2 : |p.2 - q.2| ≤ max |p.1 - q.1| |p.2 - q.2| := le_max_right _ _
  calc
    |p.1 - q.1| + |p.2 - q.2|
      ≤ max |p.1 - q.1| |p.2 - q.2| + max |p.1 - q.1| |p.2 - q.2| := add_le_add h1 h2
    _ = 2 * max |p.1 - q.1| |p.2 - q.2| := by ring

end ParameterBoxArgmax

end ENN
