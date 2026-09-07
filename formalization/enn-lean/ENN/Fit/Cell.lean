import ENN.Fit.Box

/-!
# Sub-box geometry and quadtree covers
-/

namespace ENN

open Filter MeasureTheory

namespace Fit.Cell

variable {B : Fit.Box}

/-- The underlying subset of the ambient compact parameter box. -/
def toSet (C : Fit.Cell B) : Set (Fit.region B) :=
  {p | C.alpha_lo ≤ p.1.1 ∧ p.1.1 ≤ C.alpha_hi ∧
      C.gamma_lo ≤ p.1.2 ∧ p.1.2 ≤ C.gamma_hi}

@[simp] theorem mem_toSet {C : Fit.Cell B} {p : Fit.region B} :
    p ∈ C.toSet ↔
      C.alpha_lo ≤ p.1.1 ∧ p.1.1 ≤ C.alpha_hi ∧
        C.gamma_lo ≤ p.1.2 ∧ p.1.2 ≤ C.gamma_hi :=
  Iff.rfl

theorem toSet_eq_preimage (C : Fit.Cell B) :
    C.toSet =
      Subtype.val ⁻¹' Set.Icc (C.alpha_lo, C.gamma_lo) (C.alpha_hi, C.gamma_hi) := by
  ext p
  constructor
  · intro hp
    exact ⟨⟨hp.1, hp.2.2.1⟩, hp.2.1, hp.2.2.2⟩
  · intro hp
    exact ⟨hp.1.1, hp.2.1, hp.1.2, hp.2.2⟩

theorem isCompact_toSet (C : Fit.Cell B) :
    IsCompact C.toSet := by
  rw [C.toSet_eq_preimage]
  exact (isClosed_Icc.preimage continuous_subtype_val).isCompact

theorem isClosed_toSet (C : Fit.Cell B) :
    IsClosed C.toSet := by
  rw [C.toSet_eq_preimage]
  exact isClosed_Icc.preimage continuous_subtype_val

theorem toSet_nonempty (C : Fit.Cell B) :
    C.toSet.Nonempty := by
  refine ⟨⟨(C.alpha_lo, C.gamma_lo), by
    refine ⟨⟨C.alpha_lower, C.gamma_lower⟩, ?_⟩
    exact ⟨le_trans C.alpha_bounds C.alpha_upper, le_trans C.gamma_bounds C.gamma_upper⟩⟩, ?_⟩
  simp [toSet, C.alpha_bounds, C.gamma_bounds]

/-- The whole parameter box, viewed as a subbox of itself. -/
def whole (B : Fit.Box) : Fit.Cell B where
  alpha_lo := B.alpha_lo
  alpha_hi := B.alpha_hi
  gamma_lo := B.gamma_lo
  gamma_hi := B.gamma_hi
  alpha_lower := le_rfl
  alpha_bounds := B.alpha_bounds
  alpha_upper := le_rfl
  gamma_lower := le_rfl
  gamma_bounds := B.gamma_bounds
  gamma_upper := le_rfl

@[simp] theorem whole_toSet (B : Fit.Box) :
    (whole B).toSet = Set.univ := by
  ext p
  constructor
  · intro _
    trivial
  · intro _
    exact ⟨p.2.1.1, p.2.2.1, p.2.1.2, p.2.2.2⟩

/-- Midpoint of the `α`-interval of a subbox. -/
noncomputable def alphaMid (C : Fit.Cell B) : ℝ :=
  (C.alpha_lo + C.alpha_hi) / 2

/-- Midpoint of the `γ`-interval of a subbox. -/
noncomputable def gammaMid (C : Fit.Cell B) : ℝ :=
  (C.gamma_lo + C.gamma_hi) / 2

/-- Width of the `α`-interval of a subbox. -/
def alphaWidth (C : Fit.Cell B) : ℝ :=
  C.alpha_hi - C.alpha_lo

/-- Width of the `γ`-interval of a subbox. -/
def gammaWidth (C : Fit.Cell B) : ℝ :=
  C.gamma_hi - C.gamma_lo

theorem alpha_mid_lower (C : Fit.Cell B) :
    C.alpha_lo ≤ C.alphaMid := by
  dsimp [alphaMid]
  linarith [C.alpha_bounds]

theorem alpha_mid_upper (C : Fit.Cell B) :
    C.alphaMid ≤ C.alpha_hi := by
  dsimp [alphaMid]
  linarith [C.alpha_bounds]

theorem gamma_mid_lower (C : Fit.Cell B) :
    C.gamma_lo ≤ C.gammaMid := by
  dsimp [gammaMid]
  linarith [C.gamma_bounds]

theorem gamma_mid_upper (C : Fit.Cell B) :
    C.gammaMid ≤ C.gamma_hi := by
  dsimp [gammaMid]
  linarith [C.gamma_bounds]

theorem alphaWidth_nonneg (C : Fit.Cell B) :
    0 ≤ C.alphaWidth := by
  dsimp [alphaWidth]
  linarith [C.alpha_bounds]

theorem gammaWidth_nonneg (C : Fit.Cell B) :
    0 ≤ C.gammaWidth := by
  dsimp [gammaWidth]
  linarith [C.gamma_bounds]

/-- The midpoint of both coordinate intervals, viewed as a point of the ambient
parameter rectangle. -/
noncomputable def center (C : Fit.Cell B) : Fit.region B := by
  refine ⟨(C.alphaMid, C.gammaMid), ?_⟩
  exact ⟨⟨le_trans C.alpha_lower C.alpha_mid_lower,
      le_trans C.gamma_lower C.gamma_mid_lower⟩,
    le_trans C.alpha_mid_upper C.alpha_upper,
    le_trans C.gamma_mid_upper C.gamma_upper⟩

theorem center_mem (C : Fit.Cell B) :
    C.center ∈ C.toSet := by
  dsimp [center, toSet]
  exact ⟨C.alpha_mid_lower, C.alpha_mid_upper,
    C.gamma_mid_lower, C.gamma_mid_upper⟩

theorem diameter_bound
    (C : Fit.Cell B) {p q : Fit.region B}
    (hp : p ∈ C.toSet) (hq : q ∈ C.toSet) :
    dist p q ≤ C.alphaWidth + C.gammaWidth := by
  rw [Subtype.dist_eq, Prod.dist_eq]
  have hα :
      dist p.1.1 q.1.1 ≤ C.alphaWidth := by
    simpa [alphaWidth] using
      (Real.dist_le_of_mem_Icc ⟨hp.1, hp.2.1⟩ ⟨hq.1, hq.2.1⟩)
  have hγ :
      dist p.1.2 q.1.2 ≤ C.gammaWidth := by
    simpa [gammaWidth] using
      (Real.dist_le_of_mem_Icc ⟨hp.2.2.1, hp.2.2.2⟩ ⟨hq.2.2.1, hq.2.2.2⟩)
  have hmax :
      max (dist p.1.1 q.1.1) (dist p.1.2 q.1.2) ≤ max C.alphaWidth C.gammaWidth :=
    max_le_max hα hγ
  have hsum : max C.alphaWidth C.gammaWidth ≤ C.alphaWidth + C.gammaWidth := by
    rw [max_le_iff]
    exact ⟨le_add_of_nonneg_right C.gammaWidth_nonneg,
      le_add_of_nonneg_left C.alphaWidth_nonneg⟩
  exact hmax.trans hsum

/-- Southwest child of a quadtree split. -/
noncomputable def sw (C : Fit.Cell B) : Fit.Cell B where
  alpha_lo := C.alpha_lo
  alpha_hi := C.alphaMid
  gamma_lo := C.gamma_lo
  gamma_hi := C.gammaMid
  alpha_lower := C.alpha_lower
  alpha_bounds := C.alpha_mid_lower
  alpha_upper := le_trans C.alpha_mid_upper C.alpha_upper
  gamma_lower := C.gamma_lower
  gamma_bounds := C.gamma_mid_lower
  gamma_upper := le_trans C.gamma_mid_upper C.gamma_upper

/-- Northwest child of a quadtree split. -/
noncomputable def nw (C : Fit.Cell B) : Fit.Cell B where
  alpha_lo := C.alpha_lo
  alpha_hi := C.alphaMid
  gamma_lo := C.gammaMid
  gamma_hi := C.gamma_hi
  alpha_lower := C.alpha_lower
  alpha_bounds := C.alpha_mid_lower
  alpha_upper := le_trans C.alpha_mid_upper C.alpha_upper
  gamma_lower := le_trans C.gamma_lower C.gamma_mid_lower
  gamma_bounds := C.gamma_mid_upper
  gamma_upper := C.gamma_upper

/-- Southeast child of a quadtree split. -/
noncomputable def se (C : Fit.Cell B) : Fit.Cell B where
  alpha_lo := C.alphaMid
  alpha_hi := C.alpha_hi
  gamma_lo := C.gamma_lo
  gamma_hi := C.gammaMid
  alpha_lower := le_trans C.alpha_lower C.alpha_mid_lower
  alpha_bounds := C.alpha_mid_upper
  alpha_upper := C.alpha_upper
  gamma_lower := C.gamma_lower
  gamma_bounds := C.gamma_mid_lower
  gamma_upper := le_trans C.gamma_mid_upper C.gamma_upper

/-- Northeast child of a quadtree split. -/
noncomputable def ne (C : Fit.Cell B) : Fit.Cell B where
  alpha_lo := C.alphaMid
  alpha_hi := C.alpha_hi
  gamma_lo := C.gammaMid
  gamma_hi := C.gamma_hi
  alpha_lower := le_trans C.alpha_lower C.alpha_mid_lower
  alpha_bounds := C.alpha_mid_upper
  alpha_upper := C.alpha_upper
  gamma_lower := le_trans C.gamma_lower C.gamma_mid_lower
  gamma_bounds := C.gamma_mid_upper
  gamma_upper := C.gamma_upper

theorem sw_subset (C : Fit.Cell B) :
    (C.sw).toSet ⊆ C.toSet := by
  intro p hp
  exact ⟨hp.1, le_trans hp.2.1 C.alpha_mid_upper, hp.2.2.1,
    le_trans hp.2.2.2 C.gamma_mid_upper⟩

theorem nw_subset (C : Fit.Cell B) :
    (C.nw).toSet ⊆ C.toSet := by
  intro p hp
  exact ⟨hp.1, le_trans hp.2.1 C.alpha_mid_upper,
    le_trans C.gamma_mid_lower hp.2.2.1, hp.2.2.2⟩

theorem se_subset (C : Fit.Cell B) :
    (C.se).toSet ⊆ C.toSet := by
  intro p hp
  exact ⟨le_trans C.alpha_mid_lower hp.1, hp.2.1, hp.2.2.1,
    le_trans hp.2.2.2 C.gamma_mid_upper⟩

theorem ne_subset (C : Fit.Cell B) :
    (C.ne).toSet ⊆ C.toSet := by
  intro p hp
  exact ⟨le_trans C.alpha_mid_lower hp.1, hp.2.1,
    le_trans C.gamma_mid_lower hp.2.2.1, hp.2.2.2⟩

theorem sw_alpha (C : Fit.Cell B) :
    C.sw.alphaWidth = C.alphaWidth / 2 := by
  dsimp [sw, alphaWidth, alphaMid]
  ring

theorem nw_alpha (C : Fit.Cell B) :
    C.nw.alphaWidth = C.alphaWidth / 2 := by
  dsimp [nw, alphaWidth, alphaMid]
  ring

theorem se_alpha (C : Fit.Cell B) :
    C.se.alphaWidth = C.alphaWidth / 2 := by
  dsimp [se, alphaWidth, alphaMid]
  ring

theorem ne_alpha (C : Fit.Cell B) :
    C.ne.alphaWidth = C.alphaWidth / 2 := by
  dsimp [ne, alphaWidth, alphaMid]
  ring

theorem sw_gamma (C : Fit.Cell B) :
    C.sw.gammaWidth = C.gammaWidth / 2 := by
  dsimp [sw, gammaWidth, gammaMid]
  ring

theorem nw_gamma (C : Fit.Cell B) :
    C.nw.gammaWidth = C.gammaWidth / 2 := by
  dsimp [nw, gammaWidth, gammaMid]
  ring

theorem se_gamma (C : Fit.Cell B) :
    C.se.gammaWidth = C.gammaWidth / 2 := by
  dsimp [se, gammaWidth, gammaMid]
  ring

theorem ne_gamma (C : Fit.Cell B) :
    C.ne.gammaWidth = C.gammaWidth / 2 := by
  dsimp [ne, gammaWidth, gammaMid]
  ring

theorem cover_children (C : Fit.Cell B) :
    C.toSet ⊆
      (C.sw).toSet ∪ (C.nw).toSet ∪
        (C.se).toSet ∪ (C.ne).toSet := by
  intro p hp
  by_cases hα : p.1.1 ≤ C.alphaMid
  · by_cases hγ : p.1.2 ≤ C.gammaMid
    · exact Or.inl (Or.inl (Or.inl (show p ∈ C.sw.toSet from
          ⟨hp.1, hα, hp.2.2.1, hγ⟩)))
    ·
      have hγ' : C.gammaMid ≤ p.1.2 := by linarith [hp.2.2.2]
      exact Or.inl (Or.inl (Or.inr (show p ∈ C.nw.toSet from
          ⟨hp.1, hα, hγ', hp.2.2.2⟩)))
  · by_cases hγ : p.1.2 ≤ C.gammaMid
    ·
      have hα' : C.alphaMid ≤ p.1.1 := by linarith [hp.2.1]
      exact Or.inl (Or.inr (show p ∈ C.se.toSet from
          ⟨hα', hp.2.1, hp.2.2.1, hγ⟩))
    ·
      have hα' : C.alphaMid ≤ p.1.1 := by linarith [hp.2.1]
      have hγ' : C.gammaMid ≤ p.1.2 := by linarith [hp.2.2.2]
      exact Or.inr (show p ∈ C.ne.toSet from
          ⟨hα', hp.2.1, hγ', hp.2.2.2⟩)

/-- The `j`th child in the deterministic four-way split of a subbox. -/
noncomputable def child (C : Fit.Cell B) (j : Fin 4) : Fit.Cell B :=
  match j.1 with
  | 0 => C.sw
  | 1 => C.nw
  | 2 => C.se
  | _ => C.ne

theorem alphaWidth_child (C : Fit.Cell B) (j : Fin 4) :
    (C.child j).alphaWidth = C.alphaWidth / 2 := by
  fin_cases j <;> simp [child, sw_alpha, nw_alpha,
    se_alpha, ne_alpha]

theorem gammaWidth_child (C : Fit.Cell B) (j : Fin 4) :
    (C.child j).gammaWidth = C.gammaWidth / 2 := by
  fin_cases j <;> simp [child, sw_gamma, nw_gamma,
    se_gamma, ne_gamma]

theorem child_subset (C : Fit.Cell B) (j : Fin 4) :
    (C.child j).toSet ⊆ C.toSet := by
  fin_cases j <;> simp [child]
  · exact C.sw_subset
  · exact C.nw_subset
  · exact C.se_subset
  · exact C.ne_subset

theorem cover (C : Fit.Cell B) :
    C.toSet ⊆ ⋃ j : Fin 4, (C.child j).toSet := by
  intro p hp
  by_cases hα : p.1.1 ≤ C.alphaMid
  · by_cases hγ : p.1.2 ≤ C.gammaMid
    · exact Set.mem_iUnion.mpr ⟨(0 : Fin 4), by
        simpa [child] using (show p ∈ C.sw.toSet from ⟨hp.1, hα, hp.2.2.1, hγ⟩)⟩
    · have hγ' : C.gammaMid ≤ p.1.2 := by linarith [hp.2.2.2]
      exact Set.mem_iUnion.mpr ⟨(1 : Fin 4), by
        simpa [child] using (show p ∈ C.nw.toSet from ⟨hp.1, hα, hγ', hp.2.2.2⟩)⟩
  · by_cases hγ : p.1.2 ≤ C.gammaMid
    · have hα' : C.alphaMid ≤ p.1.1 := by linarith [hp.2.1]
      exact Set.mem_iUnion.mpr ⟨(2 : Fin 4), by
        simpa [child] using (show p ∈ C.se.toSet from ⟨hα', hp.2.1, hp.2.2.1, hγ⟩)⟩
    · have hα' : C.alphaMid ≤ p.1.1 := by linarith [hp.2.1]
      have hγ' : C.gammaMid ≤ p.1.2 := by linarith [hp.2.2.2]
      exact Set.mem_iUnion.mpr ⟨(3 : Fin 4), by
        simpa [child] using (show p ∈ C.ne.toSet from ⟨hα', hp.2.1, hγ', hp.2.2.2⟩)⟩

/-- The deterministic quadtree subbox at depth `n` and index `j`. The indexing
is by base-4 words in lexicographic order. -/
noncomputable def tree (B : Fit.Box) :
    (n : ℕ) → Fin (4 ^ n) → Fit.Cell B
  | 0, _ => whole B
  | n + 1, j =>
      let parent : Fin (4 ^ n) := ⟨j.1 / 4, by
        have hj : j.1 < 4 * 4 ^ n := by
          simpa [pow_succ, Nat.mul_comm] using j.2
        exact Nat.div_lt_of_lt_mul hj⟩
      let digit : Fin 4 := ⟨j.1 % 4, Nat.mod_lt _ (by norm_num)⟩
      (tree B n parent).child digit

theorem tree_compact (B : Fit.Box) (n : ℕ) (j : Fin (4 ^ n)) :
    IsCompact ((tree B n j).toSet) :=
  (tree B n j).isCompact_toSet

theorem tree_nonempty (B : Fit.Box) (n : ℕ) (j : Fin (4 ^ n)) :
    ((tree B n j).toSet).Nonempty :=
  (tree B n j).toSet_nonempty

theorem tree_nested
    (B : Fit.Box) (n : ℕ) (j : Fin (4 ^ (n + 1))) :
    let parent : Fin (4 ^ n) := ⟨j.1 / 4, by
      have hj : j.1 < 4 * 4 ^ n := by
        simpa [pow_succ, Nat.mul_comm] using j.2
      exact Nat.div_lt_of_lt_mul hj⟩
    ((tree B (n + 1) j).toSet) ⊆ ((tree B n parent).toSet) := by
  dsimp [tree]
  exact (tree B n
    ⟨j.1 / 4, by
      have hj : j.1 < 4 * 4 ^ n := by
        simpa [pow_succ, Nat.mul_comm] using j.2
      exact Nat.div_lt_of_lt_mul hj⟩).child_subset
      ⟨j.1 % 4, Nat.mod_lt _ (by norm_num)⟩

theorem tree_child
    (B : Fit.Box) (n : ℕ) (i : Fin (4 ^ n)) (d : Fin 4) :
    let j : Fin (4 ^ (n + 1)) := ⟨d.1 + 4 * i.1, by
      rw [pow_succ, Nat.mul_comm]
      omega⟩
    tree B (n + 1) j = (tree B n i).child d := by
  let parent : Fin (4 ^ n) := ⟨(d.1 + 4 * i.1) / 4, by
    have hj : d.1 + 4 * i.1 < 4 * 4 ^ n := by
      omega
    exact Nat.div_lt_of_lt_mul hj⟩
  let digit : Fin 4 := ⟨(d.1 + 4 * i.1) % 4, Nat.mod_lt _ (by norm_num)⟩
  change (tree B n parent).child digit = (tree B n i).child d
  have hparent : parent = i := by
    apply Fin.ext
    unfold parent
    calc
      (d.1 + 4 * i.1) / 4 = (d.1 + i.1 * 4) / 4 := by ac_rfl
      _ = d.1 / 4 + i.1 := Nat.add_mul_div_right _ _ (by norm_num)
      _ = i.1 := by rw [Nat.div_eq_of_lt d.2, Nat.zero_add]
  have hdigit : digit = d := by
    apply Fin.ext
    unfold digit
    calc
      (d.1 + 4 * i.1) % 4 = (d.1 + i.1 * 4) % 4 := by ac_rfl
      _ = d.1 % 4 := by
        simp
      _ = d.1 := Nat.mod_eq_of_lt d.2
  calc
    (tree B n parent).child digit = (tree B n i).child digit := by
      simp [hparent]
    _ = (tree B n i).child d := by
      simp [hdigit]

theorem tree_alpha
    (B : Fit.Box) :
    ∀ n (j : Fin (4 ^ n)),
      (tree B n j).alphaWidth = (B.alpha_hi - B.alpha_lo) / (2 : ℝ) ^ n
  | 0, j => by
      simp [tree, whole, alphaWidth]
  | n + 1, j => by
      dsimp [tree]
      rw [alphaWidth_child]
      rw [tree_alpha B n
        ⟨j.1 / 4, by
          have hj : j.1 < 4 * 4 ^ n := by
            simpa [pow_succ, Nat.mul_comm] using j.2
          exact Nat.div_lt_of_lt_mul hj⟩]
      rw [pow_succ]
      ring

theorem tree_gamma
    (B : Fit.Box) :
    ∀ n (j : Fin (4 ^ n)),
      (tree B n j).gammaWidth = (B.gamma_hi - B.gamma_lo) / (2 : ℝ) ^ n
  | 0, j => by
      simp [tree, whole, gammaWidth]
  | n + 1, j => by
      dsimp [tree]
      rw [gammaWidth_child]
      rw [tree_gamma B n
        ⟨j.1 / 4, by
          have hj : j.1 < 4 * 4 ^ n := by
            simpa [pow_succ, Nat.mul_comm] using j.2
          exact Nat.div_lt_of_lt_mul hj⟩]
      rw [pow_succ]
      ring

theorem tree_cover
    (B : Fit.Box) (n : ℕ) :
    (Set.univ : Set (Fit.region B)) ⊆ ⋃ j : Fin (4 ^ n), (tree B n j).toSet := by
  induction n with
  | zero =>
      intro p _
      exact Set.mem_iUnion.mpr ⟨(0 : Fin 1), by
        simp [tree, Fit.Cell.whole_toSet]⟩
  | succ n ih =>
      intro p _
      have hp_up : p ∈ ⋃ j : Fin (4 ^ n), (tree B n j).toSet :=
        ih trivial
      rcases Set.mem_iUnion.mp hp_up with ⟨i, hip⟩
      have hp_child : p ∈ ⋃ d : Fin 4, ((tree B n i).child d).toSet :=
        (tree B n i).cover hip
      rcases Set.mem_iUnion.mp hp_child with ⟨d, hpd⟩
      let j : Fin (4 ^ (n + 1)) := ⟨d.1 + 4 * i.1, by
        rw [pow_succ, Nat.mul_comm]
        omega⟩
      refine Set.mem_iUnion.mpr ⟨j, ?_⟩
      simpa [j] using (show p ∈ (tree B (n + 1) j).toSet by
        simpa [j, tree_child B n i d] using hpd)

end Fit.Cell

end ENN
