import ENN.Fit.Box

/-!
# Weight and prediction stability
-/

namespace ENN

open Filter MeasureTheory

section DeterministicStability

variable {ι : Type*} [Fintype ι]

/-- The variance scale associated with a C.4 parameter pair. -/
def Fit.scale (φ : Fit.Params) (s r : ι → ℝ) (i : ι) : ℝ :=
  Variance.scale φ.alpha φ.gamma s r i

omit [Fintype ι] in
/-- Pointwise variation of the variance scale with respect to the parameter pair. -/
theorem Fit.scale_lipschitz
    (φ ψ : Fit.Params) (s r : ι → ℝ) (i : ι) :
    |Fit.scale φ s r i - Fit.scale ψ s r i|
      ≤ |φ.alpha - ψ.alpha| + |φ.gamma - ψ.gamma| * (r i) ^ 2 := by
  unfold Fit.scale Variance.scale
  calc
    |φ.alpha + s i ^ 2 + φ.gamma * r i ^ 2 - (ψ.alpha + s i ^ 2 + ψ.gamma * r i ^ 2)|
      = |(φ.alpha - ψ.alpha) + (φ.gamma - ψ.gamma) * r i ^ 2| := by ring_nf
    _ ≤ |φ.alpha - ψ.alpha| + |(φ.gamma - ψ.gamma) * r i ^ 2| := abs_add_le _ _
    _ = |φ.alpha - ψ.alpha| + |φ.gamma - ψ.gamma| * (r i) ^ 2 := by
          rw [abs_mul, abs_of_nonneg (sq_nonneg (r i))]

omit [Fintype ι] in
/-- The variance scale is uniformly bounded below on a parameter box by the
strictly positive lower `α` bound. -/
theorem Fit.scale_lower
    (B : Fit.Box) {φ : Fit.Params} (hφ : Fit.InBox B φ)
    (s r : ι → ℝ) (i : ι) :
    B.alpha_lo ≤ Fit.scale φ s r i := by
  unfold Fit.scale Variance.scale
  have hγ : 0 ≤ φ.gamma := hφ.gamma_nonneg
  have hsq : 0 ≤ (s i) ^ 2 := sq_nonneg _
  have hrsq : 0 ≤ φ.gamma * (r i) ^ 2 := mul_nonneg hγ (sq_nonneg _)
  linarith [hφ.alpha_lower, hsq, hrsq]

omit [Fintype ι] in
/-- A box bound plus pointwise radius/noise-scale bounds yields a uniform upper
bound on the variance scale. -/
theorem Fit.scale_upper
    (B : Fit.Box) {φ : Fit.Params} (hφ : Fit.InBox B φ)
    (s r : ι → ℝ) {S2 R2 : ℝ} (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2)
    (i : ι) :
    Fit.scale φ s r i ≤ B.alpha_hi + S2 + B.gamma_hi * R2 := by
  unfold Fit.scale Variance.scale
  have hγ0 : 0 ≤ φ.gamma := hφ.gamma_nonneg
  have hR20 : 0 ≤ R2 := le_trans (sq_nonneg _) (hR2 i)
  have hγhi0 : 0 ≤ B.gamma_hi := le_trans B.gamma_nonneg B.gamma_bounds
  have hmul : φ.gamma * (r i) ^ 2 ≤ B.gamma_hi * R2 := by
    exact mul_le_mul hφ.gamma_upper (hR2 i) (sq_nonneg _) hγhi0
  linarith [hφ.alpha_upper, hS2 i, hmul]

/-- Reciprocal differences are Lipschitz away from zero. -/
theorem Fit.inverse_bound
    {a b m : ℝ} (hm : 0 < m) (ha : m ≤ a) (hb : m ≤ b) :
    |a⁻¹ - b⁻¹| ≤ (m⁻¹) ^ 2 * |a - b| := by
  have ha0 : 0 < a := lt_of_lt_of_le hm ha
  have hb0 : 0 < b := lt_of_lt_of_le hm hb
  have hab : 0 < a * b := mul_pos ha0 hb0
  have hm2 : 0 < m ^ 2 := pow_pos hm 2
  have hmul_lo : m ^ 2 ≤ a * b := by
    nlinarith [ha, hb]
  calc
    |a⁻¹ - b⁻¹| = |a - b| / (a * b) := by
      have hneqa : a ≠ 0 := ha0.ne'
      have hneqb : b ≠ 0 := hb0.ne'
      have hformula : a⁻¹ - b⁻¹ = (b - a) / (a * b) := by
        field_simp [hneqa, hneqb]
      rw [hformula]
      rw [abs_div, abs_sub_comm, abs_of_pos hab]
    _ ≤ |a - b| / (m ^ 2) := by
      exact div_le_div_of_nonneg_left (abs_nonneg _) hm2 hmul_lo
    _ = (m⁻¹) ^ 2 * |a - b| := by
      field_simp [hm.ne']

omit [Fintype ι] in
/-- The raw inverse-variance weights are Lipschitz in the parameter pair on a
parameter box, under a uniform squared-radius bound. -/
theorem Fit.raw_bound [Nonempty ι]
    (B : Fit.Box) {φ ψ : Fit.Params}
    (hφ : Fit.InBox B φ) (hψ : Fit.InBox B ψ)
    (s r : ι → ℝ) {R2 : ℝ} (hR2 : ∀ i, (r i) ^ 2 ≤ R2) (i : ι) :
    |Variance.raw φ.alpha φ.gamma s r i -
        Variance.raw ψ.alpha ψ.gamma s r i|
      ≤ ((B.alpha_lo⁻¹) ^ 2 * (1 + R2)) * Fit.distance φ ψ := by
  have hlowerφ : B.alpha_lo ≤ Fit.scale φ s r i :=
    Fit.scale_lower B hφ s r i
  have hlowerψ : B.alpha_lo ≤ Fit.scale ψ s r i :=
    Fit.scale_lower B hψ s r i
  have hscale :
      |Fit.scale φ s r i - Fit.scale ψ s r i|
        ≤ (1 + R2) * Fit.distance φ ψ := by
    have hbase := Fit.scale_lipschitz φ ψ s r i
    have hR20 : 0 ≤ R2 := le_trans (sq_nonneg _) (hR2 i)
    have hstep :
        |φ.alpha - ψ.alpha| + |φ.gamma - ψ.gamma| * (r i) ^ 2
          ≤ (1 + R2) * Fit.distance φ ψ := by
      unfold Fit.distance
      nlinarith [abs_nonneg (φ.alpha - ψ.alpha), abs_nonneg (φ.gamma - ψ.gamma), hR2 i]
    exact hbase.trans hstep
  unfold Variance.raw
  have hrecip :=
    Fit.inverse_bound B.alpha_pos hlowerφ hlowerψ
  have hR20 : 0 ≤ R2 := le_trans (sq_nonneg _) (hR2 i)
  have hcoeff0 : 0 ≤ (B.alpha_lo⁻¹) ^ 2 := sq_nonneg _
  calc
    |(Fit.scale φ s r i)⁻¹ - (Fit.scale ψ s r i)⁻¹|
      ≤ (B.alpha_lo⁻¹) ^ 2 * |Fit.scale φ s r i - Fit.scale ψ s r i| :=
        hrecip
    _ ≤ (B.alpha_lo⁻¹) ^ 2 * ((1 + R2) * Fit.distance φ ψ) := by
          exact mul_le_mul_of_nonneg_left hscale hcoeff0
    _ = ((B.alpha_lo⁻¹) ^ 2 * (1 + R2)) * Fit.distance φ ψ := by ring

/-- If one normalized vector uses a denominator bounded below by `m > 0`, then
the `ℓ¹` distance between the corresponding normalizations is controlled by the
`ℓ¹` distance between the raw weights. -/
theorem Fit.normalize_bound [Nonempty ι]
    (raw raw' : ι → ℝ) (hraw'_pos : ∀ i, 0 < raw' i)
    {m : ℝ} (hm : 0 < m) (hdenom : m ≤ ∑ i, raw i) :
    ∑ i, |normalize raw i - normalize raw' i|
      ≤ 2 * (∑ i, |raw i - raw' i|) / m := by
  let A : ℝ := ∑ i, raw i
  let B : ℝ := ∑ i, raw' i
  have hA : 0 < A := lt_of_lt_of_le hm hdenom
  have hB : 0 < B := by
    simpa [B] using positive_sum raw' hraw'_pos
  have hsumDiff :
      |A - B| ≤ ∑ i, |raw i - raw' i| := by
    calc
      |A - B| = |∑ i, (raw i - raw' i)| := by
        simp [A, B, Finset.sum_sub_distrib]
      _ ≤ ∑ i, |raw i - raw' i| := Finset.abs_sum_le_sum_abs _ _
  have hterm :
      ∀ i,
        |normalize raw i - normalize raw' i|
          ≤ |raw i - raw' i| / A + raw' i * |A - B| / (A * B) := by
    intro i
    have hAi : A ≠ 0 := hA.ne'
    have hBi : B ≠ 0 := hB.ne'
    have hAB : 0 < A * B := mul_pos hA hB
    have hsplit :
        normalize raw i - normalize raw' i =
          (raw i - raw' i) / A + raw' i * (B - A) / (A * B) := by
      unfold normalize
      field_simp [A, B, hAi, hBi]
      ring
    calc
      |normalize raw i - normalize raw' i|
        = |(raw i - raw' i) / A + raw' i * (B - A) / (A * B)| := by rw [hsplit]
      _ ≤ |(raw i - raw' i) / A| + |raw' i * (B - A) / (A * B)| := abs_add_le _ _
      _ = |raw i - raw' i| / A + raw' i * |A - B| / (A * B) := by
            rw [abs_div, abs_div, abs_mul, abs_sub_comm, abs_of_pos hA, abs_of_pos hAB,
              abs_of_nonneg (hraw'_pos i).le, abs_sub_comm A B]
  have hsum₁ :
      ∑ i, |raw i - raw' i| / A = (∑ i, |raw i - raw' i|) / A := by
    simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using
      (Finset.sum_mul Finset.univ (fun i ↦ |raw i - raw' i|) A⁻¹).symm
  have hsum₂ :
      ∑ i, raw' i * |A - B| / (A * B) = |A - B| / A := by
    calc
      ∑ i, raw' i * |A - B| / (A * B)
        = ∑ i, raw' i * (|A - B| / (A * B)) := by
            refine Finset.sum_congr rfl ?_
            intro i _
            ring
      _ = (∑ i, raw' i) * (|A - B| / (A * B)) := by
            simpa [mul_comm, mul_left_comm, mul_assoc] using
              (Finset.sum_mul Finset.univ raw' (|A - B| / (A * B))).symm
      _ = B * (|A - B| / (A * B)) := by rfl
      _ = |A - B| / A := by
            field_simp [hA.ne', hB.ne']
  calc
    ∑ i, |normalize raw i - normalize raw' i|
      ≤ ∑ i, (|raw i - raw' i| / A + raw' i * |A - B| / (A * B)) := by
          exact Finset.sum_le_sum fun i _ ↦ hterm i
    _ = (∑ i, |raw i - raw' i|) / A + |A - B| / A := by
          rw [Finset.sum_add_distrib, hsum₁, hsum₂]
    _ = ((∑ i, |raw i - raw' i|) + |A - B|) / A := by ring
    _ ≤ ((∑ i, |raw i - raw' i|) + ∑ i, |raw i - raw' i|) / A := by
          exact div_le_div_of_nonneg_right
            (by
              simpa [add_comm, add_left_comm, add_assoc] using
                add_le_add_left hsumDiff (∑ i, |raw i - raw' i|))
            hA.le
    _ = 2 * (∑ i, |raw i - raw' i|) / A := by ring
    _ ≤ 2 * (∑ i, |raw i - raw' i|) / m := by
          have hnum : 0 ≤ 2 * ∑ i, |raw i - raw' i| := by positivity
          exact div_le_div_of_nonneg_left hnum hm hdenom

omit [Fintype ι] in
/-- A parameter box plus uniform squared-radius and squared-noise-scale bounds
gives a pointwise lower bound on the raw inverse-variance weights. -/
theorem Fit.raw_lower [Nonempty ι]
    (B : Fit.Box) {φ : Fit.Params} (hφ : Fit.InBox B φ)
    (s r : ι → ℝ) {S2 R2 : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2) (i : ι) :
    (B.alpha_hi + S2 + B.gamma_hi * R2)⁻¹
      ≤ Variance.raw φ.alpha φ.gamma s r i := by
  have hlower : B.alpha_lo ≤ Fit.scale φ s r i :=
    Fit.scale_lower B hφ s r i
  have hupper : Fit.scale φ s r i ≤ B.alpha_hi + S2 + B.gamma_hi * R2 :=
    Fit.scale_upper B hφ s r hS2 hR2 i
  have hscale_pos : 0 < Fit.scale φ s r i := lt_of_lt_of_le B.alpha_pos hlower
  simpa [Variance.raw, Fit.scale, one_div] using
    one_div_le_one_div_of_le hscale_pos hupper

/-- Summing the pointwise lower bound yields a lower bound on the raw
normalization denominator. -/
theorem Fit.sum_lower [Nonempty ι]
    (B : Fit.Box) {φ : Fit.Params} (hφ : Fit.InBox B φ)
    (s r : ι → ℝ) {S2 R2 : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2) :
    (Fintype.card ι : ℝ) / (B.alpha_hi + S2 + B.gamma_hi * R2)
      ≤ ∑ i, Variance.raw φ.alpha φ.gamma s r i := by
  calc
    (Fintype.card ι : ℝ) / (B.alpha_hi + S2 + B.gamma_hi * R2)
      = ∑ i : ι, (B.alpha_hi + S2 + B.gamma_hi * R2)⁻¹ := by
          simp [div_eq_mul_inv]
    _ ≤ ∑ i, Variance.raw φ.alpha φ.gamma s r i := by
          exact Finset.sum_le_sum fun i _ ↦
            Fit.raw_lower B hφ s r hS2 hR2 i

/-- Summing the pointwise raw-weight Lipschitz bound yields a finite-sample
`ℓ¹` control on the raw inverse-variance weights. -/
theorem Fit.sum_bound [Nonempty ι]
    (B : Fit.Box) {φ ψ : Fit.Params}
    (hφ : Fit.InBox B φ) (hψ : Fit.InBox B ψ)
    (s r : ι → ℝ) {R2 : ℝ} (hR2 : ∀ i, (r i) ^ 2 ≤ R2) :
    ∑ i, |Variance.raw φ.alpha φ.gamma s r i -
        Variance.raw ψ.alpha ψ.gamma s r i|
      ≤ (Fintype.card ι : ℝ) *
          (((B.alpha_lo⁻¹) ^ 2 * (1 + R2)) * Fit.distance φ ψ) := by
  calc
    ∑ i, |Variance.raw φ.alpha φ.gamma s r i -
        Variance.raw ψ.alpha ψ.gamma s r i|
      ≤ ∑ i : ι, (((B.alpha_lo⁻¹) ^ 2 * (1 + R2)) * Fit.distance φ ψ) := by
          exact Finset.sum_le_sum fun i _ ↦
            Fit.raw_bound B hφ hψ s r hR2 i
    _ = (Fintype.card ι : ℝ) *
          (((B.alpha_lo⁻¹) ^ 2 * (1 + R2)) * Fit.distance φ ψ) := by
          simp

/-- Parameterized inverse-variance weights for Appendix C.4. -/
noncomputable def Fit.weights [Nonempty ι]
    (φ : Fit.Params) (s r : ι → ℝ)
    (hα : 0 < φ.alpha) (hγ : 0 ≤ φ.gamma) :
    Weights ι :=
  Variance.weights φ.alpha φ.gamma s r hα hγ

/-- Parameterized weighted-kNN predictor for a fixed response vector. -/
noncomputable def Fit.predict [Nonempty ι]
    (φ : Fit.Params) (s r y : ι → ℝ)
    (hα : 0 < φ.alpha) (hγ : 0 ≤ φ.gamma) : ℝ :=
  Estimator.mean (Fit.weights φ s r hα hγ) y

/-- A generic finite-sample stability bound for changing only the normalized
weight vector while keeping the response values fixed. -/
theorem Fit.response_bound
    (w w' : Weights ι) (y : ι → ℝ) {Y : ℝ}
    (hY : ∀ i, |y i| ≤ Y) :
    |Estimator.mean w y - Estimator.mean w' y|
      ≤ Y * ∑ i, |w.weight i - w'.weight i| := by
  unfold Estimator.mean
  have hsum :
      ∑ i, w.weight i * y i - ∑ i, w'.weight i * y i =
        ∑ i, (w.weight i - w'.weight i) * y i := by
    calc
      ∑ i, w.weight i * y i - ∑ i, w'.weight i * y i
        = ∑ i, (w.weight i * y i - w'.weight i * y i) := by
            rw [Finset.sum_sub_distrib]
      _ = ∑ i, (w.weight i - w'.weight i) * y i := by
            refine Finset.sum_congr rfl ?_
            intro i _
            ring
  calc
    |∑ i, w.weight i * y i - ∑ i, w'.weight i * y i|
      = |∑ i, (w.weight i - w'.weight i) * y i| := by rw [hsum]
    _ ≤ ∑ i, |(w.weight i - w'.weight i) * y i| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, |w.weight i - w'.weight i| * |y i| := by
          apply Finset.sum_congr rfl
          intro i _
          rw [abs_mul]
    _ ≤ ∑ i, |w.weight i - w'.weight i| * Y := by
          apply Finset.sum_le_sum
          intro i _
          exact mul_le_mul_of_nonneg_left (hY i) (abs_nonneg _)
    _ = (∑ i, |w.weight i - w'.weight i|) * Y := by
          simpa using (Finset.sum_mul Finset.univ (fun i ↦ |w.weight i - w'.weight i|) Y).symm
    _ = Y * ∑ i, |w.weight i - w'.weight i| := by ring

/-- Weight-vector `ℓ¹` control implies predictor stability for a fixed bounded
response vector. This is the algebraic core needed for Appendix C.4 equation
`(20)` once a separate weight-stability estimate is supplied. -/
theorem Fit.budget_stability [Nonempty ι]
    (φ ψ : Fit.Params) (s r y : ι → ℝ)
    (hαφ : 0 < φ.alpha) (hγφ : 0 ≤ φ.gamma)
    (hαψ : 0 < ψ.alpha) (hγψ : 0 ≤ ψ.gamma)
    {Y Λ : ℝ} (hY0 : 0 ≤ Y) (_hΛ0 : 0 ≤ Λ)
    (hY : ∀ i, |y i| ≤ Y)
    (hbudget :
      ∑ i, |(Fit.weights φ s r hαφ hγφ).weight i -
          (Fit.weights ψ s r hαψ hγψ).weight i|
        ≤ Λ * Fit.distance φ ψ) :
    |Fit.predict φ s r y hαφ hγφ - Fit.predict ψ s r y hαψ hγψ|
      ≤ (Y * Λ) * Fit.distance φ ψ := by
  calc
    |Fit.predict φ s r y hαφ hγφ - Fit.predict ψ s r y hαψ hγψ|
      ≤ Y *
          ∑ i, |(Fit.weights φ s r hαφ hγφ).weight i -
            (Fit.weights ψ s r hαψ hγψ).weight i| := by
              exact Fit.response_bound
                (Fit.weights φ s r hαφ hγφ)
                (Fit.weights ψ s r hαψ hγψ) y hY
    _ ≤ Y * (Λ * Fit.distance φ ψ) := by
          exact mul_le_mul_of_nonneg_left hbudget hY0
    _ = (Y * Λ) * Fit.distance φ ψ := by ring

/-- Appendix C.4 equation `(19)`: on a compact parameter box, the ENN fitted
normalized weights are Lipschitz in the parameter pair, under uniform squared
radius and squared-noise-scale bounds. -/
theorem Fit.weights_l1 [Nonempty ι]
    (B : Fit.Box) {φ ψ : Fit.Params}
    (hφ : Fit.InBox B φ) (hψ : Fit.InBox B ψ)
    (s r : ι → ℝ) {S2 R2 : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2) :
    ∑ i, |(Fit.weights φ s r
          (lt_of_lt_of_le B.alpha_pos hφ.alpha_lower) hφ.gamma_nonneg).weight i -
        (Fit.weights ψ s r
          (lt_of_lt_of_le B.alpha_pos hψ.alpha_lower) hψ.gamma_nonneg).weight i|
      ≤ (2 * (B.alpha_hi + S2 + B.gamma_hi * R2) *
          ((B.alpha_lo⁻¹) ^ 2 * (1 + R2))) * Fit.distance φ ψ := by
  let rawφ : ι → ℝ := Variance.raw φ.alpha φ.gamma s r
  let rawψ : ι → ℝ := Variance.raw ψ.alpha ψ.gamma s r
  let V : ℝ := B.alpha_hi + S2 + B.gamma_hi * R2
  let C : ℝ := (B.alpha_lo⁻¹) ^ 2 * (1 + R2)
  have hcard_pos : 0 < (Fintype.card ι : ℝ) := by
    exact_mod_cast Fintype.card_pos
  have hS20 : 0 ≤ S2 := by
    let i0 : ι := Classical.choice ‹Nonempty ι›
    exact le_trans (sq_nonneg (s i0)) (hS2 i0)
  have hR20 : 0 ≤ R2 := by
    let i0 : ι := Classical.choice ‹Nonempty ι›
    exact le_trans (sq_nonneg (r i0)) (hR2 i0)
  have hVpos : 0 < V := by
    have hγhi0 : 0 ≤ B.gamma_hi := le_trans B.gamma_nonneg B.gamma_bounds
    have hαhi_pos : 0 < B.alpha_hi := lt_of_lt_of_le B.alpha_pos B.alpha_bounds
    have hmul : 0 ≤ B.gamma_hi * R2 := mul_nonneg hγhi0 hR20
    dsimp [V]
    linarith
  have hrawψ_pos : ∀ i, 0 < rawψ i := by
    intro i
    dsimp [rawψ]
    exact Variance.raw_positive ψ.alpha ψ.gamma s r
      (lt_of_lt_of_le B.alpha_pos hψ.alpha_lower) hψ.gamma_nonneg i
  have hdenom :
      (Fintype.card ι : ℝ) / V ≤ ∑ i, rawφ i := by
    simpa [rawφ, V] using
      Fit.sum_lower B hφ s r hS2 hR2
  have hrawsum :
      ∑ i, |rawφ i - rawψ i|
        ≤ (Fintype.card ι : ℝ) * (C * Fit.distance φ ψ) := by
    simpa [rawφ, rawψ, C] using
      Fit.sum_bound B hφ hψ s r hR2
  have hnorm :
      ∑ i, |normalize rawφ i - normalize rawψ i|
        ≤ 2 * (∑ i, |rawφ i - rawψ i|) / ((Fintype.card ι : ℝ) / V) :=
    Fit.normalize_bound rawφ rawψ hrawψ_pos
      (div_pos hcard_pos hVpos) hdenom
  have hcard_ne : (Fintype.card ι : ℝ) ≠ 0 := by exact_mod_cast (Fintype.card_pos).ne'
  calc
    ∑ i, |(Fit.weights φ s r
          (lt_of_lt_of_le B.alpha_pos hφ.alpha_lower) hφ.gamma_nonneg).weight i -
        (Fit.weights ψ s r
          (lt_of_lt_of_le B.alpha_pos hψ.alpha_lower) hψ.gamma_nonneg).weight i|
      = ∑ i, |normalize rawφ i - normalize rawψ i| := by
          simp [Fit.weights, Variance.weights, ofPositive, rawφ, rawψ]
    _ ≤ 2 * (∑ i, |rawφ i - rawψ i|) / ((Fintype.card ι : ℝ) / V) := hnorm
    _ ≤ 2 * ((Fintype.card ι : ℝ) * (C * Fit.distance φ ψ)) / ((Fintype.card ι : ℝ) / V) := by
          exact div_le_div_of_nonneg_right
            (mul_le_mul_of_nonneg_left hrawsum (by positivity))
            (div_nonneg hcard_pos.le hVpos.le)
    _ = (2 * V * C) * Fit.distance φ ψ := by
          field_simp [hcard_ne, hVpos.ne', V, C]
    _ = (2 * (B.alpha_hi + S2 + B.gamma_hi * R2) *
          ((B.alpha_lo⁻¹) ^ 2 * (1 + R2))) * Fit.distance φ ψ := by rfl

/-- Appendix C.4 equation `(20)`: bounded responses plus the fitted-weight
stability estimate imply finite-sample predictor stability on the parameter
box. -/
theorem Fit.predictor_l1 [Nonempty ι]
    (B : Fit.Box) {φ ψ : Fit.Params}
    (hφ : Fit.InBox B φ) (hψ : Fit.InBox B ψ)
    (s r y : ι → ℝ) {S2 R2 Y : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2)
    (hY0 : 0 ≤ Y) (hY : ∀ i, |y i| ≤ Y) :
    |Fit.predict φ s r y
        (lt_of_lt_of_le B.alpha_pos hφ.alpha_lower) hφ.gamma_nonneg -
      Fit.predict ψ s r y
        (lt_of_lt_of_le B.alpha_pos hψ.alpha_lower) hψ.gamma_nonneg|
      ≤ ((Y * (2 * (B.alpha_hi + S2 + B.gamma_hi * R2) *
          ((B.alpha_lo⁻¹) ^ 2 * (1 + R2)))) * Fit.distance φ ψ) := by
  have hS20 : 0 ≤ S2 := by
    let i0 : ι := Classical.choice ‹Nonempty ι›
    exact le_trans (sq_nonneg (s i0)) (hS2 i0)
  have hR20 : 0 ≤ R2 := by
    let i0 : ι := Classical.choice ‹Nonempty ι›
    exact le_trans (sq_nonneg (r i0)) (hR2 i0)
  have hγhi0 : 0 ≤ B.gamma_hi := le_trans B.gamma_nonneg B.gamma_bounds
  have hαhi_pos : 0 < B.alpha_hi := lt_of_lt_of_le B.alpha_pos B.alpha_bounds
  have hΛ0 :
      0 ≤ 2 * (B.alpha_hi + S2 + B.gamma_hi * R2) * ((B.alpha_lo⁻¹) ^ 2 * (1 + R2)) := by
    have hV0 : 0 ≤ B.alpha_hi + S2 + B.gamma_hi * R2 := by
      have hmul : 0 ≤ B.gamma_hi * R2 := mul_nonneg hγhi0 hR20
      linarith
    have hC0 : 0 ≤ (B.alpha_lo⁻¹) ^ 2 * (1 + R2) := by
      have hR1 : 0 ≤ 1 + R2 := by linarith
      exact mul_nonneg (sq_nonneg _) hR1
    exact mul_nonneg (mul_nonneg (by positivity) hV0) hC0
  refine Fit.budget_stability
    φ ψ s r y
    (lt_of_lt_of_le B.alpha_pos hφ.alpha_lower) hφ.gamma_nonneg
    (lt_of_lt_of_le B.alpha_pos hψ.alpha_lower) hψ.gamma_nonneg
    hY0 hΛ0 hY ?_
  exact Fit.weights_l1 B hφ hψ s r hS2 hR2

/-- The weighted-kNN predictor, restricted to the compact parameter rectangle. -/
noncomputable def Fit.box_predict [Nonempty ι]
    (B : Fit.Box) (s r y : ι → ℝ) :
    Fit.region B → ℝ := fun p ↦
  let φ : Fit.Params := Fit.Params.ofProd p.1
  have hφ : Fit.InBox B φ := by
    exact (Fit.mem_region B φ).2 (by simp [φ] at p ⊢)
  Fit.predict φ s r y
    (lt_of_lt_of_le B.alpha_pos hφ.alpha_lower) hφ.gamma_nonneg

/-- The concrete C.4 predictor is Lipschitz on the compact parameter
rectangle. -/
theorem Fit.predict_lipschitz [Nonempty ι]
    (B : Fit.Box) (s r y : ι → ℝ) {S2 R2 Y : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2)
    (hY0 : 0 ≤ Y) (hY : ∀ i, |y i| ≤ Y)
    (p q : Fit.region B) :
    dist (Fit.box_predict B s r y p) (Fit.box_predict B s r y q)
      ≤ (2 * (Y * (2 * (B.alpha_hi + S2 + B.gamma_hi * R2) *
          ((B.alpha_lo⁻¹) ^ 2 * (1 + R2))))) * dist p q := by
  let φ : Fit.Params := Fit.Params.ofProd p.1
  let ψ : Fit.Params := Fit.Params.ofProd q.1
  have hφ : Fit.InBox B φ := by
    exact (Fit.mem_region B φ).2 (by simp [φ] at p ⊢)
  have hψ : Fit.InBox B ψ := by
    exact (Fit.mem_region B ψ).2 (by simp [ψ] at q ⊢)
  have hR20 : 0 ≤ R2 := by
    let i0 : ι := Classical.choice ‹Nonempty ι›
    exact le_trans (sq_nonneg (r i0)) (hR2 i0)
  have hV0 : 0 ≤ B.alpha_hi + S2 + B.gamma_hi * R2 := by
    have hS20 : 0 ≤ S2 := by
      let i0 : ι := Classical.choice ‹Nonempty ι›
      exact le_trans (sq_nonneg (s i0)) (hS2 i0)
    have hγhi0 : 0 ≤ B.gamma_hi := le_trans B.gamma_nonneg B.gamma_bounds
    have hαhi0 : 0 ≤ B.alpha_hi := le_of_lt (lt_of_lt_of_le B.alpha_pos B.alpha_bounds)
    have hmul : 0 ≤ B.gamma_hi * R2 := mul_nonneg hγhi0 hR20
    linarith
  have hC0 : 0 ≤ 2 * (B.alpha_hi + S2 + B.gamma_hi * R2) * ((B.alpha_lo⁻¹) ^ 2 * (1 + R2)) := by
    have hR1 : 0 ≤ 1 + R2 := by linarith
    have hinner : 0 ≤ (B.alpha_lo⁻¹) ^ 2 * (1 + R2) := mul_nonneg (sq_nonneg _) hR1
    exact mul_nonneg (mul_nonneg (by positivity) hV0) hinner
  have hM0 :
      0 ≤ Y * (2 * (B.alpha_hi + S2 + B.gamma_hi * R2) *
          ((B.alpha_lo⁻¹) ^ 2 * (1 + R2))) := by
    exact mul_nonneg hY0 hC0
  have hparam : Fit.distance φ ψ ≤ 2 * dist p q := by
    simpa [φ, ψ, Subtype.dist_eq] using Fit.distance_bound p.1 q.1
  have hbase :=
    Fit.predictor_l1 B hφ hψ s r y hS2 hR2 hY0 hY
  calc
    dist (Fit.box_predict B s r y p) (Fit.box_predict B s r y q)
      = |Fit.box_predict B s r y p - Fit.box_predict B s r y q| := by
          rw [Real.dist_eq]
    _ = |Fit.predict φ s r y
          (lt_of_lt_of_le B.alpha_pos hφ.alpha_lower) hφ.gamma_nonneg -
        Fit.predict ψ s r y
          (lt_of_lt_of_le B.alpha_pos hψ.alpha_lower) hψ.gamma_nonneg| := by
          simp [Fit.box_predict, φ, ψ]
    _ ≤ (Y * (2 * (B.alpha_hi + S2 + B.gamma_hi * R2) *
          ((B.alpha_lo⁻¹) ^ 2 * (1 + R2)))) * Fit.distance φ ψ := hbase
    _ ≤ (Y * (2 * (B.alpha_hi + S2 + B.gamma_hi * R2) *
          ((B.alpha_lo⁻¹) ^ 2 * (1 + R2)))) * (2 * dist p q) := by
          exact mul_le_mul_of_nonneg_left hparam hM0
    _ = (2 * (Y * (2 * (B.alpha_hi + S2 + B.gamma_hi * R2) *
          ((B.alpha_lo⁻¹) ^ 2 * (1 + R2))))) * dist p q := by ring

/-- Hence the box-restricted predictor is continuous. -/
theorem Fit.predict_continuous [Nonempty ι]
    (B : Fit.Box) (s r y : ι → ℝ) {S2 R2 Y : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2)
    (hY0 : 0 ≤ Y) (hY : ∀ i, |y i| ≤ Y) :
    Continuous (Fit.box_predict B s r y) := by
  exact (LipschitzWith.of_dist_le' fun p q ↦
    Fit.predict_lipschitz B s r y hS2 hR2 hY0 hY p q).continuous

/-- Hence the box-restricted predictor is Borel measurable. -/
theorem Fit.predict_measurable [Nonempty ι]
    (B : Fit.Box) (s r y : ι → ℝ) {S2 R2 Y : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2)
    (hY0 : 0 ≤ Y) (hY : ∀ i, |y i| ≤ Y) :
    Measurable (Fit.box_predict B s r y) :=
  (Fit.predict_continuous B s r y hS2 hR2 hY0 hY).measurable

/-- The box-restricted predictor is almost-everywhere measurable with respect
to any measure on the parameter box. -/
theorem Fit.predict_aemeasurable [Nonempty ι]
    (B : Fit.Box) (s r y : ι → ℝ) {S2 R2 Y : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2)
    (hY0 : 0 ≤ Y) (hY : ∀ i, |y i| ≤ Y)
    (μ : Measure (Fit.region B)) :
    AEMeasurable (Fit.box_predict B s r y) μ :=
  (Fit.predict_measurable B s r y hS2 hR2 hY0 hY).aemeasurable

/-- The concrete C.4 weighted-kNN predictor attains a maximum on the compact
parameter rectangle. -/
theorem Fit.predict_max [Nonempty ι]
    (B : Fit.Box) (s r y : ι → ℝ) {S2 R2 Y : ℝ}
    (hS2 : ∀ i, (s i) ^ 2 ≤ S2) (hR2 : ∀ i, (r i) ^ 2 ≤ R2)
    (hY0 : 0 ≤ Y) (hY : ∀ i, |y i| ≤ Y) :
    ∃ p : Fit.region B, IsMaxOn (Fit.box_predict B s r y) Set.univ p := by
  obtain ⟨p, _, hpmax⟩ :=
    isCompact_univ.exists_isMaxOn Set.univ_nonempty
      (Fit.predict_continuous B s r y hS2 hR2 hY0 hY).continuousOn
  exact ⟨p, hpmax⟩

end DeterministicStability

end ENN
