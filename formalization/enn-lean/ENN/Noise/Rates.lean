import ENN.Noise.Consistency

/-!
# Radius and weight rates
-/

namespace ENN

open Filter MeasureTheory

open ProbabilityTheory Topology

section RateNotation

/-- Paper notation `uₙ ≲ vₙ`: eventually `uₙ` is bounded above by a fixed
nonnegative constant times `vₙ`. -/
def RateBound (u v : ℕ → ℝ) : Prop :=
  ∃ C, 0 ≤ C ∧ ∀ᶠ n in Filter.atTop, u n ≤ C * v n

theorem RateBound.refl (u : ℕ → ℝ) : RateBound u u := by
  exact ⟨1, zero_le_one, Filter.Eventually.of_forall fun _ ↦ by simp⟩

theorem RateBound.add
    {u₁ u₂ v₁ v₂ : ℕ → ℝ}
    (h₁ : RateBound u₁ v₁) (h₂ : RateBound u₂ v₂)
    (hv₁ : ∀ᶠ n in Filter.atTop, 0 ≤ v₁ n)
    (hv₂ : ∀ᶠ n in Filter.atTop, 0 ≤ v₂ n) :
    RateBound (fun n ↦ u₁ n + u₂ n) (fun n ↦ v₁ n + v₂ n) := by
  rcases h₁ with ⟨C₁, hC₁, h₁⟩
  rcases h₂ with ⟨C₂, hC₂, h₂⟩
  refine ⟨max C₁ C₂, hC₁.trans (le_max_left _ _), ?_⟩
  filter_upwards [h₁, h₂, hv₁, hv₂] with n hn₁ hn₂ hvn₁ hvn₂
  calc
    u₁ n + u₂ n ≤ C₁ * v₁ n + C₂ * v₂ n := add_le_add hn₁ hn₂
    _ ≤ max C₁ C₂ * v₁ n + max C₁ C₂ * v₂ n := by
      gcongr <;> simp
    _ = max C₁ C₂ * (v₁ n + v₂ n) := by ring

/-- Appendix C, Theorem 2(ii), at the paper's `≲` abstraction level. -/
theorem Noise.rate
    {mse R W rRate invK : ℕ → ℝ}
    {L σ2 : ℝ}
    (hσ2 : 0 ≤ σ2)
    (hmse_le : ∀ n, mse n ≤ L ^ 2 * R n + σ2 * W n)
    (hradius : RateBound R rRate)
    (hweight : RateBound W invK)
    (hrate0 : ∀ᶠ n in Filter.atTop, 0 ≤ rRate n)
    (hinv0 : ∀ᶠ n in Filter.atTop, 0 ≤ invK n) :
    RateBound mse (fun n ↦ rRate n + invK n) := by
  rcases hradius with ⟨Cr, hCr, hradius⟩
  rcases hweight with ⟨Cw, hCw, hweight⟩
  let C := max (L ^ 2 * Cr) (σ2 * Cw)
  have hC : 0 ≤ C :=
    (mul_nonneg (sq_nonneg L) hCr).trans (le_max_left _ _)
  refine ⟨C, hC, ?_⟩
  filter_upwards [hradius, hweight, hrate0, hinv0] with n hr hw har hak
  have hr' : L ^ 2 * R n ≤ (L ^ 2 * Cr) * rRate n := by
    calc
      L ^ 2 * R n ≤ L ^ 2 * (Cr * rRate n) :=
        mul_le_mul_of_nonneg_left hr (sq_nonneg L)
      _ = (L ^ 2 * Cr) * rRate n := by ring
  have hw' : σ2 * W n ≤ (σ2 * Cw) * invK n := by
    calc
      σ2 * W n ≤ σ2 * (Cw * invK n) :=
        mul_le_mul_of_nonneg_left hw hσ2
      _ = (σ2 * Cw) * invK n := by ring
  calc
    mse n ≤ L ^ 2 * R n + σ2 * W n := hmse_le n
    _ ≤ (L ^ 2 * Cr) * rRate n + (σ2 * Cw) * invK n := add_le_add hr' hw'
    _ ≤ C * rRate n + C * invK n := by
      dsimp [C]
      gcongr <;> simp
    _ = C * (rRate n + invK n) := by ring

/-- If both terms in the local-rate bound have the same eventual rate, their
sum has that rate as well.  This is the last algebraic step in the displayed
`n^{-2/(q+2)}` conclusion of Appendix C, Theorem 2(ii). -/
theorem Noise.balanced
    {mse rRate invK rate : ℕ → ℝ}
    (hmse : RateBound mse (fun n ↦ rRate n + invK n))
    (hradius : RateBound rRate rate)
    (hinverse : RateBound invK rate)
    (hrate0 : ∀ᶠ n in Filter.atTop, 0 ≤ rate n) :
    RateBound mse rate := by
  rcases hmse with ⟨Cm, hCm, hmse⟩
  rcases hradius with ⟨Cr, hCr, hradius⟩
  rcases hinverse with ⟨Ck, hCk, hinverse⟩
  refine ⟨Cm * (Cr + Ck), mul_nonneg hCm (add_nonneg hCr hCk), ?_⟩
  filter_upwards [hmse, hradius, hinverse, hrate0] with n hm hr hk hrate
  calc
    mse n ≤ Cm * (rRate n + invK n) := hm
    _ ≤ Cm * (Cr * rate n + Ck * rate n) := by
      gcongr
    _ = (Cm * (Cr + Ck)) * rate n := by ring

/-- Algebra behind the paper's balanced choice: when the effective neighbor
count has exponent `2 / (q + 2)`, the squared radius scale has exponent
`-2 / (q + 2)`. -/
theorem Noise.radius_exp
    {n q : ℝ} (hn : 0 < n) (hq : 0 < q) :
    ((((n ^ (2 / (q + 2) : ℝ)) / n) ^ (1 / q : ℝ)) ^ (2 : ℕ)) =
      n ^ (-2 / (q + 2) : ℝ) := by
  have hq0 : q ≠ 0 := ne_of_gt hq
  have hq2 : q + 2 ≠ 0 := by linarith
  calc
    (((n ^ (2 / (q + 2) : ℝ) / n) ^ (1 / q : ℝ)) ^ (2 : ℕ))
        = ((n ^ ((2 / (q + 2) : ℝ) - 1)) ^ (1 / q : ℝ)) ^ (2 : ℕ) := by
            rw [Real.rpow_sub hn]
            simp
    _ = (n ^ (((2 / (q + 2) : ℝ) - 1) * (1 / q))) ^ (2 : ℕ) := by
      rw [Real.rpow_mul hn.le]
    _ = n ^ ((((2 / (q + 2) : ℝ) - 1) * (1 / q)) * 2) := by
      rw [← Real.rpow_natCast, ← Real.rpow_mul hn.le]
      norm_num
    _ = n ^ (-2 / (q + 2) : ℝ) := by
      congr 1
      field_simp
      ring

/-- Under the same balanced choice, the inverse neighbor count has the same
paper rate exponent. -/
theorem Noise.inv_count_exp
    {n q : ℝ} (hn : 0 < n) :
    (n ^ (2 / (q + 2) : ℝ))⁻¹ = n ^ (-2 / (q + 2) : ℝ) := by
  calc
    (n ^ (2 / (q + 2) : ℝ))⁻¹ = n ^ (-(2 / (q + 2)) : ℝ) :=
      (Real.rpow_neg hn.le (2 / (q + 2))).symm
    _ = n ^ (-2 / (q + 2) : ℝ) := by
      congr 1
      ring

end RateNotation

end ENN
