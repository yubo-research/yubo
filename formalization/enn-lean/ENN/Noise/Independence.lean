import ENN.WeightedKNN
import Mathlib.Probability.Independence.Conditional
import Mathlib.Probability.Independence.Integration

/-!
# Conditional independence and product moments
-/

namespace ENN

open Filter MeasureTheory

open ProbabilityTheory Topology

section ConditionalIndependence

variable {Ω : Type*} [mΩ : MeasurableSpace Ω] [StandardBorelSpace Ω]
variable {μ : Measure Ω} [IsFiniteMeasure μ]

/-- Conditional independence factors conditional expectations of integrable
products.  Mathlib defines conditional independence through factorization of
the conditional joint law; this lemma converts that kernel-level statement
into the product identity used in Appendix C, Theorem 2. -/
theorem Noise.condExp_mul
    {m : MeasurableSpace Ω} (hm : m ≤ mΩ)
    {f g : Ω → ℝ}
    (hf : @Measurable Ω ℝ mΩ _ f) (hg : @Measurable Ω ℝ mΩ _ g)
    (hf_int : Integrable f μ) (hg_int : Integrable g μ)
    (hmul_int : Integrable (fun ω ↦ f ω * g ω) μ)
    (hfg : CondIndepFun m hm f g μ) :
    @condExp Ω ℝ m mΩ _ _ μ (fun ω ↦ f ω * g ω) =ᵐ[μ]
      fun ω ↦ (@condExp Ω ℝ m mΩ _ _ μ f ω) *
        (@condExp Ω ℝ m mΩ _ _ μ g ω) := by
  have hmap := ae_eq_of_ae_eq_trim
    ((condIndepFun_iff_map_prod_eq_prod_map_map (mΩ := mΩ) hf hg).mp hfg)
  have hprod_cond :=
    condExp_ae_eq_integral_condExpKernel (mΩ := mΩ) hm hmul_int
  have hf_cond := condExp_ae_eq_integral_condExpKernel (mΩ := mΩ) hm hf_int
  have hg_cond := condExp_ae_eq_integral_condExpKernel (mΩ := mΩ) hm hg_int
  filter_upwards [hmap, hprod_cond, hf_cond, hg_cond] with
      ω hmapω hprodω hfω hgω
  rw [hprodω, hfω, hgω]
  have hindep : IndepFun f g (condExpKernel (mΩ := mΩ) μ m ω) := by
    apply (indepFun_iff_map_prod_eq_prod_map_map
      hf.aemeasurable hg.aemeasurable).2
    rw [Kernel.map_apply _ (hf.prodMk hg), Kernel.prod_apply,
      Kernel.map_apply _ hf, Kernel.map_apply _ hg] at hmapω
    exact hmapω
  exact hindep.integral_mul_eq_mul_integral
    hf.aestronglyMeasurable hg.aestronglyMeasurable

/-- Conditionally independent, conditionally centered real random variables
have zero conditional cross moment. -/
theorem Noise.condExp_zero
    {m : MeasurableSpace Ω} (hm : m ≤ mΩ)
    {f g : Ω → ℝ}
    (hf : @Measurable Ω ℝ mΩ _ f) (hg : @Measurable Ω ℝ mΩ _ g)
    (hf_int : Integrable f μ) (hg_int : Integrable g μ)
    (hmul_int : Integrable (fun ω ↦ f ω * g ω) μ)
    (hfg : CondIndepFun m hm f g μ)
    (hf_zero : @condExp Ω ℝ m mΩ _ _ μ f =ᵐ[μ] 0)
    (hg_zero : @condExp Ω ℝ m mΩ _ _ μ g =ᵐ[μ] 0) :
    @condExp Ω ℝ m mΩ _ _ μ (fun ω ↦ f ω * g ω) =ᵐ[μ] 0 := by
  calc
    @condExp Ω ℝ m mΩ _ _ μ (fun ω ↦ f ω * g ω)
        =ᵐ[μ] fun ω ↦ (@condExp Ω ℝ m mΩ _ _ μ f ω) *
          (@condExp Ω ℝ m mΩ _ _ μ g ω) :=
      Noise.condExp_mul (mΩ := mΩ) hm hf hg
        hf_int hg_int hmul_int hfg
    _ =ᵐ[μ] fun _ ↦ (0 : ℝ) * 0 := hf_zero.mul hg_zero
    _ =ᵐ[μ] 0 := Filter.Eventually.of_forall fun _ ↦ by simp

end ConditionalIndependence

end ENN
