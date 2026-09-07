import ENN.PseudoBO.Axioms

/-!
# Counterexamples for the PseudoBO bridge audit

This file contains small checked counterexamples showing where the current
Appendix C bridge fails without extra hypotheses.
-/

namespace ENN
namespace PseudoBO

/-- A minimal counterexample to the naive evaluation-worthiness bridge for the
Appendix C auxiliary acquisition. Even when the query is repeatedly equal to a
sampled point and `σ = 0`, the baseline `m = inf μ = 0` can leave a strictly
positive acquisition value. -/
theorem baseline_counterexample
    {lam : ℝ} (hlam : 0 < lam) :
    ¬ Worthwhile
      (X := ℝ) (nSeq := fun _ => 1) (data := fun _ _ => (1 : ℝ))
      (fun _ x => Acq.auxiliary (fun y : ℝ => max y 0) (fun _ => 0) 0 lam x) := by
  intro hW
  let query : ℕ → ℝ := fun _ => 1
  have hnear :
      NearData (X := ℝ) (nSeq := fun _ => 1) (data := fun _ _ => (1 : ℝ)) query := by
    refine ⟨fun _ => ⟨0, by simp⟩, ?_⟩
    simp [query]
  have hzero :
      Filter.Tendsto
        (fun n ↦ Acq.auxiliary (fun y : ℝ => max y 0) (fun _ => 0) 0 lam (query n))
        Filter.atTop (nhds 0) :=
    hW (query := query) hnear
  have hconst0 : Filter.Tendsto (fun _ : ℕ => lam) Filter.atTop (nhds 0) := by
    simpa [query, Acq.auxiliary, Acq.positive] using hzero
  have hlam_tend : Filter.Tendsto (fun _ : ℕ => lam) Filter.atTop (nhds lam) := tendsto_const_nhds
  have : lam = 0 := tendsto_nhds_unique hlam_tend hconst0
  exact hlam.ne' this

/-- A fixed exploration coefficient cannot satisfy the diverging-`βₙ`
hypothesis used in Chen–Lam style UCB propositions. This is the formal core of
the fixed-`β` versus diverging-`βₙ` audit note. -/
theorem fixed_beta (β : ℝ) :
    ¬ Filter.Tendsto (fun _ : ℕ => β) Filter.atTop Filter.atTop := by
  simpa using (Filter.not_tendsto_const_atTop β Filter.atTop)

end PseudoBO
end ENN
