import ENN.ExactObservation
import ENN.AcquisitionGeometry
import ENN.WeightedKNN
import ENN.OrderedWeightedKNN
import ENN.FittedParameters
import ENN.Paper.AppendixCAsymptotics

/-!
# Paper-facing wrappers for Appendix C

This file exposes theorem names whose intent maps directly to Appendix C of
*Taking the GP Out of the Loop*. It does not hide known gaps: wrappers are only
provided for results that are actually formalized in the current development.
-/

namespace ENN.Paper

/-- C.1(i): mean error is bounded by a supplied monotone neighbor envelope. -/
abbrev Exact.approximation := @ENN.Exact.error_bound

/-- C.1(ii): uncertainty lies between the two nearest-distance bounds. -/
abbrev Exact.spacing := @ENN.Exact.spacing

/-- C.1(iii): fixed-query consistency on a compact domain for continuous signals. -/
abbrev Exact.consistency := @ENN.Exact.consistency

/-- C.1(iv): uncertainty vanishes as sampled points approach the query. -/
abbrev Exact.sneb := @ENN.Exact.sigma_limit

/-- Proposition 1(2): vanishing uncertainty and nonpositive gain limsup imply a vanishing score. -/
abbrev Acq.improvement := @ENN.Acq.vanishes

/-- C.3: conditional weighted-noise second-moment bound. -/
abbrev Noise.second_moment := @ENN.Noise.noise_var

/-- Theorem 2: conditional bias–variance bound. -/
abbrev Noise.error_bound := @ENN.Noise.mse

/-- C.3: measurable weighted sums of conditionally centered noises remain centered. -/
abbrev Noise.noise_mean := @ENN.Noise.noise_mean

/-- C.3: conditional orthogonality removes the off-diagonal noise moments. -/
abbrev Noise.noise_var_eq := @ENN.Noise.noise_var_eq

/-- C.3: a uniform second-moment bound controls the weighted noise. -/
abbrev Noise.noise_var := @ENN.Noise.noise_var

/-- Theorem 2: conditional MSE bound for supplied normalized random weights. -/
abbrev Noise.mse := @ENN.Noise.mse

/-- Theorem 2: instantiate the conditional MSE bound with ENN variance weights. -/
abbrev Noise.mse_weights := @ENN.Noise.mse_weights

/-- Theorem 2: concrete deterministic ordered neighbors with random scales and noise. -/
abbrev Noise.mse_ordered := @ENN.Noise.mse_ordered

/-- Theorem 2: supplied measurable random neighbors and a square-integrable radius. -/
abbrev Noise.mse_random := @ENN.Noise.mse_random

/-- Theorem 2: ENN weights for the supplied random-design interface. -/
abbrev Noise.mse_random_weights := @ENN.Noise.mse_random_weights

/-- C.3: conditional independence and centering imply conditional orthogonality. -/
abbrev Noise.orthogonal := @ENN.Noise.orthogonal

/-- Theorem 2: random-design ENN bound taking conditional independence directly. -/
abbrev Noise.mse_indep := @ENN.Noise.mse_indep

/-- Theorem 2(i): vanishing radius and weight budgets yield mean-square and probability limits. -/
abbrev Noise.consistency := @ENN.Noise.consistency

/-- Theorem 2(i): integrate the conditional bound, then apply the consistency transfer. -/
abbrev Noise.consistency_cond := @ENN.Noise.consistency_cond

/-- Theorem 2(ii): combine supplied bounds on the radius and weight budgets. -/
abbrev Noise.rate := @ENN.Noise.rate

/-- Theorem 2(ii): combine two terms bounded by the same final rate. -/
abbrev Noise.balanced := @ENN.Noise.balanced

/-- Theorem 2(ii): exponent algebra for the balanced squared-radius term. -/
abbrev Noise.radius_exp := @ENN.Noise.radius_exp

/-- Theorem 2(ii): exponent algebra for the inverse neighbor count. -/
abbrev Noise.inv_count_exp := @ENN.Noise.inv_count_exp

/-- C.4, equation (19): weight stability on the compact variance box. -/
abbrev Fit.weight_stability := @ENN.Fit.weights_l1

/-- C.4, equation (20): prediction stability for bounded responses. -/
abbrev Fit.predictor_stability := @ENN.Fit.predictor_l1

/-- Theorem 3: uniform objective convergence and target separation imply argmax convergence. -/
abbrev Fit.argmax_limit := @ENN.Fit.argmax_limit

/-- Theorem 3: argmax transfer using the variance parameters and their ℓ¹ distance. -/
abbrev Fit.param_argmax := @ENN.Fit.param_argmax

/-- Theorem 3: oracle probability convergence transfers under a bounded stability scale. -/
abbrev Fit.fitted_prob := @ENN.Fit.fitted_prob

/-- Theorem 3: oracle mean-square convergence transfers under weighted parameter-error control. -/
abbrev Fit.fitted_l2 := @ENN.Fit.fitted_l2

/-- C.4: construct an exact measurable maximizer of a jointly measurable continuous objective. -/
noncomputable abbrev Select.argmax := @ENN.Select.argmax

/-- C.4: measurability of the attained maximum over the compact parameter box. -/
abbrev Select.max_measurable := @ENN.Select.max_measurable

/-- C.4: measurable approximate maximizers, constructed from a dense sequence. -/
abbrev Select.exists_approx := @ENN.Select.exists_approx

/-- C.4: approximate objective values converge to the maximum under the canonical schedule. -/
abbrev Select.value_limit := @ENN.Select.schedule_limit

/-- C.4: exact measurable selection under pointwise uniqueness. -/
noncomputable abbrev Select.unique_argmax := @ENN.Select.unique_argmax

/-- C.4: approximate selectors converge to the unique maximizer. -/
abbrev Select.unique_limit := @ENN.Select.unique_limit

/-- C.4: measurable predictions using the constructed exact selector and fixed data arrays. -/
abbrev Fit.selected_measurable := @ENN.Fit.argmax_measurable

/-- C.4: measurable predictions using the unique-maximizer construction. -/
abbrev Fit.unique_measurable := @ENN.Fit.unique_measurable

/-- C.4: transfer convergence in measure from supplied selectors to predictions. -/
abbrev Fit.selected_limit := @ENN.Fit.selected_limit

/-- C.4: transfer a square-integrability bound on selector distance to prediction error. -/
abbrev Fit.selected_l2 := @ENN.Fit.selected_l2

end ENN.Paper
