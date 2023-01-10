
# Behavioral Bayesian Optimization

Optimize over objects in a metric space (instead of parameters in a parameter space).

## Example
Optimize over RL policies in the space of policy behavior.  

Capture differences in policy behaviors by a distance function, d(pi_1, pi_2). Build a GPR
surrogate of episode return over d(pi_1, pi_2). Optimize policy parameters to maximize and acquisition function
over the surrogate.

Ex distance: d(pi_1, pi_2) = distance between actions over the states visited by one or both of the policies.


