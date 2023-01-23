# Bayesian optimization


- Can you do better than space-filling initialization?
  - Can you be purely stateless? (Sobol isn't, really)
  - Can you optimize each design instead of choosing an ad-hoc/arbitrary number of initialization points?
  
- Possible design methods
  - minimax dist
  - maximin dist
  - minimal integrated posterior variance
  - toroidal distance
  - action-space distance for RL problems; non-parameter distance/similarity in general (think of simulation optimization)

