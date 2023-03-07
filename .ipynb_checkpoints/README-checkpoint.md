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

#"random", "sobol", "minimax", "minimax-toroidal", "variance", "iopt_ei", "ioptv_ei",
#"idopt", "ei", "iei", "ucb", "iucb", "ax"

    Ackley,
    Beale,
    Branin,
    Bukin,
    CrossInTray,
    DixonPrice,
    DropWave,
    EggHolder,
    Griewank,
    GrLee12,
    Hartmann,
    HolderTable,
    Levy,
    Michalewicz,
    Powell,
    Rastrigin,
    Rosenbrock,
    Shekel,
    Shubert,
    SixHumpCamel,
    Sphere,
    StybTang,
    ThreeHumpCamel,