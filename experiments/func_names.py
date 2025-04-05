from problems.benchmark_functions import all_benchmarks

funcs_nd = ["ackley", "dixonprice", "griewank", "levy", "michalewicz", "rastrigin", "rosenbrock", "sphere", "stybtang"]
funcs_1d = ["ackley", "dixonprice", "griewank", "levy", "rastrigin", "sphere", "stybtang"]




funcs_multimodal = [
    "ackley",
    "bukin",
    "crossintray",
    "dropwave",
    "eggholder",
    "grlee12",
    "griewank",
    "holdertable",
    "langerman",
    "levy",
    "levy13",
    "rastrigin",
    "schaffer2",
    "schaffer4",
    "schwefel",
    "shubert",
]
funcs_bowl = [
    "bohachevsky1",
    "perm",
    "rotatedhyperellipsoid",
    "sphere",
    "sumofdifferentpowers",
    "sum_squares",
    "trid",
]
funcs_plate = [
    "booth",
    "matyas",
    "mccormick",
    "powersum",
    "zakharov",
]
funcs_valley = [
    "threehumpcamel",
    "sixhumpcamel",
    "dixonprice",
    "rosenbrock",
]
funcs_ridges = [
    "dejong5",
    "easom",
    "michalewicz",
]
funcs_other = [
    "beale",
    "branin",
    "colville",
    # "forrester",
    "goldsteinprice",
    "hartmann",
    "permdbeta",
    "powell",
    "shekel",
    "stybtang",
]

# 51 functions
funcs_all = funcs_multimodal + funcs_bowl + funcs_plate + funcs_valley + funcs_ridges + funcs_other
