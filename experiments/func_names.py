from problems.benchmark_functions import all_benchmarks

funcs_nd = ["ackley", "dixonprice", "griewank", "levy", "michalewicz", "rastrigin", "rosenbrock", "sphere", "stybtang"]
funcs_1d = ["ackley", "dixonprice", "griewank", "levy", "rastrigin", "sphere", "stybtang"]

funcs_36 = list(all_benchmarks().keys())


funcs_multimodal = [
    "ackley",
    "bukin",
    "crossintray",
    "dropwave",
    "eggholder",
    "grlee12",
    "griewank",
    "holdertable",
    "langermann",
    "levy",
    "levy13",
    "rastrigin",
    "schaffer2",
    "schaffer4",
    "schwefel",
    "shubert",
]
funcs_bowl = [
    "bohachevsky",
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
