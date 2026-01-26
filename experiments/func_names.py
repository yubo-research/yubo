funcs_nd = [
    "ackley",
    "dixonprice",
    "griewank",
    "levy",
    "michalewicz",
    "rastrigin",
    "rosenbrock",
    "sphere",
    "stybtang",
]
funcs_1d = [
    "ackley",
    "dixonprice",
    "griewank",
    "levy",
    "rastrigin",
    "sphere",
    "stybtang",
]


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
    # "perm",
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
    # "permdbeta",
    "powell",
    "shekel",
    "stybtang",
]

# 51 functions
func_lists = [
    funcs_multimodal,
    funcs_bowl,
    funcs_plate,
    funcs_valley,
    funcs_ridges,
    funcs_other,
]
funcs_all = [f for fl in func_lists for f in fl]
# 6 Functions
func_brief = [fn[0] for fn in func_lists]
func_brief_2 = [
    "ackley",
    "rastrigin",
    "sphere",
    "trid",
    "booth",
    "mccormick",
    "dixonprice",
    "rosenbrock",
    "dejong5",
    "easom",
    "branin",
    "stybtang",
]
