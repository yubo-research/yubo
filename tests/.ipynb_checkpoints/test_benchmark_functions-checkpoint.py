import numpy as np

def _test(problem_name, num_dim, x, y):
    from problems.pure_functions import make

    fn = make(f"f:{problem_name}-{num_dim}d", seed=17)
    y_check = fn.step(x)[1]
    assert abs(y_check - y) < 1e-6, (x, y_check, y)


# def test_michalewicz():
#     _test(problem_name="michalewicz", num_dim=2, x=np.array([0.3, 0.3]), y= 0.17602377327570407)

# def test_sphere():
#     _test(problem_name="sphere", num_dim=2, x=np.array([0.3, 0.3]), y= -0.17939358802334188)
def test_sphere():
    _test(problem_name="sphere", num_dim=2, x=np.array([0, 0]), y= 0)

# def test_ackley():
#     _test(problem_name="ackley", num_dim=2, x=np.array([0.3, 0.3]), y= -18.38383765911014)

# def test_beale():
#     _test(problem_name="beale", num_dim=2, x=np.array([0.3, 0.3]), y= -131.3849191667202)

# def test_bukin():
#     _test(problem_name="bukin", num_dim=2, x=np.array([0.3, 0.3]), y= -69.86337581613658)

# def test_crossintray():
#     _test(problem_name="crossintray", num_dim=2, x=np.array([0.3, 0.3]), y= 1.725660628352792)

# def test_dropwave():
#     _test(problem_name="dropwave", num_dim=2, x=np.array([0.3, 0.3]), y= 0.24225378631433528)

# def test_dixonprice():
#     _test(problem_name="dixonprice", num_dim=2, x=np.array([0.3, 0.3]), y= -8691.931212430023)

# def test_eggholder():
#     _test(problem_name="eggholder", num_dim=2, x=np.array([0.3, 0.3]), y= -400.19934589526787)

# def test_griewank():
#     _test(problem_name="griewank", num_dim=2, x=np.array([0.3, 0.3]), y= -32.87451239862118)

# def test_holdertable():
#     _test(problem_name="holdertable", num_dim=2, x=np.array([0.3, 0.3]), y= 1.834294348679241)

# def test_levy():
#     _test(problem_name="levy", num_dim=2, x=np.array([0.3, 0.3]), y= -8.67852899186019)

# def test_rastrigin():
#     _test(problem_name="rastrigin", num_dim=2, x=np.array([0.3, 0.3]), y= -11.572523396546803)

# def test_rosenbrock():
#     _test(problem_name="rosenbrock", num_dim=2, x=np.array([0.3, 0.3]), y= -296.22523290112576)

# def test_shubert():
#     _test(problem_name="shubert", num_dim=2, x=np.array([0.3, 0.3]), y= 14.16434451333946)

# def test_sixhumpcamel():
#     _test(problem_name="sixhumpcamel", num_dim=2, x=np.array([0.3, 0.3]), y= -1.9158442961058475)

def test_threehumpcamel():
    _test(problem_name="threehumpcamel", num_dim=2, x=np.array([0.3, 0.3]), y= -6.298071583760114)

