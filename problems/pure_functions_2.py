import numpy as np

class Booth:

    #src="https://en.wikipedia.org/wiki/Test_functions_for_optimization"

    def __call__(self, x):
        return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2


class Himmelblau:

    #src="https://en.wikipedia.org/wiki/Test_functions_for_optimization"

    def __call__(self, x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    

class Matyas:

    #src="https://en.wikipedia.org/wiki/Test_functions_for_optimization"

    def __call__(self, x):
        return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]


class Zettl:

    #src = "http://georgioudakis.com/blog/a-collection-of-test-functions-for-single-objective-optimization-problems"
    
    def __call__(self, x):
        return (x[0]**2 + x[1]**2 - 2 * x[0])**2 + 0.25 * x[0]


class Sum_Squares:

    #src ="https://www.sfu.ca/~ssurjano/sumsqu.html"

    def __call__(self, x):
        x = np.array(x)
        return np.sum((x**2) * np.arange(1, 1 + 1))


class Perm:

    #src ="https://www.sfu.ca/~ssurjano/permdb.html"

    def __call__(self, x, beta=10):
        n = 2
        return sum(sum((j + beta) * (x[j]**i - 1) for j in range(n))**2 for i in range(1, n+1))


class Salomon:

    #src = https://github.com/sigopt/evalset

    def __call__(self, x):
        r = np.sqrt(sum(xi**2 for xi in x))
        return 1 - np.cos(2 * np.pi * r) + 0.1 * r


class Whitley:

    #src = "https://infinity77.net/global_optimization/test_functions_nd_W"

    def __call__(self, x):
        x = np.asarray(x)
        n = len(x)
        result = 0.0
        for i in range(n):
            for j in range(n):
                term = 100 * (x[i] ** 2 - x[j]) ** 2 + (1 - x[j]) ** 2
                result += (term ** 2 / 4000) - np.cos(term) + 1
        return result

class Brown:

    #src = https://github.com/sigopt/evalset

    def __call__(self, x):
        x = np.array(x)
        n = 2
        sum1 = np.sum(x**2 - 10)**2
        sum2 = np.prod(x**2)
        return sum1 + sum2


class Zakharov:

    #src = "https://www.sfu.ca/~ssurjano/zakharov.html"
    
    def __call__(self, x):
        x = np.array(x)
        term1 = np.sum(x**2)
        term2 = np.sum((0.5 * np.arange(1, 2 + 1) * x)**2)
        return term1 + term2