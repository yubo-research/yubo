import numpy as np

# Requirements
# all domains are [-1,1]**num_dim
# all functions should have *minima* [not maxima]
# If the function is of fixed dimension, assert it in __call__()


def all_benchmarks():
    import inspect
    import sys

    mod = sys.modules[__name__]
    all_bf = {}
    for name in dir(mod):
        obj = getattr(mod, name)
        if inspect.isclass(obj):
            all_bf[name.lower()] = obj
    return all_bf


class Sphere:
    def __call__(self, x):
        return (x**2).mean()


# 1 xi ∈ [-32.768, 32.768] ackley result [0,25]
class Ackley:
    def __init__(self):
        self.a = 20.0
        self.b = 0.2
        self.c = 2 * np.pi

    def __call__(self, x):
        x = 32.768 * x
        return -self.a * np.exp(-self.b * np.sqrt((x**2).mean())) - np.exp(np.cos(self.c * x).mean()) + self.a  # + np.e


# 2  xi ∈ [-4.5, 4.5], for all i = 1, 2. Beale result [0, 4.5*10^5]
class Beale:
    def __call__(self, x):
        assert len(x) == 2
        x = 4.5 * x
        part1 = (1.5 - x[0] + x[0] * x[1]) ** 2
        part2 = (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        part3 = (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        return part1 + part2 + part3


# 3 x1 ∈ [-5, 10], x2 ∈ [0, 15]. result [0,350] Branin
class Branin:
    def __init__(self):
        self.a = 1
        self.b = 5.1 / (4 * np.pi**2)
        self.c = 5 / np.pi
        self.r = 6
        self.s = 10
        self.t = 1 / (8 * np.pi)

    def __call__(self, x):
        assert len(x) == 2
        x1 = 7.5 * x[0] + 2.5
        x2 = 7.5 * x[1] + 7.5
        return self.a * (x2 - self.b * x1**2 + self.c * x1 - self.r) ** 2 + self.s * (1 - self.t) * np.cos(x1) + self.s


# 4 x1 ∈ [-15, -5], x2 ∈ [-3, 3]. result [0,250] Bukin
class Bukin:
    def __call__(self, x):
        assert len(x) == 2
        x0 = x[0] * 5 - 10
        x1 = x[1] * 3
        return 100.0 * np.sqrt(np.abs(x1 - 0.01 * x0**2)) + 0.01 * np.abs(x0 + 10.0)


# 5 xi ∈ [-10, 10], for all i = 1, 2. result[-2.5,-0.5] CrossInTray
class CrossInTray:
    def __call__(self, x):
        assert len(x) == 2
<<<<<<< HEAD
<<<<<<< HEAD
        x = x * 9 + 1 - self._x_0
=======
>>>>>>> main
=======
>>>>>>> main
        x0 = x[0]
        x1 = x[1]
        part1 = np.abs(np.sin(x0) * np.sin(x1) * np.exp(np.abs(100.0 - np.sqrt(x0**2 + x1**2) / np.pi))) + 1.0
        part2 = np.power(part1, 0.1)
        return -0.0001 * part2


# 6  xi ∈ [-5.12, 5.12], for all i = 1, 2. result[-1,0] DropWave
class DropWave:
    def __call__(self, x):
        assert len(x) == 2
        x = x * 5.12
        x0 = x[0]
        x1 = x[1]
        sum2 = x0**2 + x1**2
        part1 = 1.0 + np.cos(12.0 * np.sqrt(sum2))
        part2 = 0.5 * sum2 + 2.0
        return -part1 / part2


# 7 xi ∈ [-10, 10] result [0,9*10^4] DixonPrice
class DixonPrice:
    def __call__(self, x):
        x = x * 10
        part1 = (x[0] - 1) ** 2
        sum_terms = 0
        for i in range(2, len(x) + 1):
            xnew = x[i - 1]
            xold = x[i - 2]
            new = i * (2 * xnew**2 - xold) ** 2
            sum_terms += new
        return part1 + sum_terms


# 8 EggHolder xi ∈ [-512, 512], for all i = 1, 2.result [-1000,1500]
class EggHolder:
    def __call__(self, x):
        assert len(x) == 2
        x = x * 511
        x1 = x[0]
        x2 = x[1]
        part1 = -(x2 + 47.0) * np.sin(np.sqrt(np.abs(x2 + x1 / 2.0 + 47.0)))
        part2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47.0))))
        return part1 + part2


# 9 Griewank xi ∈ [-600, 600] , for all i = 1, …, d. result [0,200]
class Griewank:
    def __call__(self, x):
        x = x * 600
        part1 = np.sum(x**2 / 4000.0)
        num_dim = len(x)
        part2 = np.prod(np.cos(x / np.sqrt((1 + np.arange(num_dim)))))
        return part1 - part2 + 1


# 10 GrLee12 one-dimention x ∈ [0.5, 2.5]. result [-1,6]
class GrLee12:
    def __call__(self, x):
        assert len(x) == 1
        x = x[0] + 1.5
        return np.sin(10.0 * np.pi * x) / (2.0 * x) + (x - 1.0) ** 4


# 11 Hartmann xi ∈ (0, 1), for all i , dimention =3,4,6
class Hartmann:
    def __init__(self):
        self.A = {
            3: np.array((3, 10, 30, 0.1, 10, 35, 3, 10, 30, 0.1, 10, 35)).reshape(4, 3),
            4: np.array((10, 3, 17, 3.5, 1.7, 8, 0.05, 10, 17, 0.1, 8, 14, 3, 3.5, 1.7, 10, 17, 8, 17, 8, 0.05, 10, 0.1, 14)).reshape(4, 6),
        }
        self.P = {
            3: 10 ** (-4) * np.array((3689, 1170, 2673, 4699, 4387, 7470, 1091, 8732, 5547, 381, 5743, 8828)).reshape(4, 3),
            4: 10 ** (-4)
            * np.array(
                (1312, 1696, 5569, 124, 8283, 5886, 2329, 4135, 8307, 3736, 1004, 9991, 2348, 1451, 3522, 2883, 3047, 6650, 4047, 8828, 8732, 5743, 1091, 381)
            ).reshape(4, 6),
        }
        self.ALPHA = [1.0, 1.2, 3.0, 3.2]

    def __call__(self, x):
        num_dim = len(x)
        assert num_dim in self.A, num_dim

        A = self.A[num_dim]
        P = self.P[num_dim]

        x = 0.5 * x + 0.5
        outer = 0
        for i in range(4):
            inner = 0
            for j in range(self.num_dim):
                inner += A[i][j] * (x[j] - P[i][j]) ** 2
            new = self.ALPHA[i] * np.exp(-inner)
            outer += new
        if self.num_dim == 3:
            return -outer
        if self.num_dim == 4:
            return (1.1 - outer) / 0.839
        if self.num_dim == 6:
            return -(2.58 + outer) / 1.94


# 12 HolderTable xi ∈ [-10, 10], for all i = 1, 2. result [-20,0]
class HolderTable:
    def __call__(self, x):
        assert len(x) == 2
        x = x * 10
        x0 = x[0]
        x1 = x[1]
        part1 = np.sin(x0) * np.cos(x1)
        part2 = np.exp(np.abs(1 - np.sqrt(x0**2 + x1**2) / np.pi))
        return -np.abs(part1 * part2)


# 13 Levy  xi ∈ [-10, 10], for all i = 1, …, d. result [0,100]
class Levy:
    def __call__(self, x):
        x = 10 * x
        w = 1.0 + (x - 1.0) / 4.0
        part1 = np.sin(np.pi * w[0]) ** 2
        # part2 = np.sum(
        #     (w[:, :-1] - 1.0) ** 2
        #     * (1.0 + 10.0 * np.sin(np.pi * w[:, :-1] + 1.0) ** 2),
        #     dim=1,
        # )
        part2 = 0
        num_dim = len(x)
        for i in range(num_dim - 1):
            part2 += (w[i] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[i] + 1) ** 2)
        part3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
        return part1 + part2 + part3


#####
## BELOW ARE UNCHECKED
#####


# 14 Michalewicz xi ∈ [0, π], for all i = 1, …, d. result[-1.8,0]
class Michalewicz:
    def __call__(self, x):
        x = x * np.pi / 2 + np.pi / 2
        num_dim = len(x)
        m = 10
        sum = 0
        for i in range(num_dim):
            new = np.sin(x[i]) * (np.sin(i * x[i] ** 2 / np.pi)) ** (2 * m)
            sum += new
<<<<<<< HEAD
<<<<<<< HEAD
        return sum
=======
        return -sum
>>>>>>> main
=======
        return -sum
>>>>>>> main


# 15 Powell xi ∈ [-4, 5], for all i = 1, …, d.
class Powell:
    def __call__(self, x):
        num_dim = len(x)
        assert num_dim % 4 == 0, num_dim

        x = x * 4.5 + 0.5
        result = 0
        for i in range(num_dim // 4):
            i_ = i + 1
            part1 = (x[4 * i_ - 4] + 10.0 * x[4 * i_ - 3]) ** 2
            part2 = 5.0 * (x[4 * i_ - 2] - x[4 * i_ - 1]) ** 2
            part3 = (x[4 * i_ - 3] - 2.0 * x[4 * i_ - 2]) ** 4
            part4 = 10.0 * (x[4 * i_ - 4] - x[4 * i_ - 1]) ** 4
            result += part1 + part2 + part3 + part4

        return result


# 16 Rastrigin  xi ∈ [-5.12, 5.12], for all i = 1, …, d result[0,90]
class Rastrigin:
    def __call__(self, x):
        x = x * 5.12
        num_dim = len(x)
        return 10 * num_dim + np.sum(x**2 - 10 * np.cos(np.pi * 2 * x))


# 17 Rosenbrock  xi ∈ [-2.048, 2.048], for all i = 1, …, d. result [0,18*10^4]
class Rosenbrock:
    def __call__(self, x):
        x = x * 2.048
        part = 0
        num_dim = len(x)
        for i in range(num_dim - 1):
            part += (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
        return 100.0 * part


# 18 Shubert xi ∈ [-5.12, 5.12], for all i = 1, 2. result[-200,300]
class Shubert:
    def __call__(self, x):
        x = x * 5.12
        x0 = x[0]
        x1 = x[1]
        part1 = 0
        part2 = 0
        for i in range(1, 6):
            new1 = i * np.cos((i + 1) * x0 + i)
            new2 = i * np.cos((i + 1) * x1 + i)
            part1 += new1
            part2 += new2
        return part1 * part2


# 19 Shekel  xi ∈ [0, 10], for all i = 1, 2, 3, 4. dimentions 4
class Shekel:
    def __init__(self):
        self.beta = 0.1 * np.array((1, 2, 2, 4, 4, 6, 3, 7, 5, 5)).T
        self.C = np.array((4, 1, 8, 6, 3, 2, 5, 8, 6, 7, 4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6, 4, 1, 8, 6, 3, 2, 5, 8, 6, 7, 4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6)).reshape(
            4, 10
        )

    def __call__(self, x):
        assert len(x) == 4
        x = x * 5 + 5
        m = self.C.shape[1]
        outer = 0
        for i in range(m):
            bi = self.beta[i]
            inner = 0
            for j in range(4):
                inner += (x[j] - self.C[j][i]) ** 2
            outer += 1 / (inner + bi)
        return outer


# 20 SixHumpCamel x1 ∈ [-3, 3], x2 ∈ [-2, 2]. result [-50,200]
class SixHumpCamel:
    def __call__(self, x):
        assert len(x) == 2
        x = x
        x0 = x[0] * 3
        x1 = x[1] * 2
        return (4 - 2.1 * x0**2 + x0**4 / 3) * x0**2 + x0 * x1 + (4 * x1**2 - 4) * x1**2


# 21 StybTang  xi ∈ [-5, 5], for all i = 1, …, d. result [-100,250]
class StybTang:
    def __call__(self, x):
        x = x * 5
        return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)


# 22 ThreeHumpCamel xi ∈ [-5, 5], for all i = 1, 2. result[0,10][0,1000]
class ThreeHumpCamel:
    def __call__(self, x):
        assert len(x) == 2
        x = x * 5
        x0 = x[0]
        x1 = x[1]
<<<<<<< HEAD
<<<<<<< HEAD
        return 2.0 * x0**2 - 1.05 * x0**4 + x0**6 / 6.0 + x0 * x1 + x1**2
=======
        return 2.0 * x0**2 - 1.05 * x0**4 + x0**6 / 6.0 + x0 * x1 + x1**2
>>>>>>> main
=======
        return 2.0 * x0**2 - 1.05 * x0**4 + x0**6 / 6.0 + x0 * x1 + x1**2
>>>>>>> main
