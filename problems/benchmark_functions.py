import numpy as np


# class Sphere:
#     def __init__(self, seed, num_dim):
#         rng = np.random.default_rng(seed)
#         self._x_0 = rng.uniform(size=(num_dim,))
#         print("x0",self._x_0)
#
#     def __call__(self, x):
#         print("x",x)
#         print("result",-(((x - self._x_0) ** 2)).mean())
#         print("newx", (x - self._x_0) )
#         return -(((x - self._x_0) ** 2)).mean()

# 
import math
import numpy as np

#1 xi ∈ [-32.768, 32.768] ackley result [0,25]
class Ackley:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        a = 20.0
        b = 0.2
        c = 2 * math.pi
        x = x*30-self._x_0
        y = x[1]
        x= x[0]
        result = -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20
        return -result

#2  xi ∈ [-4.5, 4.5], for all i = 1, 2. Beale result [0, 4.5*10^5]
class Beale:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x = x*3.5-self._x_0
        x1 = x[0]
        x2 = x[1]
        part1 = (1.5 - x1 + x1 * x2) ** 2
        part2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
        part3 = (2.625 - x1 + x1 * x2 ** 3) ** 2
        result = part1 + part2 + part3
        return -result

#3 x1 ∈ [-5, 10], x2 ∈ [0, 15]. result [0,350] Branin
class Branin:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x0 = x[0]*7.5+2.5
        x1 = x[1]*7.5+7.5
        t1 = (
                x1
                - 5.1 / (4 * math.pi ** 2) * x0 ** 2
                + 5 / math.pi * x0
                - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * np.cos(x0)
        result = t1 ** 2 + t2 + 10
        return -result

#4 x1 ∈ [-15, -5], x2 ∈ [-3, 3]. result [0,250] Bukin
class Bukin:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x0 = x[0]*5-10
        x1 = x[1]*3
        part1 = 100.0 * np.sqrt(np.abs(x1 - 0.01 * x0 ** 2))
        part2 = 0.01 * np.abs(x0 + 10.0)
        result = part1 + part2
        return -result

#5 xi ∈ [-10, 10], for all i = 1, 2. result[-2.5,-0.5] CrossInTray
class CrossInTray:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x = x*9+1-self._x_0
        x0 = x[0]
        x1 = x[1]
        part1 = np.abs(np.sin(x0) * np.sin(x1)* np.exp(np.abs(100.0 - np.sqrt(x0 ** 2 + x1 ** 2) / math.pi)))+ 1.0
        part2 = np.power(part1, 0.1)
        result = -0.0001 * part2

        return result
#6  xi ∈ [-5.12, 5.12], for all i = 1, 2. result[-1,0] DropWave
class DropWave:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x = x *5.12
        x0 = x[0]
        x1 = x[1]
        sum2 = x0 ** 2+x1 **2
        part1 = 1.0 + np.cos(12.0 * np.sqrt(sum2))
        part2 = 0.5 * sum2 + 2.0
        result = -part1 / part2
        return result



# 7 xi ∈ [-10, 10] result [0,9*10^4] DixonPrice
class DixonPrice:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x = x*10
        part1 = (x[0] - 1) ** 2
        sum = 0
        for i in range(2,len(x)+1):
            xnew = x[i-1]
            xold = x[i-2]
            new = i*(2*xnew**2-xold)**2
            sum += new
        result = part1+sum
        return -result


#8 EggHolder xi ∈ [-512, 512], for all i = 1, 2.result [-1000,1500]
class EggHolder:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x = x*511 - self._x_0
        x1 = x[0]
        x2 = x[1]
        part1 = -(x2 + 47.0) * np.sin(np.sqrt(np.abs(x2 + x1 / 2.0 + 47.0)))
        part2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47.0))))
        result = part1 + part2
        return result -1500

# 9 Griewank xi ∈ [-600, 600] , for all i = 1, …, d. result [0,200]
class Griewank:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x = x *600
        part1 = np.sum(x ** 2 / 4000.0)
        d = len(x)
        part2 = 1
        for i in range(d):
            part2 = part2*np.cos(x[i]/np.sqrt(i+1))
        result = part1 - part2 +1
        return -result


#10 GrLee12 one-dimention x ∈ [0.5, 2.5]. result [-1,6]
class GrLee12:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x = (x[0]+1.5)
        result = (np.sin(10.0 * math.pi * x) / (2.0 * x) + (x - 1.0) ** 4)
        return -result-1


#11 Hartmann xi ∈ (0, 1), for all i , dimention =3,4,6
class Hartmann:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))
        self.num_dim = num_dim
        if num_dim ==3:
            self.A = np.array((3,10,30,0.1,10,35,3,10,30,0.1,10,35)).reshape(4,3)
            self.P = 10**(-4)*np.array((3689,1170,2673,4699,4387,7470,1091,8732,5547,381,5743,8828)).reshape(4,3)
        if num_dim ==4 or num_dim ==6:
            self.A = np.array((10, 3, 17, 3.5, 1.7, 8,
                             0.05, 10, 17, 0.1, 8, 14,
                             3, 3.5, 1.7, 10, 17, 8,
                             17, 8, 0.05, 10, 0.1, 14)).reshape(4,6)
            self.P = 10**(-4)*np.array((1312, 1696, 5569, 124, 8283, 5886,
                               2329, 4135, 8307, 3736, 1004, 9991,
                               2348, 1451, 3522, 2883, 3047, 6650,
                               4047, 8828, 8732, 5743, 1091, 381)).reshape(4,6)


    def __call__(self, x):
        # inner_sum = np.sum(
        #     x.new(self.A) * (x.unsqueeze(1) - 0.0001 * x.new(self.P)) ** 2
        # )
        x = 0.5*x+0.5
        ALPHA = [1.0, 1.2, 3.0, 3.2]
        # H = -np.sum(x.new(ALPHA) * np.exp(-inner_sum), dim=1)
        outer = 0
        for i in range(4):
            inner = 0
            for j in range(self.num_dim):
                inner += self.A[i][j]*(x[j]-self.P[i][j])**2
            new = ALPHA[i]*np.exp(-inner)
            outer += new
        if self.num_dim==3:
            result = -outer
            return result
        if self.num_dim==4:
            result = (1.1 - outer) / 0.839
            return result
        if self.num_dim==6:
            result = -(2.58 + outer) / 1.94
            return result




#12 HolderTable xi ∈ [-10, 10], for all i = 1, 2. result [-20,0]
class HolderTable:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x = x *9 - self._x_0
        x0 = x[0]
        x1 = x[1]
        part1 = np.sin(x0)*np.cos(x1)
        part2 = np.exp(np.abs(1-np.sqrt(x0**2+x1**2)/np.pi))
        result = -np.abs(part1*part2)
        return result




#13 Levy  xi ∈ [-10, 10], for all i = 1, …, d. result [0,100]
class Levy:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))
        self.num_dim = num_dim

    def __call__(self, x):
        x = x *9 - self._x_0
        w = 1.0 + (x - 1.0) / 4.0
        part1 = np.sin(math.pi * w[0]) ** 2
        # part2 = np.sum(
        #     (w[:, :-1] - 1.0) ** 2
        #     * (1.0 + 10.0 * np.sin(math.pi * w[:, :-1] + 1.0) ** 2),
        #     dim=1,
        # )
        part2= 0
        for i in range(self.num_dim-1):
            part2+=(w[i]-1)**2*(1+10*np.sin(np.pi*w[i]+1)**2)
        part3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * math.pi * w[-1]) ** 2)
        result =  part1 + part2 + part3
        return -result



#14 Michalewicz xi ∈ [0, π], for all i = 1, …, d. result[-1.8,0]
class Michalewicz:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x = x *np.pi/2+np.pi/2
        d = len(x)
        m = 10
        sum = 0
        for i in range(d):
            new = np.sin(x[i])*(np.sin(i*x[i]**2/np.pi))**(2*m)
            sum+=new
        result = -sum
        return result


#15 Powell xi ∈ [-4, 5], for all i = 1, …, d.
class Powell:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))
        self.num_dim = num_dim

    def __call__(self, x):
        x = x*4+1-self._x_0
        result = 0
        for i in range(self.num_dim // 4):
            i_ = i + 1
            part1 = (x[4 * i_ - 4] + 10.0 * x[4 * i_ - 3]) ** 2
            part2 = 5.0 * (x[4 * i_ - 2] - x[ 4 * i_ - 1]) ** 2
            part3 = (x[4 * i_ - 3] - 2.0 * x[4 * i_ - 2]) ** 4
            part4 = 10.0 * (x[4 * i_ - 4] - x[4 * i_ - 1]) ** 4
            result += part1 + part2 + part3 + part4
        return -result



#16 Rastrigin  xi ∈ [-5.12, 5.12], for all i = 1, …, d result[0,90]
class Rastrigin:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x = x *4.12+1-self._x_0
        d = len(x)
        part = 0
        for i in range(d):
            part += x[i]**2- 10.0 * np.cos(2.0 * math.pi * x[i])
        result = 10*d+part
        return -result



#17 Rosenbrock  xi ∈ [-2.048, 2.048], for all i = 1, …, d. result [0,18*10^4]
class Rosenbrock:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x = x *1.048+1-self._x_0
        part = 0
        d = len(x)
        for i in range(d-1):
            part += (x[i+1]-x[i]**2)**2 + (x[i]-1)**2
        result = 100.0 * part
        return -result



#18 Shubert xi ∈ [-5.12, 5.12], for all i = 1, 2. result[-200,300]
class Shubert:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x = x*4.12 +1 - self._x_0
        x0 = x[0]
        x1 = x[1]
        part1=0
        part2=0
        for i in range(1,6):
            new1 = i*np.cos((i+1)*x0+i)
            new2 = i*np.cos((i+1)*x1+i)
            part1 += new1
            part2 += new2
        result = part1*part2
        return result-300



#19 Shekel  xi ∈ [0, 10], for all i = 1, 2, 3, 4. dimentions 4
class Shekel:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))
        self.beta =  0.1 *np.array((1, 2, 2, 4, 4, 6, 3, 7, 5, 5)).T
        self.C = np.array((4, 1, 8, 6, 3, 2, 5, 8, 6, 7,
            4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6,
            4, 1, 8, 6, 3, 2, 5, 8, 6, 7,
            4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6)).reshape(4,10)

    def __call__(self, x):
        x = x*4+5 - self._x_0
        m =  self.C.shape[1]
        outer = 0
        for i in range(m):
            bi = self.beta[i]
            inner = 0
            for j in range(4):
               inner += (x[j]-self.C[j][i])**2
            outer += 1/(inner+bi)
        result = -outer
        return result


#20 SixHumpCamel x1 ∈ [-3, 3], x2 ∈ [-2, 2]. result [-50,200]
class SixHumpCamel:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x = x +1 - self._x_0
        x0 = x[0]*3
        x1 = x[1]*2
        result= (
                (4 - 2.1 * x0 ** 2 + x0 ** 4 / 3) * x0 ** 2
                + x0 * x1
                + (4 * x1 ** 2 - 4) * x1 ** 2
        )
        return -result-50

#21 StybTang  xi ∈ [-5, 5], for all i = 1, …, d. result [-100,250]
class StybTang:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x = x*4 +1 - self._x_0
        d = len(x)
        sum = 0
        for i in range(d):
            sum+= x[i]**4-16*x[i]**2+5*x[i]
        result = sum/2
        return -result-100

#22 ThreeHumpCamel xi ∈ [-5, 5], for all i = 1, 2. result[0,10][0,1000]
class ThreeHumpCamel:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x = x*4+1 - self._x_0
        x0 = x[0]
        x1 = x[1]
        result = 2.0 * x0 ** 2 - 1.05 * x0 ** 4 + x0 ** 6 / 6.0 + x0 * x1 + x1 ** 2
        return -result
