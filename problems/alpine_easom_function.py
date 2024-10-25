
import numpy as np

class alpine:

    #src = https://github.com/sigopt/evalset

    def __call__(self,x):
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x))
    
class easom:

    #src = "https://en.wikipedia.org/wiki/Test_functions_for_optimization"
    
    def __call__(self, x, y):
        return -np.cos(x) * np.cos(y) * np.exp(-(x - np.pi)**2 - (y - np.pi)**2)