import os
import numpy as np
import matplotlib.pyplot as plt 
import sympy as sp
from sympy.abc import x, y
from ecbf.utils.paths import PLOTS_PATH 
from ecbf.scripts.basecbf import BaseCBF

class TotalEnergyLimit(BaseCBF):
    def __init__(self, H_func, c): 
        '''
        This class represents provides a function that is greater than zero inside a circle with radius C.
        '''
        super().__init__()
        self.H = H_func 
        self.c = c

    def function(self, state): 
        _h = self.c - self.H(state[0], state[1])
        return _h 
     