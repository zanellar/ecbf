import os
import numpy as np
import matplotlib.pyplot as plt 
import sympy as sp
from sympy.abc import x, y
from ecbf.utils.paths import PLOTS_PATH 
from ecbf.scripts.basecbf import BaseCBF

class EnergyLimit(BaseCBF):
    def __init__(self, energy_func, c, pump=False): 
        '''
        This class represents provides a function that is greater than zero inside a circle with radius C.
        '''
        super().__init__()
        self.energy_func = energy_func 
        self.c = c

        self.pump = pump

    def function(self, state): 
        if self.pump:
            _h = self.energy_func(state[0], state[1]) - self.c
        else:
            _h = self.c - self.energy_func(state[0], state[1])
        return _h 
     