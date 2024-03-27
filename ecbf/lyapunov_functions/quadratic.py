import os
import numpy as np
import matplotlib.pyplot as plt 
import sympy as sp
from sympy.abc import x, y
from ecbf.utils.paths import PLOTS_PATH 

class Quadratic:

    def __init__(self):  
        self.target = None

    def function(self, state, target):
        x = state[0]
        y = state[1]
        _V = (x - target[0])**2 + (y - target[1])**2 
        return _V 
     