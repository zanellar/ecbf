import os
import numpy as np
import matplotlib.pyplot as plt 
import sympy as sp
from sympy.abc import x, y
from ecbf.utils.paths import PLOTS_PATH 

class BaseCBF: 

    def function(self, state): 
        raise NotImplementedError
     