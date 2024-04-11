import os
import ecbf

_PACKAGE_PATH = ecbf.__path__[0]  
 
PARAMS_PATH = os.path.join(_PACKAGE_PATH, os.pardir, "data", "params")  
PLOTS_PATH = os.path.join(_PACKAGE_PATH, os.pardir, "data", "plots")
RES_PATH = os.path.join(_PACKAGE_PATH, os.pardir, "data", "results")