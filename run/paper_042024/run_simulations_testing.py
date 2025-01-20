import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import os
 
from ecbf.defined_models.mass_spring import MassSpring
from ecbf.barrier_functions.safe_doughnut import SafeDoughnut
from ecbf.barrier_functions.safe_circle import SafeCircle
from ecbf.barrier_functions.energy_limit import EnergyLimit
from ecbf.scripts.control import Controller
from ecbf.utils.paths import PLOTS_PATH 

'''
            In this script, we will run simulations for the mass-spring system with different barrier functions 
                            !!! MAKE SEPARETE PLOTS FOR EACH SIMULATION IN PNG FORMAT!!!
'''
  
gammas = [2,4]

for gamma in gammas:

    # Use slider values in parameters
    parameter = {
        'time_horizon': 15,
        'time_step': 0.1,
        'init_state': [-8, 15],
        'target_state': None,
        'weight_input': 1,
        'cbf_gamma': gamma,
        'weight_slack': None,
        'u_max': None,
        'u_min': None  
    } 

    name = f"dampH_gamma{parameter['cbf_gamma']}"

    model = MassSpring(m=1, k=0.5, dt=parameter["time_step"], verbose=False)
 
    cbf = EnergyLimit(energy_func=model.H, c=10)

    ctrl = Controller(
        model, 
        parameter,  
        cbf=cbf
    )

    ctrl.run(save=True, name=name)
    
    # Plotting
    ctrl.plot_phase_trajectory(name=name, show=False, save=True)
    ctrl.plot_state(name=name, show=False, save=True) 
    ctrl.plot_control(name=name, show=False, save=True)
    ctrl.plot_cbf(name=name, show=False, save=True) 

####################################################################################
####################################################################################
####################################################################################


gammas = [2,4]

for gamma in gammas:

    # Use slider values in parameters
    parameter = {
        'time_horizon': 15,
        'time_step': 0.1,
        'init_state': [-3, 2],
        'target_state': None,
        'weight_input': 1,
        'cbf_gamma': gamma,
        'weight_slack': None,
        'u_max': None,
        'u_min': None  
    } 

    name = f"pumpH_gamma{parameter['cbf_gamma']}"

    model = MassSpring(m=1, k=0.5, dt=parameter["time_step"], verbose=False)
 
    cbf = EnergyLimit(energy_func=model.H, c=10, pump=True)

    ctrl = Controller(
        model, 
        parameter,  
        cbf=cbf
    )

    ctrl.run(save=True, name=name)
    
    # Plotting
    ctrl.plot_phase_trajectory(name=name, show=False, save=True)
    ctrl.plot_state(name=name, show=False, save=True) 
    ctrl.plot_control(name=name, show=False, save=True)
    ctrl.plot_cbf(name=name, show=False, save=True) 


####################################################################################
####################################################################################
####################################################################################
 
gammas = [4,8]

for gamma in gammas:

    # Use slider values in parameters
    parameter = {
        'time_horizon': 15,
        'time_step': 0.1,
        'init_state': [-8, 15],
        'target_state': None,
        'weight_input': 1,
        'cbf_gamma': gamma,
        'weight_slack': None,
        'u_max': None,
        'u_min': None  
    } 

    name = f"dumpK_gamma{parameter['cbf_gamma']}"

    model = MassSpring(m=1, k=0.5, dt=parameter["time_step"], verbose=False)
 
    cbf = EnergyLimit(energy_func=model.K, c=10, pump=False)

    ctrl = Controller(
        model, 
        parameter,  
        cbf=cbf
    )

    ctrl.run(save=True, name=name)
    
    # Plotting
    ctrl.plot_phase_trajectory(name=name, show=False, save=True)
    ctrl.plot_state(name=name, show=False, save=True) 
    ctrl.plot_control(name=name, show=False, save=True)
    ctrl.plot_cbf(name=name, show=False, save=True) 

