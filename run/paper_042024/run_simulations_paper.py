import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
  
from ecbf.defined_models.mass_spring import MassSpring
from ecbf.defined_models.double_pendulum import DoublePendulum

from ecbf.barrier_functions.safe_doughnut import SafeDoughnut
from ecbf.barrier_functions.safe_circle import SafeCircle
from ecbf.barrier_functions.energy_limit import EnergyLimit
from ecbf.scripts.control import Controller
from ecbf.utils.paths import PLOTS_PATH 
  
'''
In this script, we will run simulations for the mass-spring system and double pendulum with different barrier functions  

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
!!! PLOTS ARE COMBINED INTO ONE FIGURE FOR EACH SIMULATION IN PDF FORMAT !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
'''
  
plt.rcParams['font.size'] = 32 

colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

image_file_format = 'pdf'
 
####################################################################################
####################################################################################
####################################################################################

gammas = [2,4] 

_, axs1 = plt.subplots(1, 1, figsize=(14, 14))  
_, axs2 = plt.subplots(1, 1, figsize=(18, 6))  
_, axs2 = plt.subplots(1, 1, figsize=(18, 6))  
_, axs3 = plt.subplots(1, 1, figsize=(18, 10))  
_, axs4 = plt.subplots(1, 1, figsize=(18, 6))  
 
ylims2 = None
ylims3 = None
ylims4 = None

for i, gamma in enumerate(gammas):

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

    run_name = "dampH_massspring"

    model = MassSpring(m=2, k=0.5, dt=parameter["time_step"], verbose=False)
 
    cbf = EnergyLimit(energy_func=model.H, c=10)

    ctrl = Controller(
        model, 
        parameter,  
        cbf=cbf
    )

    ctrl.run(save=True, name=f"{run_name}_gamma{parameter['cbf_gamma']}")
     

    ######################### Plotting phase trajectory
    if i ==0:  
        ctrl.plot_phase_trajectory(name=f"$\gamma$={gamma}", state_range=[-20,20], show=False, save=False, figure=axs1, add_safe_set=True, color=colors[i], plot_end_state=False, arrow_skip=1, arrow_index=5)
    elif i < len(gammas)-1:
        ctrl.plot_phase_trajectory(name=f"$\gamma$={gamma}", state_range=[-20,20], show=False, save=False, figure=axs1, add_safe_set=False, color=colors[i], plot_end_state=False, arrow_skip=1, arrow_index=5)
    else:
        ctrl.plot_phase_trajectory(name=f"$\gamma$={gamma}", state_range=[-20,20], show=False, save=False, figure=axs1, add_safe_set=False, color=colors[i], plot_end_state=False, arrow_skip=1, arrow_index=5) 
        axs1.legend()
        axs1.legend(loc='lower right')
        file_name = f'phase_trajectory_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300)  
  
    ########################## Plotting control
    name = f"$\gamma$={gamma}"  

    ctrl.plot_control(name=name, show=False, save=False, figure=axs2, color=colors[i], ylims=None)

    # Correct ylims 
    if ylims2 is None:
        ylims2 = list(axs2.get_ylim())
    else:
        if axs2.get_ylim()[0] < ylims2[0]:
            ylims2[0] = axs2.get_ylim()[0]
        if axs2.get_ylim()[1] > ylims2[1]:
            ylims2[1] = axs2.get_ylim()[1] 

    if i == len(gammas)-1:  
        axs2.legend()
        axs2.set_ylim(ylims2)
        axs2.legend(loc='lower right')
        axs2.set_xlabel('')
        file_name = f'control_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300)

    ########################## Plotting energy

    ctrl.plot_total_energy (name=f"$\gamma$={gamma}", show=False, save=False, figure=axs3, color=colors[i], ylims=None) 

    # Correct ylims 
    if ylims3 is None:
        ylims3 = list(axs3.get_ylim())
    else:
        if axs3.get_ylim()[0] < ylims3[0]:
            ylims3[0] = axs3.get_ylim()[0]
        if axs3.get_ylim()[1] > ylims3[1]:
            ylims3[1] = axs3.get_ylim()[1] 

    if i == len(gammas)-1:  
        axs3.legend()
        axs3.legend(loc='upper right')
        axs3.set_ylim(ylims3)
        file_name = f'total_energy_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300)

    ########################## Plotting cbf
    
    ctrl.plot_cbf(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs4, color=colors[i], ylims=None)

    # Correct ylims 
    if ylims4 is None:
        ylims4 = list(axs4.get_ylim())
    else:
        if axs4.get_ylim()[0] < ylims4[0]:
            ylims4[0] = axs4.get_ylim()[0]
        if axs4.get_ylim()[1] > ylims4[1]:
            ylims4[1] = axs4.get_ylim()[1] 

    if i == len(gammas)-1:  
        axs4.legend()
        axs4.legend(loc='lower right')
        axs4.set_ylim(ylims4)
        axs4.set_xlabel('')
        file_name = f'cbf_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300)
 

####################################################################################
####################################################################################
####################################################################################


gammas = [2,4]

_, axs1 = plt.subplots(1, 1, figsize=(14, 14))  
_, axs2 = plt.subplots(1, 1, figsize=(18, 6))  
_, axs2 = plt.subplots(1, 1, figsize=(18, 6))  
_, axs3 = plt.subplots(1, 1, figsize=(18, 10))  
_, axs4 = plt.subplots(1, 1, figsize=(18, 6))  

ylims2 = None
ylims3 = None
ylims4 = None

for i, gamma in enumerate(gammas):

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

    run_name = "pumpH_massspring" 

    model = MassSpring(m=2, k=0.5, dt=parameter["time_step"], verbose=False)
 
    cbf = EnergyLimit(energy_func=model.H, c=10, pump=True)

    ctrl = Controller(
        model, 
        parameter,  
        cbf=cbf
    )

    ctrl.run(save=True, name=f"{run_name}_gamma{parameter['cbf_gamma']}")
    
    # Plotting  
    arrow_index = 2

    ######################### Plotting phase trajectory
    if i ==0:  
        ctrl.plot_phase_trajectory(name=f"$\gamma$={gamma}", state_range=[-20,20], show=False, save=False, figure=axs1, add_safe_set=True, color=colors[i], plot_end_state=False, arrow_skip=1, arrow_index=arrow_index)
    elif i < len(gammas)-1:
        ctrl.plot_phase_trajectory(name=f"$\gamma$={gamma}", state_range=[-20,20], show=False, save=False, figure=axs1, add_safe_set=False, color=colors[i], plot_end_state=False, arrow_skip=1, arrow_index=arrow_index)
    else:
        ctrl.plot_phase_trajectory(name=f"$\gamma$={gamma}", state_range=[-20,20], show=False, save=False, figure=axs1, add_safe_set=False, color=colors[i], plot_end_state=False, arrow_skip=1, arrow_index=arrow_index) 
        axs1.legend()
        axs1.legend(loc='lower right')
        file_name = f'phase_trajectory_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300)  
  
    ########################## Plotting control
    name = f"$\gamma$={gamma}"  
    
    ctrl.plot_control(name=name, show=False, save=False, figure=axs2,color=colors[i], ylims=None)

    # Correct ylims 
    if ylims2 is None:
        ylims2 = list(axs2.get_ylim())
    else:
        if axs2.get_ylim()[0] < ylims2[0]:
            ylims2[0] = axs2.get_ylim()[0]
        if axs2.get_ylim()[1] > ylims2[1]:
            ylims2[1] = axs2.get_ylim()[1] 

    if i == len(gammas)-1:  
        axs2.legend()
        axs2.set_ylim(ylims2)
        axs2.legend(loc='upper right')
        axs2.set_xlabel('')
        file_name = f'control_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300)

    ########################## Plotting energy
        
    ctrl.plot_total_energy (name=f"$\gamma$={gamma}", show=False, save=False, figure=axs3, color=colors[i], ylims=None)

    # Correct ylims 
    if ylims3 is None:
        ylims3 = list(axs3.get_ylim())
    else:
        if axs3.get_ylim()[0] < ylims3[0]:
            ylims3[0] = axs3.get_ylim()[0]
        if axs3.get_ylim()[1] > ylims3[1]:
            ylims3[1] = axs3.get_ylim()[1] 

    if i == len(gammas)-1:  
        axs3.legend()
        axs3.legend(loc='lower right')
        axs3.set_ylim(ylims3)
        file_name = f'total_energy_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300) 

    ########################## Plotting cbf
        
    ctrl.plot_cbf(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs4, color=colors[i], ylims=None)
        
    # Correct ylims 
    if ylims4 is None:
        ylims4 = list(axs4.get_ylim())
    else:
        if axs4.get_ylim()[0] < ylims4[0]:
            ylims4[0] = axs4.get_ylim()[0]
        if axs4.get_ylim()[1] > ylims4[1]:
            ylims4[1] = axs4.get_ylim()[1]  

    if i == len(gammas)-1:  
        axs4.legend()
        axs4.legend(loc='lower right')
        axs4.set_ylim(ylims4)
        axs4.set_xlabel('')
        file_name = f'cbf_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300) 

####################################################################################
####################################################################################
####################################################################################
 
# gammas = [0.2, 0.5, 1]
gammas = [0.1, 0.5, 10]
# gammas = [0.5, 2, 50]

_, axs1 = plt.subplots(1, 1, figsize=(14, 14))  
_, axs2 = plt.subplots(1, 1, figsize=(18, 6))  
_, axs2 = plt.subplots(1, 1, figsize=(18, 6))  
_, axs3 = plt.subplots(1, 1, figsize=(18, 10))  
_, axs4 = plt.subplots(1, 1, figsize=(18, 6))  

ylims2 = None
ylims3 = None
ylims4 = None

for i, gamma in enumerate(gammas):

    # Use slider values in parameters
    parameter = {
        'time_horizon': 100,
        'time_step': 0.1,
        'init_state': [-15, 15],
        'target_state': None,
        'weight_input': 1,
        'cbf_gamma': gamma,
        'weight_slack': None,
        'u_max': None,
        'u_min': None  
    } 

    run_name = "dampKout_massspring" 

    model = MassSpring(m=2, k=0.5, dt=parameter["time_step"], verbose=False)
 
    cbf = EnergyLimit(energy_func=model.K, c=20, pump=False)

    ctrl = Controller(
        model, 
        parameter,  
        cbf=cbf
    )

    ctrl.run(save=True, name=f"{run_name}_gamma{parameter['cbf_gamma']}")
     
    # Plotting  
    arrow_index = 10

    ######################### Plotting phase trajectory
    if i ==0:  
        ctrl.plot_phase_trajectory(name=f"$\gamma$={gamma}", state_range=[-20,20], show=False, save=False, figure=axs1, add_safe_set=True, color=colors[i], plot_end_state=False, arrow_skip=1, arrow_index=arrow_index)
    elif i < len(gammas)-1:
        ctrl.plot_phase_trajectory(name=f"$\gamma$={gamma}", state_range=[-20,20], show=False, save=False, figure=axs1, add_safe_set=False, color=colors[i], plot_end_state=False, arrow_skip=1, arrow_index=arrow_index)
    else:
        ctrl.plot_phase_trajectory(name=f"$\gamma$={gamma}", state_range=[-20,20], show=False, save=False, figure=axs1, add_safe_set=False, color=colors[i], plot_end_state=False, arrow_skip=1, arrow_index=arrow_index) 
        axs1.legend()
        axs1.legend(loc='lower right')
        file_name = f'phase_trajectory_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300)  
  
    ########################## Plotting control
    name = f"$\gamma$={gamma}"  
    
    ctrl.plot_control(name=name, show=False, save=False, figure=axs2, color=colors[i], ylims=None) 

    # Correct ylims 
    if ylims2 is None:
        ylims2 = list(axs2.get_ylim())
    else:
        if axs2.get_ylim()[0] < ylims2[0]:
            ylims2[0] = axs2.get_ylim()[0]
        if axs2.get_ylim()[1] > ylims2[1]:
            ylims2[1] = axs2.get_ylim()[1] 

    if i == len(gammas)-1:  
        axs2.legend()
        axs2.legend(loc='lower right')
        axs2.set_ylim(ylims2)
        axs2.set_xlabel('')
        file_name = f'control_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300)
        

    ########################## Plotting energy 

    ctrl.plot_total_energy (name=f"$\gamma$={gamma}", show=False, save=False, figure=axs3, color=colors[i], ylims=None) 
    
    # Correct ylims 
    if ylims3 is None:
        ylims3 = list(axs3.get_ylim())
    else:
        if axs3.get_ylim()[0] < ylims3[0]:
            ylims3[0] = axs3.get_ylim()[0]
        if axs3.get_ylim()[1] > ylims3[1]:
            ylims3[1] = axs3.get_ylim()[1] 

    if i == len(gammas)-1:  
        axs3.legend()
        axs3.legend(loc='upper right')
        axs3.set_ylim(ylims3)
        file_name = f'total_energy_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300) 

    ########################## Plotting cbf 

    ctrl.plot_cbf(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs4, color=colors[i], ylims=None) 

    # Correct ylims 
    if ylims4 is None:
        ylims4 = list(axs4.get_ylim())
    else:
        if axs4.get_ylim()[0] < ylims4[0]:
            ylims4[0] = axs4.get_ylim()[0]
        if axs4.get_ylim()[1] > ylims4[1]:
            ylims4[1] = axs4.get_ylim()[1] 

    if i == len(gammas)-1:  
        axs4.legend()
        axs2.legend(loc='upper right')
        axs4.set_ylim(ylims4)
        axs4.set_xlabel('')
        file_name = f'cbf_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300) 


####################################################################################
####################################################################################
####################################################################################
 
# gammas = [0.2, 0.5, 1] 
gammas = [0.1, 0.5, 10]
# gammas = [0.5, 2, 50]

_, axs1 = plt.subplots(1, 1, figsize=(14, 14))  
_, axs2 = plt.subplots(1, 1, figsize=(18, 6))  
_, axs2 = plt.subplots(1, 1, figsize=(18, 6))  
_, axs3 = plt.subplots(1, 1, figsize=(18, 10))  
_, axs4 = plt.subplots(1, 1, figsize=(18, 6))  

ylims2 = None
ylims3 = None
ylims4 = None

for i, gamma in enumerate(gammas):

    # Use slider values in parameters
    parameter = {
        'time_horizon': 100,
        'time_step': 0.1,
        'init_state': [-15, 0.1],
        'target_state': None,
        'weight_input': 1,
        'cbf_gamma': gamma,
        'weight_slack': None,
        'u_max': None,
        'u_min': None  
    } 

    run_name = "dampKin_massspring" 

    model = MassSpring(m=2, k=0.5, dt=parameter["time_step"], verbose=False)
 
    cbf = EnergyLimit(energy_func=model.K, c=20, pump=False)

    ctrl = Controller(
        model, 
        parameter,  
        cbf=cbf
    )

    ctrl.run(save=True, name=f"{run_name}_gamma{parameter['cbf_gamma']}")
     
    # Plotting  
    arrow_index = 13

    ######################### Plotting phase trajectory
    if i ==0:  
        ctrl.plot_phase_trajectory(name=f"$\gamma$={gamma}", state_range=[-20,20], show=False, save=False, figure=axs1, add_safe_set=True, color=colors[i], plot_end_state=False, arrow_skip=1, arrow_index=arrow_index)
    elif i < len(gammas)-1:
        ctrl.plot_phase_trajectory(name=f"$\gamma$={gamma}", state_range=[-20,20], show=False, save=False, figure=axs1, add_safe_set=False, color=colors[i], plot_end_state=False, arrow_skip=1, arrow_index=arrow_index)
    else:
        ctrl.plot_phase_trajectory(name=f"$\gamma$={gamma}", state_range=[-20,20], show=False, save=False, figure=axs1, add_safe_set=False, color=colors[i], plot_end_state=False, arrow_skip=1, arrow_index=arrow_index) 
        axs1.legend()
        axs1.legend(loc='lower right')
        file_name = f'phase_trajectory_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300)  
  
    ########################## Plotting control
    name = f"$\gamma$={gamma}"   

    ctrl.plot_control(name=name, show=False, save=False, figure=axs2, color=colors[i], ylims=None) 

    # Correct ylims 
    if ylims2 is None:
        ylims2 = list(axs2.get_ylim())
    else:
        if axs2.get_ylim()[0] < ylims2[0]:
            ylims2[0] = axs2.get_ylim()[0]
        if axs2.get_ylim()[1] > ylims2[1]:
            ylims2[1] = axs2.get_ylim()[1]  
            
    if i == len(gammas)-1:  
        axs2.legend()
        axs2.legend(loc='lower right')
        axs2.set_ylim(ylims2)
        axs2.set_xlabel('')
        file_name = f'control_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300)

    ########################## Plotting energy 

    ctrl.plot_total_energy (name=f"$\gamma$={gamma}", show=False, save=False, figure=axs3, color=colors[i], ylims=None) 

    # Correct ylims 
    if ylims3 is None:
        ylims3 = list(axs3.get_ylim())
    else:
        if axs3.get_ylim()[0] < ylims3[0]:
            ylims3[0] = axs3.get_ylim()[0]
        if axs3.get_ylim()[1] > ylims3[1]:
            ylims3[1] = axs3.get_ylim()[1] 

    if i == len(gammas)-1:  
        axs3.legend()
        axs3.legend(loc='upper right')
        axs3.set_ylim(ylims3)
        file_name = f'total_energy_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300) 

    ########################## Plotting cbf 

    ctrl.plot_cbf(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs4, color=colors[i], ylims=None) 

    # Correct ylims 
    if ylims4 is None:
        ylims4 = list(axs4.get_ylim())
    else:
        if axs4.get_ylim()[0] < ylims4[0]:
            ylims4[0] = axs4.get_ylim()[0]
        if axs4.get_ylim()[1] > ylims4[1]:
            ylims4[1] = axs4.get_ylim()[1] 

    if i == len(gammas)-1:  
        axs4.legend()
        axs2.legend(loc='upper right')
        axs4.set_ylim(ylims4)
        axs4.set_xlabel('')
        file_name = f'cbf_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300) 



####################################################################################
####################################################################################
####################################################################################
  
gammas = [0.1, 10] 

_, axs1 = plt.subplots(1, 1, figsize=(14, 14))  
_, axs2 = plt.subplots(1, 1, figsize=(18, 6))  
_, axs2 = plt.subplots(1, 1, figsize=(18, 6))  
_, axs3 = plt.subplots(1, 1, figsize=(18, 10))  
_, axs4 = plt.subplots(1, 1, figsize=(18, 6))    

ylims2 = None
ylims3 = None
ylims4 = None

for i, gamma in enumerate(gammas):

    # Use slider values in parameters
    parameter = {
        'time_horizon': 10,
        'time_step': 0.01,
        'init_state': [3.14,3.14,0.001,0.001],
        'target_state': None,
        'weight_input': 1,
        'cbf_gamma': gamma,
        'weight_slack': None,
        'u_max': None,
        'u_min': None  
    } 

    run_name = "dampKin_doublependulum" 

    model = DoublePendulum(m1=2, m2=2, l1=1, l2=1, dt=parameter["time_step"], verbose=False)
 
    cbf = EnergyLimit(energy_func=model.K, c=10, pump=False, num_states=model.num_states)

    ctrl = Controller(
        model, 
        parameter,  
        cbf=cbf
    )

    ctrl.run(save=True, name=f"{run_name}_gamma{parameter['cbf_gamma']}")
    
    # Plotting  

    ########################## Plotting control
    name = f"$, \gamma$={gamma}"  
    
    ctrl.plot_control(name=name, show=False, save=False, figure=axs2, color=colors[i], ylims=None) 

    # Correct ylims 
    # if ylims2 is None:
    #     ylims2 = list(axs2.get_ylim())
    # else:
    #     if axs2.get_ylim()[0] < ylims2[0]:
    #         ylims2[0] = axs2.get_ylim()[0]
    #     if axs2.get_ylim()[1] > ylims2[1]:
    #         ylims2[1] = axs2.get_ylim()[1] 

    if i == len(gammas)-1:  
        axs2.legend()
        axs2.legend(loc='upper right')
        axs2.set_ylim([-10,44])
        axs2.set_xlabel('')
        file_name = f'control_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300) 

    ########################## Plotting energy 

    ctrl.plot_total_energy (name=f"$\gamma$={gamma}", show=False, save=False, figure=axs3, color=colors[i], ylims=None) 

    # Correct ylims 
    if ylims3 is None:
        ylims3 = list(axs3.get_ylim())
    else:
        if axs3.get_ylim()[0] < ylims3[0]:
            ylims3[0] = axs3.get_ylim()[0]
        if axs3.get_ylim()[1] > ylims3[1]:
            ylims3[1] = axs3.get_ylim()[1] 

    if i == len(gammas)-1:  
        axs3.legend()
        axs3.legend(loc='upper right')
        axs3.set_ylim(ylims3)
        file_name = f'total_energy_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300) 

    ########################## Plotting cbf
    ctrl.plot_cbf(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs4, color=colors[i], ylims=None)

    print("@@@@@@@@", axs4.get_ylim(), ylims4) 
    
    # Correct ylims 
    if ylims4 is None:
        ylims4 = list(axs4.get_ylim())
    else:
        if axs4.get_ylim()[0] < ylims4[0]:
            ylims4[0] = axs4.get_ylim()[0]
        if axs4.get_ylim()[1] > ylims4[1]:
            ylims4[1] = axs4.get_ylim()[1] 


    if i == len(gammas)-1:  
        axs4.legend()
        axs4.legend(loc='lower right')
        axs4.set_ylim(ylims4)
        axs4.set_xlabel('')
        file_name = f'cbf_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300) 
 

####################################################################################
####################################################################################
####################################################################################
  
gammas = [0.1, 10] 

_, axs1 = plt.subplots(1, 1, figsize=(14, 14))  
_, axs2 = plt.subplots(1, 1, figsize=(18, 6))  
_, axs2 = plt.subplots(1, 1, figsize=(18, 6))  
_, axs3 = plt.subplots(1, 1, figsize=(18, 10))  
_, axs4 = plt.subplots(1, 1, figsize=(18, 6))    

ylims2 = None
ylims3 = None
ylims4 = None

for i, gamma in enumerate(gammas):

    # Use slider values in parameters
    parameter = {
        'time_horizon': 10,
        'time_step': 0.01,
        'init_state': [3.14,3.14,6.28,6.28],
        'target_state': None,
        'weight_input': 1,
        'cbf_gamma': gamma,
        'weight_slack': None,
        'u_max': None,
        'u_min': None  
    } 

    run_name = "dampKout_doublependulum" 

    model = DoublePendulum(m1=2, m2=2, l1=1, l2=1, dt=parameter["time_step"], verbose=False)
 
    cbf = EnergyLimit(energy_func=model.K, c=10, pump=False, num_states=model.num_states)

    ctrl = Controller(
        model, 
        parameter,  
        cbf=cbf
    )

    ctrl.run(save=True, name=f"{run_name}_gamma{parameter['cbf_gamma']}")
     
    # Plotting  

    ########################## Plotting control
    name = f"$, \gamma$={gamma}"   

    ctrl.plot_control(name=name, show=False, save=False, figure=axs2, color=colors[i], ylims=None) 
    
    # # Correct ylims 
    # if ylims2 is None:
    #     ylims2 = list(axs2.get_ylim())
    # else:
    #     if axs2.get_ylim()[0] < ylims2[0]:
    #         ylims2[0] = axs2.get_ylim()[0]
    #     if axs2.get_ylim()[1] > ylims2[1]:
    #         ylims2[1] = axs2.get_ylim()[1] 

    if i == len(gammas)-1:  
        axs2.legend()
        axs2.legend(loc='lower right')
        axs2.set_ylim([-44,33])
        axs2.set_xlabel('')
        file_name = f'control_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300) 

    ########################## Plotting energy 

    ctrl.plot_total_energy (name=f"$\gamma$={gamma}", show=False, save=False, figure=axs3, color=colors[i], ylims=None) 

    # Correct ylims 
    if ylims3 is None:
        ylims3 = list(axs3.get_ylim())
    else:
        if axs3.get_ylim()[0] < ylims3[0]:
            ylims3[0] = axs3.get_ylim()[0]
        if axs3.get_ylim()[1] > ylims3[1]:
            ylims3[1] = axs3.get_ylim()[1] 

    if i == len(gammas)-1:  
        axs3.legend()
        axs3.legend(loc='upper right')
        axs3.set_ylim(ylims3)
        file_name = f'total_energy_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300) 

    ########################## Plotting cbf 

    ctrl.plot_cbf(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs4, color=colors[i], ylims=None) 

    # Correct ylims 
    if ylims4 is None:
        ylims4 = list(axs4.get_ylim())
    else:
        if axs4.get_ylim()[0] < ylims4[0]:
            ylims4[0] = axs4.get_ylim()[0]
        if axs4.get_ylim()[1] > ylims4[1]:
            ylims4[1] = axs4.get_ylim()[1] 

    if i == len(gammas)-1:  
        axs4.legend()
        axs4.legend(loc='lower right')
        axs4.set_xlabel('')
        axs4.set_ylim(ylims4)
        file_name = f'cbf_{run_name}.{image_file_format}'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300) 

