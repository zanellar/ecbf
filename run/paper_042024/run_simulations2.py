import matplotlib.pyplot as plt
import os


from ecbf.defined_models.mass_spring import MassSpring
from ecbf.barrier_functions.safe_doughnut import SafeDoughnut
from ecbf.barrier_functions.safe_circle import SafeCircle
from ecbf.barrier_functions.energy_limit import EnergyLimit
from ecbf.scripts.control import Controller
from ecbf.utils.paths import PLOTS_PATH 
  
colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

image_file_format = 'png'

def plots(i, N, ctrl, run_name, axs):

    # Plotting phase trajectory
    if i ==0:  
        ctrl.plot_phase_trajectory(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs[0], add_safe_set=True, color=colors[i], plot_end_state=False, arrow_index=10)
    elif i == N-1:
        ctrl.plot_phase_trajectory(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs[0], add_safe_set=False, color=colors[i], plot_end_state=False, arrow_index=10) 
        axs[0].legend()
        file_name = f'phase_trajectory_{run_name}.png'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300)  
    else:
        ctrl.plot_phase_trajectory(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs[0], add_safe_set=False, color=colors[i], plot_end_state=False, arrow_index=10)


    # Plotting state
    if i ==0:  
        ctrl.plot_state(name=f", $\gamma$={gamma}", show=False, save=False, figure=axs[1], color=colors[i])
    elif i == N-1:
        ctrl.plot_state(name=f", $\gamma$={gamma}", show=False, save=False, figure=axs[1], color=colors[i])
        axs[1].legend()
        file_name = f'state_{run_name}.png'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300)
    else:
        ctrl.plot_state(name=f", $\gamma$={gamma}", show=False, save=False, figure=axs[1],color=colors[i])

    # Plotting control
    if i ==0:  
        ctrl.plot_control(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs[2], color=colors[i])
    elif i == N-1:
        ctrl.plot_control(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs[2], color=colors[i])
        axs[2].legend()
        file_name = f'control_{run_name}.png'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300)
    else:
        ctrl.plot_control(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs[2],color=colors[i])

    # Plotting energy
    if i ==0:  
        ctrl.plot_energy_openloop(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs[3], color=colors[i])
    elif i == N-1:
        ctrl.plot_energy_openloop(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs[3], color=colors[i])
        axs[3].legend()
        file_name = f'energy_openloop_{run_name}.png'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300)
    else:
        ctrl.plot_energy_openloop(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs[3],color=colors[i])

    # Plotting cbf
    if i ==0:  
        ctrl.plot_cbf(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs[4], color=colors[i])
    elif i == N-1:
        ctrl.plot_cbf(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs[4], color=colors[i])
        axs[4].legend()
        file_name = f'cbf_{run_name}.png'
        plt.savefig(os.path.join(PLOTS_PATH, file_name), format=image_file_format, dpi=300)
    else:
        ctrl.plot_cbf(name=f"$\gamma$={gamma}", show=False, save=False, figure=axs[4],color=colors[i])

####################################################################################
####################################################################################
####################################################################################

gammas = [2,4] 

_, axs1 = plt.subplots(1, 1, figsize=(8, 8))  
_, axs2 = plt.subplots(1, 1, figsize=(8, 8))  
_, axs3 = plt.subplots(1, 1, figsize=(8, 8))  
_, axs4 = plt.subplots(1, 1, figsize=(8, 8))  
_, axs5 = plt.subplots(1, 1, figsize=(8, 8))  
 
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

    run_name = "dampH"

    model = MassSpring(m=2, k=0.5, dt=parameter["time_step"], verbose=False)
 
    cbf = EnergyLimit(energy_func=model.H, c=10)

    ctrl = Controller(
        model, 
        parameter,  
        cbf=cbf
    )

    ctrl.run(save=True, name=f"{run_name}_gamma{parameter['cbf_gamma']}")
    
    # Plotting 
    plots(i, len(gammas), ctrl, run_name, axs = [axs1, axs2, axs3, axs4, axs5])

####################################################################################
####################################################################################
####################################################################################


gammas = [2,4]

_, axs1 = plt.subplots(1, 1, figsize=(8, 8))  
_, axs2 = plt.subplots(1, 1, figsize=(8, 8))  
_, axs3 = plt.subplots(1, 1, figsize=(8, 8))  
_, axs4 = plt.subplots(1, 1, figsize=(8, 8))  
_, axs5 = plt.subplots(1, 1, figsize=(8, 8))  

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

    run_name = "pumpH" 

    model = MassSpring(m=2, k=0.5, dt=parameter["time_step"], verbose=False)
 
    cbf = EnergyLimit(energy_func=model.H, c=10, pump=True)

    ctrl = Controller(
        model, 
        parameter,  
        cbf=cbf
    )

    ctrl.run(save=True, name=f"{run_name}_gamma{parameter['cbf_gamma']}")
    
    # Plotting
    plots(i, len(gammas), ctrl, run_name, axs = [axs1, axs2, axs3, axs4, axs5])


####################################################################################
####################################################################################
####################################################################################
 
# gammas = [0.2, 0.5, 1]
gammas = [0.1, 0.5, 10]
# gammas = [0.5, 2, 50]

_, axs1 = plt.subplots(1, 1, figsize=(8, 8))  
_, axs2 = plt.subplots(1, 1, figsize=(8, 8))  
_, axs3 = plt.subplots(1, 1, figsize=(8, 8))  
_, axs4 = plt.subplots(1, 1, figsize=(8, 8))  
_, axs5 = plt.subplots(1, 1, figsize=(8, 8))  

for i, gamma in enumerate(gammas):

    # Use slider values in parameters
    parameter = {
        'time_horizon': 60,
        'time_step': 0.1,
        'init_state': [-15, 15],
        'target_state': None,
        'weight_input': 1,
        'cbf_gamma': gamma,
        'weight_slack': None,
        'u_max': None,
        'u_min': None  
    } 

    run_name = "dampKout" 

    model = MassSpring(m=2, k=0.5, dt=parameter["time_step"], verbose=False)
 
    cbf = EnergyLimit(energy_func=model.K, c=20, pump=False)

    ctrl = Controller(
        model, 
        parameter,  
        cbf=cbf
    )

    ctrl.run(save=True, name=f"{run_name}_gamma{parameter['cbf_gamma']}")
    
    # Plotting
    plots(i, len(gammas), ctrl, run_name, axs = [axs1, axs2, axs3, axs4, axs5])


####################################################################################
####################################################################################
####################################################################################
 
# gammas = [0.2, 0.5, 1] 
gammas = [0.1, 0.5, 10]
# gammas = [0.5, 2, 50]

_, axs1 = plt.subplots(1, 1, figsize=(8, 8))  
_, axs2 = plt.subplots(1, 1, figsize=(8, 8))  
_, axs3 = plt.subplots(1, 1, figsize=(8, 8))  
_, axs4 = plt.subplots(1, 1, figsize=(8, 8))  
_, axs5 = plt.subplots(1, 1, figsize=(8, 8))  

for i, gamma in enumerate(gammas):

    # Use slider values in parameters
    parameter = {
        'time_horizon': 60,
        'time_step': 0.1,
        'init_state': [-15, 0.1],
        'target_state': None,
        'weight_input': 1,
        'cbf_gamma': gamma,
        'weight_slack': None,
        'u_max': None,
        'u_min': None  
    } 

    run_name = "dampKin" 

    model = MassSpring(m=2, k=0.5, dt=parameter["time_step"], verbose=False)
 
    cbf = EnergyLimit(energy_func=model.K, c=20, pump=False)

    ctrl = Controller(
        model, 
        parameter,  
        cbf=cbf
    )

    ctrl.run(save=True, name=f"{run_name}_gamma{parameter['cbf_gamma']}")
    
    # Plotting
    plots(i, len(gammas), ctrl, run_name, axs = [axs1, axs2, axs3, axs4, axs5])

