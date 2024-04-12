import tkinter as tk 
import matplotlib.pyplot as plt

from ecbf.defined_models.mass_spring import MassSpring
from ecbf.defined_models.pendulum import Pendulum
from ecbf.defined_models.double_pendulum import DoublePendulum
from ecbf.barrier_functions.safe_doughnut import SafeDoughnut
from ecbf.barrier_functions.safe_circle import SafeCircle
from ecbf.barrier_functions.energy_limit import EnergyLimit 
from ecbf.scripts.control import Controller

# Create a new Tk root widget
root = tk.Tk()
 
# Create Spinboxes for parameters with labels
tk.Label(root, text='Time Horizon').pack()
time_horizon_value = tk.Spinbox(root, from_=1, to=100, textvariable=tk.IntVar(value=15))
time_horizon_value.pack()

tk.Label(root, text='Time Step').pack()
time_step_value = tk.Spinbox(root, from_=0.01, to=1, textvariable=tk.DoubleVar(value=0.1), increment=0.01)
time_step_value.pack()

tk.Label(root, text='m').pack()
m_value = tk.Spinbox(root, from_=-20, to=20, textvariable=tk.DoubleVar(value=2), increment=0.1)
m_value.pack()

tk.Label(root, text='k').pack()
k_value = tk.Spinbox(root, from_=-20, to=20, textvariable=tk.DoubleVar(value=0.5), increment=0.1)
k_value.pack()

tk.Label(root, text='q0').pack()
init_state_value1 = tk.Spinbox(root, from_=-10, to=10, textvariable=tk.DoubleVar(value=-12), increment=0.1)
init_state_value1.pack()

tk.Label(root, text='p0').pack()
init_state_value2 = tk.Spinbox(root, from_=-10, to=10, textvariable=tk.DoubleVar(value=3), increment=0.1)
init_state_value2.pack()
 
tk.Label(root, text='CBF Gamma').pack()
cbf_gamma_value = tk.Spinbox(root, from_=0, to=10, textvariable=tk.DoubleVar(value=1), increment=0.01)
cbf_gamma_value.pack()
  
tk.Label(root, text='c (offset CBF)').pack()
c_value = tk.Spinbox(root, from_=-1000, to=1000, textvariable=tk.IntVar(value=20))
c_value.pack()
  

limit_K_only_var = tk.BooleanVar()
u_max_check = tk.Checkbutton(root, text="Limit Kinetic Energy Only", variable=limit_K_only_var)
u_max_check.pack()
 
pump_var = tk.BooleanVar()
pump_check = tk.Checkbutton(root, text="Energy Pump", variable=pump_var)
pump_check.pack()

# Create a button to run the controller
def run_controller():

    # close all figures
    plt.close('all')

    # Use slider values in parameters
    parameter = {
        'time_horizon': float(time_horizon_value.get()),
        'time_step': float(time_step_value.get()),
        'init_state': [float(init_state_value1.get()), float(init_state_value2.get())],
        'target_state': None,
        'weight_input': 1,
        'weight_slack': None,
        'clf_lambda': None, 
        'u_max': None,
        'u_min': None,
        'cbf_gamma': float(cbf_gamma_value.get())
    }

    model = MassSpring(m=float(m_value.get()), k=float(k_value.get()), dt=parameter["time_step"], verbose=False) 

    # Control Barrier Function 
    if limit_K_only_var.get():
        energy_func = model.K
    else:
        energy_func = model.H
    cbf = EnergyLimit(energy_func=energy_func, c=float(c_value.get()), pump=pump_var.get())  
 
    # Create a controller
    ctrl = Controller(
        model, 
        parameter,  
        cbf=cbf 
    )

    # Run the controller
    ctrl.run()

    # Show the plots
    ctrl.show(
        ctrl.plot_phase_trajectory, 
        ctrl.plot_state,
        ctrl.plot_total_energy ,
        ctrl.plot_cbf_constraint,
        ctrl.plot_control,
        ctrl.plot_cbf,
        subplots=(3, 2)
    ) 

 

run_button = tk.Button(root, text="Run Controller", command=run_controller)
run_button.pack()

# Run the Tk event loop
root.mainloop()