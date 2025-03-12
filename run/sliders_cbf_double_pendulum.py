import tkinter as tk 
import matplotlib.pyplot as plt

from ecbf.defined_models.mass_spring import MassSpring
from ecbf.defined_models.pendulum import Pendulum
from ecbf.defined_models.double_pendulum import DoublePendulum
from ecbf.barrier_functions.safe_doughnut import SafeDoughnut
from ecbf.barrier_functions.safe_circle import SafeCircle
from ecbf.barrier_functions.energy_limit import EnergyLimit 
from ecbf.scripts.control import Controller

import numpy as np 

# Create a new Tk root widget
root = tk.Tk()
 
# Create Spinboxes for parameters with labels
tk.Label(root, text='Time Horizon').pack()
time_horizon_value = tk.Spinbox(root, from_=1, to=100, textvariable=tk.IntVar(value=10))
time_horizon_value.pack()

tk.Label(root, text='Time Step').pack()
time_step_value = tk.Spinbox(root, from_=0.001, to=0.1, textvariable=tk.DoubleVar(value=0.01), increment=0.001)
time_step_value.pack()

tk.Label(root, text='m1').pack()
m1_value = tk.Spinbox(root, from_=-20, to=20, textvariable=tk.DoubleVar(value=1.5), increment=0.1)
m1_value.pack()

tk.Label(root, text='m2').pack()
m2_value = tk.Spinbox(root, from_=-20, to=20, textvariable=tk.DoubleVar(value=1.5), increment=0.1)
m2_value.pack()

tk.Label(root, text='l1').pack()
l1_value = tk.Spinbox(root, from_=-20, to=20, textvariable=tk.DoubleVar(value=1), increment=0.1)
l1_value.pack()

tk.Label(root, text='l2').pack()
l2_value = tk.Spinbox(root, from_=-20, to=20, textvariable=tk.DoubleVar(value=1), increment=0.1)
l2_value.pack()

tk.Label(root, text='b1').pack()
b1_value = tk.Spinbox(root, from_=-20, to=20, textvariable=tk.DoubleVar(value=1), increment=0.1)
b1_value.pack()

tk.Label(root, text='b2').pack()
b2_value = tk.Spinbox(root, from_=-20, to=20, textvariable=tk.DoubleVar(value=1), increment=0.1)
b2_value.pack()

tk.Label(root, text='init q1').pack()
init_q1_value = tk.Spinbox(root, from_=-10, to=10, textvariable=tk.DoubleVar(value=1.57), increment=0.1)
init_q1_value.pack()

tk.Label(root, text='init q2').pack()
init_q2_value = tk.Spinbox(root, from_=-10, to=10, textvariable=tk.DoubleVar(value=1.57), increment=0.1)
init_q2_value.pack()

tk.Label(root, text='init p1').pack()
init_p1_value = tk.Spinbox(root, from_=-10, to=10, textvariable=tk.DoubleVar(value=1.57), increment=0.1)
init_p1_value.pack()

tk.Label(root, text='init p2').pack()
init_p2_value = tk.Spinbox(root, from_=-10, to=10, textvariable=tk.DoubleVar(value=1.57), increment=0.1)
init_p2_value.pack()
  
tk.Label(root, text='CBF Gamma').pack()
cbf_gamma_value = tk.Spinbox(root, from_=0, to=10, textvariable=tk.DoubleVar(value=1), increment=0.01)
cbf_gamma_value.pack() 

tk.Label(root, text='c (offset CBF)').pack()
c_value = tk.Spinbox(root, from_=-1000, to=1000, textvariable=tk.IntVar(value=10))
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
    print([float(init_q1_value.get()), float(init_q2_value.get()), float(init_p1_value.get()), float(init_p2_value.get())])
    parameter = {
        'time_horizon': float(time_horizon_value.get()),
        'time_step': float(time_step_value.get()),
        'init_state': [float(init_q1_value.get()), float(init_q2_value.get()), float(init_p1_value.get()), float(init_p2_value.get())],
        'target_state': None,
        'weight_input': 1,
        'weight_slack': None,
        'clf_lambda': None, 
        'u_max': None,
        'u_min': None,
        'cbf_gamma': float(cbf_gamma_value.get())
    }
 
    model = DoublePendulum(
        m1=float(m1_value.get()), 
        m2=float(m2_value.get()), 
        l1=float(l1_value.get()), 
        l2=float(l2_value.get()),  
        b1=float(b1_value.get()),
        b2=float(b2_value.get()),
        dt=parameter["time_step"], 
        verbose=False
    )


    if limit_K_only_var.get():
        energy_func = model.K
    else:
        energy_func = model.H

    cbf = EnergyLimit(num_states=4, energy_func=energy_func, c=float(c_value.get()), pump=pump_var.get())  
 
    ctrl = Controller(
        model, 
        parameter,  
        cbf=cbf,
        # regularization=2
    )

    ctrl.run()


    ctrl.show(
        # ctrl.plot_phase_trajectory, 
        # ctrl.plot_state,
        ctrl.plot_total_energy ,
        ctrl.plot_kinetic_energy ,
        ctrl.plot_potential_energy ,
        ctrl.plot_cbf_constraint,
        ctrl.plot_control,
        ctrl.plot_cbf,
        subplots=(3, 2)
    ) 

    x = np.array(ctrl.xt)
    traj_angles = [[x[0][i], x[1][i]] for i in range(x.shape[1])] 
    model.visualize(traj_angles, skip=50)
 

run_button = tk.Button(root, text="Run Controller", command=run_controller)
run_button.pack()

# Run the Tk event loop
root.mainloop()