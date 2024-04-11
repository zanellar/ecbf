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

tk.Label(root, text='q0').pack()
init_state_value1 = tk.Spinbox(root, from_=-10, to=10, textvariable=tk.DoubleVar(value=-12), increment=0.1)
init_state_value1.pack()

tk.Label(root, text='p0').pack()
init_state_value2 = tk.Spinbox(root, from_=-10, to=10, textvariable=tk.DoubleVar(value=3), increment=0.1)
init_state_value2.pack()

# tk.Label(root, text='q*').pack()
# target_state_value1 = tk.Spinbox(root, from_=-20, to=20, textvariable=tk.DoubleVar(value=0), increment=0.1)
# target_state_value1.pack()

# tk.Label(root, text='p*').pack()
# target_state_value2 = tk.Spinbox(root, from_=-20, to=20, textvariable=tk.DoubleVar(value=0), increment=0.1)
# target_state_value2.pack()

tk.Label(root, text='Weight Input').pack()
weight_input_value = tk.Spinbox(root, from_=0, to=100, textvariable=tk.DoubleVar(value=1), increment=0.1)
weight_input_value.pack()

tk.Label(root, text='CBF Gamma').pack()
cbf_gamma_value = tk.Spinbox(root, from_=0, to=10, textvariable=tk.DoubleVar(value=1), increment=0.01)
cbf_gamma_value.pack()

tk.Label(root, text='Weight Slack').pack()
weight_slack_value = tk.Spinbox(root, from_=0, to=10, textvariable=tk.DoubleVar(value=0), increment=0.1)
weight_slack_value.pack()

tk.Label(root, text='U Max').pack()
u_max_value = tk.Spinbox(root, from_=0, to=100, textvariable=tk.IntVar(value=100))
u_max_value.pack()

tk.Label(root, text='U Min').pack()
u_min_value = tk.Spinbox(root, from_=-100, to=0, textvariable=tk.IntVar(value=-100))
u_min_value.pack()

tk.Label(root, text='c (offset CBF)').pack()
c_value = tk.Spinbox(root, from_=-1000, to=1000, textvariable=tk.IntVar(value=20))
c_value.pack()
 
# Create checkboxes for u_min and u_max
# u_min_var = tk.BooleanVar()
# u_min_check = tk.Checkbutton(root, text="Use U Min", variable=u_min_var)
# u_min_check.pack()

# u_max_var = tk.BooleanVar()
# u_max_check = tk.Checkbutton(root, text="Use U Max", variable=u_max_var)
# u_max_check.pack()
 
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
        'target_state': [0, 0],
        'weight_input': float(weight_input_value.get()),
        'weight_slack': float(weight_slack_value.get()),
        'clf_lambda': None, 
        'u_max': float(u_max_value.get()),
        'u_min': float(u_min_value.get()),
        'cbf_gamma': float(cbf_gamma_value.get())
    }

    model = MassSpring(m=2, k=0.5, dt=parameter["time_step"], verbose=False)
    # model = Pendulum(m=2, l=0.5, dt=parameter["time_step"], verbose=False)
    # model = DoublePendulum(m1=1, m2=1, l1=1, l2=1, dt=parameter["time_step"], verbose=False)

    # cbf = SafeDoughnut(C1=20, C2=10) 
    # cbf = SafeCircle(r=9, c=[0, 0])  
    cbf = EnergyLimit(energy_func=model.K, c=float(c_value.get()), pump=pump_var.get()) 
    # cbf = EnergyLimit(energy_func=model.H, c=float(c_value.get()), pump=pump_var.get()) 
 
    ctrl = Controller(
        model, 
        parameter,  
        cbf=cbf 
    )

    ctrl.run()


    ctrl.show(
        # ctrl.plot_phase_trajectory, 
        # ctrl.plot_state,
        ctrl.plot_energy_openloop,
        ctrl.plot_cbf_constraint,
        ctrl.plot_control,
        ctrl.plot_cbf,
        subplots=(3, 2)
    ) 

 

run_button = tk.Button(root, text="Run Controller", command=run_controller)
run_button.pack()

# Run the Tk event loop
root.mainloop()