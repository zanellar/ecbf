import tkinter as tk
from ecbf.defined_models.mass_spring import MassSpring
from ecbf.scripts.control import Controller
from ecbf.lyapunov_functions.quadratic import Quadratic
import matplotlib.pyplot as plt

# Create a new Tk root widget
root = tk.Tk()

# Create sliders for parameters
time_horizon_slider = tk.Scale(root, from_=1, to=100, resolution=1, orient='horizontal', label='Time Horizon')
time_horizon_slider.set(15)
time_horizon_slider.pack()

time_step_slider = tk.Scale(root, from_=0.001, to=0.1, resolution=0.01, orient='horizontal', label='Time Step')
time_step_slider.set(0.1)
time_step_slider.pack()

init_state_slider1 = tk.Scale(root, from_=-20, to=20, resolution=0.1, orient='horizontal', label='q0')
init_state_slider1.set(8)
init_state_slider1.pack()

init_state_slider2 = tk.Scale(root, from_=-20, to=20, resolution=0.1, orient='horizontal', label='p0')
init_state_slider2.set(0)
init_state_slider2.pack()

target_state_slider1 = tk.Scale(root, from_=-20, to=20, resolution=0.1, orient='horizontal', label='q*')
target_state_slider1.set(3)
target_state_slider1.pack()

target_state_slider2 = tk.Scale(root, from_=-20, to=20, resolution=0.1, orient='horizontal', label='p*')
target_state_slider2.set(0)
target_state_slider2.pack()

weight_input_slider = tk.Scale(root, from_=0, to=10, resolution=0.1, orient='horizontal', label='Weight Input')
weight_input_slider.set(1)
weight_input_slider.pack()

clf_lambda_slider = tk.Scale(root, from_=0, to=1, resolution=0.01, orient='horizontal', label='CLF Lambda')
clf_lambda_slider.set(0.1)
clf_lambda_slider.pack()

weight_slack_slider = tk.Scale(root, from_=0, to=10, resolution=0.1, orient='horizontal', label='Weight Slack')
weight_slack_slider.set(0.1)
weight_slack_slider.pack()


u_max_slider = tk.Scale(root, from_=0, to=100, resolution=1, orient='horizontal', label='U Max')
u_max_slider.set(100)
u_max_slider.pack()

u_min_slider = tk.Scale(root, from_=-100, to=0, resolution=1, orient='horizontal', label='U Min')
u_min_slider.set(-100)
u_min_slider.pack()
 

# Create checkboxes for u_min and u_max
u_min_var = tk.BooleanVar()
u_min_check = tk.Checkbutton(root, text="Use U Min", variable=u_min_var)
u_min_check.pack()

u_max_var = tk.BooleanVar()
u_max_check = tk.Checkbutton(root, text="Use U Max", variable=u_max_var)
u_max_check.pack()

# Create a button to run the controller
def run_controller():

    # close all figures
    plt.close('all')

    # Use slider values in parameters
    parameter = {
        'time_horizon': time_horizon_slider.get(),
        'time_step': time_step_slider.get(),
        'init_state': [init_state_slider1.get(), init_state_slider2.get()],
        'target_state': [target_state_slider1.get(), target_state_slider2.get()],
        'weight_input': weight_input_slider.get(),
        'weight_slack': weight_slack_slider.get(),
        'clf_lambda': clf_lambda_slider.get(), 
        'u_max': u_max_slider.get() if u_max_var.get() else None,
        'u_min': u_min_slider.get() if u_min_var.get() else None,
        'cbf_gamma': None,
        'cbf_gamma0': None,
    } 

    model = MassSpring(m=1, k=0.5, dt=parameter["time_step"], verbose=False)

    clf = Quadratic()

    ctrl = Controller(
        model, 
        parameter, 
        clf=clf.function 
    )

    ctrl.run()

    ctrl.show(
        ctrl.plot_phase_trajectory, 
        ctrl.plot_state,
        ctrl.plot_energy_openloop,
        ctrl.plot_energy_closeloop,
        ctrl.plot_control,
        ctrl.plot_clf,
        subplots=(3, 2) 
    ) 

 
 
run_button = tk.Button(root, text="Run Controller", command=run_controller)
run_button.pack()

# Run the Tk event loop
root.mainloop()