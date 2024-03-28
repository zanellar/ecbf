import matplotlib.pyplot as plt

from ecbf.defined_models.mass_spring import MassSpring
from ecbf.barrier_functions.safe_doughnut import SafeDoughnut
from ecbf.barrier_functions.safe_circle import SafeCircle
from ecbf.barrier_functions.total_energy_limit import TotalEnergyLimit
from ecbf.scripts.control import Controller
 
# close all figures
plt.close('all')

# Use slider values in parameters
parameter = {
    'time_horizon': 100,
    'time_step': 0.1,
    'init_state': [8, 8],
    'target_state': [0, 0],
    'weight_input': 1,
    'cbf_gamma': 10,
    'weight_slack': None,
    'u_max': None,
    'u_min': None  
} 

model = MassSpring(m=1, k=0.5, dt=parameter["time_step"], verbose=False)

# cbf = SafeDoughnut(C1=20, C2=10) 
# cbf = SafeCircle(r=10, c=[5, 0])  
cbf = TotalEnergyLimit(H_func=model.H, c=5)

ctrl = Controller(
    model, 
    parameter,  
    cbf=cbf
)

ctrl.run()


ctrl.show(
    ctrl.plot_phase_trajectory, 
    ctrl.plot_state,
    ctrl.plot_energy_openloop,
    ctrl.plot_energy_closeloop,
    ctrl.plot_control,
    ctrl.plot_cbf,
    subplots=(3, 2)
) 

# ctrl.plot_phase_trajectory()
# ctrl.plot_state() 
# ctrl.plot_energy_closeloop()
# ctrl.plot_control()
# ctrl.plot_cbf()

ctrl.plot_energy_closeloop(ylims=[0, 100], show=False)
plt.hlines(cbf.c, 0, parameter["time_horizon"], colors='r', linestyles='dashed')
plt.show()