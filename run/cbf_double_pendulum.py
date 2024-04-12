import matplotlib.pyplot as plt

from ecbf.defined_models.double_pendulum import DoublePendulum
from ecbf.barrier_functions.safe_doughnut import SafeDoughnut
from ecbf.barrier_functions.safe_circle import SafeCircle
from ecbf.barrier_functions.energy_limit import EnergyLimit
from ecbf.scripts.control import Controller
 
# close all figures
plt.close('all')

# Use slider values in parameters
parameter = {
    'time_horizon': 10,
    'time_step': 0.01,
    'init_state': [1.57,1.57,1.57,1.57],
    'target_state': None,
    'weight_input': 1,
    'cbf_gamma': 1,
    'weight_slack': None,
    'u_max': None,
    'u_min': None  
} 

model = DoublePendulum(m1=1, m2=1, l1=1, l2=1, dt=parameter["time_step"], verbose=False)

# cbf = SafeDoughnut(C1=20, C2=10) 
# cbf = SafeCircle(r=10, c=[5, 0])  
cbf = EnergyLimit(energy_func=model.H, c=1, num_states=model.num_states, pump=False)

ctrl = Controller(
    model, 
    parameter,  
    cbf=cbf,
    regularization=2
)

ctrl.run()


ctrl.show(
    # ctrl.plot_phase_trajectory, 
    ctrl.plot_state,
    ctrl.plot_total_energy ,
    ctrl.plot_cbf_constraint,
    ctrl.plot_control,
    ctrl.plot_cbf,
    subplots=(3, 2)
) 
 
# ctrl.plot_state() 
# ctrl.plot_total_energy_closeloop()
# ctrl.plot_control()
# ctrl.plot_cbf()

# ctrl.plot_total_energy_closeloop( show=False)
# plt.hlines(cbf.c, 0, parameter["time_horizon"], colors='r', linestyles='dashed')
# plt.show()