
from ecbf.scripts.sim import Simulator
from ecbf.defined_models.double_pendulum import DoublePendulum

# Define the system parameters
m1 = 1
m2 = 1
l1 = 1
l2 = 1
dt = 0.01

# Create the mass-spring system
model = DoublePendulum(m1=m1, m2=m2, l1=l1, l2=l2, dt=dt, verbose=True)

# Define the initial state
x0 = [3.14,3.14,0,0]

# Define the input
u = [[0],[0]]

# Create the simulator
simulator = Simulator(model)

# Simulate the system
n_steps = 1500
x_traj, y_traj = simulator.simulate(x0, u, n_steps)
print("simulation done")

# # Plot the phase trajectory
# simulator.plot_phase_trajectory(x_traj)
# print("phase plto done")

# # Plot the phase portrait
# xlim = [-2, 2]
# ylim = [-2, 2]
# simulator.plot_phase_portrait(x_traj, xlim, ylim) 
# print("phase plto done")

# # Animate the phase trajectory
# simulator.animate_phase_trajectory(x_traj)
# print("animation done")

# Plot the energy trajectory
simulator.plot_energy(x_traj, ylims=[0, 100])
print("energy plot done")

