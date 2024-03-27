
from ecbf.sys.sim import Simulator
from ecbf.models.mass_spring import MassSpring

# Define the system parameters
m = 1
k = 0.5
dt = 0.1

# Create the mass-spring system
mass_spring = MassSpring(m, k, dt, verbose=True)

# Define the initial state
x0 = [1, 0]

# Define the input
u = 0

# Create the simulator
simulator = Simulator(mass_spring)

# Simulate the system
n_steps = 150
x_traj, y_traj = simulator.simulate(x0, u, n_steps)
print("simulation done")

# Plot the phase trajectory
simulator.plot_phase_trajectory(x_traj)
print("phase plto done")

# Plot the phase portrait
xlim = [-2, 2]
ylim = [-2, 2]
simulator.plot_phase_portrait(x_traj, xlim, ylim) 
print("phase plto done")

# Animate the phase trajectory
simulator.animate_phase_trajectory(x_traj)
print("animation done")

# Plot the energy trajectory
simulator.plot_energy(x_traj)
print("energy plot done")

