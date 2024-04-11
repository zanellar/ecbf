
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
matplotlib.rcParams['figure.dpi'] = 200


class Simulator():

    def __init__(self, model):
        '''
        This class simulates the trajectory of a port-Hamiltonian system in canonic form (x=[q,p]). 
        The simulator is initialized with the following parameters:
        - model: instance of the PHSystemCanonic class
        '''
        self.model = model

    def simulate(self, x0, u, n_steps):
        '''
        This function simulates the system for n_steps.
        
        Inputs:
        - x0: initial state (list)
        - u: input vector (list)
        - n_steps: number of steps (int)
        '''
        x = x0
        x_traj = [x]
        y_traj = []
        K_traj = []
        V_traj = []

        for i in range(n_steps):
            x, y = self.model.step(x, u)
            x_traj.append(x)
            y_traj.append(y)
            q = x[0] if self.model.scalar else x[0:self.model.num_states//2]
            p = x[1] if self.model.scalar else x[self.model.num_states//2:]
            K_traj.append(self.model.K(q,p))
            V_traj.append(self.model.V(q,p))
            print(f'step {i+1}/{n_steps}')
        return x_traj, y_traj
    
    def plot_phase_trajectory(self, state_traj, add_to_protrait=False):
        '''
        This function plots the phase trajectory of the system.

        Inputs:
        - state_traj: list of states (list)
        '''
        q_traj = [x[0] for x in state_traj]
        p_traj = [x[1] for x in state_traj]
  
        # Plot the phase trajectory
        plt.plot(q_traj, p_traj, color='black')

        # Add arrows to indicate the direction of motion
        for i in range(0, len(q_traj) - 1, 5):  # Stop one step earlier
            plt.quiver(q_traj[i], p_traj[i], q_traj[i+1]-q_traj[i], p_traj[i+1]-p_traj[i], angles='xy', scale_units='xy', scale=1, color='black')

        # Plot the initial and final states with a higher z-order
        plt.scatter([q_traj[0]], [p_traj[0]], color='green', label='Initial state', zorder=5)
        plt.scatter([q_traj[-1]], [p_traj[-1]], color='red', label='Final state', zorder=5)

        # Add text close to the initial and final states with an offset
        plt.text(q_traj[0], p_traj[0], '$x(0)$', verticalalignment='bottom', horizontalalignment='right')
        plt.text(q_traj[-1], p_traj[-1], '$x(T)$', verticalalignment='bottom', horizontalalignment='right')
        
        if not add_to_protrait: 
            plt.grid(True)
            plt.xlabel('q')
            plt.ylabel('p')
            plt.title('Phase space trajectory')
            plt.show()

    def plot_phase_portrait(self, state_traj, qlim, plim):
        # Create a grid of points
        q = np.linspace(qlim[0], qlim[1], 20)
        p = np.linspace(plim[0], plim[1], 20)
        q_grid, p_grid = np.meshgrid(q, p)

        # Calculate the system's dynamics at each point on the grid
        U, V = np.zeros(q_grid.shape), np.zeros(p_grid.shape)
        NI, NJ = q_grid.shape
        for i in range(NI):
            for j in range(NJ):
                q = q_grid[i, j]
                p = p_grid[i, j]
                state = np.array([q, p])  # Create a state vector
                x, y = self.model.step(state, np.zeros((self.model.num_inputs,)))  # Pass the entire state vector
                U[i,j] = x[0] - q
                V[i,j] = x[1] - p

        # Normalize the arrows so their size represents their speed
        N = np.sqrt(U**2+V**2)
        U = U/N
        V = V/N

        # Plot the vector field
        plt.quiver(q_grid, p_grid, U, V, color='blue', alpha=0.5)

        # Plot the phase trajectories 
        self.plot_phase_trajectory(state_traj, add_to_protrait=True)


        plt.grid(True)
        plt.xlim(qlim)
        plt.ylim(plim)
        plt.xlabel('q')
        plt.ylabel('p')
        plt.show()

    def animate_phase_trajectory(self, state_traj):
        fig, ax = plt.subplots()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('q')
        ax.set_ylabel('p')
        ax.set_title('Phase space trajectory')

        # Create the line object
        line, = ax.plot([], [], color='black')

        # Create the initial state point
        point, = ax.plot([], [], 'go')

        # Create the final state point
        point_final, = ax.plot([], [], 'ro')

        def init():
            line.set_data([], [])
            point.set_data([], [])
            point_final.set_data([], [])
            return line, point, point_final

        def animate(i):
            q = state_traj[i][0]
            p = state_traj[i][1]
            line.set_data([x[0] for x in state_traj[:i+1]], [x[1] for x in state_traj[:i+1]])
            point.set_data(q, p)
            point_final.set_data(state_traj[-1][0], state_traj[-1][1])
            return line, point, point_final

        ani = animation.FuncAnimation(fig, animate, frames=len(state_traj), init_func=init, blit=True)
        plt.show()

    def plot_energy(self, state_traj):
        # Calculate the energy of each state
        energy_traj = []
        for state in state_traj:
            q = state[0] if self.model.scalar else state[0:self.model.num_states//2]
            p = state[1] if self.model.scalar else state[self.model.num_states//2:]
            state = np.array([q, p]) 
            energy = self.model.K(q,p) + self.model.V(q,p)
            energy_traj.append(energy) 

        # Plot the energy trajectory
        plt.plot(energy_traj, color='black')
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.xlim([0, len(state_traj)])
        plt.ylim([round(min(energy_traj),3), round(max(energy_traj),3)])
        plt.title('Energy trajectory')
        plt.show()