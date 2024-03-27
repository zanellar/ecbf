import numpy as np
import sympy as sp
import casadi as ca
import matplotlib 
#matplotlib.use('TkAgg')  # Do this BEFORE importing matplotlib.pyplotimport matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from sympy.utilities.lambdify import lambdify

matplotlib.rcParams['figure.dpi'] = 200
 
class PHSystemCanonic():
    def __init__(self, D, B, K, V, dt, verbose=False): 
        '''
        This class implements a generic port-Hamiltonian system in canonic form (x=[q,p]) using SymPy.

        The system is defined by the following equations:
            dx =  F * dH(x) + G * u
            y = G' * dH(x)

        where:
        - x = [q,p] is the state vector
        - u is the input vector
        - y is the output vector
        - F = [0, I; -I, -D] is the state transition matrix
        - G = [0; B] is the input matrix
        - dH = [dHdq, dHdp] is gradient of the Hamiltonian H(x)=V(x)+K(x)
            with V the potential energy and K the kinetic energy

        The system is integrated using the Runge-Kutta 4th order method.

        The system is initialized with the following parameters:
        - D: matrix n x n of the state transition matrix F (list)
        - B: matrix n x m of the input matrix G (list)
        - K: function that returns the kinetic energy K(x) (function with symbolic expression)
        - V: function that returns the potential energy V(x) (function with symbolic expression)
        - dt: sampling time for the integration (float)
        - verbose: if true, prints "q" and "p" every step
        '''
        self.K = K
        self.V = V
        self.D = D
        self.B = B
        self.dt = dt
        self.verbose = verbose

        # State shape 
        self.num_states = 1 if np.array(D).ndim < 1 else np.array(D).shape[0]  # Number of states
        self.num_inputs = 1 if np.array(B).ndim < 1 else np.array(B).shape[1]  # Number of inputs

        # Define the symbolic variables
        q, p = sp.symbols('q p')
        self.q = q
        self.p = p
        x = sp.Matrix([q, p]) 

        # Define the Hamiltonian
        self.H = self.K() + self.V()

        # Define the gradient of the Hamiltonian 
        dHdq = sp.diff(self.H, q)
        dHdp = sp.diff(self.H, p)
        dH = sp.Matrix([dHdq, dHdp]) 

        # Define the state transition matrix F
        F = sp.Matrix([[0, 1], [-1, -self.D]])
        self.F = F

        # Define the input matrix G
        G = sp.Matrix([[0], [self.B]])
        self.G = G

        # Define the system equations
        dx = F @ dH
        y = G.T @ dH

        # Define the system as a function
        self._system = lambdify([x], dx)
        self._output = lambdify([x], y)

    
    def reset(self):
        '''
        This function resets the system.
        '''
        raise NotImplementedError
    
    def _integrate_rk4(self, x, u):
        '''
        This function integrates the system using the Runge-Kutta 4th order method.
        '''
        state_shape = np.array(x).shape  

        k1 = self._system(x).reshape(state_shape)
        k2 = self._system(x + self.dt/2 * k1).reshape(state_shape) 
        k3 = self._system(x + self.dt/2 * k2).reshape(state_shape) 
        k4 = self._system(x + self.dt * k3).reshape(state_shape) 
        x_next = x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4) 
        return x_next
    
    def step(self, x, u):
        '''
        This function performs one step of the system.
        '''
        x_next = self._integrate_rk4(x, u)
        y = self._output(x)
        if self.verbose:
            print(f'q: {x_next[0]}')
            print(f'p: {x_next[1]}')
        return x_next, y
    
############################################################################################
############################################################################################
############################################################################################
    
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
            K_traj.append(self.model.K())
            V_traj.append(self.model.V())
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
            q = state[0]
            p = state[1]
            state = np.array([q, p])
            self.model.q = q
            self.model.p = p
            energy = self.model.K() + self.model.V()
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

############################################################################################
############################################################################################
############################################################################################
        
class MassSpring(PHSystemCanonic):
    def __init__(self, m, k, dt, verbose=False):
        '''
        This class implements a mass-spring system using SymPy. 
        The system is initialized with the following parameters:
        - m: mass of the system (float)
        - k: spring constant (float)
        - c: damping coefficient (float)
        - dt: sampling time for the integration (float)
        - verbose: if true, prints "q" and "p" every step
        '''
        self.m = m
        self.k = k 
 
        super().__init__(D=0, B=1, K=self.K, V=self.V, dt=dt, verbose=verbose)

    def K(self): 
        # Kinetic energy
        K = self.m/2 * self.p**2
        return K
    
    def V(self):
        # Potential energy
        V = self.k/2 * self.q**2
        return V
     

if __name__ == '__main__':

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

