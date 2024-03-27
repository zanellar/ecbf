import numpy as np
import sympy as sp 
from sympy.utilities.lambdify import lambdify
  
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
        self.num_states = 2 if np.array(D).ndim < 1 else np.array(D).shape[0]*2  # Number of states
        self.num_inputs = 1 if np.array(B).ndim < 1 else np.array(B).shape[1]    # Number of inputs

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
        self._dH = sp.Matrix([dHdq, dHdp]) 
        self.dH = lambdify([q, p], self._dH)

        # Define the state transition matrix F 
        self.F = np.array([[0, 1], [-1, -self.D]])

        # Define the input matrix G 
        self.G = np.array([[0], [self.B]])
    
    def dynamics(self, x, u):
        '''
        This function returns the dynamics of the system.
        ''' 

        dx = self.F @ self.dH(x[0], x[1]) + self.G @ u

        return dx    

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

        k1 = self.dynamics(x, u)
        k1 = k1.reshape(state_shape) 

        k2 = self.dynamics(x + self.dt/2 * k1, u)
        k2 = k2.reshape(state_shape)

        k3= self.dynamics(x + self.dt/2 * k2, u)
        k3 = k3.reshape(state_shape)

        k4 = self.dynamics(x + self.dt * k3, u)
        k4 = k4.reshape(state_shape)

        x_next = x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4) 

        return x_next
    
    def step(self, x, u):
        '''
        This function performs one step of the system.
        '''

        u = np.array(u).reshape(self.num_inputs, 1)

        # Integrate the system
        x_next = self._integrate_rk4(x, u) 
        y = self.G.T @ self.dH(x_next[0], x_next[1])

        if self.verbose:
            print(f'q: {x_next[0]}')
            print(f'p: {x_next[1]}')

        return x_next, y