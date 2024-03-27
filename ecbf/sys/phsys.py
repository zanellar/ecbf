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