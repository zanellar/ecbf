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
        self._D = D
        self._B = B
        self.dt = dt
        self.verbose = verbose

        # State shape 
        self.num_states = 2 if np.array(D).ndim < 1 else np.array(D).shape[0]*2  # Number of states
        self.num_inputs = 1 if np.array(B).ndim < 1 else np.array(B).shape[1]    # Number of inputs

        # Check if the system is scalar
        self.scalar = True if np.array(D).ndim < 1 else False

        # Define the symbolic variables
        if self.scalar:
            q, p = sp.symbols('q p')
        else:
            q = sp.Matrix(sp.symbols(f'q:{self.num_states//2}'))
            p = sp.Matrix(sp.symbols(f'p:{self.num_states//2}'))

        self.q = q
        self.p = p
        x = sp.Matrix([q, p]) 

        # Define the Hamiltonian
        self._H = self.K() + self.V() 
        
        self.H = lambdify([q, p], self._H)
  
        # Define the gradient of the Hamiltonian 
        _dHdq = sp.diff(self._H, q)
        _dHdp = sp.diff(self._H, p)
        self._dH = sp.Matrix([_dHdq, _dHdp]) 
        self.dH = lambdify([q, p], self._dH)

        # Define the state transition matrix F 
        if self.scalar:
            self._F = np.array([[0, 1], [-1, -self._D]])
        else:
            self._F = np.block([[np.zeros((self.num_states//2, self.num_states//2)), np.eye(self.num_states//2)], [-np.eye(self.num_states//2), -np.array(self._D)]])

        # Define the input matrix G 
        if self.scalar:
            self._G = np.array([[0], [self._B]])
        else:
            self._G = np.block([[np.zeros((self.num_states//2, self.num_inputs))], [np.array(self._B)]])

        # Make Kinetic and Potential energy functions callable
        self.K = lambdify([q, p], self.K())
        self.V = lambdify([q, p], self.V())
 
    def get_energy(self, x):
        '''
        This function returns the energy of the system.
        ''' 
        q = x[0] if self.scalar else x[0:self.num_states//2]
        p = x[1] if self.scalar else x[self.num_states//2:] 
        return self.H(q,p)
    
    def dynamics(self, x, u):
        '''
        This function returns the dynamics of the system.
        ''' 
        q = x[0] if self.scalar else x[0:self.num_states//2]
        p = x[1] if self.scalar else x[self.num_states//2:] 
        
        # print("@@@@@@@@")
        # print(q)
        # print(p)
        # print(self.dH(q,p))
        # print(self._F)
        # print(self._G)
        # print(u)

        dx = self._F @ self.dH(q,p) + self._G @ u

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

        k3 = self.dynamics(x + self.dt/2 * k2, u)
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
        q = x_next[0] if self.scalar else x_next[0:self.num_states//2]
        p = x_next[1] if self.scalar else x_next[self.num_states//2:]
        y = self._G.T @ self.dH(q,p)

        if self.verbose:
            print(f'q: {x_next[0]}')
            print(f'p: {x_next[1]}')

        return x_next, y