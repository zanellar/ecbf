
from ecbf.scripts.phsys import PHSystemCanonic
import sympy as sp

class DoublePendulum(PHSystemCanonic):

    GRAVITY = 9.81

    def __init__(self, m1, m2, l1, l2, dt, verbose=False):
        '''
        This class implements a double pendulum system using SymPy. 
        The system is initialized with the following parameters:
        - m1: mass of the first pendulum (float)
        - m2: mass of the second pendulum (float)
        - l1: length of the first pendulum (float)
        - l2: length of the second pendulum (float)
        - dt: sampling time for the integration (float)
        - verbose: if true, prints "q" and "p" every step
        '''
        self.m1 = m1
        self.m2 = m2

        self.l1 = l1
        self.l2 = l2
 
        super().__init__(D=[[0,0],[0,0]], B=[[1,0],[0,1]], K=self.K, V=self.V, dt=dt, verbose=verbose)
        # super().__init__(D=[[0,0],[0,0]], B=[[1],[1]], K=self.K, V=self.V, dt=dt, verbose=verbose)

    def K(self): 
        # Kinetic energy 
        K1 = 0.5 * self.l1**2 * self.p[0]**2 / self.m1  
        K2 = 0.5 * self.m2 * (self.l1**2 * self.p[0]**2 / self.m1**2  + self.l2**2 * self.p[1]**2 / self.m2**2 + 2 * self.l1 * self.l2 * self.p[0] * self.p[1] * sp.cos(self.q[0] - self.q[1]) / (self.m1 * self.m2))
        return K1 + K2
    
    def V(self):
        # Potential energy
        V1 = - self.m1 * self.GRAVITY * self.l1 * sp.cos(self.q[0]) 
        V2 = - self.m2 * self.GRAVITY * (self.l1 * sp.cos(self.q[0]) + self.l2 * sp.cos(self.q[1])) 
        return V1 + V2
     
 