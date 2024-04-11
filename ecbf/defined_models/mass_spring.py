
from ecbf.scripts.phsys import PHSystemCanonic

class MassSpring(PHSystemCanonic):
    def __init__(self, m, k, dt, verbose=False):
        '''
        This class implements a mass-spring system using SymPy. 
        The system is initialized with the following parameters:
        - m: mass of the system (float)
        - k: spring constant (float) 
        - dt: sampling time for the integration (float)
        - verbose: if true, prints "q" and "p" every step
        '''
        self.m = m
        self.k = k 
 
        super().__init__(D=0, B=1, K=self.K, V=self.V, dt=dt, verbose=verbose)

    def K(self): 
        # Kinetic energy
        K =1/(2 * self.m) * self.p**2
        return K
    
    def V(self):
        # Potential energy
        V = self.k/2 * self.q**2
        return V
     
 