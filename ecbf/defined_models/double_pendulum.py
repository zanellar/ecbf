
from ecbf.scripts.phsys import PHSystemCanonic
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

class DoublePendulum(PHSystemCanonic):

    GRAVITY = 9.81

    def __init__(self, m1, m2, l1, l2, dt, b1=0, b2=0, verbose=False):
        '''
        This class implements a double pendulum system using SymPy. 
        The system is initialized with the following parameters:
        - m1: mass of the first pendulum (float)
        - m2: mass of the second pendulum (float)
        - l1: length of the first pendulum (float)
        - l2: length of the second pendulum (float)
        - dt: sampling time for the integration (float)
        - b1: dumping coefficient first pendulum (float) 
        - b2: dumping coefficient second pendulum (float) 
        - verbose: if true, prints "q" and "p" every step
        '''
        self.m1 = m1
        self.m2 = m2

        self.l1 = l1
        self.l2 = l2

        self.b1 = b1
        self.b2 = b2
 
        super().__init__(
            D=[[self.b1,0],
               [0,self.b2]], 
            B=[[1,0],
               [0,1]], 
            K=self.K, 
            V=self.V, 
            dt=dt, 
            verbose=verbose
        ) 
 

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
       
    def visualize(self, traj_angles, skip=1, color=False):
        '''
        This function visualizes the evolution of double pendulum system. Darker colors indicate later time steps.   

        Inputs:
        - traj_angles: list of angles (q1,q2) for each time step (list). For example, [ [q1(0),q2(0)], [q1(1),q2(1)], ...]
        - skip: number of steps to skip between each frame (int)
        '''

        self.fig, self.ax = plt.subplots() 
        self.ax.clear()

        theta1, theta2 = self.q 
  
        for i in range(0, len(traj_angles), skip):
            theta1, theta2 = traj_angles[i]
            x1 = self.l1 * np.sin(theta1)
            y1 = -self.l1 * np.cos(theta1)
            x2 = x1 + self.l2 * np.sin(theta2)
            y2 = y1 - self.l2 * np.cos(theta2)
            if color:
                colormap = plt.cm.viridis(i / len(traj_angles)) # Use viridis colormap
                alpha = 1
            else:
                colormap = plt.cm.Greys(i / len(traj_angles))  # Use grayscale colormap
                alpha = i / len(traj_angles) 
            self.ax.plot([0, x1, x2], [0, y1, y2], color=colormap, alpha=alpha) # Draw the pendulum
            self.ax.plot(x1, y1, 'o', color=colormap, alpha=alpha) # Draw the first joint
            self.ax.plot(x2, y2, 'o', color=colormap, alpha=alpha) # Draw the second joint

        self.ax.set_xlim(-self.l1 - self.l2, self.l1 + self.l2)
        self.ax.set_ylim(-self.l1 - self.l2, self.l1 + self.l2)
        self.ax.set_aspect('equal')
        self.ax.grid(True) 
        plt.draw()
        plt.pause(0.01)