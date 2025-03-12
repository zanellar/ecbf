import os
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from ecbf.scripts.phsys import PHSystemCanonic
from ecbf.utils.paths import PLOTS_PATH

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
    
    ##################################################################################################################################################
    ##################################################################################################################################################
    ##################################################################################################################################################
    
    def visualize(self, traj_angles, skip=1, name="", show=True, save=False, figure=None, color="black"):
        '''
        This function visualizes the evolution of double pendulum system. Darker colors indicate later time steps.   

        Inputs:
        - traj_angles: list of angles (q1,q2) for each time step (list). For example, [ [q1(0),q2(0)], [q1(1),q2(1)], ...]
        - skip: number of steps to skip between each frame (int)
        '''

        if figure is None:
            plt.figure()
        else:
            plt.sca(figure) 

        theta1, theta2 = self.q 
  
        for i in range(0, len(traj_angles), skip):
            theta1, theta2 = traj_angles[i]
            x1 = self.l1 * np.sin(theta1)
            y1 = -self.l1 * np.cos(theta1)
            x2 = x1 + self.l2 * np.sin(theta2)
            y2 = y1 - self.l2 * np.cos(theta2)
            if color == "viridis":
                color = plt.cm.viridis(i / len(traj_angles)) # Use viridis colormap
                alpha = 1 
            else:
                alpha = i / len(traj_angles)  

            
            plt.plot([0, x1, x2], [0, y1, y2], color=color, alpha=alpha, linewidth=8) # Draw the pendulum
            if i == len(traj_angles) - 1:
                plt.plot([0, x1, x2], [0, y1, y2], color=color, alpha=alpha, linewidth=8, label=name) # Draw the pendulum (add the label only for the last time step to be included in the legend)

            plt.plot(x1, y1, 'o', color=color, alpha=alpha) # Draw the first joint
            plt.plot(x2, y2, 'o', color=color, alpha=alpha) # Draw the second joint

        plt.xlim(-self.l1 - self.l2, self.l1 + self.l2)
        plt.ylim(-self.l1 - self.l2, self.l1 + self.l2) 
        plt.grid(True)  
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.xlabel('x')
        # plt.ylabel('y') 
        # plt.draw()
        
        if show:
            plt.show()

        if save:
            file_name = 'cartesian.png' if name == '' else f'cartesian_{name}.png'
            plt.savefig(os.path.join(PLOTS_PATH, file_name), format='png', dpi=300)  