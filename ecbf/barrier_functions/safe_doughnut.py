import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sympy as sp
from sympy.abc import x, y
from ecbf.utils.paths import PLOTS_PATH
from sympy.utilities.lambdify import lambdify

class SafeDoughnut:
    def __init__(self, C1=20, C2=10): 
        '''
        This class represents provides a function that is greater than zero inside a doughnut shape with inner radius C2 and outer radius C1.
        '''
        self.C1 = C1
        self.C2 = C2 

    def function(self, state):
        x = state[0]
        y = state[1]
        _h = -(x**2 + y**2 + 0.1*y - self.C2**2) * (x**2 + y**2 + 0.1*y - self.C1**2)  
        return _h 
     
    def plot(self, op=1):
        x_vals = np.linspace(-self.C1*1.05, self.C1*1.05, 500)
        y_vals = np.linspace(-self.C1*1.05, self.C1*1.05, 500)
        x_vals, y_vals = np.meshgrid(x_vals, y_vals)

        z_vals = self.function([x_vals, y_vals])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if op==1: 
            ax.plot_surface(x_vals, y_vals, z_vals, cmap='viridis', alpha=0.8)

        if op==2:
            # Create two separate arrays for the regions where the function is over and under the plane
            z_vals_over = np.where(z_vals > 0, z_vals, np.nan)
            z_vals_under = np.where(z_vals <= 0, z_vals, np.nan)

            # Plot the regions separately
            ax.plot_surface(x_vals, y_vals, z_vals_over, color='green', alpha=0.8)
            ax.plot_surface(x_vals, y_vals, z_vals_under, color='red', alpha=0.8)

        # Plot the plane at z=0
        z_plane = np.zeros((500, 500))
        ax.plot_surface(x_vals, y_vals, z_plane, color='black', alpha=0.3, rstride=50, cstride=50)

        # Highlight the intersection
        ax.contour(x_vals, y_vals, z_vals, [0], colors='black')

        plt.savefig(os.path.join(PLOTS_PATH, 'barrier_function.png'))  # Save the figure before showing it 

        # Plot the 2D contour
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        contour = ax2.contour(x_vals, y_vals, z_vals, levels=[0], colors='black')
        ax2.set_title('2D Contour Plot')

        # Fill the areas where the function is over the plane in green and under the plane in red
        ax2.contourf(x_vals, y_vals, z_vals, levels=[-np.inf, 0], colors='grey', alpha=0.3)

        plt.savefig(os.path.join(PLOTS_PATH, 'barrier_function_contour.png'))  # Save the figure before showing it 

        plt.show()


if __name__ == '__main__':
    plotter = SafeDoughnut()
    plotter.plot(op=1)
    plotter.plot(op=2) 