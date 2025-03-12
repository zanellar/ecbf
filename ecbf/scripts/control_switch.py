import os
import numpy as np
import sympy as sp
import casadi as ca
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from sympy.utilities.lambdify import lambdify

from ecbf.utils.paths import PLOTS_PATH, RES_PATH
from ecbf.scripts.control import Controller
 
class ControllerSwitch(Controller):

    def __init__(self, model, parameter, clf=None, cbf=None, regularization=0): 
        '''
        This class implements a controller that uses a CLF and a CBF to control a system. 
        '''
        super().__init__(model, parameter, clf, cbf, regularization)

    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
        
    def run(self, u_ref=0, save= False, name='', switch_on=0, switch_off=None):
        '''
        Run the controller and solve the QP problem at each time step.  

        :param u_ref: reference control input
        :param save: if True, save the results in RES_PATH
        :param name: name of the file to save the results
        :param switch_on: time step when the controller switches on
        :param switch_off: time step when the controller switches off
        '''

        for t in range(self.time_steps):
 
            if t % 100 == 0:
                print(f't = {t}')
 
            # if u_ref is a scalar convert it to an array repeting the value for each control input
            if isinstance(u_ref, (int, float)):
                u_ref = np.array([u_ref]*self.num_inputs, dtype=np.float64)
            else:
                u_ref = np.array(u_ref, dtype=np.float64)

            if t >= switch_on and (switch_off is None or t < switch_off): 
                u, delta, clf, cbf, self.feas = self.solve_qp(self.current_state, u_ref, t)
            else:
                u = u_ref.reshape((self.num_inputs,-1))
                delta = 0
                clf = 0
                cbf = 0
                self.feas = True

                self.cbf_constrant_value = 0

            if not self.feas:
                print('\nThis problem is infeasible!\n') 
                break
            else:
                pass

            self.xt[:, t] = np.copy(self.current_state)
            self.ut[:, t] = u.flatten()
            self.slackt[:, t] = delta
            self.clf_t[:, t] = clf
            self.cbf_t[:, t] = cbf 
            self.cbf_ct[:, t] = self.cbf_constrant_value

            # Compute the energy of the closed-loop system  
            self.closeloop_total_energy_t[:, t] = self.model.get_energy(self.current_state) 
            self.closeloop_potential_energy_t[:, t] = self.model.get_energy(self.current_state, type='V')
            self.closeloop_kinetic_energy_t[:, t] = self.model.get_energy(self.current_state, type='K')

            # Compute the energy of the open-loop system 
            state = self.current_state + self.target_state 
            self.total_energy_t[:, t] = self.model.get_energy(state)  
            self.potential_energy_t[:, t] = self.model.get_energy(state, type='V')
            self.kinetic_energy_t[:, t] = self.model.get_energy(state, type='K')
 
            self.current_state, current_output = self.model.step(self.current_state, u) 

            if t % 100 == 0:
                print(f'x = {self.current_state}, u = {u}')

        print('Finish the solve of qp with clf!')

        # Save all the data in RES_PATH
        if save:
            np.save(os.path.join(RES_PATH, f'{name}_xt.npy'), self.xt)
            np.save(os.path.join(RES_PATH, f'{name}_ut.npy'), self.ut)
            np.save(os.path.join(RES_PATH, f'{name}_slackt.npy'), self.slackt)
            np.save(os.path.join(RES_PATH, f'{name}_clf_t.npy'), self.clf_t)
            np.save(os.path.join(RES_PATH, f'{name}_cbf_t.npy'), self.cbf_t)
            np.save(os.path.join(RES_PATH, f'{name}_total_energy_t.npy'), self.total_energy_t)
            np.save(os.path.join(RES_PATH, f'{name}_closeloop_total_energy_t.npy'), self.closeloop_total_energy_t)
 