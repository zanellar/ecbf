import os
import numpy as np
import sympy as sp
import casadi as ca
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200
#matplotlib.use('TkAgg')  # Do this BEFORE importing matplotlib.pyplotimport matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from sympy.utilities.lambdify import lambdify

from ecbf.utils.paths import PLOTS_PATH
from ecbf.models.mass_spring import MassSpring

class Controller():

    def __init__(self, model, parameter, symbolic_clf=None, symbolic_cbf=None): 

        self.parameter = parameter 

 
        self.T = self.parameter['time_horizon']
        self.dt = self.parameter['time_step']
        self.time_steps = int(np.ceil(self.T / self.dt))
        self.weight_input = self.parameter['weight_input']
        self.weight_slack = self.parameter['weight_slack']
        self.clf_lambda = self.parameter['clf_lambda']
        self.cbf_gamma = self.parameter['cbf_gamma']
        self.u_max = self.parameter['u_max']
        self.u_min = self.parameter['u_min']
        self.init_state = self.parameter['init_state']
        self.target_state = self.parameter['target_state'] 
        self.gamma0 = self.parameter['cbf_gamma0']
        self.current_state = self.init_state 

        ########### Symbolic State ###########
  
        _x, _y = sp.symbols('x y')  # define symbolic representation
        self._state = sp.Matrix([_x, _y])  # row vector

        _ex, _ey = sp.symbols('ex, ey,')
        self._target_state = sp.Matrix([_ex, _ey])  # symbols
 
        ########### Dynamics ###########

        self.model = model

        self.state_dim = self.model.num_states
        self.control_dim = self.model.num_inputs
        
        ########### CLF ###########

        self._clf = symbolic_clf

        if self._clf is not None: 
            
            # Derivative of CLF w.r.t the state x
            dx_clf = sp.Matrix([self._clf]).jacobian(self._state)
            dx_H = sp.Matrix([self.model.H]).jacobian(self._state)
 
            # Lie derivatives of CLF  w.r.t f(x)=F*dHdx(x)
            self._dLie_f_clf = dx_clf @ self.model.F @ dx_H

            # Lie derivatives of CLF  w.r.t g(x)=G
            self._dLie_g_clf = dx_clf @ self.model.G 

            # Make the symbolic functions callable
            self.clf = lambdify([self._state, self._target_state], self._clf)
            self.dLie_f_clf = lambdify([self._state, self._target_state], self._dLie_f_clf)
            self.dLie_g_clf = lambdify([self._state, self._target_state], self._dLie_g_clf)

        ########### CBF ###########
 
        self._cbf = symbolic_cbf

        if self._cbf is not None:
 
            # Derivative of CBF w.r.t the state x
            dx_cbf = sp.Matrix([self._cbf]).jacobian(self._state)
            dx_H = sp.Matrix([self.model.H]).jacobian(self._state)
 
            # Lie derivatives of CBF  w.r.t f(x)=F*dHdx(x)
            self._dLie_f_cbf = dx_cbf @ self.model.F @ dx_H

            # Lie derivatives of CBF  w.r.t g(x)=G
            self._dLie_g_cbf = dx_cbf @ self.model.G 

            # Make the symbolic functions callable
            self.cbf = lambdify([self._state], self._cbf)
            self.dLie_f_cbf = lambdify([self._state], self._dLie_f_cbf)
            self.dLie_g_cbf = lambdify([self._state], self._dLie_g_cbf)

   
        ########### Solver ###########
        opts_setting = {
            'ipopt.max_iter': 200,
            'ipopt.print_level': 1,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }

        self.opti = ca.Opti()
        self.opti.solver('ipopt', opts_setting)
        self.u = self.opti.variable(self.control_dim)
        self.slack = self.opti.variable()
 
        ########### Data ###########
        self.xt = np.zeros((self.state_dim, self.time_steps))
        self.ut = np.zeros((self.control_dim, self.time_steps))
        self.slackt = np.zeros((1, self.time_steps))
        self.clf_t = np.zeros((1, self.time_steps))
        self.cbf_t = np.zeros((1, self.time_steps))
        self.energy_t = np.zeros((1, self.time_steps))
 

    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
        
    def solve_qp(self, current_state, u_ref):
        '''
        Solve the QP problem (using Casadi) 
        opt_u = argmin[ u'Hu + (u-u_ref)'H(u-u_ref) + slack^2]
                    subject to:
                    Lie_f_V + Lie_g_V * u + lambda * V <= slack
                    Lie_f_h + Lie_g_h * u + gamma * h >= 0

        where V is the CLF, h is the CBF, u_ref is the reference control input, H is the weight matrix, lambda and gamma are the design parameters
 
        :param current_state: current state of the system
        :param u_ref: reference control input
        :return: optimal control input, slack, clf, cbf, feasible 
        '''

        # empty the constraint set
        self.opti.subject_to()

        # objective function
        self.H = self.weight_input * np.eye(self.control_dim)
        self.obj = .5 * (self.u - u_ref).T @ self.H @ (self.u - u_ref)
        self.obj = self.obj + self.weight_slack * self.slack ** 2
        self.opti.minimize(self.obj)

        # constraints
        self.opti.subject_to(self.opti.bounded(self.u_min, self.u, self.u_max))
        self.opti.subject_to(self.opti.bounded(-np.inf, self.slack, np.inf))
 
        # CLF constraint
        clf = None
        if self._clf is not None:
            clf = self.clf(current_state, self.target_state)
            dLie_f_clf = self.dLie_f_clf(current_state, self.target_state)
            dLie_g_clf = self.dLie_g_clf(current_state, self.target_state)
 
            # LfV + LgV * u + lambda * V <= slack
            self.opti.subject_to(dLie_f_clf + (dLie_g_clf @ self.u)[0][0] + self.clf_lambda * clf - self.slack <= 0)

        # CBF constraint
        cbf = None
        if self._cbf is not None:
            cbf = self.cbf(current_state)
            dLie_f_cbf = self.dLie_f_cbf(current_state)
            dLie_g_cbf = self.dLie_g_cbf(current_state)

            # Lfh + Lgh * u + gamma * h >= 0
            self.opti.subject_to(dLie_f_cbf + (dLie_g_cbf @ self.u)[0][0] + self.cbf_gamma * cbf >= 0)

        # optimize the Qp problem
        try:
            sol = self.opti.solve()
            self.feasible = True
            optimal_control = sol.value(self.u)
            slack = sol.value(self.slack)

            return optimal_control, slack, clf, cbf, self.feasible
        except:

            print(self.opti.return_status())
            self.feasible = False

            return None, None, clf, cbf, self.feasible


    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
        
    def run(self):

        for t in range(self.time_steps):
            if t % 100 == 0:
                print(f't = {t}')

            u_ref = np.array([0])
            u, delta, clf, cbf, feas = self.solve_qp(self.current_state, u_ref)

            if not feas:
                print('This problem is infeasible!')
                break
            else:
                pass

            self.xt[:, t] = np.copy(self.current_state)
            self.ut[:, t] = u
            self.slackt[:, t] = delta
            self.clf_t[:, t] = clf
            self.cbf_t[:, t] = cbf 
            self.model.q = self.current_state[0]
            self.model.p = self.current_state[1]
            H = self.model.K() + self.model.V() 
            self.energy_t[:, t] = H

            self.current_state, current_output = self.model.step(self.current_state, u)

        print('Finish the solve of qp with clf!')


    #######################################################################################################################
    #######################################################################################################################
    ####################################################################################################################### 

    def plot_state(self):
        t = np.arange(0, self.T, self.dt)
        q = self.xt[0]
        p = self.xt[1]

        plt.grid() 
        plt.plot(t, q, 'k', linewidth=3, label='q', linestyle='-')
        plt.plot(t, p, 'b', linewidth=3, label='p', linestyle='-')
        plt.legend()
        plt.title('State Variable')
        plt.ylabel('q, p')

        plt.show()
        # plt.savefig('slack.png', format='png', dpi=300) 

    #######################################################################################################################

    def plot_energy(self):
        t = np.arange(0, self.T, self.dt) 
        energy = self.energy_t.flatten()

        plt.grid() 
        plt.plot(t, energy, linewidth=3, color='b')
        plt.xlabel('Time')
        plt.ylabel('Energy') 
        plt.ylim([round(float(min(energy)),3), round(float(max(energy)),3)])
        plt.title('Energy trajectory')
        plt.show()

        plt.show()
        # plt.savefig('clf.png', format='png', dpi=300)

    #######################################################################################################################

    def plot_phase_trajectory(self, add_to_protrait=False):
        '''
        This function plots the phase trajectory of the system.

        Inputs:
        - state_traj: list of states (list)
        ''' 
        q_traj = [q for q in self.xt[0]]
        p_traj = [p for p in self.xt[1]]
  
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

    #######################################################################################################################

    def animate_phase_trajectory(self):
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

        state_traj = self.xt.T

        def animate(i): 
  
            q = state_traj[i][0]
            p = state_traj[i][1]
            line.set_data([x[0] for x in state_traj[:i+1]], [x[1] for x in state_traj[:i+1]])
            point.set_data(q, p)
            point_final.set_data(state_traj[-1][0], state_traj[-1][1])
            return line, point, point_final

        ani = animation.FuncAnimation(fig, animate, frames=len(state_traj), init_func=init, blit=True)
        plt.show()

    #######################################################################################################################

    def plot_slack(self):
        t = np.arange(0, self.T, self.dt)
        slack = self.slackt.flatten()

        plt.grid() 
        plt.plot(t, slack, linewidth=3, color='orange')
        plt.title('Slack')
        plt.ylabel('delta')

        plt.show()
        # plt.savefig('slack.png', format='png', dpi=300)
        

    #######################################################################################################################

    def plot_clf(self):
        t = np.arange(0, self.T, self.dt)
        clf = self.clf_t.flatten()

        plt.grid()        
        plt.plot(t, clf, linewidth=3, color='cyan')
        plt.title('clf')
        plt.ylabel('V(x)')

        plt.show()
        # plt.savefig('clf.png', format='png', dpi=300)
        

    #######################################################################################################################

    def plot_cbf(self):
        t = np.arange(0, self.T, self.dt)
        cbf = self.cbf_t.flatten()

        plt.grid()
        plt.plot(t, cbf, linewidth=3, color='cyan')
        plt.title('cbf')
        plt.ylabel('h(x)')

        plt.show()
        # plt.savefig('clf.png', format='png', dpi=300)
        

    #######################################################################################################################

    def plot_control(self):
        u_max = self.parameter['u_max']
        t = np.arange(0, self.T, self.dt)
        control = self.ut.flatten()

        plt.grid() 
        plt.plot(t, control, 'b', linewidth=3, label='w', linestyle='-')
        plt.plot(t, u_max * np.ones(t.shape[0]), 'k', linewidth=3, label='Bound', linestyle='--')
        plt.plot(t, -u_max * np.ones(t.shape[0]), 'k', linewidth=3, linestyle='--') 
        plt.title('State Variable')
        plt.ylabel('q, p') 
        plt.title('control')
        plt.ylabel('u')
        plt.legend(loc='upper left')

        plt.show()
        # plt.savefig('control.png', format='png', dpi=300)
        

##############################################################################################################
##############################################################################################################
##############################################################################################################
        
if __name__ == "__main__":
    parameter = {
        'time_horizon': 15,
        'time_step': 0.1,
        'init_state': [1, 0],
        'target_state': [0, 0],
        'weight_input': 0.5,
        'weight_slack': 10,
        'clf_lambda': 1.0, 
        'u_max': 5,
        'u_min': -5, 
        'cbf_gamma': 3,
        'cbf_gamma0': 1,
    } 

    model = MassSpring(m=1, k=0.5, dt=parameter["time_step"], verbose=False)

    symbolic_cbf = None
    symbolic_clf = None

    ctrl = Controller(model, parameter, symbolic_clf=symbolic_clf, symbolic_cbf=symbolic_cbf)

    ctrl.run()
 
    ctrl.plot_control()
    ctrl.plot_state() 
    # ctrl.plot_energy()
    # ctrl.plot_phase_trajectory()
    # ctrl.animate_phase_trajectory()
    
    # ctrl.plot_clf()
    ctrl.plot_cbf()
