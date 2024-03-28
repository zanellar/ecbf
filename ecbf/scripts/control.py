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

class Controller():

    def __init__(self, model, parameter, clf=None, cbf=None): 
        '''
        This class implements a controller that uses a CLF and a CBF to control a system.
        
        opt_u, opt_delta = argmin[ 0.5*u'*W*u + b*delta^2]
                            
                            subject to:
                                Lie_f_V + Lie_g_V * u  <= - lambda * V + delta
                                Lie_f_h + Lie_g_h * u  >= - gamma * h
                                u_min <= u <= u_max 
                                delta >= 0
 
        where: 
        - V is the CLF
        - h is the CBF
        - u_ref is the reference control input
        - W is the weight matrix
        - gamma is CBF modulating factor (higher gamma means that the CBF activates only getting closer to the boundary of the safe set)
        - lambda is CLF modulating factor 
        - delta is the relaxation/slack variable that ensures the feasibility of the QP (to guarantee safety one must set delta > 0)
        - b is a value that penalizes violations of the CLF constraint
 

        :param model: the model of the system
        :param parameter: a dictionary containing the parameters of the controller, containing:
            - time_horizon: the time horizon of the simulation
            - time_step: the time step of the simulation 
            - weight_input: the weight matrix for the control input (W in the QP)
            - weight_slack: the weight for the slack variable (b in the QP)
            - clf_lambda: the modulating factor for the CLF (lambda in the QP)
            - cbf_gamma: the modulating factor for the CBF (gamma in the QP)
            - u_max: the maximum control input
            - u_min: the minimum control input
            - init_state: the initial state of the system
            - target_state: the target state of the system                        
        :param clf: the CLF function
        :param cbf: the CBF function
        '''

        self.parameter = parameter 

        self.T = self.parameter['time_horizon'] if 'time_horizon' in self.parameter else 10
        self.dt = self.parameter['time_step'] if 'time_step' in self.parameter else 0.1
        self.weight_input = self.parameter['weight_input'] if 'weight_input' in self.parameter else 1
        self.weight_slack = self.parameter['weight_slack'] if 'weight_slack' in self.parameter else None
        self.clf_lambda = self.parameter['clf_lambda'] if 'clf_lambda' in self.parameter else 0.1
        self.cbf_gamma = self.parameter['cbf_gamma'] if 'cbf_gamma' in self.parameter else 0.1
        self.u_max = self.parameter['u_max'] if 'u_max' in self.parameter else None
        self.u_min = self.parameter['u_min'] if 'u_min' in self.parameter else None
        self.target_state = self.parameter['target_state'] if 'target_state' in self.parameter else None
        self.init_state = self.parameter['init_state'] if 'init_state' in self.parameter else [0, 0]

        # ------------------------------------

        self.time_steps = int(np.ceil(self.T / self.dt)) 
        self.target_state = [0, 0] if self.target_state is None else np.array(self.target_state)
        self.init_state = np.array(self.init_state) - self.target_state # convert to error state
        self.current_state = self.init_state

        ########### Symbolic State ###########
  
        _x, _y = sp.symbols('x y')  # define symbolic representation
        self._state = sp.Matrix([_x, _y])  # row vector

        _xd, _yd = sp.symbols('xd yd')
        self._target_state = sp.Matrix([_xd, _yd])  # row vector
 
        ########### Dynamics ###########

        self.model = model

        self.state_dim = self.model.num_states
        self.control_dim = self.model.num_inputs
        
        ########### CLF ###########

        self._clf = clf 

        if self._clf is not None: 

            _clf = self._clf(self._state, self._target_state)
            self.clf = lambdify([self._state, self._target_state], _clf)
            
            # Derivative of CLF w.r.t the state x
            dx_clf = sp.Matrix([_clf]).jacobian(self._state).T
            dx_H = sp.Matrix([self.model.H]).jacobian(self._state).T
 
            # Lie derivatives of CLF  w.r.t f(x)=F*dHdx(x)
            self._dLie_f_clf = dx_clf.T @ self.model.F @ dx_H

            # Lie derivatives of CLF  w.r.t g(x)=G
            self._dLie_g_clf = dx_clf.T @ self.model.G 

            # Make the symbolic functions callable
            self.dLie_f_clf = lambdify([self._state, self._target_state], self._dLie_f_clf)
            self.dLie_g_clf = lambdify([self._state, self._target_state], self._dLie_g_clf)

        ########### CBF ###########
 
        self._cbf = cbf

        if self._cbf is not None:

            _cbf = self._cbf(self._state)
            self.cbf = lambdify([self._state], _cbf)
 
            # Derivative of CBF w.r.t the state x
            dx_cbf = sp.Matrix([_cbf]).jacobian(self._state).T
            dx_H = sp.Matrix([self.model.H]).jacobian(self._state).T
              
            # Lie derivatives of CBF  w.r.t f(x)=F*dHdx(x)
            self._dLie_f_cbf = dx_cbf.T @ self.model.F @ dx_H

            # Lie derivatives of CBF  w.r.t g(x)=G
            self._dLie_g_cbf = dx_cbf.T @ self.model.G 

            # Make the symbolic functions callable
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
        self.feas = True

        ########### Data ###########
        self.xt = np.zeros((self.state_dim, self.time_steps))
        self.ut = np.zeros((self.control_dim, self.time_steps))
        self.slackt = np.zeros((1, self.time_steps))
        self.clf_t = np.zeros((1, self.time_steps))
        self.cbf_t = np.zeros((1, self.time_steps))
        self.openloop_energy_t = np.zeros((1, self.time_steps))
        self.closeloop_energy_t = np.zeros((1, self.time_steps))
 

    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
        
    def solve_qp(self, current_state, u_ref, t):
        '''
        Solve the QP problem (using Casadi) 

        :param current_state: current state of the system
        :param u_ref: reference control input
        :return: optimal control input, slack, clf, cbf, feasible 
        '''

        # empty the constraint set
        self.opti.subject_to()

        use_slack = False
        if self.weight_slack is not None:
            if self.weight_slack > 0:
                use_slack = True

        # objective function
        self.H = self.weight_input * np.eye(self.control_dim)
        self.obj = .5 * (self.u - u_ref).T @ self.H @ (self.u - u_ref)
        if use_slack:
            self.obj = self.obj + self.weight_slack * self.slack ** 2

        self.opti.minimize(self.obj)

        # constraints
        if self.u_min is not None and self.u_max is not None:
            self.opti.subject_to(self.opti.bounded(self.u_min, self.u, self.u_max))
        
        if use_slack:
            self.opti.subject_to(self.opti.bounded(-np.inf, self.slack, np.inf))
 
        # CLF constraint
        clf = None
        if self._clf is not None:
            clf = self.clf(current_state, self.target_state)
            dLie_f_clf = self.dLie_f_clf(current_state, self.target_state)
            dLie_g_clf = self.dLie_g_clf(current_state, self.target_state)
 
            # LfV + LgV * u + lambda * V <= slack
            if dLie_g_clf !=0: # check if there is a control input in the constraint
                if use_slack:
                    self.opti.subject_to(dLie_f_clf + dLie_g_clf @ self.u + self.clf_lambda * clf - self.slack <= 0)
                else: 
                    self.opti.subject_to(dLie_f_clf + dLie_g_clf @ self.u + self.clf_lambda * clf <= 0)
            else:
                print(f'No control input in the CLF constraint at t = {t}!')

        # CBF constraint
        cbf = None
        if self._cbf is not None:
            cbf = self.cbf(current_state)
            dLie_f_cbf = self.dLie_f_cbf(current_state)
            dLie_g_cbf = self.dLie_g_cbf(current_state)

            # Lfh + Lgh * u + gamma * h >= 0
            self.opti.subject_to(dLie_f_cbf + dLie_g_cbf @ self.u + self.cbf_gamma * cbf >= 0)

        # optimize the Qp problem
        try:
            sol = self.opti.solve()
            self.feasible = True
            optimal_control = sol.value(self.u) 
            if use_slack:
                slack = sol.value(self.slack)
            else:
                slack = None

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

            self.model.q = self.current_state[0]
            self.model.p = self.current_state[1]
            H = self.model.K() + self.model.V() 
            self.closeloop_energy_t[:, t] = H

            self.model.q = self.current_state[0] + self.target_state[0]
            self.model.p = self.current_state[1] + self.target_state[1]
            H = self.model.K() + self.model.V() 
            self.openloop_energy_t[:, t] = H

            if t % 100 == 0:
                print(f't = {t}')

            u_ref = np.array([0])
            u, delta, clf, cbf, self.feas = self.solve_qp(self.current_state, u_ref, t)
            print(f't = {t}, u = {u}, delta = {delta}, clf = {clf}, cbf = {cbf}') 

            if not self.feas:
                print('\nThis problem is infeasible!\n') 
                break
            else:
                pass

            self.xt[:, t] = np.copy(self.current_state)
            self.ut[:, t] = u
            self.slackt[:, t] = delta
            self.clf_t[:, t] = clf
            self.cbf_t[:, t] = cbf 

            self.current_state, current_output = self.model.step(self.current_state, u)

        print('Finish the solve of qp with clf!')


    #######################################################################################################################
    #######################################################################################################################
    ####################################################################################################################### 
        
    def show(self, *args, save=False, subplots=False):
        plt.close('all')

        if subplots:
            fig, axs = plt.subplots(subplots[0], subplots[1], figsize=(8, 6 * len(args))) 

            for i, arg in enumerate(args):
                arg(show=False, save=save, figure=axs[i // subplots[1]][i % subplots[1]])
        else: 
            for i, arg in enumerate(args):
                plt.figure(i)
                arg(show=False, save=save)

        plt.tight_layout()
        plt.show()

    def plot_state(self, show=True, save=False, figure=None):
        if figure is None:
            plt.figure()
        else:
            plt.sca(figure)

        t = np.arange(0, self.T, self.dt)

        q = self.xt[0]  
        p = self.xt[1] 

        # convert from error state
        q = np.array(q) + self.target_state[0]
        p = np.array(p) + self.target_state[1]

        plt.grid() 
        plt.plot(t, q, 'k', linewidth=3, label='q', linestyle='-')
        plt.plot(t, p, 'b', linewidth=3, label='p', linestyle='-')
        plt.legend()
        #plt.title('State Variable')
        plt.ylabel('q, p')

        if show:
            plt.show()

        if save:
            plt.savefig(os.path.join(PLOTS_PATH, 'state.png'), format='png', dpi=300)

    #######################################################################################################################

    def plot_energy_openloop(self, show=True, save=False, figure=None):
        if figure is None:
            plt.figure()
        else:
            plt.sca(figure)
            
        t = np.arange(0, self.T, self.dt) 
        energy = self.openloop_energy_t.flatten()

        plt.grid() 
        plt.plot(t, energy, linewidth=3, color='b')
        plt.xlabel('Time')
        plt.ylabel('Open-loop Energy') 
        plt.ylim([round(float(min(energy)),3), round(float(max(energy)),3)])
        #plt.title('Open-loop Energy')

        if show:
            plt.show()

        if save:
            plt.savefig(os.path.join(PLOTS_PATH, 'op_energy.png'), format='png', dpi=300)

    
    #######################################################################################################################

    def plot_energy_closeloop(self, show=True, save=False, figure=None):
        if figure is None:
            plt.figure()
        else:
            plt.sca(figure)
            
        t = np.arange(0, self.T, self.dt) 
        energy = self.closeloop_energy_t.flatten()

        plt.grid() 
        plt.plot(t, energy, linewidth=3, color='b')
        plt.xlabel('Time')
        plt.ylabel('Closed-loop Energy') 
        plt.ylim([round(float(min(energy)),3), round(float(max(energy)),3)])
        #plt.title('Closed-loop Energy')

        if show:
            plt.show()

        if save:
            plt.savefig(os.path.join(PLOTS_PATH, 'cl_energy.png'), format='png', dpi=300)


    #######################################################################################################################

    def plot_phase_trajectory(self, add_safe_set=True, state_range=[-20, 20], show=True, save=False, figure=None):
        if figure is None:
            plt.figure()
        else:
            plt.sca(figure) 

        q_traj = [q for q in self.xt[0]]  
        p_traj = [p for p in self.xt[1]] 

        # Convert from error state 
        q_traj = np.array(q_traj) + self.target_state[0]
        p_traj = np.array(p_traj) + self.target_state[1]
  
        # Plot the phase trajectory
        plt.plot(q_traj, p_traj, color='black')

        # Add arrows to indicate the direction of motion
        for i in range(0, len(q_traj) - 1, 5):  # Stop one step earlier
            plt.quiver(q_traj[i], p_traj[i], q_traj[i+1]-q_traj[i], p_traj[i+1]-p_traj[i], angles='xy', scale_units='xy', scale=1, color='black')

        # Plot the initial and final with a higher z-order
        plt.scatter([q_traj[0]], [p_traj[0]], color='green', label='Initial state', zorder=5)
        plt.scatter([q_traj[-1]], [p_traj[-1]], marker='o', facecolors='none', edgecolors='red', label='Final state', zorder=5)

        # Plot the target state with a star
        if self.target_state is not None:
            plt.scatter(self.target_state[0], self.target_state[1], marker='*', color='red', label='Target state', zorder=5)

        # Add text close to the initial and final states with an offset
        plt.text(q_traj[0], p_traj[0], '$x(0)$', verticalalignment='bottom', horizontalalignment='right')
        plt.text(q_traj[-1], p_traj[-1], '$x(T)$', verticalalignment='bottom', horizontalalignment='right')
         
        if self._cbf is not None and add_safe_set: 

            # q_vals = np.linspace(min(self.xt[0])*2, max(self.xt[0])*2, 500)
            # p_vals = np.linspace(min(self.xt[1])*2, max(self.xt[1])*2, 500)
            q_vals = np.linspace(state_range[0], state_range[1], 500)
            p_vals = np.linspace(state_range[0], state_range[1], 500)
            q_vals, p_vals = np.meshgrid(q_vals, p_vals) 

            cbf_vals = self._cbf([q_vals, p_vals])

            # Plot the 2D contour 
            plt.contour(q_vals, p_vals, cbf_vals, levels=[0], colors='blue') 

            # Fill the areas where the function is over the plane in grey and under the plane in white
            plt.contourf(q_vals, p_vals, cbf_vals, levels=[-np.inf, 0], colors='grey', alpha=0.3)
 
        if not self.feas:
            plt.text(0, 0, 'Infeasible', verticalalignment='bottom', horizontalalignment='right', color='red', fontsize=18)

        plt.grid(True)
        plt.xlabel('q')
        plt.ylabel('p')
        #plt.title('Phase space trajectory')
        
        if show:
            plt.show()

        if save:
            plt.savefig(os.path.join(PLOTS_PATH, 'phase_trajectory.png'), format='png', dpi=300)
 

    #######################################################################################################################

    def plot_slack(self, show=True, save=False, figure=None):
        if figure is None:
            plt.figure()
        else:
            plt.sca(figure)

        t = np.arange(0, self.T, self.dt)
        slack = self.slackt.flatten()

        plt.grid() 
        plt.plot(t, slack, linewidth=3, color='b')
        #plt.title('Slack')
        plt.ylabel('delta')

        if show:
            plt.show()

        if save:
            plt.savefig(os.path.join(PLOTS_PATH, 'slack.png'), format='png', dpi=300)
                    

    #######################################################################################################################

    def plot_clf(self, show=True, save=False, figure=None):
        if figure is None:
            plt.figure()
        else:
            plt.sca(figure)
             
        t = np.arange(0, self.T, self.dt)
        clf = self.clf_t.flatten()

        plt.grid()        
        plt.plot(t, clf, linewidth=3, color='b')
        #plt.title('clf')
        plt.ylabel('V(x)')

        if show:
            plt.show()

        if save:
            plt.savefig(os.path.join(PLOTS_PATH, 'clf.png'), format='png', dpi=300)

    #######################################################################################################################

    def plot_cbf(self, show=True, save=False, figure=None):
        if figure is None:
            plt.figure()
        else:
            plt.sca(figure)
            
        t = np.arange(0, self.T, self.dt)
        cbf = self.cbf_t.flatten()

        plt.grid()
        plt.plot(t, cbf, linewidth=3, color='b')
        #plt.title('cbf')
        plt.ylabel('h(x)')

        if show:
            plt.show()

        if save:
            plt.savefig(os.path.join(PLOTS_PATH, 'cbf.png'), format='png', dpi=300)
        

    #######################################################################################################################

    def plot_control(self, show=True, save=False, figure=None):
        if figure is None:
            plt.figure()
        else:
            plt.sca(figure)
            
        u_max = self.parameter['u_max']
        t = np.arange(0, self.T, self.dt)
        control = self.ut.flatten()

        plt.grid() 
        plt.plot(t, control, 'b', linewidth=3, label='w', linestyle='-')
        if u_max is not None:
            plt.plot(t, u_max * np.ones(t.shape[0]), 'k', linewidth=3, label='Bound', linestyle='--')
            plt.plot(t, -u_max * np.ones(t.shape[0]), 'k', linewidth=3, linestyle='--') 
        plt.ylabel('q, p') 
        #plt.title('control')
        plt.ylabel('u')
        plt.legend(loc='upper left')
 
        if show:
            plt.show()

        if save:
            plt.savefig(os.path.join(PLOTS_PATH, 'control.png'), format='png', dpi=300)

        
    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
    
    def animate_phase_trajectory(self, add_safe_set=False, state_range=[-20, 20], show=True, save=False):
        fig, ax = plt.subplots() 
        ax.set_xlabel('q')
        ax.set_ylabel('p')
        ax.set_title('Phase space trajectory')

        if add_safe_set: 
            # q_vals = np.linspace(min(self.xt[0])*2, max(self.xt[0])*2, 500)
            # p_vals = np.linspace(min(self.xt[1])*2, max(self.xt[1])*2, 500)
            q_vals = np.linspace(state_range[0], state_range[1], 500) + self.target_state[0]
            p_vals = np.linspace(state_range[0], state_range[1], 500) + self.target_state[1]
            q_vals, p_vals = np.meshgrid(q_vals, p_vals)  

            cbf_vals = self._cbf([q_vals, p_vals])

            # Plot the intersection of the cfb with the plane
            ax.contour(q_vals, p_vals, cbf_vals, levels=[0], colors='blue') 

            # Fill the areas where the function is over the plane in grey 
            ax.contourf(q_vals, p_vals, cbf_vals, levels=[-np.inf, 0], colors='grey', alpha=0.3)

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
            point.set_data([q], [p])
            point_final.set_data([state_traj[-1][0]], [state_traj[-1][1]])
            return line, point, point_final


        ani = animation.FuncAnimation(fig, animate, frames=len(state_traj), init_func=init, blit=True)
        
        if show:
            plt.show()

        if save:
            ani.save(os.path.join(PLOTS_PATH, 'phase_trajectory.gif'), writer='imagemagick', fps=60)

