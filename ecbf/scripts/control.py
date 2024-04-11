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

from ecbf.utils.paths import PLOTS_PATH, RES_PATH

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
        self.no_target_defined = True if self.target_state is None else False
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

        self.clf_tool = clf 
        if clf is not None:
            self._clf = clf.function

        if self.clf_tool is not None: 

            _clf = self._clf(self._state, self._target_state)
            self.clf = lambdify([self._state, self._target_state], _clf)
            
            # Derivative of CLF w.r.t the state x
            _dx_clf = sp.Matrix([_clf]).jacobian(self._state).T
            _dx_H = sp.Matrix([self.model._H]).jacobian(self._state).T
 
            # Lie derivatives of CLF  w.r.t f(x)=F*dHdx(x)
            self._dLie_f_clf = _dx_clf.T @ self.model._F @ _dx_H

            # Lie derivatives of CLF  w.r.t g(x)=G
            self._dLie_g_clf = _dx_clf.T @ self.model._G 

            # Make the symbolic functions callable
            self.dLie_f_clf = lambdify([self._state, self._target_state], self._dLie_f_clf)
            self.dLie_g_clf = lambdify([self._state, self._target_state], self._dLie_g_clf)

        ########### CBF ###########

        self.cbf_tool = cbf
        if cbf is not None:
            self._cbf = cbf.function

        if self.cbf_tool is not None:

            _cbf = self._cbf(self._state)
            self.cbf = lambdify([self._state], _cbf)
 
            # Derivative of CBF w.r.t the state x
            _dx_cbf = sp.Matrix([_cbf]).jacobian(self._state).T
            _dx_H = sp.Matrix([self.model._H]).jacobian(self._state).T
              
            # Lie derivatives of CBF  w.r.t f(x)=F*dHdx(x)
            self._dLie_f_cbf = _dx_cbf.T @ self.model._F @ _dx_H

            # Lie derivatives of CBF  w.r.t g(x)=G
            self._dLie_g_cbf = _dx_cbf.T @ self.model._G 

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
        self.W = self.weight_input * np.eye(self.control_dim)
        self.obj = .5 * (self.u - u_ref).T @ self.W @ (self.u - u_ref)
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
        if self.clf_tool is not None:
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


        # sigma = 1 # TODO
        # beta = 1-np.exp(-self.current_state[1]**2/sigma) 

        # CBF constraint
        cbf = None
        if self.cbf_tool is not None:
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
        
    def run(self, u_ref=[0], save= False, name=''):
        '''
        Run the controller and solve the QP problem at each time step.  

        :param u_ref: reference control input
        '''

        for t in range(self.time_steps):

            if t % 100 == 0:
                print(f't = {t}')

            u_ref = np.array(u_ref)
            u, delta, clf, cbf, self.feas = self.solve_qp(self.current_state, u_ref, t)
            # print(f't = {t}, u = {u}, delta = {delta}, clf = {clf}, cbf = {cbf}') 

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

            # Compute the energy of the open-loop system 
            H = self.model.get_energy(self.current_state + self.target_state) 
            self.openloop_energy_t[:, t] = H

            # Compute the energy of the closed-loop system 
            H = self.model.get_energy(self.current_state)
            self.closeloop_energy_t[:, t] = H

            self.current_state, current_output = self.model.step(self.current_state, u)

        print('Finish the solve of qp with clf!')

        # Save all the data in RES_PATH
        if save:
            np.save(os.path.join(RES_PATH, f'{name}_xt.npy'), self.xt)
            np.save(os.path.join(RES_PATH, f'{name}_ut.npy'), self.ut)
            np.save(os.path.join(RES_PATH, f'{name}_slackt.npy'), self.slackt)
            np.save(os.path.join(RES_PATH, f'{name}_clf_t.npy'), self.clf_t)
            np.save(os.path.join(RES_PATH, f'{name}_cbf_t.npy'), self.cbf_t)
            np.save(os.path.join(RES_PATH, f'{name}_openloop_energy_t.npy'), self.openloop_energy_t)
            np.save(os.path.join(RES_PATH, f'{name}_closeloop_energy_t.npy'), self.closeloop_energy_t)




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

    #######################################################################################################################

    def plot_phase_trajectory(self, add_safe_set=True, plot_end_state = True, state_range=[-20, 20], show=True, save=False, figure=None, name = '', color='blue', arrow_skip=10, arrow_clif=0):
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
        plt.plot(q_traj, p_traj, color=color, label=name)

        # Add arrows to indicate the direction of motion
        for i in range(0, len(q_traj) - 1, arrow_skip):  # Stop one step earlier
            if i>arrow_clif:
                plt.annotate('', xy=(q_traj[i+1], p_traj[i+1]), xytext=(q_traj[i], p_traj[i]),
                            arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle='Simple,tail_width=0.5,head_width=0.5,head_length=0.5'))    
                       
        # Plot the initial and final with a higher z-order without a label
        plt.scatter([q_traj[0]], [p_traj[0]], color='black', zorder=5)
        if plot_end_state:
            plt.scatter([q_traj[-1]], [p_traj[-1]], marker='o', facecolors='none', edgecolors='black', zorder=5 , color=color)

        # Plot the target state with a star
        if self.no_target_defined is False:
            plt.scatter(self.target_state[0], self.target_state[1], marker='*', color='black', label='Target state', zorder=5)

        # Add text close to the initial and final states with an offset
        plt.text(q_traj[0], p_traj[0], '$x(0)$', verticalalignment='bottom', horizontalalignment='right')
        if plot_end_state:
            plt.text(q_traj[-1], p_traj[-1], '$x(T)$', verticalalignment='bottom', horizontalalignment='right', color=color)
         
        if self.cbf_tool is not None and add_safe_set: 

            # q_vals = np.linspace(min(self.xt[0])*2, max(self.xt[0])*2, 500)
            # p_vals = np.linspace(min(self.xt[1])*2, max(self.xt[1])*2, 500)
            q_vals = np.linspace(state_range[0], state_range[1], 500)
            p_vals = np.linspace(state_range[0], state_range[1], 500)
            q_vals, p_vals = np.meshgrid(q_vals, p_vals) 

            cbf_vals = self._cbf([q_vals, p_vals])

            # Plot the 2D contour 
            plt.contour(q_vals, p_vals, cbf_vals, levels=[0], colors='black', linewidths=5, alpha=0.3) 

            # Fill the areas where the function is over the plane in grey and under the plane in white
            plt.contourf(q_vals, p_vals, cbf_vals, levels=[-np.inf, 0], colors='grey', alpha=0.3)
 
        if not self.feas:
            plt.text(0, 0, 'Infeasible', verticalalignment='bottom', horizontalalignment='right', color='red', fontsize=18)

        plt.grid(True)
        plt.xlabel('q [m]')
        plt.ylabel('p [m]')
        #plt.title('Phase space trajectory')
        
        if show:
            plt.show()

        if save:
            file_name = 'phase_trajectory.png' if name == '' else f'phase_trajectory_{name}.png'
            plt.savefig(os.path.join(PLOTS_PATH, file_name), format='png', dpi=300)  

  
    #######################################################################################################################

    def plot_state(self, show=True, save=False, figure=None, name = '', color='blue'):
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

        plt.plot(t, q, linewidth=3, label='q'+name, linestyle='-', color=color)
        plt.plot(t, p,  linewidth=3, label='p'+name, linestyle='--', color=color)

        # plt.legend()
        plt.grid(True)
        #plt.title('State Variable')
        plt.ylabel('q, p [m]')
        plt.xlabel('time [s]')

        if show:
            plt.show()

        if save:
            file_name = 'state.png' if name == '' else f'state_{name}.png'
            plt.savefig(os.path.join(PLOTS_PATH, file_name), format='png', dpi=300)

    #######################################################################################################################

    def plot_energy_openloop(self, show=True, save=False, figure=None, ylims=None, name = '', color='blue'):
        if figure is None:
            plt.figure()
        else:
            plt.sca(figure)
            
        t = np.arange(0, self.T, self.dt) 
        energy = self.openloop_energy_t.flatten()

        plt.plot(t, energy, linewidth=3, color=color, label=name)
 
        plt.grid(True) 
        plt.xlabel('time [s]')
        plt.ylabel('H [J]') 
        if ylims is not None:
            plt.ylim(ylims)
        else:
            plt.ylim([round(float(min(energy)),3), round(float(max(energy)),3)])
        #plt.title('Open-loop Energy')

        if show:
            plt.show()

        if save:
            file_name = 'op_energy.png' if name == '' else f'op_energy_{name}.png'
            plt.savefig(os.path.join(PLOTS_PATH, file_name), format='png', dpi=300) 

    
    #######################################################################################################################

    def plot_energy_closeloop(self, show=True, save=False, figure=None, ylims=None, name = '', color='blue'):
        if figure is None:
            plt.figure()
        else:
            plt.sca(figure)
            
        t = np.arange(0, self.T, self.dt) 
        energy = self.closeloop_energy_t.flatten()

        plt.plot(t, energy, linewidth=3, color=color, label=name)
 
        plt.grid() 
        plt.xlabel('time [s]')
        plt.ylabel('H [J]') 
        if ylims is not None:
            plt.ylim(ylims)
        else:
            plt.ylim([round(float(min(energy)),3), round(float(max(energy)),3)])
        #plt.title('Closed-loop Energy')

        if show:
            plt.show()

        if save:
            file_name = 'cl_energy.png' if name == '' else f'cl_energy_{name}.png'
            plt.savefig(os.path.join(PLOTS_PATH, file_name), format='png', dpi=300)  


    #######################################################################################################################

    def plot_slack(self, show=True, save=False, figure=None, name = ''):
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
            file_name = 'slack.png' if name == '' else f'slack_{name}.png'
            plt.savefig(os.path.join(PLOTS_PATH, file_name), format='png', dpi=300)  
                    

    #######################################################################################################################

    def plot_clf(self, show=True, save=False, figure=None, ylims=None, name = '', color='blue'):
        if figure is None:
            plt.figure()
        else:
            plt.sca(figure)
             
        t = np.arange(0, self.T, self.dt)
        clf = self.clf_t.flatten()

        plt.plot(t, clf, linewidth=3, color=color , label=name)

        #plt.title('clf')
        plt.grid(True)
        plt.ylabel('V')
        plt.xlabel('time [s]')
        if ylims is not None:
            plt.ylim(ylims)
        else:
            plt.ylim([round(float(min(clf)),3), round(float(max(clf)),3)])
 
        if show:
            plt.show()

        if save:
            file_name = 'clf.png' if name == '' else f'clf_{name}.png'
            plt.savefig(os.path.join(PLOTS_PATH, file_name), format='png', dpi=300)  

    #######################################################################################################################

    def plot_cbf(self, show=True, save=False, figure=None, ylims=None, name = '', color='blue'):
        if figure is None:
            plt.figure()
        else:
            plt.sca(figure)
            
        t = np.arange(0, self.T, self.dt)
        cbf = self.cbf_t.flatten()

        plt.plot(t, cbf, linewidth=3, color=color, label=name)

        #plt.title('cbf')
        plt.grid(True)
        plt.ylabel('h')
        plt.xlabel('time [s]')
        if ylims is not None:
            plt.ylim(ylims)
        else:
            plt.ylim([round(float(min(cbf)),3), round(float(max(cbf)),3)])

        if show:
            plt.show()

        if save:
            file_name = 'cbf.png' if name == '' else f'cbf_{name}.png'
            plt.savefig(os.path.join(PLOTS_PATH, file_name), format='png', dpi=300)  
        

    #######################################################################################################################

    def plot_control(self, show=True, save=False, figure=None, name = '', color='blue'):
        if figure is None:
            plt.figure()
        else:
            plt.sca(figure)
            
        u_max = self.parameter['u_max']
        t = np.arange(0, self.T, self.dt)
        control = self.ut.flatten()

        plt.grid() 
        plt.plot(t, control, linewidth=3, label=name, linestyle='-', color=color)
        if u_max is not None:
            plt.plot(t, u_max * np.ones(t.shape[0]), 'k', linewidth=3, label='Bound', linestyle='--')
            plt.plot(t, -u_max * np.ones(t.shape[0]), 'k', linewidth=3, linestyle='--') 

        plt.grid(True)
        #plt.title('control')
        plt.xlabel('time [s]') 
        plt.ylabel('u')
        plt.legend(loc='upper left')
 
        if show:
            plt.show()

        if save:
            file_name = 'control.png' if name == '' else f'control_{name}.png'
            plt.savefig(os.path.join(PLOTS_PATH, file_name), format='png', dpi=300)  

        
    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
    
    def animate_phase_trajectory(self, add_safe_set=False, state_range=[-20, 20], show=True, save=False, name = ''):
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
            file_name = 'phase_trajectory.gif' if name == '' else f'phase_trajectory_{name}.gif' 
            ani.save(os.path.join(PLOTS_PATH, file_name), writer='imagemagick', fps=60)

