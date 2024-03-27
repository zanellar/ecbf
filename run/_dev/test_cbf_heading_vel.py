import numpy as np
import sympy as sp
import casadi as ca
import matplotlib
#matplotlib.use('TkAgg')  # Do this BEFORE importing matplotlib.pyplotimport matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from sympy.utilities.lambdify import lambdify

matplotlib.rcParams['figure.dpi'] = 200


class clf_cbf_dubins():
    def __init__(self) -> None:
        """
        robot_state: [x, y, theta]
        target_state: [x, y, theta]
        control: [velocity]
        """
        self.parameter = {
            'T': 12,
            'step_time': 0.1,
            'init_state': [0.0, 5.0, 0.0],
            'target_state': [10.0, 1.0, 0.0],
            'weight_input': 0.5,
            'weight_slack': 10,
            'clf_lambda': 1.0,
            'velocity': 1.0,
            'u_max': 5,
            'u_min': -5,
            'obstacle_radius': 1,
            'obstacle_center': [6, 3.0],
            'cbf_gamma': 1,
            'cbf_gamma0': 1,
        }
        self.state_dim = 3
        self.control_dim = 1

        self.T = self.parameter['T']
        self.step_time = self.parameter['step_time']
        self.time_steps = int(np.ceil(self.T / self.step_time))
        self.weight_input = self.parameter['weight_input']
        self.weight_slack = self.parameter['weight_slack']
        self.clf_lambda = self.parameter['clf_lambda']
        self.cbf_gamma = self.parameter['cbf_gamma']
        self.u_max = self.parameter['u_max']
        self.u_min = self.parameter['u_min']
        self.init_state = self.parameter['init_state']
        self.target_state_algebra = self.parameter['target_state']
        self.obs_state_algebra = self.parameter['obstacle_center']
        self.obs_radius = self.parameter['obstacle_radius']
        self.constant_v = self.parameter['velocity']
        self.gamma0 = self.parameter['cbf_gamma0']
        self.current_state = self.init_state
        self.robot_radius = 0.3

        x, y, theta = sp.symbols('x y theta')  # define symbolic representation
        self.state = sp.Matrix([x, y, theta])  # row vector

        e_x, e_y, e_theta = sp.symbols('e_x, e_y, e_theta')
        self.target_state = sp.Matrix([e_x, e_y, e_theta])  # symbols

        o_x, o_y = sp.symbols('o_x, o_y')
        self.obs_state = sp.Matrix([o_x, o_y])

        # cons_v = sp.symbols('v')
        # self.constant_v = sp.Matrix([cons_v])

        self.f = None
        self.g = None
        self.f_symbolic = None
        self.g_symbolic = None

        self.clf = None
        self.clf_symbolic = None
        self.lf_clf = None
        self.lf_clf_symbolic = None
        self.lg_clf = None
        self.lg_clf_symbolic = None

        self.cbf = None
        self.cbf_symbolic = None
        self.lf_cbf = None
        self.lf_cbf_symbolic = None
        self.lg_cbf = None
        self.lg_cbf_symbolic = None

        self.init_system()  # optimize

        self.opti = ca.Opti()

        # solver
        opts_setting = {
            'ipopt.max_iter': 200,
            'ipopt.print_level': 1,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }

        self.opti.solver('ipopt', opts_setting)
        self.u = self.opti.variable(self.control_dim)
        self.slack = self.opti.variable()

        self.obj = None  # cost function
        self.feasible = None  # feedback
        self.H = None

        # storage
        self.xt = np.zeros((self.state_dim, self.time_steps))
        self.ut = np.zeros((self.control_dim, self.time_steps))
        self.slackt = np.zeros((1, self.time_steps))
        self.clf_t = np.zeros((1, self.time_steps))
        self.cbf_t = np.zeros((1, self.time_steps))

        # plot
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(10, 10)
        self.robot_body = None
        self.robot_arrow = None
        self.target_body = None
        self.target_arrow = None
        self.obs = None

    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
    def init_system(self):
        # define the system symbolic model
        self.f_symbolic = sp.Matrix([self.parameter['velocity'] * sp.cos(self.state[2]),
                                     self.parameter['velocity'] * sp.sin(self.state[2]),
                                     0
                                     ])
        self.f = lambdify([self.state], self.f_symbolic)

        self.g_symbolic = sp.Matrix([[0], [0], [1]])
        self.g = lambdify([self.state], self.g_symbolic)

        # clf
        self.clf_symbolic = (sp.cos(self.state[2]) * (self.state[1] - self.target_state[1]) - sp.sin(
            self.state[2]) * (self.state[0] - self.target_state[0])) ** 2
        self.clf = lambdify([self.state, self.target_state], self.clf_symbolic)

        dx_clf = sp.Matrix([self.clf_symbolic]).jacobian(self.state)
        self.lf_clf_symbolic = (dx_clf @ self.f_symbolic)
        self.lf_clf = lambdify([self.state, self.target_state], self.lf_clf_symbolic)

        self.lg_clf_symbolic = dx_clf @ self.g_symbolic
        self.lg_clf = lambdify([self.state, self.target_state], self.lg_clf_symbolic)
        # cbf
        distance = (self.state[0] - self.obs_state[0]) ** 2 + (
                self.state[1] - self.obs_state[1]) ** 2 - (self.obs_radius + self.robot_radius) ** 2

        # deriDistance = 2 * (self.state[0] - self.obs_state[0]) * 1 * sp.cos(self.state[2]) + 2 * (
        #         self.state[1] - self.obs_state[1]) * 1 * sp.sin(self.state[2])

        deriDistance = (sp.Matrix([distance]).jacobian(self.state) @ self.f_symbolic)[0, 0]

        self.cbf_symbolic = deriDistance + self.gamma0 * distance
        self.cbf = lambdify([self.state, self.obs_state], self.cbf_symbolic)

        dx_cbf = sp.Matrix([self.cbf_symbolic]).jacobian(self.state)
        self.lf_cbf_symbolic = dx_cbf @ self.f_symbolic
        self.lf_cbf = lambdify([self.state, self.obs_state], self.lf_cbf_symbolic)

        self.lg_cbf_symbolic = dx_cbf @ self.g_symbolic
        self.lg_cbf = lambdify([self.state, self.obs_state], self.lg_cbf_symbolic)

    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
    def dynamic(self, x, u):
        return self.f(x) + self.g(x) @ np.array(u).reshape(self.control_dim, -1)


    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
    def get_next_state(self, current_state, u, dt):
        # Fourth-order Rungekutta method
        f1 = self.dynamic(current_state, u).T[0]
        f2 = self.dynamic(current_state + dt * f1 / 2, u).T[0]
        f3 = self.dynamic(current_state + dt * f2 / 2, u).T[0]
        f4 = self.dynamic(current_state + dt * f3, u).T[0]
        next_state = current_state + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
        return next_state


    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
    def clf_qp(self, current_state, u_ref):
        """
        current_state: [x, y, theta]
        """
        # empty the constraint set
        self.opti.subject_to()

        # objective function
        self.H = self.weight_input * np.eye(self.control_dim)
        self.obj = .5 * (self.u - u_ref).T @ self.H @ (self.u - u_ref)
        self.obj = self.obj + self.weight_slack * self.slack ** 2
        self.opti.minimize(self.obj)

        clf = self.clf(current_state, self.target_state_algebra)
        lf_clf = self.lf_clf(current_state, self.target_state_algebra)
        lg_clf = self.lg_clf(current_state, self.target_state_algebra)

        cbf = self.cbf(current_state, self.obs_state_algebra)
        lf_cbf = self.lf_cbf(current_state, self.obs_state_algebra)
        lg_cbf = self.lg_cbf(current_state, self.obs_state_algebra)

        # constraint
        self.opti.subject_to(self.opti.bounded(self.u_min, self.u, self.u_max))
        self.opti.subject_to(self.opti.bounded(-np.inf, self.slack, np.inf))

        # LfV + LgV * u + lambda * V <= slack
        self.opti.subject_to(lf_clf + (lg_clf @ self.u)[0][0] + self.clf_lambda * clf - self.slack <= 0)

        # Lfh + Lgh * u + gamma * h >= 0
        self.opti.subject_to(lf_cbf + (lg_cbf @ self.u)[0][0] + self.cbf_gamma * cbf >= 0)

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
    def qp_solve(self):
        for t in range(self.time_steps):
            if t % 100 == 0:
                print(f't = {t}')

            u_ref = np.array([0])
            u, delta, clf, cbf, feas = self.clf_qp(self.current_state, u_ref)

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

            self.current_state = self.get_next_state(self.current_state, u, self.step_time)

        print('Finish the solve of qp with clf!')


    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
    def render(self): 
        self.ax.set_xlim(min(self.target_state_algebra[0],self.init_state[0])-5, max(self.target_state_algebra[0],self.init_state[0])+5)  # wrt the specific problem
        self.ax.set_ylim(min(self.target_state_algebra[1],self.init_state[1])-5, max(self.target_state_algebra[1],self.init_state[1])+5)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")

        self.animation_init()
        position_x = self.init_state[0]
        position_y = self.init_state[1]

        self.robot_body = mpatches.Circle(xy=(position_x, position_y), radius=self.robot_radius, color='r')
        self.ax.add_patch(self.robot_body)

        self.robot_arrow = mpatches.Arrow(position_x,
                                          position_y,
                                          self.robot_radius * np.cos(self.init_state[2]),
                                          self.robot_radius * np.sin(self.init_state[2]),
                                          width=0.10)
        self.ax.add_patch(self.robot_arrow)

        self.ani = animation.FuncAnimation(self.fig,
                                           func=self.animation_loop,
                                           frames=self.xt.shape[1],
                                           init_func=self.animation_init,
                                           interval=200,
                                           repeat=False)
        plt.grid('--')
        plt.show()
        self.ani.save('animation.gif', writer='pillow')
        plt.close()

    def animation_init(self):
        position_x = self.target_state_algebra[0]
        position_y = self.target_state_algebra[1]

        pos_obs_x = self.obs_state_algebra[0]
        pos_obs_y = self.obs_state_algebra[1]

        self.target_body = mpatches.Circle(xy=(position_x, position_y), radius=self.robot_radius, color='b')
        self.obs = mpatches.Circle(xy=(pos_obs_x, pos_obs_y), radius=self.obs_radius, color='k')

        self.ax.add_patch(self.target_body)
        self.ax.add_patch(self.obs)

        self.target_arrow = mpatches.Arrow(position_x, position_y,
                                           self.robot_radius * np.cos(self.target_state_algebra[2]),
                                           self.robot_radius * np.sin(self.target_state_algebra[2]),
                                           width=0.10,
                                           color='black')
        self.ax.add_patch(self.target_arrow)
        return self.ax.patches + self.ax.texts + self.ax.artists

    def animation_loop(self, indx):
        self.robot_arrow.remove()
        self.robot_body.remove()

        position_x = self.xt[:, indx][0]
        position_y = self.xt[:, indx][1]
        orientation = self.xt[:, indx][2]

        self.robot_body = mpatches.Circle(xy=(position_x, position_y), radius=self.robot_radius, color='r')
        self.ax.add_patch(self.robot_body)

        self.robot_arrow = mpatches.Arrow(position_x,
                                          position_y,
                                          self.robot_radius * np.cos(orientation),
                                          self.robot_radius * np.sin(orientation),
                                          width=0.10,
                                          color='black')
        self.ax.add_patch(self.robot_arrow)

        if indx != 0:
            # past trajecotry
            x_list = [self.xt[:, indx - 1][0], self.xt[:, indx][0]]
            y_list = [self.xt[:, indx - 1][1], self.xt[:, indx][1]]
            self.ax.plot(x_list, y_list, color='b', )

        return self.ax.patches + self.ax.texts + self.ax.artists

    def show_state(self):
        t = np.arange(0, self.T, self.step_time)
        plt.grid()

        plt.plot(t, self.xt[0], 'k', linewidth=3, label='x', linestyle='-')
        plt.plot(t, self.xt[1], 'b', linewidth=3, label='y', linestyle='-')
        plt.plot(t, self.xt[2], 'g', linewidth=3, label='theta', linestyle='-')

        plt.legend()

        plt.title('State Variable')
        plt.ylabel('x, y, theta')

        plt.show()
        # plt.savefig('slack.png', format='png', dpi=300)
        plt.close(self.fig)

    def show_slack(self):
        t = np.arange(0, self.T, self.step_time)
        plt.grid()

        plt.plot(t, self.slackt[0], linewidth=3, color='orange')
        plt.title('Slack')
        plt.ylabel('delta')

        plt.show()
        # plt.savefig('slack.png', format='png', dpi=300)
        plt.close(self.fig)
        # print(self.slackt[0])

    def show_clf(self):
        t = np.arange(0, self.T, self.step_time)
        plt.grid()

        plt.plot(t, self.clf_t[0], linewidth=3, color='cyan')
        plt.title('clf')
        plt.ylabel('V(x)')

        plt.show()
        # plt.savefig('clf.png', format='png', dpi=300)
        plt.close(self.fig)

    def show_cbf(self):
        t = np.arange(0, self.T, self.step_time)
        plt.grid()

        plt.plot(t, self.cbf_t[0], linewidth=3, color='cyan')
        plt.title('cbf')
        plt.ylabel('h(x)')

        plt.show()
        # plt.savefig('clf.png', format='png', dpi=300)
        plt.close(self.fig)

    def show_control(self):
        u_max = self.parameter['u_max']
        t = np.arange(0, self.T, self.step_time)
        plt.grid()

        plt.plot(t, self.ut[0], 'b', linewidth=3, label='w', linestyle='-')
        plt.plot(t, u_max * np.ones(t.shape[0]), 'k', linewidth=3, label='Bound', linestyle='--')
        plt.plot(t, -u_max * np.ones(t.shape[0]), 'k', linewidth=3, linestyle='--')

        plt.title('State Variable')
        plt.ylabel('x, y, theta')

        plt.title('control')
        plt.ylabel('u')
        plt.legend(loc='upper left')

        plt.show()
        # plt.savefig('control.png', format='png', dpi=300)
        plt.close(self.fig)

    def show_traj(self):
        t = np.arange(0, self.T, self.step_time)
        plt.grid()
        position_x_g = self.target_state_algebra[0]
        position_y_g = self.target_state_algebra[1]
        position_x = self.init_state[0]
        position_y = self.init_state[1]
        position_x_t = self.xt[0][-1]
        position_y_t = self.xt[1][-1]
        pos_obs_x = self.obs_state_algebra[0]
        pos_obs_y = self.obs_state_algebra[1]

        self.target_body = mpatches.Circle(xy=(position_x_g, position_y_g), radius=self.robot_radius, color='b')
        self.robot_body = mpatches.Circle(xy=(position_x, position_y), radius=self.robot_radius, color='r')
        self.obs = mpatches.Circle(xy=(pos_obs_x, pos_obs_y), radius=self.obs_radius, edgecolor='k', fill=False,
                                   linewidth=3)
        self.robot_body_final = mpatches.Circle(xy=(position_x_t, position_y_t), radius=self.robot_radius, color='r')
        self.ax.add_patch(self.robot_body)
        self.ax.add_patch(self.robot_body_final)
        self.ax.add_patch(self.target_body)
        self.ax.add_patch(self.obs)

        self.robot_arrow = mpatches.Arrow(position_x,
                                          position_y,
                                          self.robot_radius * np.cos(self.init_state[2]),
                                          self.robot_radius * np.sin(self.init_state[2]),
                                          width=0.10)
        self.ax.add_patch(self.robot_arrow)

        plt.plot(self.xt[0], self.xt[1], 'b', linewidth=3, label='Trajectory', linestyle='-')

        for i in range(0, len(t), 3):
            pos_x = self.xt[0][i]
            pos_y = self.xt[1][i]
            self.robot_body_i = mpatches.Circle(xy=(pos_x, pos_y), radius=self.robot_radius, color='r',
                                                alpha=0.3)
            self.ax.add_patch(self.robot_body_i)

        plt.legend()
        plt.title('Trajectory')
        plt.ylabel('x, y, theta')

        plt.axis("equal")
        plt.show()
        # plt.savefig('slack.png', format='png', dpi=300)
        plt.close(self.fig)


if __name__ == "__main__":
    test_target = clf_cbf_dubins()
    test_target.qp_solve()

    # test_target.render()
    test_target.show_traj()

    test_target.show_control()
    test_target.show_state()
    # test_target.show_slack()
    test_target.show_clf()
    test_target.show_cbf()
