import numpy as np
import casadi as cs

class Ismpc:
  def __init__(self, initial, footstep_planner, params):
    # parameters
    self.params = params
    self.N = params['N']
    self.delta = params['world_time_step']
    self.h = params['h']
    self.eta = params['eta']
    self.foot_size = params['foot_size']
    self.mass = params['mass']
    self.initial = initial
    self.footstep_planner = footstep_planner
    self.sigma = lambda t, t0, t1: np.clip((t - t0) / (t1 - t0), 0, 1) # piecewise linear sigmoidal function

    # lip model matrices
    self.A_lip = np.array([[0, 1, 0], [self.eta**2, 0, -self.eta**2], [0, 0, 0]])
    self.B_lip = np.array([[0], [0], [1]])

    # dynamics -- An: lambda is an anonymus function in Python, input dim depends on the user-- cs.vercat stacks these component vertically
    # An: x= [pcom_x,pcom_y,pcom_z, vcom_x,vcom_y,vcom_z, hw_x,hw_y,hw_z] hw:angular momentum
    # An: TBD: How to treat the disturbance force theta, the cross product calculation
    # An: We do not need to include the disturbance model into the mpc solver
    
    self.f = lambda x, u: cs.vertcat(
        x[3:6] ,
        1/self.mass @ u[0:3]+np.array([0, 0,- params['g']]),
      self.A_lip @ x[6:9] + self.B_lip @ u[2] + np.array([0, - params['g'], 0]),
    )

    # optimization problem
    self.opt = cs.Opti('conic')
    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000, "verbose": False}
    self.opt.solver("osqp", p_opts, s_opts)
    
    #An: Create optimization variable
    self.U = self.opt.variable(3, self.N)
    self.CoM = self.opt.variable(3, self.N + 1)
    self.dCoM = self.opt.variable(3, self.N + 1)
    self.dhw = self.opt.variable(3, self.N + 1)
    self.thetahat = self.opt.variable(3, self.N + 1)
    self.pos_contact_left= self.opt.variable(3, self.N + 1)
    self.pos_contact_right= self.opt.variable(3, self.N + 1)
    self.vel_contact_left= self.opt.variable(3, self.N + 1)
    self.vel_contact_right= self.opt.variable(3, self.N + 1)

    #An: Concatenate them in to the state matrix (3*num_state, self.N+1)
    self.state= cs.vercat(self.CoM,self.dCoM,self.dhw,self.thetahat,
                          self.pos_contact_left,self.pos_contact_right,self.vel_contact_left,self.vel_contact_right)

    #An: Create optimization params that change during simulation time
    self.x0_param = self.opt.parameter(9)
    self.zmp_x_mid_param = self.opt.parameter(self.N)
    self.zmp_y_mid_param = self.opt.parameter(self.N)
    self.zmp_z_mid_param = self.opt.parameter(self.N)

    self.com_ref = self.opt.parameter(3,self.N)
    self.v_com_ref = self.opt.parameter(3,self.N)

    # An: to track the contact status of the left and right foot 1 means in contact
    self.contact_left = self.opt.paremeter(1)
    self.contact_right = self.opt.paremeter(1)
    # An: Left contact force And right contact force
    self.force_contact_left = self.opt.paremeter(3,self.N)
    self.force_contact_right = self.opt.paremeter(3,self.N)
    # An: Setup multiple shooting
    for i in range(self.N):
      self.opt.subject_to(self.state[:,i+1]== self.state[:,i]+
                           self.delta*self.centroidal_dynamic(self.state,self.com_ref,self.v_com_ref,self.contact_left,self.contact_right,self.force_contact_left,self.force_contact_right))
    

    for i in range(self.N):
      self.opt.subject_to(self.X[:, i + 1] == self.X[:, i] + self.delta * self.f(self.X[:, i], self.U[:, i]))

    cost = cs.sumsqr(self.U) + \
           100 * cs.sumsqr(self.X[2, 1:].T - self.zmp_x_mid_param) + \
           100 * cs.sumsqr(self.X[5, 1:].T - self.zmp_y_mid_param) + \
           100 * cs.sumsqr(self.X[8, 1:].T - self.zmp_z_mid_param)

    self.opt.minimize(cost)

    # zmp constraints
    self.opt.subject_to(self.X[2, 1:].T <= self.zmp_x_mid_param + self.foot_size / 2.)
    self.opt.subject_to(self.X[2, 1:].T >= self.zmp_x_mid_param - self.foot_size / 2.)
    self.opt.subject_to(self.X[5, 1:].T <= self.zmp_y_mid_param + self.foot_size / 2.)
    self.opt.subject_to(self.X[5, 1:].T >= self.zmp_y_mid_param - self.foot_size / 2.)
    self.opt.subject_to(self.X[8, 1:].T <= self.zmp_z_mid_param + self.foot_size / 2.)
    self.opt.subject_to(self.X[8, 1:].T >= self.zmp_z_mid_param - self.foot_size / 2.)

    # initial state constraint
    self.opt.subject_to(self.X[:, 0] == self.x0_param)

    # stability constraint with periodic tail
    self.opt.subject_to(self.X[1, 0     ] + self.eta * (self.X[0, 0     ] - self.X[2, 0     ]) == \
                        self.X[1, self.N] + self.eta * (self.X[0, self.N] - self.X[2, self.N]))
    self.opt.subject_to(self.X[4, 0     ] + self.eta * (self.X[3, 0     ] - self.X[5, 0     ]) == \
                        self.X[4, self.N] + self.eta * (self.X[3, self.N] - self.X[5, self.N]))
    self.opt.subject_to(self.X[7, 0     ] + self.eta * (self.X[6, 0     ] - self.X[8, 0     ]) == \
                        self.X[7, self.N] + self.eta * (self.X[6, self.N] - self.X[8, self.N]))

    # state
    self.x = np.zeros(9)
    self.lip_state = {'com': {'pos': np.zeros(3), 'vel': np.zeros(3), 'acc': np.zeros(3)},
                      'zmp': {'pos': np.zeros(3), 'vel': np.zeros(3)}}

  def solve(self, current, t):
    self.x = np.array([current['com']['pos'][0], current['com']['vel'][0], current['zmp']['pos'][0],
                       current['com']['pos'][1], current['com']['vel'][1], current['zmp']['pos'][1],
                       current['com']['pos'][2], current['com']['vel'][2], current['zmp']['pos'][2]])
    
    mc_x, mc_y, mc_z = self.generate_moving_constraint(t)

    # solve optimization problem
    self.opt.set_value(self.x0_param, self.x)
    self.opt.set_value(self.zmp_x_mid_param, mc_x)
    self.opt.set_value(self.zmp_y_mid_param, mc_y)
    self.opt.set_value(self.zmp_z_mid_param, mc_z)

    sol = self.opt.solve()
    self.x = sol.value(self.X[:,1])
    self.u = sol.value(self.U[:,0])

    self.opt.set_initial(self.U, sol.value(self.U))
    self.opt.set_initial(self.X, sol.value(self.X))

    # create output LIP state
    # Change the index to take out the result bcz of different order defined in the dynamics model
    self.lip_state['com']['pos'] = np.array([self.x[0], self.x[1], self.x[2]])
    self.lip_state['com']['vel'] = np.array([self.x[3], self.x[4], self.x[5]])
    self.lip_state['zmp']['pos'] = np.array([self.x[2], self.x[5], self.x[8]])
    self.lip_state['zmp']['vel'] = self.u
    self.lip_state['com']['acc'] = self.eta**2 * (self.lip_state['com']['pos'] - self.lip_state['zmp']['pos']) + np.hstack([0, 0, - self.params['g']])

    contact = self.footstep_planner.get_phase_at_time(t)
    if contact == 'ss':
      contact = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']

    return self.lip_state, contact
  
  def generate_moving_constraint(self, t):
    mc_x = np.full(self.N, (self.initial['lfoot']['pos'][3] + self.initial['rfoot']['pos'][3]) / 2.)
    mc_y = np.full(self.N, (self.initial['lfoot']['pos'][4] + self.initial['rfoot']['pos'][4]) / 2.)
    time_array = np.array(range(t, t + self.N))
    for j in range(len(self.footstep_planner.plan) - 1):
      fs_start_time = self.footstep_planner.get_start_time(j)
      ds_start_time = fs_start_time + self.footstep_planner.plan[j]['ss_duration']
      fs_end_time = ds_start_time + self.footstep_planner.plan[j]['ds_duration']
      fs_current_pos = self.footstep_planner.plan[j]['pos'] if j > 0 else np.array([mc_x[0], mc_y[0]])
      fs_target_pos = self.footstep_planner.plan[j + 1]['pos']
      mc_x += self.sigma(time_array, ds_start_time, fs_end_time) * (fs_target_pos[0] - fs_current_pos[0])
      mc_y += self.sigma(time_array, ds_start_time, fs_end_time) * (fs_target_pos[1] - fs_current_pos[1])

    return mc_x, mc_y, np.zeros(self.N)
  #An's function: Compute the centroidal dynamic
  # Input:
  # 1. state: com, dcom, angularmomentum, thetahat: casadi opt.variable | contact[pos: casadi opt.variable,onGround:logic]
  #(here I am writing for contact with one vertex --> develop later for multiple vertex)
  # 2. input: contact force: casadi opt.variable
  # 3. input: CoM ref, vCoM ref
  # Output:
  # State derivative formular for multiple shooting  
  def centroidal_dynamic(self, state, CoM_ref, vCoM_ref,contact_lef, contact_right, force_left, force_right,input):
    k1=1
    mass = self.params('mass')
    g = np.array([0, 0,- self.params['g']])
    com=state[0:3]
    pos_left= state[12:15]
    pos_right= state[15:18]
    vel_left= state[18:21]
    vel_right= state[21:24]
    #size=self.N
    # Centroidal dynamic with disturbance estimator theta hat, contact dynamics
    dcom=state[3:6]#state[3],state[4],state[5]
    ddcom= 1/mass*(g+force_left*contact_lef+force_right*contact_right)
    dhw= np.cross(com-pos_left)*force_left+np.cross(com-pos_right)*force_right
    v_left= (1-contact_lef)*vel_left
    v_right= (1-contact_right)*vel_right
    dthetahat= k1*(com-CoM_ref)+dcom-vCoM_ref

    return cs.vertcart(dcom,ddcom,dhw,dthetahat,v_left,v_right)