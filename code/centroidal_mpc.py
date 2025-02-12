import numpy as np
import casadi as cs

class centroidal_mpc:
  def __init__(self, initial, footstep_planner, params, CoM_ref):
    # parameters
    self.params = params
    self.N = params['N']-90
    self.delta = params['world_time_step']
    self.h = params['h']
    self.eta = params['eta']
    self.foot_size = params['foot_size']
    self.mass = params['mass']
    self.g = params['g']
    self.initial = initial
    self.footstep_planner = footstep_planner
    self.sigma = lambda t, t0, t1: np.clip((t - t0) / (t1 - t0), 0, 1) # piecewise linear sigmoidal function
    self.k1=0.1
    self.k2=0.5

    # lip model matrices
    # self.A_lip = np.array([[0, 1, 0], [self.eta**2, 0, -self.eta**2], [0, 0, 0]])
    # self.B_lip = np.array([[0], [0], [1]])

    #An: Get the CoM_ref data from Daniele --> thanks
    #self.CoM_ref_planner=CoM_ref

    self.pos_com_ref_x= CoM_ref['pos_x']
    self.pos_com_ref_y= CoM_ref['pos_y']
    self.pos_com_ref_z= CoM_ref['pos_z']

    self.vel_com_ref_x= CoM_ref['vel_x']
    self.vel_com_ref_y= CoM_ref['vel_y']
    self.vel_com_ref_z= CoM_ref['vel_z']

    self.acc_com_ref_x= CoM_ref['acc_x']
    self.acc_com_ref_y= CoM_ref['acc_y']
    self.acc_com_ref_z= CoM_ref['acc_z']
    print("shape_com:")
    print(self.acc_com_ref_z[300])
    print(self.acc_com_ref_z.shape)
    #An: Get all the foot step ref from foot step planner over time stamp
    self.pos_contact_ref_l= footstep_planner.contacts_ref['contact_left']
    self.pos_contact_ref_r= footstep_planner.contacts_ref['contact_right']
    print("shape_contact:")
    print(self.pos_contact_ref_l)
    
    with open("pos_contact_ref_l", "w") as file:
      file.writelines(" \n".join(map(str, self.pos_contact_ref_l)))

    #print(self.pos_contact_ref_l.shape)

    #print(CoM_ref)
    # optimization problem setup
    self.opt = cs.Opti('conic')
    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000, "verbose": False}
    
    self.opt.solver("osqp", p_opts, s_opts)
    
    
  #An: Create optimization variable: prefix "opti_" denotes as symbolic variable
    # An: Left vel And right vel: Components of control input -> Decide by mpc
    self.opti_vel_contact_l= self.opt.variable(3, self.N)
    self.opti_vel_contact_r= self.opt.variable(3, self.N)
    # An: Left contact force And right contact force: Components of control input -> Decide by mpc
    self.opti_force_contact_l = self.opt.variable(3,self.N)
    self.opti_force_contact_r = self.opt.variable(3,self.N)

    self.U = cs.vertcat(self.opti_force_contact_l,self.opti_force_contact_r,
                        self.opti_vel_contact_l,self.opti_vel_contact_r)

    #Define the state, centroidal dynamic model. thetahat plays a role as an observer
    self.opti_CoM = self.opt.variable(3, self.N + 1)
    self.opti_dCoM = self.opt.variable(3, self.N + 1)
    self.opti_hw = self.opt.variable(3, self.N + 1)
    self.opti_thetahat = self.opt.variable(3, self.N + 1)
    self.opti_pos_contact_l= self.opt.variable(3, self.N + 1)
    self.opti_pos_contact_r= self.opt.variable(3, self.N + 1)
    
    #An: Concatenate them in to the state matrix (3*num_state, self.N+1)
    self.opti_state= cs.vertcat(self.opti_CoM,self.opti_dCoM,self.opti_hw,self.opti_thetahat,
                          self.opti_pos_contact_l,self.opti_pos_contact_r)

  #An: Create optimization params that change during simulation time
    #An Initial state at the beginning of the mpc horizion
    self.opti_x0_param = self.opt.parameter(18) # update every step based on the current value obtained by the simulator (column vector)
    
    #An: Reference Com trj that is a C2 function
    self.opti_com_ref = self.opt.parameter(3*3,self.N) #including pos x,y,z, vel x,y,z, acc x,y,z ref, update every step based on the pre-planner
    #An: Reference contact points, taken from the footstep planner
    self.opti_pos_contact_l_ref = self.opt.parameter(3,self.N)
    self.opti_pos_contact_r_ref = self.opt.parameter(3,self.N)
    
    # self.v_com_ref = self.opt.parameter(3,self.N)
    # self.acc_com_ref = self.opt.parameter(3,self.N)

    #An: to track the contact status of the left and right foot 1 means in contact
    # If the horizon of mpc N=100 then the contact phase will not change it this horizon because ds+ss>=100
    # If N>100, contact phase has changed, need more number of param to capture that change
    # The change in the contact phase inside the mpc horizon will affect the dynamic constraint
    self.opti_contact_left = self.opt.parameter(1)
    self.opti_contact_right = self.opt.parameter(1)

    #An: Setup multiple shooting:
    #An: Dynamic constraints
    self.opt.subject_to(self.opti_state[:,0]==self.opti_x0_param)
    for i in range(self.N):
      self.opt.subject_to(self.opti_state[:,i+1]== self.opti_state[:,i]+
                           self.delta*self.centroidal_dynamic(self.opti_state[:,i],self.opti_com_ref[:,i],self.opti_contact_left,self.opti_contact_right,self.U[:,i]))
    
    #An: Formulate the change coordinate
    z1= self.opti_CoM[:,1:]-self.opti_com_ref[0:3,:]
    z2= self.k1*z1+self.opti_dCoM[:,1]-self.opti_com_ref[3:6,:]
    
    force_sum = self.opti_force_contact_l+self.opti_force_contact_r
    gravity = cs.GenDM_zeros(3)
    gravity[2]=-self.g
    #An: Adaptive force u_n
    u_n= self.k1*self.k1*z1-(self.k1+self.k2)*z2- gravity +self.opti_com_ref[6:9,:]-self.opti_thetahat[:,1:]

    #An: Lyapunov stability constrains
    for i in range(self.N):
      self.opt.subject_to(-z1[:,i].T@(self.k1*z1[:,i])-z2[:,i].T@(self.k2*z2[:,i])+z1[:,i].T@z2[:,i]+z2[:,i].T@(force_sum-u_n)<=0.0)

    #An: angular momentum constraint:
    for i in range(self.N):
      self.opt.subject_to(self.opti_hw[:,i].T@self.opti_hw[:,i]<=0.1)
    


    
    # Define the cost function
    # still lack of the components to minimize the deviation of forces at the foot vertices (aka foot corners)
    cost = 10*cs.sumsqr(self.opti_hw[:,1:]) + \
           1*cs.sumsqr(self.opti_CoM[:,1:]-self.opti_com_ref[0:3,:])+\
           10*cs.sumsqr(self.opti_pos_contact_l[:,1:]-self.opti_pos_contact_l_ref)+\
           10*cs.sumsqr(self.opti_pos_contact_r[:,1:]-self.opti_pos_contact_r_ref)
           

    self.opt.minimize(cost)

    #An: initialize the state space to collect the real time state value from the simulator
    self.current_state = np.zeros(3*6)
    self.lip_state = {'com': {'pos': np.zeros(3), 'vel': np.zeros(3), 'acc': np.zeros(3)},
                      'zmp': {'pos': np.zeros(3), 'vel': np.zeros(3)}}
    
# Solve the mpc every time step --> That will be very tough
# Main tasks are updating the current state at the beginning of the horizon and
# let the mpc compute the state in the rest of the horizon
#
  def solve(self, current, t):
    #array = row vector
    self.current_state = np.array([current['com']['pos'][0],       current['com']['pos'][1],       current['com']['pos'][2],
                                   current['com']['vel'][0],       current['com']['vel'][1],       current['com']['vel'][2],
                                   current['hw']['val'][0],        current['hw']['val'][1],        current['hw']['val'][2],
                                   current['theta_hat']['val'][0], current['theta_hat']['val'][1], current['theta_hat']['val'][2],
                                   current['lfoot']['pos'][0],     current['lfoot']['pos'][1],     current['lfoot']['pos'][2],
                                   current['rfoot']['pos'][0],     current['rfoot']['pos'][1],     current['rfoot']['pos'][2],])
    
    
    #An: Update the initial state contrainst
    self.opt.set_value(self.opti_x0_param, self.current_state)

    # mc_x, mc_y, mc_z = self.generate_moving_constraint(t)

    #An: Extract the status of the contacts
    #An: Update the "real time" contact phase from contact planner list
    #t= self.time
    contact_status = self.footstep_planner.get_phase_at_time(t)
    #An: Update contact status to the model constrains
    #'ds'
    self.opt.set_value(self.opti_contact_left, 1)
    self.opt.set_value(self.opti_contact_right, 1)
    if contact_status == 'ss':
      self.opt.set_value(self.opti_contact_left, 0)
      self.opt.set_value(self.opti_contact_right, 1)
      contact_status = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']
      if contact_status=='lfoot':
        self.opt.set_value(self.opti_contact_left, 1)
        self.opt.set_value(self.opti_contact_right, 0)

    #An: Update CoM_ref value for every step t and in an entire horizon N=100
    pos_com_ref_x= self.pos_com_ref_x[t+1:t+1+self.N]
    pos_com_ref_y= self.pos_com_ref_y[t+1:t+1+self.N]
    pos_com_ref_z= self.pos_com_ref_z[t+1:t+1+self.N]

    vel_com_ref_x= self.vel_com_ref_x[t+1:t+1+self.N]
    vel_com_ref_y= self.vel_com_ref_y[t+1:t+1+self.N]
    vel_com_ref_z= self.vel_com_ref_z[t+1:t+1+self.N]

    acc_com_ref_x= self.acc_com_ref_x[t+1:t+1+self.N]
    acc_com_ref_y= self.acc_com_ref_y[t+1:t+1+self.N]
    acc_com_ref_z= self.acc_com_ref_z[t+1:t+1+self.N]

    com_ref_sample_horizon= np.vstack((pos_com_ref_x,pos_com_ref_y,pos_com_ref_z,
                                      vel_com_ref_x,vel_com_ref_y,vel_com_ref_z,
                                      acc_com_ref_x,acc_com_ref_y,acc_com_ref_z))

    self.opt.set_value(self.opti_com_ref,com_ref_sample_horizon)

    #An: Update pos_contact_ref value for every step t and in an entire horizon N=100
    
    pos_contact_ref_l = self.pos_contact_ref_l[t+1:t+1+self.N].T
    #print(pos_contact_ref_l)
    pos_contact_ref_r = self.pos_contact_ref_r[t+1:t+1+self.N].T

    self.opt.set_value(self.opti_pos_contact_l_ref,pos_contact_ref_l)
    self.opt.set_value(self.opti_pos_contact_r_ref,pos_contact_ref_r)
    # solve optimization problem
    
    # self.opt.set_value(self.zmp_x_mid_param, mc_x)
    # self.opt.set_value(self.zmp_y_mid_param, mc_y)
    # self.opt.set_value(self.zmp_z_mid_param, mc_z)

    sol = self.opt.solve()
    self.x = sol.value(self.opti_state[:,1])
    self.u = sol.value(self.U[:,0])

    self.opt.set_initial(self.U, sol.value(self.U))
    self.opt.set_initial(self.opti_state, sol.value(self.opti_state))

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
  def centroidal_dynamic(self, state,CoM_ref,contact_lef, contact_right,input):
    k1=self.k1
    mass = self.mass
    g = np.array([0, 0,- self.g])
    #print(g)
    g=g.T
    gravity = cs.GenDM_zeros(3)
    gravity[2]=-self.g
    #Extract states
    com=state[0:3]
    pos_left= state[12:15]
    pos_right= state[15:18]
    #Extract inputs
    force_left=input[0:3]
    force_right=input[3:6]

    vel_left= input[6:9]
    vel_right= input[9:12]
    
    CoM_ref_pos= cs.vertcat(CoM_ref[0],CoM_ref[1],CoM_ref[2])
    #CoM_ref_pos=CoM_ref_pos.T

    CoM_ref_vel= cs.vertcat(CoM_ref[3],CoM_ref[4],CoM_ref[5])
    #CoM_ref_vel=CoM_ref_vel.T
    #size=self.N
    # Centroidal dynamic with disturbance estimator theta hat, contact dynamics
    dcom=state[3:6] #state[3],state[4],state[5]
    ddcom= g+1/mass*(force_left*contact_lef+force_right*contact_right)
    dhw= cs.cross(com-pos_left,force_left)*contact_lef+cs.cross(com-pos_right,force_right)*contact_right
    v_left= (1-contact_lef)*vel_left
    v_right= (1-contact_right)*vel_right
    dthetahat= k1*(com-CoM_ref_pos)+dcom-CoM_ref_vel

    return cs.vertcat(dcom,ddcom,dhw,dthetahat,v_left,v_right)