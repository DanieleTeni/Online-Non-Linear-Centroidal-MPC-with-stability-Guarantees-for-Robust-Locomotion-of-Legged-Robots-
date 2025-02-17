import numpy as np
import casadi as cs

class centroidal_mpc:
  def __init__(self, initial, footstep_planner, params, CoM_ref,foot_ref):
    # parameters
    self.params = params
    self.N = 25
    self.delta = params['world_time_step']
    self.h = params['h']
    self.eta = params['eta']
    self.foot_size = params['foot_size']
    self.mass = params['mass']
    self.g = params['g']
    self.initial = initial
    self.footstep_planner = footstep_planner
    # self.sigma = lambda t, t0, t1: np.clip((t - t0) / (t1 - t0), 0, 1) # piecewise linear sigmoidal function
    self.k1=0.1
    self.k2=0.5
    self.left=foot_ref['lfoot']['pos']
    self.left_vel=foot_ref['lfoot']['vel']
    self.right=foot_ref['rfoot']['pos']
    self.right_vel=foot_ref['rfoot']['vel']


    self.pos_com_ref_x= CoM_ref['pos_x']
    self.pos_com_ref_y= CoM_ref['pos_y']
    self.pos_com_ref_z= CoM_ref['pos_z']

    self.vel_com_ref_x= CoM_ref['vel_x']
    self.vel_com_ref_y= CoM_ref['vel_y']
    self.vel_com_ref_z= CoM_ref['vel_z']

    self.acc_com_ref_x= CoM_ref['acc_x']
    self.acc_com_ref_y= CoM_ref['acc_y']
    self.acc_com_ref_z= CoM_ref['acc_z']

    self.opt = cs.Opti('nlp')  ##  coinic not indicated for non linear constrain   so used nlp or something else
    p_opts = {"expand": True}
    # s_opts = {"max_iter": 1000}
    s_opts = {"max_iter": 1000, "print_level": 5}  # 0 to suppress output, 1 for basic, 5 for detailed


    
    self.opt.solver("ipopt", p_opts, s_opts)
    
    
  #An: Create optimization variable: prefix "opti_" denotes as symbolic variable
    # An: Left vel And right vel: Components of control input -> Decide by mpc




    self.U = self.opt.variable(12, self.N)
    self.X = self.opt.variable(15, self.N)  #19 if include theta hat
    ######  self.X contain state of the system :                                        ||||||    self.U input  of system
    #####  total 18 state                                                               ||||||     total 6 +6 = 12 state
    ######  X[0 : 3] = x,y,z component of Com                                           ||||||     
    ######  X[3 : 6]  = x,y,z component of dcom*self.mas = linear momuntum              ||||||     U[0:3] force acting on left foot
    ######  X[6 : 9]  = x,y,z componet of hw (angular momentum)                         ||||||     U[3:6] force acting on right foot
    ######  X[9 : 12]  = p(s) rappresent the position of the left foot at the state s   ||||||     U[6:9] velocity of left foot   (not sure we need it)
    ######  X[12 : 15]  = p(s) rappresent the position of the right foot at the state s ||||||     U[9:12] velocity of right foot  (not sure we need it)
    ######  X[15 : 18] = theta hat 
    self.x0_param = self.opt.parameter(15)

    self.z1 = self.opt.parameter(self.N)
    self.z2=self.opt.parameter(self.N)
    gravity = cs.DM([0, 0, -self.g])



    #gravity = np.hstack([0, 0, - self.params['g']]


    self.ref_traj=self.opt.parameter(9,self.N)   # [0:3,:] are x,y,z component of the reference com
                                                  ## [0:6,:]  //     //                //   reference velocity com
                                                  ## [6 :9,:]  //    //               //     reference acceleration com
    
    
    self.ref_point=self.opt.parameter(12,self.N)   # [0:3] contain position left foot accordig to reference
                                                  # [3:6] //    //          right  //                 // 
                                                  # [6:9]  //    velocity   left  //                 //
                                                  # [9:12]  //    velocity   right  //                 //




    # self.Gamma=self.opt.parameter(2,self.N)
    self.Gamma_real=self.opt.parameter(2,self.N)

    # tau_L = cs.cross(self.X[9:12, i] - self.X[0:3, i], self.U[0:3, i]) 
    # tau_R = cs.cross(self.X[12:15, i] - self.X[0:3, i], self.U[3:6, i]) 
    # self.opt.subject_to(self.X[6:9, i + 1] == self.X[6:9, i] + self.delta * (tau_L + tau_R))


    ######## DYNAMIC EQUATION ###############################
    for i in range(self.N-1):
      self.opt.subject_to(self.X[0:3, i + 1] == self.X[0:3, i] + self.delta *self.X[3:6,i])
      self.opt.subject_to(self.X[3:6, i + 1] == self.X[3:6, i] + self.delta *(self.U[0:3,i]+self.U[3:6,i])/self.mass +gravity)   ###not sure ...............
      self.opt.subject_to(self.X[6:9, i + 1] == self.X[6:9, i] + self.delta *(cs.cross(self.X[9:12,i]-self.X[0:3,i],self.U[0:3,i])+cs.cross(self.X[12:15, i] - self.X[0:3, i], self.U[3:6, i])))
      self.opt.subject_to(self.X[9:12, i+1] == self.X[9:12, i] + self.delta* self.U[6:9, i])
      self.opt.subject_to(self.X[12:15,i+1]== self.X[12:15,i] +self.delta*self.U[9:12,i])
######################################################################




     ##change of variable ############ààà
    z1= self.X[0:3,:]-self.ref_traj[0:3,:]
    z2= self.k1*z1+self.X[3:6,:]-self.ref_traj[3:6,:]

      ####### const function  ####################

    cost = 10 * cs.sumsqr(self.X[6:9,:]) + \
           10 * cs.sumsqr(z1)+ \
           1000 * cs.sumsqr(self.X[9:12,:] - 
                      self.ref_point[0:3,:] ) + \
            100 * cs.sumsqr(self.X[12:15,:]- 
                      self.ref_point[6:9,:])+ \
            5 * cs.sumsqr(self.U[6:9,:]-self.ref_point[6:9,:]) + \
            5* cs.sumsqr(self.U[9:12,:]-self.ref_point[9:12,:])
                           


      #############################

    #An: Adaptive force u_n
    u_n= self.k1*self.k1*z1-(self.k1+self.k2)*z2- gravity +self.ref_traj[6:9,:]#-self.opti_thetahat[:,1:]

              # 
    for i in range(self.N):
      self.opt.subject_to(-z1[:,i].T@(self.k1*z1[:,i])-z2[:,i].T@(self.k2*z2[:,i])+z1[:,i].T@z2[:,i]+z2[:,i].T@(self.U[0:3,i]+self.U[3:6,i]-u_n)<=0.0)

    for i in range(1,self.N) :
      self.opt.subject_to(self.X[6:9,i].T@self.X[6:9,i] <= self.X[6:9,i-1].T@self.X[6:9,i-1])


    #An: initialize the state space to collect the real time state value from the simulator
    self.x = np.zeros(15)
    self.controller = {'com': {'pos': np.zeros(3), 'vel': np.zeros(3), 'acc': np.zeros(3)},
                      'hw':{'val':np.zeros(3),'derivative':np.zeros(3)},
                      'lfoot' :{'pos': np.zeros(3 ),'vel':np.zeros(3)},
                      'rfoot' :{'pos': np.zeros(3 ),'vel':np.zeros(3)},
                      'input_forces': {'u1':np.zeros(3),'u2':np.zeros(3)}
                      
                      }
#
  def solve(self, current, t):
    #array = row vector
    self.current_state = np.array([current['com']['pos'][0],       current['com']['pos'][1],       current['com']['pos'][2],
                                   current['com']['vel'][0],       current['com']['vel'][1],       current['com']['vel'][2],
                                   current['hw']['val'][0],        current['hw']['val'][1],        current['hw']['val'][2],
                                   #current['theta_hat']['val'][0], current['theta_hat']['val'][1], current['theta_hat']['val'][2],
                                    current['lfoot']['pos'][0],     current['lfoot']['pos'][1],     current['lfoot']['pos'][2],
                                    current['rfoot']['pos'][0],     current['rfoot']['pos'][1],     current['rfoot']['pos'][2]]
                                    )
    com_ref_sample_horizon=np.zeros((9, self.N))
    for i,name in enumerate([self.pos_com_ref_x,self.pos_com_ref_y,self.pos_com_ref_z,self.vel_com_ref_x,self.vel_com_ref_y,self.vel_com_ref_z,self.acc_com_ref_x,self.acc_com_ref_y,self.acc_com_ref_z]):
      com_ref_sample_horizon[i,:]=name[t:t+self.N]
    point_ref_sample_horizon=np.zeros((12, self.N))
    for i,name in enumerate([self.left,self.right,self.left_vel,self.right_vel]):
      point_ref_sample_horizon[i:i+3,:]=name[:,t:t+self.N]
    
    
    #An: Update the initial state contrainst
    # self.opt.set_value(self.X, self.current_state)
    

    # Gamma_ = np.zeros((2, self.N)) 
    #  for i in range(self.N):                                                      ######### in the real code we have to use self.Gamma 
    #     Gamma_[0, i] = self.X[11, i] < 0.01, 1, 0)                                ###########  instead of self.Gamma_real 
    # #    Gamma_[1, i] = cs.if_else(self.X[14, i] < 0.01, 1, 0)


    # self.opt.set_value(self.Gamma, Gamma_)

    Gamma_real = np.zeros((2, self.N))
    for i in range(self.N):
      Gamma_real[0, i] = 0 if self.left[2,i] < 0.0001 else 1
      Gamma_real[1, i] = 0 if self.right[2,i] < 0.0001 else 1

    self.opt.set_value(self.Gamma_real, Gamma_real)
    
    
    self.opt.set_value(self.ref_point,point_ref_sample_horizon)

    self.opt.set_value(self.x0_param, self.x)


    self.opt.set_value(self.ref_traj,com_ref_sample_horizon)


    sol = self.opt.solve()
    self.x = sol.value(self.X[:,1])
    self.u = sol.value(self.U[:,0])


    self.opt.set_initial(self.U, sol.value(self.U))
    self.opt.set_initial(self.X, sol.value(self.X))



    # create output LIP state
    # Change the index to take out the result bcz of different order defined in the dynamics model
    self.controller['com']['pos'] = np.array([self.x[0], self.x[1], self.x[2]])
    self.controller['com']['vel'] = np.array([self.x[3], self.x[4], self.x[5]])
    self.controller['hw']['val']=np.array([self.x[6], self.x[7], self.x[8]])
    self.controller['lfoot']['pos'] = np.array([self.x[9], self.x[10], self.x[11]])
    self.controller['rfoot']['pos'] = np.array([self.x[12], self.x[13], self.x[14]])
    self.controller['lfoot']['vel'] = np.array([self.u[6], self.u[7], self.u[8]])
    self.controller['rfoot']['vel'] = np.array([self.u[9], self.u[10], self.u[11]])
    self.controller['input_forces']['u1'] = np.array([self.u[0], self.u[1], self.u[2]])
    self.controller['input_forces']['u2'] = np.array([self.u[3], self.u[4], self.u[5]])
    self.controller['com']['acc'] =  (self.controller['com']['vel'] + self.controller['input_forces']['u1']+self.controller['input_forces']['u2']) + np.hstack([0, 0, - self.params['g']])

    contact = self.footstep_planner.get_phase_at_time(t)
    if contact == 'ss':
      contact = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']

    return self.controller, contact
  