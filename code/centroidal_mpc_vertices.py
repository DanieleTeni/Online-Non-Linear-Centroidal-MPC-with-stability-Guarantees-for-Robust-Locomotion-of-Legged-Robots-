import numpy as np
import casadi as cs
import os
from scipy.spatial.transform import Rotation as R

class centroidal_mpc:
  def __init__(self, initial, footstep_planner, params, CoM_ref, contact_trj_l, contact_trj_r):
    # parameters
    self.params = params
    self.N = params['N']
    self.delta = params['world_time_step']*params['mpc_rate']
    self.h = params['h']
    self.eta = params['eta']
    self.foot_size = params['foot_size']
    self.mass = params['mass']
    print("total mass:")
    print(self.mass)
    self.g = params['g']
    self.initial = initial
    self.footstep_planner = footstep_planner
   
    self.debug_folder= "Debug"
    self.debug = 0
    self.update_contact_flag = 0
    #Change of coordinates Gains
    
    self.k1=4# 4-0.1 rate 1// 5-0.1 rate 10 6-1
    self.k2=0.1
    if params['mpc_rate'] == 10:
      self.k1=5
      self.k2=0.2

    self.mpc_rate = params['mpc_rate']
    self.correction=1
    #(t+self.N-self.correction)
    self.update_swing_trj=0
   
  
    self.next_des_pose_swing_MPC_at_69=np.zeros(3)
    #To build the Non-slippage constraints
    mu= 0.5
    
    
    self.A=cs.DM([[1,0.,-mu],
                  [-1,0.,-mu],
                  [0.,1,-mu],
                  [0.,-1,-mu] ])
    self.b=cs.GenDM_zeros(4)

    # Foot dimensions (in meters)
    FOOT_LENGTH = 0.25  # 25 cm
    FOOT_WIDTH  = 0.13  # 13 cm

    # Define the foot vertices in local coordinates (relative to the foot center)
    self.foot_polygon_local = np.array([
        [ FOOT_LENGTH/2,  FOOT_WIDTH/2,  0.0],  # Front-right corner
        [ FOOT_LENGTH/2, -FOOT_WIDTH/2,  0.0],  # Front-left corner
        [-FOOT_LENGTH/2, -FOOT_WIDTH/2,  0.0],  # Back-left corner
        [-FOOT_LENGTH/2,  FOOT_WIDTH/2,  0.0]   # Back-right corner
    ])


    #Get the CoM_ref data
    self.pos_com_ref_x= CoM_ref['pos_x']
    self.pos_com_ref_y= CoM_ref['pos_y']
    self.pos_com_ref_z= CoM_ref['pos_z']

    self.vel_com_ref_x= CoM_ref['vel_x']
    self.vel_com_ref_y= CoM_ref['vel_y']
    self.vel_com_ref_z= CoM_ref['vel_z']

    self.acc_com_ref_x= CoM_ref['acc_x']
    self.acc_com_ref_y= CoM_ref['acc_y']
    self.acc_com_ref_z= CoM_ref['acc_z']
    
    #Get all the foot step ref from foot step planner over time stamp
    self.pose_contact_ref_l= footstep_planner.position_contacts_ref['contact_left']
    self.pose_contact_ref_r= footstep_planner.position_contacts_ref['contact_right']

    self.pos_contact_ref_l = self.pose_contact_ref_l[:, 3:6]  # Position [x, y, z]
    self.pos_contact_ref_r = self.pose_contact_ref_r[:, 3:6]  # Position [x, y, z]

    self.rotvec_contact_ref_l = self.pose_contact_ref_l[:,2]  # angle around z
    self.rotvec_contact_ref_r = self.pose_contact_ref_r[:,2]  # angle around z

    if self.debug==1:
      file_path=os.path.join(self.debug_folder, "MPC_pose_contact_ref")        
      with open(file_path, "w") as file:
        # for i in range(len(self.pose_contact_ref_l)):  # Loop through all time steps
        #     # Extract rotation vector and position for the left foot
        #     rotvec_l = self.rotvec_contact_ref_l[i]
        #     pos_l = self.pos_contact_ref_l[i]

        #     # Extract rotation vector and position for the right foot
        #     rotvec_r = self.rotvec_contact_ref_r[i]
        #     pos_r = self.pos_contact_ref_r[i]

        #     # Format the line as requested
        #     line = f"{i}   lfoot: {rotvec_l.tolist()} {pos_l.tolist()}   rfoot: {rotvec_r.tolist()} {pos_r.tolist()}\n"

        #     # Write the formatted line to file
        #     file.write(line)
        for i in range(self.N):
          file.write("\n".join(map(str, self.footstep_planner.plan[i]['pos'])))
          file.write("end")

    
    #store the (ang,pos,vel,acc) data of the foot over time, they changes every 1u and they are the ref.
    self.contact_trj_l=contact_trj_l
    self.contact_trj_r=contact_trj_r
    #to access at the position of the feets we need      self.pre_left_traj[2499][0]['pos'][3:6]} last position
    if self.debug==1:
      file_path=os.path.join(self.debug_folder, "contact_trj_from_centroidal_MPC")        
      with open(file_path, "w") as file:
        for i in range(len(self.contact_trj_l)):  # Assumo che abbiano la stessa lunghezza
          left_pos = " ".join(map(str, self.contact_trj_l[i][0]['pos'][0:6]))
          left_vel = " ".join(map(str, self.contact_trj_l[i][0]['vel'][0:6]))
          right_pos = " ".join(map(str, self.contact_trj_r[i][0]['pos'][0:6]))
          right_vel = " ".join(map(str, self.contact_trj_r[i][0]['vel'][0:6]))

          file.write(f"({i})\tLfoot_POSE: {left_pos}\tRfoot_POSE: {right_pos}\n")
          file.write(f"({i})\tLfoot_VEL: {left_vel}\tRfoot_VEL: {right_vel}\n\n")

    
    # optimization problem setup
    self.opt = cs.Opti()
    p_opts = {"expand": True,"print_time":False}
    s_opts = {"max_iter": 1000,"print_level": False,"tol":0.001}
    #Set up a proper optimal solver
    self.opt.solver('ipopt',p_opts,s_opts) 

  
    
    #INPUT variables, to be optimized     
    self.opti_v1l_force=self.opt.variable(3,self.N)
    self.opti_v2l_force=self.opt.variable(3,self.N)
    self.opti_v3l_force=self.opt.variable(3,self.N)
    self.opti_v4l_force=self.opt.variable(3,self.N)
    self.opti_v1r_force=self.opt.variable(3,self.N)
    self.opti_v2r_force=self.opt.variable(3,self.N)
    self.opti_v3r_force=self.opt.variable(3,self.N)
    self.opti_v4r_force=self.opt.variable(3,self.N)
    self.opti_vel_contact_l= self.opt.variable(3, self.N)
    self.opti_vel_contact_r= self.opt.variable(3, self.N)
    self.opti_omega_contact_l= self.opt.variable(1,self.N)
    self.opti_omega_contact_r= self.opt.variable(1,self.N)


    self.U = cs.vertcat(self.opti_v1l_force,self.opti_v2l_force,self.opti_v3l_force,self.opti_v4l_force,
                        self.opti_v1r_force,self.opti_v2r_force,self.opti_v3r_force,self.opti_v4r_force,
                        self.opti_vel_contact_l,self.opti_vel_contact_r,
                        self.opti_omega_contact_l,self.opti_omega_contact_r)
    
    #STATE -> Variables object of the optimization problem (will impose two constraints)
    self.opti_CoM = self.opt.variable(3, self.N + 1)
    self.opti_dCoM = self.opt.variable(3, self.N + 1)
    self.opti_hw = self.opt.variable(3, self.N + 1)
    self.opti_thetahat = self.opt.variable(3, self.N + 1)
    self.opti_ang_contact_l= self.opt.variable(1, self.N + 1)
    self.opti_ang_contact_r= self.opt.variable(1, self.N + 1)
    self.opti_pos_contact_l= self.opt.variable(3, self.N + 1)
    self.opti_pos_contact_r= self.opt.variable(3, self.N + 1)
    
    self.opti_state= cs.vertcat(self.opti_CoM,self.opti_dCoM,self.opti_hw,self.opti_thetahat,
                          self.opti_ang_contact_l,self.opti_pos_contact_l,
                          self.opti_ang_contact_r,self.opti_pos_contact_r)

                        #Paramteres Needed to solve the optimization problem
    #INITIAL STATE
    self.opti_x0_param = self.opt.parameter(20) # update every step based on the current value obtained by the simulator (column vector)
    #CoM trajectories for the change of coordinates 
    self.opti_com_ref = self.opt.parameter(3+3+3,self.N) #including pos x,y,z, vel x,y,z, acc x,y,z ref, update every step based on the pre-planner
    #DESIRED POSITION OF THE CONTACT POINT and orientation - to put into cost function
    self.opti_pos_contact_l_ref = self.opt.parameter(3,self.N)
    self.opti_pos_contact_r_ref = self.opt.parameter(3,self.N)
    self.opti_ang_contact_l_ref = self.opt.parameter(1,self.N)
    self.opti_ang_contact_r_ref = self.opt.parameter(1,self.N)
    #GAMMA_L
    self.opti_contact_left = self.opt.parameter(1,self.N+1)
    #GAMMA_R
    self.opti_contact_right = self.opt.parameter(1,self.N+1)


    #CONSTRAINTS ON THE STATE, centroidal dynamics
    self.opt.subject_to(self.opti_state[:,0]==self.opti_x0_param) #Initial constraint
    #Centroidal Dynamic constraints in all the horizon self.N
    for i in range(self.N):
      self.opt.subject_to(self.opti_state[:,i+1] == self.opti_state[:,i]+
                           self.delta*self.centroidal_dynamic(self.opti_state[:,i],self.opti_com_ref[:,i],
                                        self.opti_contact_left[i],self.opti_contact_right[i],self.U[:,i]))
    
    #Change of coordinates at the first step
    self.z1_mat = cs.MX.zeros(3,self.N)
    self.z2_mat = cs.MX.zeros(3,self.N)
    self.u_n_mat = cs.MX.zeros(3,self.N)
    

    gravity = cs.GenDM_zeros(3)
    gravity[2]=-self.g

    
    for i in range(self.N):        
      self.z1_mat[:,i] = self.opti_CoM[:,i+1]-self.opti_com_ref[0:3,i]
      self.z2_mat[:,i] = self.k1*(self.z1_mat[:,i])+(self.opti_dCoM[:,i+1]-self.opti_com_ref[3:6,i])

    for i in range(self.N):
      self.u_n_mat[:,i]=-(self.k1+self.k2)*self.z2_mat[:,i]+\
                        self.k1*self.k1*self.z1_mat[:,i]-gravity+self.opti_com_ref[6:9,i]*1-self.opti_thetahat[:,i]/self.mass

    #Total linear force evaluated by the MPCs
    self.Vl_mat = cs.MX.zeros(3,self.N)
    self.Vr_mat = cs.MX.zeros(3,self.N)
    for i in range (self.N):
      self.Vl_mat[:,i]=(self.opti_v1l_force[:,i]+self.opti_v2l_force[:,i]+self.opti_v3l_force[:,i]+self.opti_v4l_force[:,i])*self.opti_contact_left[i]/self.mass
      self.Vr_mat[:,i]=(self.opti_v1r_force[:,i]+self.opti_v2r_force[:,i]+self.opti_v3r_force[:,i]+self.opti_v4r_force[:,i])*self.opti_contact_right[i]/self.mass

    # Lyapunov stability constrains
    for i in range(1*self.N):  
      self.opt.subject_to(-self.z1_mat[:,i].T@(self.k1*self.z1_mat[:,i])-self.z2_mat[:,i].T@(self.k2*self.z2_mat[:,i])+\
                          self.z1_mat[:,i].T@self.z2_mat[:,i]+self.z2_mat[:,i].T@((self.Vl_mat[:,i]+self.Vr_mat[:,i])-self.u_n_mat[:,i])<=0.0)

    # angular momentum constraint:
    for i in range(1):
      self.opt.subject_to(self.opti_hw[:,i].T@self.opti_hw[:,i]>=self.opti_hw[:,i+1].T@self.opti_hw[:,i+1])
    #self.opt.subject_to(self.opti_hw[:,i].T@self.opti_hw[:,i]<=100)
    # for i in range(self.N):
    #   self.opt.subject_to(self.opti_hw[:,i].T@self.opti_hw[:,i]<=100)  
    max_vert_force = 180
    for i in range(self.N):
      self.opt.subject_to(self.opti_CoM[2,i]<=0.76)      

      num_vertices_for_a_foot=4
     

      # Apply friction cone constraints for linear forces
      self.opt.subject_to(self.A @ (self.opti_v1l_force[:,i]) * self.opti_contact_left[i] <= self.b)
      self.opt.subject_to(self.A @ (self.opti_v2l_force[:,i]) * self.opti_contact_left[i] <= self.b)
      self.opt.subject_to(self.A @ (self.opti_v3l_force[:,i]) * self.opti_contact_left[i] <= self.b)
      self.opt.subject_to(self.A @ (self.opti_v4l_force[:,i]) * self.opti_contact_left[i] <= self.b)
      
      self.opt.subject_to(self.A @ (self.opti_v1r_force[:,i]) * self.opti_contact_right[i] <= self.b)
      self.opt.subject_to(self.A @ (self.opti_v2r_force[:,i]) * self.opti_contact_right[i] <= self.b)
      self.opt.subject_to(self.A @ (self.opti_v3r_force[:,i]) * self.opti_contact_right[i] <= self.b)
      self.opt.subject_to(self.A @ (self.opti_v4r_force[:,i]) * self.opti_contact_right[i] <= self.b)
    
      self.opt.subject_to( self.opti_v1l_force[2,i] * self.opti_contact_left[i] >= 0)
      self.opt.subject_to( self.opti_v2l_force[2,i] * self.opti_contact_left[i] >= 0)
      self.opt.subject_to( self.opti_v3l_force[2,i] * self.opti_contact_left[i] >= 0)
      self.opt.subject_to( self.opti_v4l_force[2,i] * self.opti_contact_left[i] >= 0)

      self.opt.subject_to( self.opti_v1r_force[2,i] * self.opti_contact_right[i] >= 0)
      self.opt.subject_to( self.opti_v2r_force[2,i] * self.opti_contact_right[i] >= 0)
      self.opt.subject_to( self.opti_v3r_force[2,i] * self.opti_contact_right[i] >= 0)
      self.opt.subject_to( self.opti_v4r_force[2,i] * self.opti_contact_right[i] >= 0)

      #constarin on the maximum deviation of the foot pose from the desired position
      #ROTATION MAT MISSING 
    for i in range(self.N):
     self.opt.subject_to((self.opti_pos_contact_l[0,i+1]-self.opti_pos_contact_l_ref[0,i])*self.opti_contact_left[i+1]<=0.01 )
     self.opt.subject_to((self.opti_pos_contact_l[0,i+1]-self.opti_pos_contact_l_ref[0,i])*self.opti_contact_left[i+1]>=-0.01 )
     self.opt.subject_to((self.opti_pos_contact_l[1,i+1]-self.opti_pos_contact_l_ref[1,i])*self.opti_contact_left[i+1]<=0.005 )
     self.opt.subject_to((self.opti_pos_contact_l[1,i+1]-self.opti_pos_contact_l_ref[1,i])*self.opti_contact_left[i+1]>=-0.005 )
     self.opt.subject_to((self.opti_pos_contact_l[2,i+1]-self.opti_pos_contact_l_ref[2,i])*self.opti_contact_left[i+1]<=0.00005 )
     self.opt.subject_to((self.opti_pos_contact_l[2,i+1]-self.opti_pos_contact_l_ref[2,i])*self.opti_contact_left[i+1]>=-0.00005 )

     self.opt.subject_to((self.opti_pos_contact_r[0,i+1]-self.opti_pos_contact_r_ref[0,i])*self.opti_contact_right[i+1]<=0.01 )
     self.opt.subject_to((self.opti_pos_contact_r[0,i+1]-self.opti_pos_contact_r_ref[0,i])*self.opti_contact_right[i+1]>=-0.01 )
     self.opt.subject_to((self.opti_pos_contact_r[1,i+1]-self.opti_pos_contact_r_ref[1,i])*self.opti_contact_right[i+1]<=0.005 )
     self.opt.subject_to((self.opti_pos_contact_r[1,i+1]-self.opti_pos_contact_r_ref[1,i])*self.opti_contact_right[i+1]>=-0.005 )
     self.opt.subject_to((self.opti_pos_contact_r[2,i+1]-self.opti_pos_contact_r_ref[2,i])*self.opti_contact_right[i+1]<=0.00005 )
     self.opt.subject_to((self.opti_pos_contact_r[2,i+1]-self.opti_pos_contact_r_ref[2,i])*self.opti_contact_right[i+1]>=-0.00005 )

    
    #To create the term Tu of the paper
    self.aux_forces_average_l_mat = cs.MX.zeros(3,self.N)
    self.aux_forces_average_r_mat = cs.MX.zeros(3,self.N)
    for i in range(self.N):
      self.aux_forces_average_l_mat[:,i]=(1/num_vertices_for_a_foot)*self.Vl_mat[:,i]*self.opti_contact_left[i]*self.mass
      self.aux_forces_average_r_mat[:,i]=(1/num_vertices_for_a_foot)*self.Vr_mat[:,i]*self.opti_contact_right[i]*self.mass
    

    #to have a print
    self.centroidal_dynamics_in_t = 0.01*self.centroidal_dynamic(self.opti_state[:, 0],self.opti_com_ref[:,0],
            self.opti_contact_left[0], self.opti_contact_right[0], self.U[:, 0] )

    

    # Force rate of change
    force_change_rate_v1l = cs.diff(self.opti_v1l_force.T).T
    force_change_rate_v2l = cs.diff(self.opti_v2l_force.T).T
    force_change_rate_v3l = cs.diff(self.opti_v3l_force.T).T
    force_change_rate_v4l = cs.diff(self.opti_v4l_force.T).T
    
    force_change_rate_v1r = cs.diff(self.opti_v1r_force.T).T
    force_change_rate_v2r = cs.diff(self.opti_v2r_force.T).T
    force_change_rate_v3r = cs.diff(self.opti_v3r_force.T).T
    force_change_rate_v4r = cs.diff(self.opti_v4r_force.T).T

    print(f'size of force rate {force_change_rate_v1l.shape}')

    weight_com_z = cs.MX.zeros(self.N)
    weight_com_const  = 2000 # 2000 rate 1
    weight_com_z_min = weight_com_const/2
    for  i in range(self.N):
      weight_com_z[i] = (weight_com_const- weight_com_z_min) * cs.exp(-i) + weight_com_z_min
    
    #Define the cost function
   
    cost = 0  

    for i in range(self.N):
        cost += 1000*cs.sumsqr(self.opti_hw[:,i]) + \
              1*cs.sumsqr(self.opti_CoM[0,i+1]-self.opti_com_ref[0,i]) + \
              1*cs.sumsqr(self.opti_CoM[1,i+1]-self.opti_com_ref[1,i]) + \
              weight_com_z[i]*cs.sumsqr(self.opti_CoM[2,i+1]-self.opti_com_ref[2,i]) + \
              1000*cs.sumsqr((self.opti_pos_contact_l[:,i+1]-self.opti_pos_contact_l_ref[:,i])*self.opti_contact_left[i+1]) + \
              1000*cs.sumsqr((self.opti_pos_contact_r[:,i+1]-self.opti_pos_contact_r_ref[:,i])*self.opti_contact_right[i+1]) + \
              1000*cs.sumsqr((self.opti_ang_contact_l[i+1]-self.opti_ang_contact_l_ref[i])*self.opti_contact_left[i+1]) + \
              1000*cs.sumsqr((self.opti_ang_contact_r[i+1]-self.opti_ang_contact_r_ref[i])*self.opti_contact_right[i+1]) + \
              10*cs.sumsqr(self.aux_forces_average_l_mat[:,i]-self.opti_v1l_force[:,i])*self.opti_contact_left[i] + \
              10*cs.sumsqr(self.aux_forces_average_l_mat[:,i]-self.opti_v2l_force[:,i])*self.opti_contact_left[i] + \
              10*cs.sumsqr(self.aux_forces_average_l_mat[:,i]-self.opti_v3l_force[:,i])*self.opti_contact_left[i] + \
              10*cs.sumsqr(self.aux_forces_average_l_mat[:,i]-self.opti_v4l_force[:,i])*self.opti_contact_left[i] + \
              10*cs.sumsqr(self.aux_forces_average_r_mat[:,i]-self.opti_v1r_force[:,i])*self.opti_contact_right[i] + \
              10*cs.sumsqr(self.aux_forces_average_r_mat[:,i]-self.opti_v2r_force[:,i])*self.opti_contact_right[i] + \
              10*cs.sumsqr(self.aux_forces_average_r_mat[:,i]-self.opti_v3r_force[:,i])*self.opti_contact_right[i] + \
              10*cs.sumsqr(self.aux_forces_average_r_mat[:,i]-self.opti_v4r_force[:,i])*self.opti_contact_right[i] +\
              10*cs.sumsqr(self.opti_v1l_force[:,i])*(1-self.opti_contact_left[i]) + \
              10*cs.sumsqr(self.opti_v2l_force[:,i])*(1-self.opti_contact_left[i]) + \
              10*cs.sumsqr(self.opti_v3l_force[:,i])*(1-self.opti_contact_left[i]) + \
              10*cs.sumsqr(self.opti_v4l_force[:,i])*(1-self.opti_contact_left[i]) + \
              10*cs.sumsqr(self.opti_v1r_force[:,i])*(1-self.opti_contact_right[i]) + \
              10*cs.sumsqr(self.opti_v2r_force[:,i])*(1-self.opti_contact_right[i]) + \
              10*cs.sumsqr(self.opti_v3r_force[:,i])*(1-self.opti_contact_right[i]) + \
              10*cs.sumsqr(self.opti_v4r_force[:,i])*(1-self.opti_contact_right[i]) + \
              0*cs.sumsqr(self.Vl_mat[2,i]-self.g)*self.opti_contact_left[i] + \
              0*cs.sumsqr(self.Vr_mat[2,i]-self.g)*self.opti_contact_right[i] 
    
    weight_f_rate = 1
    if self.mpc_rate== 10:
      weight_f_rate =0

    for i in range(self.N-1):
        cost += weight_f_rate*cs.sumsqr(force_change_rate_v1l[2,i])*self.opti_contact_left[i] + \
              weight_f_rate*cs.sumsqr(force_change_rate_v2l[2,i])*self.opti_contact_left[i] + \
              weight_f_rate*cs.sumsqr(force_change_rate_v3l[2,i])*self.opti_contact_left[i] + \
              weight_f_rate*cs.sumsqr(force_change_rate_v4l[2,i])*self.opti_contact_left[i] + \
              weight_f_rate*cs.sumsqr(force_change_rate_v1r[2,i])*self.opti_contact_right[i] + \
              weight_f_rate*cs.sumsqr(force_change_rate_v2r[2,i])*self.opti_contact_right[i] + \
              weight_f_rate*cs.sumsqr(force_change_rate_v3r[2,i])*self.opti_contact_right[i] + \
              weight_f_rate*cs.sumsqr(force_change_rate_v4r[2,i])*self.opti_contact_right[i] 

    self.opt.minimize(cost)

    # initialize the state space to collect the real time state value from the simulator
    self.current_state = np.zeros(3*6)
    # CoM_acc as the ff for the inverse dynamic controller
    self.model_state = {'com': {'pos': np.zeros(3), 'vel': np.zeros(3), 'acc': np.zeros(3)},
                        'hw' : {'val': np.zeros(3), 'dot':np.zeros(3)},
                  'theta_hat': {'val': np.zeros(3)},
          'ang_contact_left' : {'val': np.zeros(3)},
          'pos_contact_left' : {'val': np.zeros(3)},
          'ang_contact_right': {'val': np.zeros(3)},
          'pos_contact_right': {'val': np.zeros(3)},
          'mpc_new_contact' :{'val':np.zeros(3)},
          'counter':{'val':0} ,}




  def centroidal_dynamic(self, state,CoM_ref,contact_left, contact_right,input):
    CoM_ref_pos= cs.vertcat(CoM_ref[0],CoM_ref[1],CoM_ref[2])
    CoM_ref_vel= cs.vertcat(CoM_ref[3],CoM_ref[4],CoM_ref[5])
    CoM_ref_acc= cs.vertcat(CoM_ref[6],CoM_ref[7],CoM_ref[8])
    
    k1=self.k1
    k2=self.k2
    mass = self.mass
    gravity = cs.GenDM_zeros(3)
    gravity[2]=-self.g
    #Extract states
    com_pos=state[0:3]
    com_vel=state[3:6]
    #hw
    theta_hat=state[9:12]
    ang_rotvec_l=state[12]
    pos_lc=state[13:16]
    ang_rotvec_r=state[16]
    pos_rc=state[17:20]

    #Extract inputs
    v1l= input[0:3]
    v2l= input[3:6]
    v3l= input[6:9]
    v4l= input[9:12]
    v1r= input[12:15]
    v2r= input[15:18]
    v3r= input[18:21]
    v4r= input[21:24]
    vel_left= input[24:27]  
    vel_right= input[27:30] 
    omega_left= input[30]
    omega_right= input[31]

    Vl=(v1l+v2l+v3l+v4l)*contact_left
    Vr=(v1r+v2r+v3r+v4r)*contact_right

    z1=com_pos-CoM_ref_pos
    z2=k1*z1+(com_vel-CoM_ref_vel)

    

    def compute_foot_vertices(pos, rotvec):
      #yaw = rotvec[2]
      yaw = rotvec
      c = cs.cos(yaw)
      s = cs.sin(yaw)
      R_z = cs.vertcat(
          cs.horzcat(c, -s,  0),
          cs.horzcat(s,  c,  0),
          cs.horzcat(0,  0,  1)
      )
      foot_vertices_world_list = []
      for v in self.foot_polygon_local: 
          v_const = cs.DM(v)  # (3x1)
          vert_world = R_z @ v_const + pos
          foot_vertices_world_list.append(vert_world)
      foot_vertices_world = cs.hcat(foot_vertices_world_list).T
      return foot_vertices_world #(4,3) matrix

    left_foot_vertices = compute_foot_vertices(pos_lc,ang_rotvec_l)
    right_foot_vertices = compute_foot_vertices(pos_rc,ang_rotvec_r)
    
    v1l_pos=left_foot_vertices[0,:].T
    v2l_pos=left_foot_vertices[1,:].T
    v3l_pos=left_foot_vertices[2,:].T
    v4l_pos=left_foot_vertices[3,:].T

    v1r_pos=right_foot_vertices[0,:].T
    v2r_pos=right_foot_vertices[1,:].T
    v3r_pos=right_foot_vertices[2,:].T
    v4r_pos=right_foot_vertices[3,:].T

    
    torque_l=contact_left*(cs.cross(v1l_pos-com_pos,v1l)+cs.cross(v2l_pos-com_pos,v2l)+cs.cross(v3l_pos-com_pos,v3l)+cs.cross(v4l_pos-com_pos,v4l))
    torque_r=contact_right*(cs.cross(v1r_pos-com_pos,v1r)+cs.cross(v2r_pos-com_pos,v2r)+cs.cross(v3r_pos-com_pos,v3r)+cs.cross(v4r_pos-com_pos,v4r))
                    



    # Centroidal dynamic with disturbance estimator theta hat, contact dynamics
    dcom=com_vel 
    ddcom= gravity+(1/mass)*(Vl+Vr+theta_hat*0)
    dhw= (torque_l)+(torque_r)
    v_left= (1-contact_left)*vel_left
    v_right= (1-contact_right)*vel_right
    omega_l=(1-contact_left)*omega_left
    omega_r=(1-contact_right)*omega_right
    dthetahat= z2/1/self.mass

    return cs.vertcat(dcom,ddcom,dhw,dthetahat,omega_l,v_left,omega_r,v_right)














# Solve the mpc every time step --> That will be very tough
# Main tasks are updating the current state at the beginning of the horizon and
# let the mpc compute the state in the rest of the horizon

  def solve(self, current, t):
    print(f'time in solve():{t}')
    self.current_state = np.array([current['com']['pos'][0],       current['com']['pos'][1],       current['com']['pos'][2],
                                   current['com']['vel'][0],       current['com']['vel'][1],       current['com']['vel'][2],
                      current['hw']['val'][0],        current['hw']['val'][1],   current['hw']['val'][2],
                      self.model_state['theta_hat']['val'][0], self.model_state['theta_hat']['val'][1], self.model_state['theta_hat']['val'][2],
                                   current['lfoot']['pos'][2],
                                   current['lfoot']['pos'][3],     current['lfoot']['pos'][4],     0,
                                   current['rfoot']['pos'][2],
                                   current['rfoot']['pos'][3],     current['rfoot']['pos'][4],     0])
                                   
      # UPDATE THE INITIAL STATE
    
    if t<200:
      contact_left_current = self.pos_contact_ref_l[t]
      contact_right_current = self.pos_contact_ref_r[t] 
    else:
      index = self.footstep_planner.get_step_index_at_time(t-70)
      if self.params['first_swing'] == 'lfoot':
        contact_left_current = self.footstep_planner.plan[index + (index % 2)]['pos']
        contact_right_current = self.footstep_planner.plan[index + (index - 1) % 2]['pos']
      else:
        contact_left_current = self.footstep_planner.plan[index + (index - 1) % 2]['pos']
        contact_right_current = self.footstep_planner.plan[index + (index % 2)]['pos']
    
    print(f'Contact_left_current:{contact_left_current}')
    print(f'Contact_right_current:{contact_right_current}')

    self.current_state[13:16] = contact_left_current
    self.current_state[17:20] = contact_right_current

    self.opt.set_value(self.opti_x0_param, self.current_state)
    print(f'Current_Pos_foot_left:{self.current_state[12:16]}')
    print(f'Current_Pos_foot_right:{self.current_state[16:20]}')
   # UPDATE  the contact status L/R over N
    contact_status_l=np.empty((0, 1))
    contact_status_r=np.empty((0, 1))
    for i in range(self.N+1):
      contact_status = self.footstep_planner.get_phase_at_time(t+i*self.mpc_rate)
      if contact_status== 'ds':
        contact_status_l_i=np.array([[1]])
        contact_status_r_i=np.array([[1]])
      else :#contact_status=='ss':
        contact_status_foot = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t+i*self.mpc_rate)]['foot_id']
        if contact_status_foot=='lfoot':
          contact_status_l_i=np.array([[1]])
          contact_status_r_i=np.array([[0]])
        else: 
          contact_status_l_i=np.array([[0]])
          contact_status_r_i=np.array([[1]])
      contact_status_l=np.vstack((contact_status_l,contact_status_l_i))
      contact_status_r=np.vstack((contact_status_r,contact_status_r_i))

    self.opt.set_value(self.opti_contact_left, contact_status_l)
    self.opt.set_value(self.opti_contact_right, contact_status_r)
    print("planned contact status left -right entire horizon")
    print(contact_status_l.T)
    print(contact_status_r.T)
    if self.debug==1:  
      file_path=os.path.join(self.debug_folder, "update contact status left in entire horizon")
      with open(file_path, "w") as file:
        file.writelines("\n".join(map(str, contact_status_l)))
      file_path=os.path.join(self.debug_folder, "update contact status right in entire horizon")  
      with open(file_path, "w") as file:
        file.writelines("\n".join(map(str, contact_status_r)))


    #Update CoM_ref value for every step t and in an entire horizon N=100
    idx=1
    pos_com_ref_x= cs.DM.zeros(self.N)
    pos_com_ref_y= cs.DM.zeros(self.N)
    pos_com_ref_z= cs.DM.zeros(self.N)

    vel_com_ref_x= cs.DM.zeros(self.N)
    vel_com_ref_y= cs.DM.zeros(self.N)
    vel_com_ref_z= cs.DM.zeros(self.N)

    acc_com_ref_x= cs.DM.zeros(self.N)
    acc_com_ref_y= cs.DM.zeros(self.N)
    acc_com_ref_z= cs.DM.zeros(self.N)

    pos_contact_ref_l = cs.DM.zeros(3,self.N)
    pos_contact_ref_r = cs.DM.zeros(3,self.N)
    ang_contact_ref_l = cs.DM.zeros(3,self.N)
    ang_contact_ref_r = cs.DM.zeros(3,self.N)
    for i in range(self.N):
      
      pos_com_ref_x[i] = self.pos_com_ref_x[t+(idx+i)*self.mpc_rate]
      pos_com_ref_y[i] = self.pos_com_ref_y[t+(idx+i)*self.mpc_rate]
      pos_com_ref_z[i] = self.pos_com_ref_z[t+(idx+i)*self.mpc_rate]

      vel_com_ref_x[i] = self.vel_com_ref_x[t+(idx+i)*self.mpc_rate]
      vel_com_ref_y[i] = self.vel_com_ref_y[t+(idx+i)*self.mpc_rate]
      vel_com_ref_z[i] = self.vel_com_ref_z[t+(idx+i)*self.mpc_rate]

      acc_com_ref_x[i] = self.acc_com_ref_x[t+(idx+i)*self.mpc_rate]
      acc_com_ref_y[i] = self.acc_com_ref_y[t+(idx+i)*self.mpc_rate]
      acc_com_ref_z[i] = self.acc_com_ref_z[t+(idx+i)*self.mpc_rate]

      #UPDATE THE DESIRED FEET POSITIONS *and orientation*
      # print(f'pos_contact_ref_l:{self.pos_contact_ref_l[t+idx+i*10]}')
      pos_contact_ref_l[:,i] = self.pos_contact_ref_l[t+(idx+i)*self.mpc_rate].T
      pos_contact_ref_r[:,i] = self.pos_contact_ref_r[t+(idx+i)*self.mpc_rate].T
      ang_contact_ref_l[:,i] = self.rotvec_contact_ref_l[t+(idx+i)*self.mpc_rate].T
      ang_contact_ref_r[:,i] = self.rotvec_contact_ref_r[t+(idx+i)*self.mpc_rate].T

    com_ref_sample_horizon= np.vstack((pos_com_ref_x.T,pos_com_ref_y.T,pos_com_ref_z.T,
                                      vel_com_ref_x.T,vel_com_ref_y.T,vel_com_ref_z.T,
                                      acc_com_ref_x.T,acc_com_ref_y.T,acc_com_ref_z.T))
    
    self.opt.set_value(self.opti_com_ref,com_ref_sample_horizon)

    
    
    print(f'Pos_contact_ref_l:{pos_contact_ref_l[:,0]}')
    print(f'Pos_contact_ref_r:{pos_contact_ref_r[:,0]}')
    for i in range(self.N):
      self.opt.set_value(self.opti_pos_contact_l_ref[:,i],pos_contact_ref_l[:,i])
      self.opt.set_value(self.opti_pos_contact_r_ref[:,i],pos_contact_ref_r[:,i])
      self.opt.set_value(self.opti_ang_contact_l_ref[i],ang_contact_ref_l[i])
      self.opt.set_value(self.opti_ang_contact_r_ref[i],ang_contact_ref_r[i])
   
  #UPDATING COMPLETED

  # solve optimization problem
    try:
      sol = self.opt.solve()
    
    except RuntimeError as e:
       
        self.opt.debug.show_infeasibilities(1e-6)
        self.opt.debug.value()


    self.x = sol.value(self.opti_state[:,1]) #desired state at the next time
    #print(f'x_next_des:{self.x}')
    self.u = sol.value(self.U[:,0])
    self.x_collect=sol.value(self.opti_state)

    centroidal_dynamics_in_t = self.opt.value(self.centroidal_dynamics_in_t)
   
    if self.debug==1:
      file_path=os.path.join(self.debug_folder, "mpc contact lfoot result in entire horizon") 
      with open(file_path, "w") as file:
        file.writelines(" \n".join(map(str, self.x_collect[9:12,:])))

      file_path=os.path.join(self.debug_folder, "mpc contact rfoot result in entire horizon")     
      with open(file_path, "w") as file:
        file.writelines(" \n".join(map(str, self.x_collect[12:15,:])))

    self.opt.set_initial(self.U, sol.value(self.U))
    self.opt.set_initial(self.opti_state, sol.value(self.opti_state))

    Vl_mat=self.u[0:3]+self.u[3:6]+self.u[6:9]+self.u[9:12]
    Vr_mat=self.u[12:15]+self.u[15:18]+self.u[18:21]+self.u[21:24]
    theta_hat = self.x[9:12]
    CoM_acc=(1/self.mass)*(contact_status_l[0]*Vl_mat+contact_status_r[0]*Vr_mat+theta_hat*0).T+np.array([0, 0,- self.g])

    #update the return structure
    self.model_state['com']['pos'] = np.array([self.x[0], self.x[1], self.x[2]])
    self.model_state['com']['vel'] = np.array([self.x[3], self.x[4], self.x[5]])
    self.model_state['com']['acc'] = CoM_acc
    self.model_state['hw']['val'] = np.array([self.x[6], self.x[7], self.x[8]])
    self.model_state['hw']['dot'] = centroidal_dynamics_in_t[6:9]*self.delta*self.mpc_rate
    self.model_state['theta_hat']['val'] = np.array([self.x[9], self.x[10], self.x[11]])
    self.model_state['ang_contact_left']['val'] = self.x[12]
    self.model_state['pos_contact_left']['val'] = np.array([self.x[13], self.x[14], self.x[15]])
    self.model_state['ang_contact_right']['val'] = self.x[16]
    self.model_state['pos_contact_right']['val'] = np.array([self.x[17], self.x[18], self.x[19]])
    self.model_state['counter']['val']=0

    print(f'Pos_contact_left_horizon:{self.x_collect[13:16,:]}')
    print(f'Pos_contact_right_horizon:{self.x_collect[17:20,:]}')
    print(f'Theta_hat:{self.x[9:12]}')

  
    if self.params['update_contact']=='YES':
      update_step = 1
      if self.mpc_rate==10:
        update_step = 1
      contact_phase_current= self.footstep_planner.get_phase_at_time(t)
      contact_phase_next_after_the_horizon= self.footstep_planner.get_phase_at_time(t+self.N*self.mpc_rate-update_step)
      if contact_phase_current=='ss' and contact_phase_next_after_the_horizon=='ds' and self.update_contact_flag== 0:
      #Update new contact position at this specific moment
        print("Time to update contact list")
        self.update_contact_flag=1
        self.model_state['counter']['val'] = self.update_contact_flag
        current_contact_foot = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']
        if current_contact_foot=='lfoot':
          self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)+1]['pos'] = self.x_collect[17:20,self.N]#mean the swing foot is the right one-> update the contact position of the swing foot
          self.model_state['mpc_new_contact']['val'] = self.x_collect[17:20,self.N]
        else:
          self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)+1]['pos'] = self.x_collect[13:16,self.N]
          self.model_state['mpc_new_contact']['val'] = self.x_collect[13:16,self.N]
      if contact_phase_current=='ds':
        self.update_contact_flag=0 #reset the flag

    # self.model_state['next_des_pose_swing_MPC_at_69']['val'] = np.array(self.next_des_pose_swing_MPC_at_69)
    
    contact = self.footstep_planner.get_phase_at_time(t)
    if contact == 'ss':
      contact = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']

    return self.model_state, contact
  
  def reset_update_swing_trj(self):
    self.update_swing_trj=0