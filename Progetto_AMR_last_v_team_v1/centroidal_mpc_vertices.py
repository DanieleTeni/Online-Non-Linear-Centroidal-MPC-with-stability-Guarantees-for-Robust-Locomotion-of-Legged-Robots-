import numpy as np
import casadi as cs
import os
from scipy.spatial.transform import Rotation as R

class centroidal_mpc:
  def __init__(self, initial, footstep_planner, params, CoM_ref, contact_trj_l, contact_trj_r):
    # parameters
    self.params = params
    self.N = params['N']-90
    self.delta = params['world_time_step']
    self.h = params['h']
    self.eta = params['eta']
    self.foot_size = params['foot_size']
    self.mass = params['mass']
    print("total mass:")
    print(self.mass)
    self.g = params['g']
    self.initial = initial
    self.footstep_planner = footstep_planner
    self.sigma = lambda t, t0, t1: np.clip((t - t0) / (t1 - t0), 0, 1) # piecewise linear sigmoidal function
    self.debug_folder= "Debug"
    self.debug = 0
    #Change of coordinates Gains
    #self.k1=6#3 
    #self.k2=1
    self.k1=20
    self.k2=15


    #to consider self.update_time=0 we need to impose self.update_time=100
    self.update_time=61
    self.correction=1
    #(t+self.N-self.correction)
    self.update_swing_trj=0
    default_ss_duration = params['ss_duration']
    self.times_in_which_we_use_the_modified_swing_foot_trj= default_ss_duration - self.update_time
    print(f'self.times_in_which_we_use_the_modified_swing_foot_trj:{self.times_in_which_we_use_the_modified_swing_foot_trj}')

    self.next_des_pose_swing=np.zeros(6) # o un array di zeri

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


    #An: Get the CoM_ref data from Daniele --> thanks
    self.pos_com_ref_x= CoM_ref['pos_x']
    self.pos_com_ref_y= CoM_ref['pos_y']
    self.pos_com_ref_z= CoM_ref['pos_z']

    self.vel_com_ref_x= CoM_ref['vel_x']
    self.vel_com_ref_y= CoM_ref['vel_y']
    self.vel_com_ref_z= CoM_ref['vel_z']

    self.acc_com_ref_x= CoM_ref['acc_x']
    self.acc_com_ref_y= CoM_ref['acc_y']
    self.acc_com_ref_z= CoM_ref['acc_z']
    
    #An: Get all the foot step ref from foot step planner over time stamp
    self.pose_contact_ref_l= footstep_planner.position_contacts_ref['contact_left']
    self.pose_contact_ref_r= footstep_planner.position_contacts_ref['contact_right']

    self.pos_contact_ref_l = self.pose_contact_ref_l[:, 3:6]  # Position [x, y, z]
    self.pos_contact_ref_r = self.pose_contact_ref_r[:, 3:6]  # Position [x, y, z]

    self.rotvec_contact_ref_l = self.pose_contact_ref_l[:, 0:3]  # Rotation vector [rx, ry, rz]
    self.rotvec_contact_ref_r = self.pose_contact_ref_r[:, 0:3]  # Rotation vector [rx, ry, rz]

    if self.debug==1:
      file_path=os.path.join(self.debug_folder, "MPC_pose_contact_ref")        
      with open(file_path, "w") as file:
        for i in range(len(self.pose_contact_ref_l)):  # Loop through all time steps
            # Extract rotation vector and position for the left foot
            rotvec_l = self.rotvec_contact_ref_l[i]
            pos_l = self.pos_contact_ref_l[i]

            # Extract rotation vector and position for the right foot
            rotvec_r = self.rotvec_contact_ref_r[i]
            pos_r = self.pos_contact_ref_r[i]

            # Format the line as requested
            line = f"{i}   lfoot: {rotvec_l.tolist()} {pos_l.tolist()}   rfoot: {rotvec_r.tolist()} {pos_r.tolist()}\n"

            # Write the formatted line to file
            file.write(line)

    
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
    s_opts = {"max_iter": 20000,"print_level": False,"tol":0.002}
    #Set up a proper optimal solver
    self.opt.solver('ipopt',p_opts,s_opts) #An: Use different solver, refer to C++ code

    #  # optimization problem
    # self.opt = cs.Opti('conic')
    # p_opts = {"expand": True}
    # s_opts = {"max_iter": 1000, "verbose": False}
    # self.opt.solver("osqp", p_opts, s_opts)
    
    
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
    self.opti_omega_contact_l= self.opt.variable(3,self.N)
    self.opti_omega_contact_r= self.opt.variable(3,self.N)


    self.U = cs.vertcat(self.opti_v1l_force,self.opti_v2l_force,self.opti_v3l_force,self.opti_v4l_force,
                        self.opti_v1r_force,self.opti_v2r_force,self.opti_v3r_force,self.opti_v4r_force,
                        self.opti_vel_contact_l,self.opti_vel_contact_r,
                        self.opti_omega_contact_l,self.opti_omega_contact_r)
    
    #STATE -> Variables object of the optimization problem (will impose two constraints)
    self.opti_CoM = self.opt.variable(3, self.N + 1)
    self.opti_dCoM = self.opt.variable(3, self.N + 1)
    self.opti_hw = self.opt.variable(3, self.N + 1)
    self.opti_thetahat = self.opt.variable(3, self.N + 1)
    self.opti_ang_contact_l= self.opt.variable(3, self.N + 1)
    self.opti_ang_contact_r= self.opt.variable(3, self.N + 1)
    self.opti_pos_contact_l= self.opt.variable(3, self.N + 1)
    self.opti_pos_contact_r= self.opt.variable(3, self.N + 1)
    
    self.opti_state= cs.vertcat(self.opti_CoM,self.opti_dCoM,self.opti_hw,self.opti_thetahat,
                          self.opti_ang_contact_l,self.opti_pos_contact_l,
                          self.opti_ang_contact_r,self.opti_pos_contact_r)

                        #Paramteres Needed to solve the optimization problem
    #INITIAL STATE
    self.opti_x0_param = self.opt.parameter(24) # update every step based on the current value obtained by the simulator (column vector)
    #CoM trajectories for the change of coordinates 
    self.opti_com_ref = self.opt.parameter(3+3+3,self.N) #including pos x,y,z, vel x,y,z, acc x,y,z ref, update every step based on the pre-planner
    #DESIRED POSITION OF THE CONTACT POINT and orientation - to put into cost function
    self.opti_ang_contact_l_ref = self.opt.parameter(3,self.N)
    self.opti_pos_contact_l_ref = self.opt.parameter(3,self.N)
    self.opti_ang_contact_r_ref = self.opt.parameter(3,self.N)
    self.opti_pos_contact_r_ref = self.opt.parameter(3,self.N)

    self.opti_ang_contact_trj_l_ref = self.opt.parameter(3,self.N)
    self.opti_pos_contact_trj_l_ref = self.opt.parameter(3,self.N)
    self.opti_ang_contact_trj_r_ref = self.opt.parameter(3,self.N)
    self.opti_pos_contact_trj_r_ref = self.opt.parameter(3,self.N)

    #GAMMA_L
    self.opti_contact_left = self.opt.parameter(1,self.N)
    #GAMMA_R
    self.opti_contact_right = self.opt.parameter(1,self.N)


    #CONSTRAINTS ON THE STATE, centroidal dynamics
    self.opt.subject_to(self.opti_state[:,0]==self.opti_x0_param) #An: Initial constraint
    #An: Centroidal Dynamic constraints in all the horizon self.N
    for i in range(self.N):
      self.opt.subject_to(self.opti_state[:,i+1] == self.opti_state[:,i]+
                           self.delta*self.centroidal_dynamic(self.opti_state[:,i],self.opti_com_ref[:,i],
                                        self.opti_contact_left[i],self.opti_contact_right[i],self.U[:,i]))
    
    #Change of coordinates at the first step
    self.z1_mat = cs.MX.zeros(3,self.N)
    self.z2_mat = cs.MX.zeros(3,self.N)
    self.u_n_mat = cs.MX.zeros(3,self.N)
    self.u_n_partial_mat = cs.MX.zeros(3,self.N) #partial /by n° of vertices
    self.u_n_partial_leg_mat = cs.MX.zeros(3,self.N) #partial /by n° of feet
    self.u_n_partial_legl_mat = cs.MX.zeros(3,self.N)
    self.u_n_partial_legr_mat = cs.MX.zeros(3,self.N)

    gravity = cs.GenDM_zeros(3)
    gravity[2]=-self.g

    
    for i in range(self.N):        
      self.z1_mat[:,i] = self.opti_CoM[:,i+1]-self.opti_com_ref[0:3,i]
      self.z2_mat[:,i] = self.k1*(self.z1_mat[:,i])+(self.opti_dCoM[:,i+1]-self.opti_com_ref[3:6,i])

    for i in range(self.N):
      self.u_n_mat[:,i]=-(self.k1+self.k2)*self.z2_mat[:,i]+\
                        self.k1*self.k1*self.z1_mat[:,i]-gravity+self.opti_com_ref[6:9,i]-self.opti_thetahat[:,i]

    #Total linear force evaluated by the MPCs
    self.Vl_mat = cs.MX.zeros(3,self.N)
    self.Vr_mat = cs.MX.zeros(3,self.N)
    for i in range (self.N):
      self.Vl_mat[:,i]=(self.opti_v1l_force[:,i]+self.opti_v2l_force[:,i]+self.opti_v3l_force[:,i]+self.opti_v4l_force[:,i])*self.opti_contact_left[i]/self.mass
      self.Vr_mat[:,i]=(self.opti_v1r_force[:,i]+self.opti_v2r_force[:,i]+self.opti_v3r_force[:,i]+self.opti_v4r_force[:,i])*self.opti_contact_right[i]/self.mass

    #An: Lyapunov stability constrains
    for i in range(self.N):  
      self.opt.subject_to(-self.z1_mat[:,i].T@(self.k1*self.z1_mat[:,i])-self.z2_mat[:,i].T@(self.k2*self.z2_mat[:,i])+\
                          self.z1_mat[:,i].T@self.z2_mat[:,i]+self.z2_mat[:,i].T@((self.Vl_mat[:,i]+self.Vr_mat[:,i])-self.u_n_mat[:,i])<=0.0)

    #An: angular momentum constraint:
    for i in range(self.N):
      #self.opt.subject_to(self.opti_hw[:,i].T@self.opti_hw[:,i]>=self.opti_hw[:,i+1].T@self.opti_hw[:,i+1])
      self.opt.subject_to(cs.fabs(self.opti_hw[0,i])<=10)
      self.opt.subject_to(cs.fabs(self.opti_hw[1,i])<=10)
      self.opt.subject_to(cs.fabs(self.opti_hw[2,i])<=5)
    #self.opt.subject_to(self.opti_hw[:,i].T@self.opti_hw[:,i]<=100)
    # for i in range(self.N):
    #   self.opt.subject_to(self.opti_hw[:,i].T@self.opti_hw[:,i]<=100)  

    for i in range(self.N):
      self.opt.subject_to(self.opti_CoM[2,i]<=0.77)      

      num_vertices_for_a_foot=4
      # sum=num_vertices_for_a_foot*(self.opti_contact_left[i]+self.opti_contact_right[i])
      # self.u_n_partial_mat[:,i]=self.u_n_mat[:,i]/sum
      
      # self.u_n_partial_leg_mat[:,i]=self.u_n_mat[:,i]/(self.opti_contact_left[i]+self.opti_contact_right[i])
      # self.u_n_partial_legl_mat[:,i]=self.u_n_partial_leg_mat[:,i]*self.opti_contact_left[i]
      # self.u_n_partial_legr_mat[:,i]=self.u_n_partial_leg_mat[:,i]*self.opti_contact_right[i]

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
      #self.opt.subject_to(self.opti_pos_contact_l[:,i] - self.opti_pos_contact_l_ref[:,i] <=0.08 )
      #self.opt.subject_to(self.opti_pos_contact_r[:,i] - self.opti_pos_contact_r_ref[:,i] >=-0.08 )

      #self.opt.subject_to(self.opti_ang_contact_l[:,i] - self.opti_ang_contact_l_ref[:,i] <=0.08 )
      #self.opt.subject_to(self.opti_ang_contact_r[:,i] - self.opti_ang_contact_r_ref[:,i] >=-0.08 )
      for j in range(3):  # x, y, z
          self.opt.subject_to((self.opti_pos_contact_l[j, i] - self.opti_pos_contact_trj_l_ref[j, i])*self.opti_contact_left[i] <= 0.024)
          self.opt.subject_to((self.opti_pos_contact_l[j, i] - self.opti_pos_contact_trj_l_ref[j, i])*self.opti_contact_left[i] >= -0.024)

          self.opt.subject_to((self.opti_pos_contact_r[j, i] - self.opti_pos_contact_trj_r_ref[j, i])*self.opti_contact_right[i] <= 0.024)
          self.opt.subject_to((self.opti_pos_contact_r[j, i] - self.opti_pos_contact_trj_r_ref[j, i])*self.opti_contact_right[i] >= -0.024)

            #for orientation 0.02rad= 1.43° , 
          self.opt.subject_to((self.opti_ang_contact_l[j, i] - self.opti_ang_contact_trj_l_ref[j, i])*self.opti_contact_left[i] <= 0.025)
          self.opt.subject_to((self.opti_ang_contact_l[j, i] - self.opti_ang_contact_trj_l_ref[j, i])*self.opti_contact_left[i] >= -0.025)

          self.opt.subject_to((self.opti_ang_contact_r[j, i] - self.opti_ang_contact_trj_r_ref[j, i])*self.opti_contact_right[i] <= 0.025)
          self.opt.subject_to((self.opti_ang_contact_r[j, i] - self.opti_ang_contact_trj_r_ref[j, i])*self.opti_contact_right[i] >= -0.025)


    #To create the term Tu of the paper
    self.aux_forces_average_l_mat = cs.MX.zeros(3,self.N)
    self.aux_forces_average_r_mat = cs.MX.zeros(3,self.N)
    for i in range(self.N):
      self.aux_forces_average_l_mat[:,i]=(1/num_vertices_for_a_foot)*self.Vl_mat[:,i]*self.opti_contact_left[i]*self.mass
      self.aux_forces_average_r_mat[:,i]=(1/num_vertices_for_a_foot)*self.Vr_mat[:,i]*self.opti_contact_right[i]*self.mass
    

    #to have a print
    self.centroidal_dynamics_in_t = 0.01*self.centroidal_dynamic(self.opti_state[:, 0],self.opti_com_ref[:,0],
            self.opti_contact_left[0], self.opti_contact_right[0], self.U[:, 0] )

    

    #An: additional constraint in the z-torque
    # fx_l=self.opti_force_contact_l[0,:]
    # fy_l=self.opti_force_contact_l[1,:]
    # fz_l=self.opti_force_contact_l[2,:]
    # tau_x_l=self.opti_torque_contact_l[0,:]
    # tau_y_l=self.opti_torque_contact_l[1,:]
    # tau_z_l=self.opti_torque_contact_l[2,:]

    # fx_r=self.opti_force_contact_r[0,:]
    # print("size fx_r")
    # print(fx_r.shape)
    # fy_r=self.opti_force_contact_r[1,:]
    # fz_r=self.opti_force_contact_r[2,:]
    # tau_x_r=self.opti_torque_contact_r[0,:]
    # tau_y_r=self.opti_torque_contact_r[1,:]
    # tau_z_r=self.opti_torque_contact_r[2,:]

    # tau_z_min_l= -mu*2*d*fz_l+cs.fabs(d*fx_l-mu*tau_x_l)+cs.fabs(d*fy_l-mu*tau_y_l)
    # tau_z_max_l=  mu*2*d*fz_l-cs.fabs(d*fx_l-mu*tau_x_l)-cs.fabs(d*fy_l-mu*tau_y_l)

    # tau_z_min_r= -mu*2*d*fz_r+cs.fabs(d*fx_r-mu*tau_x_r)+cs.fabs(d*fy_r-mu*tau_y_r)
    # tau_z_max_r=  mu*2*d*fz_r-cs.fabs(d*fx_r-mu*tau_x_r)-cs.fabs(d*fy_r-mu*tau_y_r)

    #self.opt.subject_to(tau_z_l<=tau_z_max_l)
    #self.opt.subject_to(tau_z_l>=tau_z_min_l)

    #self.opt.subject_to(tau_z_r<=tau_z_max_r)
    #self.opt.subject_to(tau_z_r>=tau_z_min_r)
    
    #An: Define the cost function
    # still lack of the components to minimize the deviation of forces at the foot vertices (aka foot corners)
    cost = 0  # Inizializza il costo totale

    for i in range(self.N):
        cost += 200*cs.sumsqr(self.opti_hw[:,i+1]) + \
              600*cs.sumsqr(self.opti_CoM[2,i]-0.74)  +\
              400*cs.sumsqr(self.opti_CoM[0,i+1]-self.opti_com_ref[0,i]) + \
              400*cs.sumsqr(self.opti_CoM[1,i+1]-self.opti_com_ref[1,i]) + \
              600*cs.sumsqr(self.opti_CoM[2,i+1]-self.opti_com_ref[2,i]) + \
              600*cs.sumsqr((self.opti_pos_contact_l[:,i]-self.opti_pos_contact_l_ref[:,i])*self.opti_contact_left[i]) + \
              600*cs.sumsqr((self.opti_pos_contact_r[:,i]-self.opti_pos_contact_r_ref[:,i])*self.opti_contact_right[i]) + \
              600*cs.sumsqr((self.opti_ang_contact_l[:,i]-self.opti_ang_contact_l_ref[:,i])*self.opti_contact_left[i]) + \
              600*cs.sumsqr((self.opti_ang_contact_r[:,i]-self.opti_ang_contact_r_ref[:,i])*self.opti_contact_right[i]) + \
              1*cs.sumsqr(self.aux_forces_average_l_mat[:,i]-self.opti_v1l_force[:,i])*self.opti_contact_left[i] + \
              1*cs.sumsqr(self.aux_forces_average_l_mat[:,i]-self.opti_v2l_force[:,i])*self.opti_contact_left[i] + \
              1*cs.sumsqr(self.aux_forces_average_l_mat[:,i]-self.opti_v3l_force[:,i])*self.opti_contact_left[i] + \
              1*cs.sumsqr(self.aux_forces_average_l_mat[:,i]-self.opti_v4l_force[:,i])*self.opti_contact_left[i] + \
              1*cs.sumsqr(self.aux_forces_average_r_mat[:,i]-self.opti_v1r_force[:,i])*self.opti_contact_right[i] + \
              1*cs.sumsqr(self.aux_forces_average_r_mat[:,i]-self.opti_v2r_force[:,i])*self.opti_contact_right[i] + \
              1*cs.sumsqr(self.aux_forces_average_r_mat[:,i]-self.opti_v3r_force[:,i])*self.opti_contact_right[i] + \
              1*cs.sumsqr(self.aux_forces_average_r_mat[:,i]-self.opti_v4r_force[:,i])*self.opti_contact_right[i]  +\
              400* self.opti_hw[:,i+1].T @ self.opti_hw[:,i+1] - self.opti_hw[:,i].T @ self.opti_hw[:,i]


    self.opt.minimize(cost)

    #An: initialize the state space to collect the real time state value from the simulator
    self.current_state = np.zeros(3*6)
    #An: CoM_acc as the ff for the inverse dynamic controller
    self.model_state = {'com': {'pos': np.zeros(3), 'vel': np.zeros(3), 'acc': np.zeros(3)},
                        'hw' : {'val': np.zeros(3), 'dot':np.zeros(3)},
                  'theta_hat': {'val': np.zeros(3)},
          'ang_contact_left' : {'val': np.zeros(3)},
          'pos_contact_left' : {'val': np.zeros(3)},
          'ang_contact_right': {'val': np.zeros(3)},
          'pos_contact_right': {'val': np.zeros(3)},
          'ang_vel_left'     : {'val': np.zeros(3)},
          'lin_vel_left'     : {'val': np.zeros(3)},
          'ang_vel_right'    : {'val': np.zeros(3)},
          'lin_vel_right'    : {'val': np.zeros(3)},
          'next_des_pose_swing' :{'val':np.zeros(6)},
          'swing_foot': {'lfoot': 0 , 'rfoot': 0},
          'update_swing_trj':{'val':0},
          'counter':{'val':0} ,
          'x_collect_right':{'val':np.zeros((6, self.N))},
          'x_collect_left':{'val':np.zeros((6, self.N))},
          'N':{'val':self.N},
          'correction':{'val':self.correction}
          }




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
    #theta
    ang_rotvec_l=state[12:15]
    pos_lc=state[15:18]
    ang_rotvec_r=state[18:21]
    pos_rc=state[21:24]

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
    omega_left= input[30:33]
    omega_right= input[33:36]

    Vl=(v1l+v2l+v3l+v4l)*contact_left
    Vr=(v1r+v2r+v3r+v4r)*contact_right

    z1=com_pos-CoM_ref_pos
    z2=k1*z1+(com_vel-CoM_ref_vel)

    # u_n=k1*k1*z1-(k1+k2)*z2-gravity+CoM_ref_acc
    # sum=contact_left+contact_right
    # u_n_l=u_n/sum*contact_left
    # u_n_r=u_n/sum*contact_right

    def compute_foot_vertices(pos, rotvec):
      yaw = rotvec[2]
      c = cs.cos(yaw)
      s = cs.sin(yaw)
      R_z = cs.vertcat(
          cs.horzcat(c, -s,  0),
          cs.horzcat(s,  c,  0),
          cs.horzcat(0,  0,  1)
      )
      foot_vertices_world_list = []
      for v in self.foot_polygon_local:  # v is an array NumPy of shape (3,)
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

    
    # torque_of_lfoot=contact_left*(cs.cross(pos_lc-com_pos,mass*u_n_l+Vl))
    # torque_of_rfoot=contact_right*(cs.cross(pos_rc-com_pos,mass*u_n_r+Vr))
    torque_l=contact_left*(cs.cross(v1l_pos-com_pos,v1l)+cs.cross(v2l_pos-com_pos,v2l)+cs.cross(v3l_pos-com_pos,v3l)+cs.cross(v4l_pos-com_pos,v4l))
    torque_r=contact_right*(cs.cross(v1r_pos-com_pos,v1r)+cs.cross(v2r_pos-com_pos,v2r)+cs.cross(v3r_pos-com_pos,v3r)+cs.cross(v4r_pos-com_pos,v4r))
                      #vectors from the center of the foot to the vertices
    #torque_l = contact_left * (cs.cross(pos_lc - v1l_pos, v1l) + cs.cross(pos_lc - v2l_pos, v2l) + 
                          # cs.cross(pos_lc - v3l_pos, v3l) + cs.cross(pos_lc - v4l_pos, v4l))

    #torque_r = contact_right * (cs.cross(pos_rc - v1r_pos, v1r) + cs.cross(pos_rc - v2r_pos, v2r) + 
                          # cs.cross(pos_rc - v3r_pos, v3r) + cs.cross(pos_rc - v4r_pos, v4r))

    # Centroidal dynamic with disturbance estimator theta hat, contact dynamics
    dcom=com_vel 
    ddcom= gravity+(1/mass)*(Vl+Vr)
    dhw= (torque_l)+(torque_r)
    v_left= (1-contact_left)*vel_left
    v_right= (1-contact_right)*vel_right
    omega_l=(1-contact_left)*omega_left
    omega_r=(1-contact_right)*omega_right
    dthetahat= z2

    return cs.vertcat(dcom,ddcom,dhw,dthetahat,omega_l,v_left,omega_r,v_right)














#An: Solve the mpc every time step --> That will be very tough
# Main tasks are updating the current state at the beginning of the horizon and
# let the mpc compute the state in the rest of the horizon

  def solve(self, current, t):
    print(f'time in solve():{t}')
    self.current_state = np.array([current['com']['pos'][0],       current['com']['pos'][1],       current['com']['pos'][2],
                                   current['com']['vel'][0],       current['com']['vel'][1],       current['com']['vel'][2],
                      current['hw']['val'][0],        current['hw']['val'][1],   current['hw']['val'][2],
                      self.model_state['theta_hat']['val'][0], self.model_state['theta_hat']['val'][1], self.model_state['theta_hat']['val'][2],
                                   current['lfoot']['pos'][0],     current['lfoot']['pos'][1],     current['lfoot']['pos'][2],
                                   current['lfoot']['pos'][3],     current['lfoot']['pos'][4],     current['lfoot']['pos'][5],
                                   current['rfoot']['pos'][0],     current['rfoot']['pos'][1],     current['rfoot']['pos'][2],
                                   current['rfoot']['pos'][3],     current['rfoot']['pos'][4],     current['rfoot']['pos'][5]])
                                   
    # UPDATE THE INITIAL STATE   
    self.opt.set_value(self.opti_x0_param, self.current_state) 
    self.opt.set_initial(self.opti_state[:,0], self.current_state)

   # UPDATE  the contact status L/R over N
    contact_status_l=np.empty((0, 1))
    contact_status_r=np.empty((0, 1))
    for i in range(self.N):
      contact_status = self.footstep_planner.get_phase_at_time(t+i)
      if contact_status== 'ds':
        contact_status_l_i=np.array([[1]])
        contact_status_r_i=np.array([[1]])
      else :#contact_status=='ss':
        contact_status_foot = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t+i)]['foot_id']
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

    if self.debug==1:  
      file_path=os.path.join(self.debug_folder, "update contact status left in entire horizon")
      with open(file_path, "w") as file:
        file.writelines("\n".join(map(str, contact_status_l)))
      file_path=os.path.join(self.debug_folder, "update contact status right in entire horizon")  
      with open(file_path, "w") as file:
        file.writelines("\n".join(map(str, contact_status_r)))


    #An: Update CoM_ref value for every step t and in an entire horizon N=100
    idx=0
    pos_com_ref_x= self.pos_com_ref_x[t+idx:t+idx+self.N]
    pos_com_ref_y= self.pos_com_ref_y[t+idx:t+idx+self.N]
    pos_com_ref_z= self.pos_com_ref_z[t+idx:t+idx+self.N]

    vel_com_ref_x= self.vel_com_ref_x[t+idx:t+idx+self.N]
    vel_com_ref_y= self.vel_com_ref_y[t+idx:t+idx+self.N]
    vel_com_ref_z= self.vel_com_ref_z[t+idx:t+idx+self.N]

    acc_com_ref_x= self.acc_com_ref_x[t+idx:t+idx+self.N]
    acc_com_ref_y= self.acc_com_ref_y[t+idx:t+idx+self.N]
    acc_com_ref_z= self.acc_com_ref_z[t+idx:t+idx+self.N]

    com_ref_sample_horizon= np.vstack((pos_com_ref_x,pos_com_ref_y,pos_com_ref_z,
                                      vel_com_ref_x,vel_com_ref_y,vel_com_ref_z,
                                      acc_com_ref_x,acc_com_ref_y,acc_com_ref_z))
    
    self.opt.set_value(self.opti_com_ref,com_ref_sample_horizon)

    #UPDATE THE DESIRED FEET POSITIONS *and orientation*
    pos_contact_ref_l = self.pos_contact_ref_l[t:t+self.N].T
    pos_contact_ref_r = self.pos_contact_ref_r[t:t+self.N].T
    ang_contact_ref_l = self.rotvec_contact_ref_l[t:t+self.N].T
    ang_contact_ref_r = self.rotvec_contact_ref_r[t:t+self.N].T

    for i in range(self.N):
      self.opt.set_value(self.opti_pos_contact_l_ref[:,i],pos_contact_ref_l[:,i])
      self.opt.set_value(self.opti_pos_contact_r_ref[:,i],pos_contact_ref_r[:,i])
      self.opt.set_value(self.opti_ang_contact_l_ref[:,i],ang_contact_ref_l[:,i])
      self.opt.set_value(self.opti_ang_contact_r_ref[:,i],ang_contact_ref_r[:,i])

      self.opt.set_value(self.opti_ang_contact_trj_l_ref[:,i],self.contact_trj_l[t+i][0]['pos'][0:3].T)
      self.opt.set_value(self.opti_pos_contact_trj_l_ref[:,i],self.contact_trj_l[t+i][0]['pos'][3:6].T)
      self.opt.set_value(self.opti_ang_contact_trj_r_ref[:,i],self.contact_trj_r[t+i][0]['pos'][0:3].T)
      self.opt.set_value(self.opti_pos_contact_trj_r_ref[:,i],self.contact_trj_r[t+i][0]['pos'][3:6].T)
  
   #UPDATING COMPLETED

  # solve optimization problem
    try:
      sol = self.opt.solve()
    
    except RuntimeError as e:
        print("Errore nell'ottimizzazione:", e)
        # Mostra le constraint non soddisfatte:
        print("Last candidate solution:", e)
        print(f'      self.opti_CoM   = [0,2]\n'
              f'      self.opti_dCoM  = [3,5]\n'
              f'      self.opti_hw    = [6,8]\n'
              f'      self.opti_thetahat  = [9,11]\n'
              f'      self.opti_ang_contact_l = [12,14]\n'
              f'      self.opti_ang_contact_r = [15,17]\n'
              f'      self.opti_pos_contact_l = [18,20]\n'
              f'      self.opti_pos_contact_r = [21,23]\n')
        
        print("===== Elenco Constraints =====")

        print("0..23 (24 eq) -> Stato iniziale")
        print(f"24..{24+24*self.N-1} (24*N eq) -> Vincoli di dinamica")
        print(f"{24+24*self.N}..{24+25*self.N-1} (N ineq) -> Lyapunov constraint")
        print(f"{24+25*self.N}..{24+26*self.N-1} (N ineq) -> Momento angolare ")
        print(f"{24+26*self.N}..{24+27*self.N-1} (N ineq) -> CoM z <= 0.76")
        print(f"{24+27*self.N}..{24+27*self.N+32*self.N-1} (32*N ineq) -> Coni di attrito")
        print(f"{24+27*self.N+32*self.N}..{24+27*self.N+32*self.N+8*self.N-1} (8*N ineq) -> Forza normale >= 0")
        print(f"{24+27*self.N+32*self.N+8*self.N}..{24+27*self.N+32*self.N+8*self.N+24*self.N-1} (24*N ineq) -> Pos/orient foot")
        self.opt.debug.show_infeasibilities(1e-6)
        self.opti.debug.value()        
        
                                                                   
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
    
    CoM_acc=(1/self.mass)*(contact_status_l[0]*Vl_mat+contact_status_r[0]*Vr_mat).T+np.array([0, 0,- self.g])

    #update the return structure
    self.model_state['com']['pos'] = np.array([self.x[0], self.x[1], self.x[2]])
    self.model_state['com']['vel'] = np.array([self.x[3], self.x[4], self.x[5]])
    self.model_state['com']['acc'] = CoM_acc
    self.model_state['hw']['val'] = np.array([self.x[6], self.x[7], self.x[8]])
    self.model_state['hw']['dot'] = centroidal_dynamics_in_t[6:9]
    self.model_state['theta_hat']['val'] = np.array([self.x[9], self.x[10], self.x[11]])
    self.model_state['ang_contact_left']['val'] = np.array([self.x[12], self.x[13], self.x[14]])
    self.model_state['pos_contact_left']['val'] = np.array([self.x[15], self.x[16], self.x[17]])
    self.model_state['ang_contact_right']['val'] = np.array([self.x[18], self.x[19], self.x[20]])
    self.model_state['pos_contact_right']['val'] = np.array([self.x[21], self.x[22], self.x[23]])
    self.model_state['ang_vel_left']['val'] = np.array([self.u[24],self.u[25],self.u[26]])
    self.model_state['lin_vel_left']['val'] = np.array([self.u[27],self.u[28],self.u[29]])
    self.model_state['ang_vel_right']['val'] = np.array([self.u[30],self.u[31],self.u[32]])
    self.model_state['lin_vel_right']['val'] = np.array([self.u[33],self.u[34],self.u[35]])

    contact = self.footstep_planner.get_phase_at_time(t)
    if contact == 'ss':
      contact = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']

    self.next_des_pose_swing=np.zeros(6)
    #An: Need to add here the output of mpc for contact pos -> update them to the footstep planner list
    if self.params['update_contact']=='YES' :
      print(f't:{t} , update_time:{self.update_time} , t%update_time:{t%self.update_time}')
      if t%self.update_time==0:

        if t!=0 :
          self.update_time=100+self.update_time
          contact_phase_current= self.footstep_planner.get_phase_at_time(t)
          contact_phase_next_after_the_horizon= self.footstep_planner.get_phase_at_time(t+self.N)
          if t>199:
            self.update_swing_trj=1
            if contact_phase_current=='ss' and contact_phase_next_after_the_horizon=='ds' :
            #Ritrieve next MPC optimal contact position for the swing foot
              print("Time to update contact list")
              current_contact_foot = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']
              if current_contact_foot=='lfoot':
                self.model_state['swing_foot']['rfoot'] = 1
                self.model_state['swing_foot']['lfoot'] = 0
                print(f'update of right foot contact pos from :{self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)+1]['pos']}')
                self.next_des_pose_swing = self.x_collect[18:24,self.N-self.correction]#mean the swing foot is the right one-> update the contact position of the swing foot
                print(f'to :{self.x_collect[18:24,self.N-self.correction]}')
              else:
                self.model_state['swing_foot']['rfoot'] = 0
                self.model_state['swing_foot']['lfoot'] = 1
                print(f'update of left foot contact pos from :{self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)+1]['pos']}')
                self.next_des_pose_swing = self.x_collect[12:18,self.N-self.correction]
                print(f'to :{self.x_collect[12:18,self.N-self.correction]}')

    self.model_state['next_des_pose_swing']['val'] = np.array(self.next_des_pose_swing)
    self.model_state['update_swing_trj']['val'] = self.update_swing_trj
    self.model_state['counter']['val'] = self.times_in_which_we_use_the_modified_swing_foot_trj
    self.model_state['x_collect_right']['val'] = self.x_collect[18:24,:]
    self.model_state['x_collect_left']['val'] = self.x_collect[12:18,:]
    
    return self.model_state, contact


  def reset_update_swing_trj(self):
    self.update_swing_trj=0
  
