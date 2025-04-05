import numpy as np
import dartpy as dart
import copy
from utils import *
import os
import centroidal_mpc_vertices
import footstep_planner_vertices 
import inverse_dynamics as id
import foot_trajectory_generator as ftg
from logger import Logger
from logger2 import Logger2
from logger3 import Logger3 
import new


debug_folder= "Debug"
class Hrp4Controller(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world, hrp4):
        super(Hrp4Controller, self).__init__(world)
        self.world = world
        self.hrp4 = hrp4
        self.time = 0
        self.params = {
            'g': 9.81,
            'h': 0.72,
            'foot_size': 0.1,
            'step_height': 0.02,
            'world_time_step': world.getTimeStep(),            
            'ss_duration': int(0.7/world.getTimeStep()),
            'ds_duration': int(0.3/world.getTimeStep()),
            'first_swing': 'lfoot',
            'µ': 0.5,
            'N': 100,
            'dof': self.hrp4.getNumDofs(),
            'mass': self.hrp4.getMass(), #An: Add the mass of the robot as a default param
            'update_contact':'YES'
        }
        self.params['eta'] = np.sqrt(self.params['g'] / self.params['h'])
        
        self.sim_update_swing_trj=0
        self.counter=0
        self.update_swing_trj=0
        
        model='full_model'   #model could be 'full model'

        #momentum = 'torso + base + l_hip'  ## if you want to consire angular momentum only of that joint
        #momentum='torso'
        #momentum='base'                 
        #momentum='semi'

        momentum='full'
        real_walk = 'NO'     #  or 'YES' if you want that the self.desired position are the one compute by mpc
        Angular_update='NO'   # or 'YES'  if ypu want that the angolar momentum is updated by the  formula h = Iw  
        acc='NO'    #if acc = 0 then self.desired[torso or base] = np.zeros(3)
                    #if acc = YES   then i update the angular velocity derivative by inverting  the formula 
                    ##      dwh = I@dw + np.cros(w,I@w)      and so compute the desired dw
        track='torso'        # or base

        self.preferences=[model,momentum ,real_walk,Angular_update,acc,track] 
                
        # robot links
        self.lsole = hrp4.getBodyNode('l_sole')
        self.rsole = hrp4.getBodyNode('r_sole')
        self.torso = hrp4.getBodyNode('torso')
        self.base  = hrp4.getBodyNode('body')
        self.l_hip_p= hrp4.getBodyNode('L_HIP_P_LINK')

        for i in range(hrp4.getNumJoints()):
            joint = hrp4.getJoint(i)
            dim = joint.getNumDofs()

            # set floating base to passive, everything else to torque
            if   dim == 6: joint.setActuatorType(dart.dynamics.ActuatorType.PASSIVE)
            elif dim == 1: joint.setActuatorType(dart.dynamics.ActuatorType.FORCE)

        # set initial configuration
        initial_configuration = {'CHEST_P': 0., 'CHEST_Y': 0., 'NECK_P': 0., 'NECK_Y': 0., \
                                 'R_HIP_Y': 0., 'R_HIP_R': -3., 'R_HIP_P': -25., 'R_KNEE_P': 50., 'R_ANKLE_P': -25., 'R_ANKLE_R':  3., \
                                 'L_HIP_Y': 0., 'L_HIP_R':  3., 'L_HIP_P': -25., 'L_KNEE_P': 50., 'L_ANKLE_P': -25., 'L_ANKLE_R': -3., \
                                 'R_SHOULDER_P': 4., 'R_SHOULDER_R': -8., 'R_SHOULDER_Y': 0., \
                                 'L_SHOULDER_P': 4., 'L_SHOULDER_R':  8.}
        
        #to open a bit the legs
        #initial_configuration['R_HIP_R'] = -5  
        #initial_configuration['L_HIP_R'] = +5 

        for joint_name, value in initial_configuration.items():
            self.hrp4.setPosition(self.hrp4.getDof(joint_name).getIndexInSkeleton(), value * np.pi / 180.)

        # position the robot on the ground
        lsole_pos = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        rsole_pos = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        self.hrp4.setPosition(3, - (lsole_pos[0] + rsole_pos[0]) / 2.)
        self.hrp4.setPosition(4, - (lsole_pos[1] + rsole_pos[1]) / 2.)
        self.hrp4.setPosition(5, - (lsole_pos[2] + rsole_pos[2]) / 2.)

   
        # initialize state
        self.initial = self.retrieve_state()
        self.contact = 'lfoot' if self.params['first_swing'] == 'rfoot' else 'rfoot' # there is a dummy footstep
        self.desired = copy.deepcopy(self.initial)
        self.com_ref = copy.deepcopy(self.initial)
        self.com_ref = copy.deepcopy(self.initial)

        # selection matrix for redundant dofs
        redundant_dofs = [ \
            "NECK_Y", "NECK_P", \
            "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P", \
            "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P"]
        
        # initialize inverse dynamics
        self.id = id.InverseDynamics(self.hrp4, redundant_dofs)

             # initialize footstep planner
        reference = [(0.1, 0., 0)] * 5 + [(0.1, 0., -0.0)] * 10 + [(0.2, 0., 0.)] * 10   
        #omega in radiants
        self.footstep_planner = footstep_planner_vertices.FootstepPlanner(reference,self.initial['lfoot']['pos'],self.initial['rfoot']['pos'],self.params)
    
        # initialize foot trajectory generator
        self.foot_trajectory_generator = ftg.FootTrajectoryGenerator(self.initial,self.footstep_planner,self.params)
        
        pre_feet_traj= self.foot_trajectory_generator.generate_feet_trajectories_pre()
        self.pre_left_traj=pre_feet_traj['lfoot']
        self.pre_right_traj=pre_feet_traj['rfoot']

        file_path=os.path.join(debug_folder, "Pos Lfoot pre trj")
        with open(file_path, "w") as file:
            for i in range(len(self.pre_left_traj)):
                file.writelines(" ".join(map(str, self.pre_left_traj[i][0]['pos'][3:6]))+ "\n")
            
        file_path=os.path.join(debug_folder, "Pos Rfoot pre trj")
        with open(file_path, "w") as file:
            for i in range(len(self.pre_left_traj)):
                file.writelines(" ".join(map(str, self.pre_right_traj[i][0]['pos'][3:6]))+ "\n")

      
        self.ref=new.references(self.foot_trajectory_generator,self.footstep_planner,0)  
        #new.plot_2d_ref(self.ref,self.footstep_planner)  #create a new image of CoM evolution
        #new.plot_9_subplots(self.ref,self.footstep_planner)
        #new.plot_3d_ref(self.ref,self.footstep_planner)
        #new.plot_com_xy_trajectory(self.ref)


             # initialize MPC controller
        self.contact_ref= self.footstep_planner.position_contacts_ref

        self.centroidal_mpc=centroidal_mpc_vertices.centroidal_mpc(self.initial,self.footstep_planner,self.params,self.ref,
                                                                    self.pre_left_traj,self.pre_right_traj)
                
        #For Logger2
        self.mpc_desired_feet = {'lfoot': {'ang': np.zeros(3),'pos': np.zeros(3)},
                                 'rfoot': {'ang': np.zeros(3),'pos': np.zeros(3)} }

        self.actual_feet_pose ={'lfoot': {'ang': np.zeros(3),'pos': np.zeros(3)},
                                'rfoot': {'ang': np.zeros(3),'pos': np.zeros(3)} }

        # initialize logger and plots
        self.logger = Logger(self.initial)
        self.logger.initialize_plot(frequency=10)
        self.logger2 = Logger2(self.initial,self.footstep_planner)
        self.logger2.initialize_plot(frequency=1)
        self.logger3 = Logger3(self.initial)
        self.logger3.initialize_plot(frequency=10)

        

        

    def customPreStep(self):
        self.current = self.retrieve_state() 
        # get references using mpc SOLVE THE MPC
        robot_state, contact= self.centroidal_mpc.solve(self.current, self.time)

        self.desired['com']['pos'] = robot_state['com']['pos']
        self.desired['com']['vel'] = robot_state['com']['vel']
        self.desired['com']['acc'] = robot_state['com']['acc']
        self.desired['hw']['val'] = robot_state['hw']['val']
        next_des_pose_swing = robot_state['next_des_pose_swing']['val']
        predicted_rfoot_pos=robot_state['x_collect_right']['val'] 
        predicted_lfoot_pos=robot_state['x_collect_left']['val'] 

        self.com_ref['com']['pos'][0] = self.ref['pos_x'][self.time]
        self.com_ref['com']['pos'][1] = self.ref['pos_y'][self.time]
        self.com_ref['com']['pos'][2] = self.ref['pos_z'][self.time]

        # get foot trajectories
        feet_trajectories = self.foot_trajectory_generator.generate_feet_trajectories_at_time(self.time)
        for foot in ['lfoot', 'rfoot']:
            for key in ['pos', 'vel', 'acc']:
                self.desired[foot][key] = feet_trajectories[foot][key]

        actual_feet_trajectories = self.foot_trajectory_generator.generate_feet_trajectories_at_time(self.time-1)

        self.update_swing_trj =  robot_state['update_swing_trj']['val'] 
        
        #print(f'from simulation.py -> self.footstep_planner.plan:{self.footstep_planner.plan}')

        swing_foot=''
        
        if self.params['update_contact']=='YES' :
            if ( robot_state['swing_foot']['lfoot']== 1 and robot_state['swing_foot']['rfoot']== 0 ):
                swing_foot = 'lfoot' 
            elif ( robot_state['swing_foot']['rfoot']== 1 and robot_state['swing_foot']['lfoot']== 0 ):
                swing_foot = 'rfoot' 
            elif ( robot_state['swing_foot']['rfoot']== 0 and robot_state['swing_foot']['lfoot']== 0 ):
                swing_foot = self.params['first_swing']

            print(f'swing_foot:{swing_foot}, self.counter:{self.counter}\n self.update_swing_trj:{self.update_swing_trj} , self.sim_update_swing_trj:{self.sim_update_swing_trj} , next_des_pose_swing:{next_des_pose_swing} ')
            
            
            if self.update_swing_trj==1 and self.sim_update_swing_trj==0:
                self.counter=robot_state['counter']['val']
                self.sim_update_swing_trj=1
                #self.foot_trajectory_generator.generate_modified_swing_foot_trajectories(self.time,self.current[swing_foot]['pos'],self.current[swing_foot]['vel'],self.current[swing_foot]['acc'],next_des_pose_swing)
                nominal_feet_trj = self.foot_trajectory_generator.generate_feet_trajectories_at_time(self.time+robot_state['N']['val']-robot_state['correction']['val'])
                planed_contact_pose=nominal_feet_trj[swing_foot]['pos']
                self.foot_trajectory_generator.generate_modified_swing_foot_trajectories_bounded(self.time,self.current[swing_foot]['pos'],self.current[swing_foot]['vel'],self.current[swing_foot]['acc'],next_des_pose_swing,planed_contact_pose)
                #def generate_modified_swing_foot_trajectories(self,time,actual_pose,actual_vel,actual_acc,new_pose):
                print(f'Generation of the trajectory- actual_pose:{actual_feet_trajectories[swing_foot]['pos']}')
                print(f'actual_pose returned by retrieve_State()->\nactual_rfoot_pos:{self.current[swing_foot]['pos']}')
                new_feet_trajectories = self.foot_trajectory_generator.generate_modified_feet_trajectories_at_time(self.time)
                
            if self.update_swing_trj==1 and self.sim_update_swing_trj==1:
                if swing_foot=='lfoot':
                    print(f'predicted_right_foot_pos:{predicted_lfoot_pos[3:6,:]}')
                    self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(self.time)+1]['pos'] = predicted_lfoot_pos[3:6,robot_state['N']['val']-robot_state['correction']['val']]
                    self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(self.time)+1]['ang'] = predicted_lfoot_pos[0:3,robot_state['N']['val']-robot_state['correction']['val']]
                                
                if swing_foot=='rfoot':
                    print(f'predicted_right_root_pos:{predicted_rfoot_pos[3:6,:]}')   
                    self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(self.time)+1]['pos'] = predicted_rfoot_pos[3:6,robot_state['N']['val']-robot_state['correction']['val']]#mean the swing foot is the right one-> update the contact position of the swing foot
                    self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(self.time)+1]['ang'] = predicted_rfoot_pos[0:3,robot_state['N']['val']-robot_state['correction']['val'] ]#mean the swing foot is the right one-> update the contact position of the swing foot
                
                new_feet_trajectories = self.foot_trajectory_generator.generate_modified_feet_trajectories_at_time(self.time)
                print(f'new_feet_trajectories at time {self.time}:{new_feet_trajectories}')
                self.desired[swing_foot]['pos'] = new_feet_trajectories['pos']
                self.desired[swing_foot]['vel'] = new_feet_trajectories['vel']
                self.desired[swing_foot]['acc'] = new_feet_trajectories['acc']

                self.counter-=1
                if self.counter<0:
                    self.centroidal_mpc.reset_update_swing_trj()
                    self.update_swing_trj=0
                    self.sim_update_swing_trj=0

        
        # set torso and base references to the average of the feet
        for link in ['torso', 'base']:
            for key in ['pos', 'vel', 'acc']:
                self.desired[link][key] = (self.desired['lfoot'][key][:3] + self.desired['rfoot'][key][:3]) / 2.



        if self.preferences[2] == 'YES'  and self.preferences[0] != 'simple' :
                self.desired['lfoot']['pos'][ 0 ] = 0                
                self.desired['lfoot']['pos'][ 1 ] = 0               
                #self.desired['lfoot']['pos'][ 2 ] = robot_state['ang_contact_left']['val'][2] 
                self.desired['lfoot']['pos'][ 2 ] = 0
                self.desired['lfoot']['pos'][3:6] = robot_state['pos_contact_left']['val'] 

                self.desired['rfoot']['pos'][ 0 ] = 0                
                self.desired['rfoot']['pos'][ 1 ] = 0 
                #self.desired['rfoot']['pos'][ 2 ] = robot_state['ang_contact_right']['val'][2]
                self.desired['rfoot']['pos'][ 2 ] = 0
                self.desired['rfoot']['pos'][3:6] = robot_state['pos_contact_right']['val']

                self.desired['lfoot']['vel'][ 0 ] = 0
                self.desired['lfoot']['vel'][ 1 ] = 0                
                #self.desired['lfoot']['vel'][ 2 ] = robot_state['ang_vel_left']['val'][2]
                self.desired['lfoot']['vel'][ 2 ] = 0
                self.desired['lfoot']['vel'][3:6] = robot_state['lin_vel_left']['val']

                self.desired['rfoot']['vel'][ 0 ] = 0
                self.desired['rfoot']['vel'][ 1 ] = 0                
                #self.desired['rfoot']['vel'][ 2 ] = robot_state['lin_vel_right']['val'][2]
                self.desired['rfoot']['vel'][ 2 ] = 0
                self.desired['rfoot']['vel'][3:6] = robot_state['lin_vel_right']['val']

                #print("Real walk enable") 
                #if self.preferences[0] == 'full_model' :
                #   self.desired['lfoot']['pos'][0:3] = robot_state['ang_contact_left']['val'] 
                #   self.desired['rfoot']['pos'][0:3] = robot_state['ang_contact_right']['val']
                #   self.desired['lfoot']['vel'][0:3] = robot_state['ang_vel_left']['val'] 
                #   self.desired['rfoot']['vel'][0:3] = robot_state['ang_vel_right']['val']
                   

        inertia=self.current['inertia']['value'] 

        if self.preferences[3] == "YES"  and self.preferences[5] != 'base ':
            self.desired['torso']['vel']=np.linalg.inv(inertia)@robot_state['hw']['val']
        if self.preferences[3] == "YES"  and self.preferences[5] == 'base ':
            self.desired['base']['vel']=np.linalg.inv(inertia)@robot_state['hw']['val']
        if self.preferences[4] == 0  and self.preferences[5] != 'base ':
            self.desired['torso']['acc']=np.zeros(3)
        elif self.preferences[4] == 0  and self.preferences[5] == 'base ':
          self.desired['base']['acc']= np.zeros(3)
        if self.preferences[4] == 'YES'  and self.preferences[5] != 'base ':
            self.desired['torso']['acc']=np.linalg.inv(inertia)@ (robot_state['hw']['dot']-np.cross(self.desired['torso']['vel'],inertia@ self.desired['torso']['vel']))
        if self.preferences[4] == 'YES'  and self.preferences[5] == 'base ':
            self.desired['torso']['acc']=np.linalg.inv(inertia)@ (robot_state['hw']['dot']-np.cross(self.desired['base']['vel'],inertia@ self.desired['base']['vel']))
                    
    
        # get torque commands using inverse dynamics
        commands = self.id.get_joint_torques(self.desired, self.current, contact) 
        
        # set acceleration commands
        for i in range(self.params['dof'] - 6):
            self.hrp4.setCommand(i + 6, commands[i])

        # Store the desired foot positions from MPC (even if preferences[2] == 'NO')
        # the MPC return the next desired position 
        self.mpc_desired_feet = {'lfoot': {'ang': robot_state['ang_contact_left']['val']  , 'pos': robot_state['pos_contact_left']['val']},
                                 'rfoot': {'ang': robot_state['ang_contact_right']['val'] , 'pos': robot_state['pos_contact_right']['val']} }

        #get the actual pose after we apply the joint commands
        self.actual_feet_pose=self.get_actual_feet_pose()

        # log and plot
        self.logger.log_data( self.desired,self.current)
        self.logger.update_plot(self.time)

        self.logger2.log_data(self.current, self.mpc_desired_feet,self.actual_feet_pose)
        self.logger2.update_plot(self.time)

        self.logger3.log_data(self.desired,self.current)
        self.logger3.update_plot(self.time)

        self.time += 1# the clock that counts the time
        print(self.time)





    def retrieve_state(self):
        # com and torso pose (orientation and position)
        com_position = self.hrp4.getCOM()
        torso_orientation = get_rotvec(self.hrp4.getBodyNode('torso').getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).rotation())
        base_orientation  = get_rotvec(self.hrp4.getBodyNode('body' ).getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).rotation())

        # feet poses (orientation and position)
        l_foot_transform = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_orientation = get_rotvec(l_foot_transform.rotation())
        l_foot_position = l_foot_transform.translation()
        left_foot_pose = np.hstack((l_foot_orientation, l_foot_position))

        r_foot_transform = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_orientation = get_rotvec(r_foot_transform.rotation())
        r_foot_position = r_foot_transform.translation()
        right_foot_pose = np.hstack((r_foot_orientation, r_foot_position))

        # velocities
        com_velocity = self.hrp4.getCOMLinearVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        torso_angular_velocity = self.hrp4.getBodyNode('torso').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        base_angular_velocity = self.hrp4.getBodyNode('body').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_spatial_velocity = self.lsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_spatial_velocity = self.rsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())

        # print("Torso angl m")
        # print(self.torso.getAngularMomentum(com_position))
        # print("Base angl m")
        # print(self.base.getAngularMomentum(com_position))
        # print("Lfoot angl m")
        # print(self.lsole.getAngularMomentum(com_position))
        # print("Rfoot angl m")
        # print(self.rsole.getAngularMomentum(com_position))
        # print("L_hip_p angl m")
        # print(self.l_hip_p.getAngularMomentum(com_position))


        # Get angular momentum
        if self.preferences[1] == 'torso + base + l_hip' :
         angular_momentum_at_com=    self.torso.getAngularMomentum(com_position)+\
                                    self.base.getAngularMomentum(com_position)*1+\
                                    self.lsole.getAngularMomentum(com_position)*0+\
                                    self.rsole.getAngularMomentum(com_position)*0+\
                                    self.l_hip_p.getAngularMomentum(com_position)
        if self.preferences[1] == 'torso' :
            angular_momentum_at_com=    self.torso.getAngularMomentum(com_position)
        if self.preferences[1] == 'base' :
            angular_momentum_at_com=self.base.getAngularMomentum(com_position)
        if self.preferences[1]=='semi' :
                  angular_momentum_at_com=    self.torso.getAngularMomentum(com_position)+\
                                    self.base.getAngularMomentum(com_position)+\
                                    self.lsole.getAngularMomentum(com_position)+\
                                    self.rsole.getAngularMomentum(com_position)+\
                                    self.l_hip_p.getAngularMomentum(com_position)
        if self.preferences[1] == 'full' :
            angular_momentum_at_com=np.zeros(3)
            for body in hrp4.getBodyNodes():
                w_R_link_i=body.getWorldTransform().rotation()
                angular_momentum_at_com+=w_R_link_i@body.getAngularMomentum(w_R_link_i.T@(-com_position+body.getCOM()))
                #angular_momentum_at_com+=body.getAngularMomentum(com_position)
        
        
        if self.preferences[5] != 'base':
         w_R_com=torso_orientation
        else : w_R_com=base_orientation
        ###############
        inertia_at_com = np.zeros((3, 3))  #CoM_I_CoM
        for body in self.hrp4.getBodyNodes():
            mass_i = body.getMass()
            if mass_i == 0:
                continue
                
            c_i = body.getCOM(withRespectTo=dart.dynamics.Frame.World())#Position of the CoM of link_i
            # distance = vector from Com_pos to c_i
            delta_i = c_i - com_position 
            inertia_i_local = body.getInertia().getMoment()  #inertia matrix wrt local frame
            w_R_i = body.getTransform().rotation() #{w}^R_{i} --> Rot mat. describing orient. of link_i frame wrt World frame

            CoM_R_i=w_R_com.T @ w_R_i            # {CoM}^R_{i}={w}^R_{CoM}.T @ {w}^R_{i}

            # Generalized Huygen-steiner th says:
                # [ CoM_I_i=com_R_i * i_I_i * com_R_i^T ] + mass *[ ||dist||^2*Id-dist @ dist.T ]
            
                    #Note CoM_I_i=com_R_i * i_I_i * com_R_i^T=CoM_I_CoM   ok

            inertia_i_wrt_CoM = CoM_R_i @ inertia_i_local @ CoM_R_i.T   # CoM_I_i=com_R_i*i_I_i*com_R_i^T
            inertia_translated = inertia_i_wrt_CoM + mass_i * (
                np.dot(delta_i, delta_i) * np.identity(3) - np.outer(delta_i, delta_i) # + mass*||dist||^2*Id-dist @ dist.T 
             )

            inertia_at_com += inertia_translated
        #print(f'inertia_at_com:\n{inertia_at_com}')
           
        
        # create state dict
        return {
            'lfoot': {'pos': left_foot_pose,
                      'vel': l_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'rfoot': {'pos': right_foot_pose,
                      'vel': r_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'com'  : {'pos': com_position,
                      'vel': com_velocity,
                      'acc': np.zeros(3)},
            'torso': {'pos': torso_orientation,
                      'vel': torso_angular_velocity,
                      'acc': np.zeros(3)},
            'base' : {'pos': base_orientation,
                      'vel': base_angular_velocity,
                      'acc': np.zeros(3)},
            'joint': {'pos': self.hrp4.getPositions(),
                      'vel': self.hrp4.getVelocities(),
                      'acc': np.zeros(self.params['dof'])},
            'hw'   : {'val': angular_momentum_at_com},
            'theta_hat':{'val':np.zeros(3)},
            'inertia':{'value': inertia_at_com}    
        }
    
    def get_actual_feet_pose(self):
        l_foot_transform = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_orientation = get_rotvec(l_foot_transform.rotation())
        l_foot_position = l_foot_transform.translation()

        r_foot_transform = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_orientation = get_rotvec(r_foot_transform.rotation())
        r_foot_position = r_foot_transform.translation()
           
        return {'lfoot':{'ang': l_foot_orientation,'pos': l_foot_position},
                'rfoot':{'ang': r_foot_orientation,'pos': r_foot_position} }




if __name__ == "__main__":
    world = dart.simulation.World()

    urdfParser = dart.utils.DartLoader()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hrp4   = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "hrp4.urdf"))
    ground = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "ground.urdf"))
    box = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "box.urdf"))
    world.addSkeleton(hrp4)
    world.addSkeleton(ground)
    world.addSkeleton(box)
    world.setGravity([0, 0, -9.81])
    world.setTimeStep(0.01)

    # set default inertia
    default_inertia = dart.dynamics.Inertia(1e-8, np.zeros(3), 1e-10 * np.identity(3))
    for body in hrp4.getBodyNodes():
        if body.getMass() == 0.0:
            body.setMass(1e-8)
            body.setInertia(default_inertia)

    node = Hrp4Controller(world, hrp4)

    # create world node and add it to viewer
    viewer = dart.gui.osg.Viewer()
    node.setTargetRealTimeFactor(100) # speed up the visualization by 10x
    node.setTargetRealTimeFactor(100) # speed up the visualization by 10x
    viewer.addWorldNode(node)

    #viewer.setUpViewInWindow(0, 0, 1920, 1080)
    viewer.setUpViewInWindow(0, 0, 1280, 720)
    #viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([5., -1., 1.5],
                                 [1.,  0., 0.5],
                                 [0.,  0., 1. ])
    viewer.run()
