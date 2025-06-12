import numpy as np
import dartpy as dart
import copy
from utils import *
import os
import centroidal_mpc_vertices_payload as centroidal_mpc_vertices

import footstep_planner_vertices 
import inverse_dynamics_payload as id
import foot_trajectory_generator as ftg
from logger import Logger
from logger2 import Logger2
from logger3 import Logger3 
from logger_theta import Logger_theta
from functions import *


debug_folder= "code/Debug"
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
            'ss_duration': 7*10,
            'ds_duration': 3*10,
            'first_swing': 'rfoot',
            'µ': 0.5,
            'N': 10,
            'dof': self.hrp4.getNumDofs(),
            'mass': self.hrp4.getMass(),
            'update_contact': 'YES',
            'mpc_rate': 1  #  rate at which the MPC is updated ( 10 means after every 10 time steps)
        }
        self.counter=0
        self.mpc_robot_state = np.zeros(34)
        self.mpc_contact = np.zeros(6)
      

        self.params['eta'] = np.sqrt(self.params['g'] / self.params['h'])
        
        print("time_step:")
        print(self.params['world_time_step'])
              
        # robot links
        self.lsole = hrp4.getBodyNode('l_sole')
        self.rsole = hrp4.getBodyNode('r_sole')
        self.torso = hrp4.getBodyNode('torso')
        self.base  = hrp4.getBodyNode('body')
        self.l_hip_p= hrp4.getBodyNode('L_HIP_P_LINK')

        # for i in range(hrp4.getNumJoints()):
        #     joint = hrp4.getJoint(i)
        #     dim = joint.getNumDofs()

        #     # set floating base to passive, everything else to torque
        #     if   dim == 6: joint.setActuatorType(dart.dynamics.ActuatorType.PASSIVE)
        #     elif dim == 1: joint.setActuatorType(dart.dynamics.ActuatorType.FORCE)

        # set initial configuration
        initial_configuration = {'CHEST_P': 0., 'CHEST_Y': 0., 'NECK_P': 0., 'NECK_Y': 0., \
                                 'R_HIP_Y': 0., 'R_HIP_R': -3., 'R_HIP_P': -25., 'R_KNEE_P': 50., 'R_ANKLE_P': -25., 'R_ANKLE_R':  3., \
                                 'L_HIP_Y': 0., 'L_HIP_R':  3., 'L_HIP_P': -25., 'L_KNEE_P': 50., 'L_ANKLE_P': -25., 'L_ANKLE_R': -3., \
                                 'R_SHOULDER_P': 4., 'R_SHOULDER_R': -8., 'R_SHOULDER_Y': 0., \
                                 'L_SHOULDER_P': 4., 'L_SHOULDER_R':  8.}

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
       

        # selection matrix for redundant dofs
        redundant_dofs = [ \
            "NECK_Y", "NECK_P", \
            "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P", \
            "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P"]
        
        # initialize inverse dynamics
        self.id = id.InverseDynamics(self.hrp4, redundant_dofs)

             # initialize footstep planner
        reference = reference = [(0.15, 0., 0)] * 5 + [(0.15, 0.0, 0)] * 3 +[(0.15, 0.0, 0)] * 3 + [(0.13, 0, 0)] * 4 + [(0.1, 0., 0)] * 2 +[(0.,0,0)]*3
    
        self.footstep_planner = footstep_planner_vertices.FootstepPlanner(
            reference,
            self.initial['lfoot']['pos'],
            self.initial['rfoot']['pos'],
            self.params
            )
       
    
        # initialize foot trajectory generator
        self.foot_trajectory_generator = ftg.FootTrajectoryGenerator(
            self.initial, 
            self.footstep_planner, 
            self.params
            )
        
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

      
        self.ref=references(self.foot_trajectory_generator,self.footstep_planner)  
        print("ref_length:")
        #print(len(self.ref['pos_x']))
        #self.ref=references(self.foot_trajectory_generator,self.footstep_planner,1)  FOR SEE GRAHP
        
             # initialize MPC controller
        self.contact_ref= self.footstep_planner.position_contacts_ref
         
        print("foot_step_plan")
        for i in range(len(self.footstep_planner.plan)):
            print(self.footstep_planner.plan[i])
            print()
    
       
        self.centroidal_mpc=centroidal_mpc_vertices.centroidal_mpc(
            self.initial, 
            self.footstep_planner, 
            self.params,
            self.ref,
            self.pre_left_traj,
            self.pre_right_traj
          )
       
        #For Logger2
        self.mpc_desired_feet = {
            'lfoot': {
                'ang': np.zeros(3),
                'pos': np.zeros(3)
            },
            'rfoot': {
                'ang': np.zeros(3),
                'pos': np.zeros(3)
            }
        }

        self.actual_feet_pose ={
            'lfoot': {
                'ang': np.zeros(3),
                'pos': np.zeros(3)
            },
            'rfoot': {
                'ang': np.zeros(3),
                'pos': np.zeros(3)
            }
        }


        # initialize logger and plots
        self.logger = Logger(self.initial)
        self.logger.initialize_plot(frequency=10)
        self.logger2 = Logger2(self.initial,self.footstep_planner)
        self.logger2.initialize_plot(frequency=10)
        self.logger3 = Logger3(self.initial)
        self.logger3.initialize_plot(frequency=10)
        self.logger_theta = Logger_theta(self.initial)
        self.logger_theta.initialize_plot(frequency=10)

        feet_trajectories = self.foot_trajectory_generator.generate_feet_trajectories_at_time(self.time)
        for foot in ['lfoot', 'rfoot']:
            self.logger2.draw_desired_swing_foot_position(feet_trajectories[foot]['pos'])


        
    def customPreStep(self):
        # create current and desired states
        if  self.time >1300 and self.time < 1400:
            force = np.array([.0, 0.0, -0.0])  # Newtons
            self.base.addExtForce(force)
            self.torso.addExtForce(force)
        self.current = self.retrieve_state() 
       
        if self.time % self.params['mpc_rate'] == 0:
            self.mpc_robot_state, self.mpc_contact= self.centroidal_mpc.solve(self.current, self.time)
        #lip_state, contact = self.centroidal_mpc.solve(self.current, self.time)
        
        robot_state = self.mpc_robot_state
        contact = self.mpc_contact
        self.desired['com']['pos'] = robot_state['com']['pos']
        self.desired['com']['vel'] = robot_state['com']['vel']
        self.desired['com']['acc'] = robot_state['com']['acc']
        self.desired['hw']['val'] = robot_state['hw']['val']
        self.desired['theta_hat']['val']=robot_state['theta_hat']['val']


        
        
        com_ref=np.zeros(3)
        com_ref[0] = self.ref['pos_x'][self.time]
        com_ref[1] = self.ref['pos_y'][self.time]
        com_ref[2] = self.ref['pos_z'][self.time]
        self.counter   = robot_state['counter']['val']
        self.com_ref['com']['pos'] = com_ref
       
        # get foot trajectories
        feet_trajectories = self.foot_trajectory_generator.generate_feet_trajectories_at_time(self.time)
        for foot in ['lfoot', 'rfoot']:
            for key in ['pos', 'vel', 'acc']: 
            #for key in ['vel', 'acc']:
                self.desired[foot][key] = feet_trajectories[foot][key]
            print(f'Pos_contact_desired {foot}')
            print(self.desired[foot]['pos'][3:6])

        print("Next contact pos in contact list:")
        print(self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(self.time)+1]['pos'])

       
        file_path=os.path.join(debug_folder, "MPC_pose_contact_ref")        
        with open(file_path, "w") as file:
            for i in range(20):
                file.write("\n".join(map(str, self.footstep_planner.plan[i]['pos'].T)) + "\n")
                file.write("end"+ "\n")
         
      

        if self.counter==1:
            self.mpc_robot_state['counter']['val'] = 0
            self.centroidal_mpc.reset_update_swing_trj()
            self.update_swing_trj=0
            self.sim_update_swing_trj=0
            if contact == 'lfoot':
                swing_foot = 'rfoot'
            else:
                swing_foot = 'lfoot'

            new_contact_feet_pose = self.foot_trajectory_generator.generate_feet_trajectories_at_time(self.time+self.params['N']*self.params['mpc_rate'])
            self.logger2.draw_desired_swing_foot_position(new_contact_feet_pose[swing_foot]['pos'])
        
         # To draw the real contact position
        if self.time > 2*(self.params['ds_duration']+self.params['ss_duration'])-self.params['ds_duration']:
            if self.footstep_planner.get_phase_at_time(self.time) == 'ds' and self.footstep_planner.get_phase_at_time(self.time + self.params['ds_duration']-1) == 'ds':
                contact_pre_ds = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(self.time-1)]['foot_id']
                print(f'Contact phase: {contact_pre_ds}')
                if contact_pre_ds == 'lfoot':
                    self.logger2.draw_current_feet(self.current['rfoot'])
                if contact_pre_ds == 'rfoot':
                    self.logger2.draw_current_feet(self.current['lfoot'])
            
        # set torso and base references to the average of the feet
        for link in ['torso', 'base']:
            for key in ['pos', 'vel', 'acc']:
                self.desired[link][key] = (self.desired['lfoot'][key][:3] + self.desired['rfoot'][key][:3]) / 2.


      
    
        # get torque commands using inverse dynamics
        commands = self.id.get_joint_torques(self.desired, self.current, contact) 
        
        # set acceleration commands
        for i in range(self.params['dof'] - 6):
            self.hrp4.setCommand(i + 6, commands[i])
        

        self.mpc_desired_feet = {'lfoot': {'ang': np.zeros(3)  , 'pos': robot_state['pos_contact_left']['val']},
                                 'rfoot': {'ang': np.zeros(3)  , 'pos': robot_state['pos_contact_right']['val']} }

        #get the actual pose after we apply the joint commands
        self.actual_feet_pose=self.get_actual_feet_pose()
        # log and plot
        self.logger.log_data( self.desired,self.com_ref)
        self.logger.update_plot(self.time)
        self.logger_theta.log_data(self.desired)
        self.logger_theta.update_plot(self.time)
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

        # compute total contact force
        force = np.zeros(3)
        for contact in world.getLastCollisionResult().getContacts():
            force += contact.force
            #print(contact.point)
       

        # Get angular momentum

      
        angular_momentum_at_com=np.zeros(3)
        for body in hrp4.getBodyNodes():
                w_R_link_i=body.getWorldTransform().rotation()
                #angular_momentum_at_com+=w_R_link_i@body.getAngularMomentum((-com_position+body.getCOM()))
                angular_momentum_at_com+=-w_R_link_i@body.getAngularMomentum(w_R_link_i.T@(com_position-body.getWorldTransform().translation()))
        
          
        
       
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
    hrp4   = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "hrp4_payload.urdf"))
    ground = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "ground.urdf"))
    #stair = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "stairs.urdf"))
    box = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "box.urdf"))
    world.addSkeleton(hrp4)
    world.addSkeleton(ground)
    world.addSkeleton(box)
    #world.addSkeleton(stair)
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
