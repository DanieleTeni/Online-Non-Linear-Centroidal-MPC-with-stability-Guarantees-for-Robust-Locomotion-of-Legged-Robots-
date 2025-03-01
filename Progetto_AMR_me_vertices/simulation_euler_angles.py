import numpy as np
import dartpy as dart
import copy
from utils import *
import os
import ismpc
import centroidal_mpc
import footstep_planner
import inverse_dynamics as id
import filter
import foot_trajectory_generator as ftg
from logger import Logger
import new
from scipy.spatial.transform import Rotation as R

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
        }
        self.params['eta'] = np.sqrt(self.params['g'] / self.params['h'])
        print("time_step:")
        print(self.params['world_time_step'])
              
        # robot links
        self.lsole = hrp4.getBodyNode('l_sole')
        self.rsole = hrp4.getBodyNode('r_sole')
        self.torso = hrp4.getBodyNode('torso')
        self.base  = hrp4.getBodyNode('body')

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
                                 'R_SHOULDER_P': 4., 'R_SHOULDER_R': -8., 'R_SHOULDER_Y': 0., 'R_ELBOW_P': -25., \
                                 'L_SHOULDER_P': 4., 'L_SHOULDER_R':  8., 'L_SHOULDER_Y': 0., 'L_ELBOW_P': -25.}

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
        reference = [(0.07, 0., 0)] * 5 + [(0.05, 0., -0.0)] * 10 + [(0.05, 0., 0.)] * 10
        self.footstep_planner = footstep_planner.FootstepPlanner(
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

        with open("Pos Lfoot pre trj", "w") as file:
            for i in range(len(self.pre_left_traj)):
                file.writelines(" ".join(map(str, self.pre_left_traj[i][0]['pos'][3:6]))+ "\n")
            
        
        with open("Pos Rfoot pre trj", "w") as file:
            for i in range(len(self.pre_left_traj)):
                file.writelines(" ".join(map(str, self.pre_right_traj[i][0]['pos'][3:6]))+ "\n")

      
        self.ref=new.references(self.foot_trajectory_generator,self.footstep_planner)  
        print("ref_length:")
        #print(len(self.ref['pos_x']))
        #self.ref=new.references(self.foot_trajectory_generator,self.footstep_planner,1)  FOR SEE GRAHP
        new.plot_2d_ref(self.ref,self.footstep_planner)  #create a new image of CoM evolution
        new.plot_9_subplots(self.ref,self.footstep_planner)
        new.plot_3d_ref(self.ref,self.footstep_planner)


             # initialize MPC controller
        self.contact_ref= self.footstep_planner.position_contacts_ref
        # print("contact Ref")
        # print(self.contact_ref['contact_left'][199])
        # print(self.contact_ref['contact_left'][201])
        # print(self.contact_ref['contact_right'][199])
        # print(self.contact_ref['contact_right'][300])
        #print(self.contact_ref['contact_right'][199].shape[1])       
        print("foot_step_plan")
        for i in range(len(self.footstep_planner.plan)):
            print(self.footstep_planner.plan[i])
            print()
        self.mpc = ismpc.Ismpc(
            self.initial, 
            self.footstep_planner, 
            self.params,
            self.ref
            )

        self.centroidal_mpc=centroidal_mpc.centroidal_mpc(
            self.initial, 
            self.footstep_planner, 
            self.params,
            self.ref,
            self.pre_left_traj,
            self.pre_right_traj
        )
        #self.logger = Logger(self.initial)
        #self.logger.initialize_plot(frequency=10)
        







    def customPreStep(self):
        print(f'Simulation.py- Custom PreStep- time:{self.time}')
        self.current = self.retrieve_state() 
        print(f'Simulation.py-self.retrieve_state()- time:{self.current}')


        #update kalman filter

        # get references using mpc SOLVE THE MPC
        robot_state, contact= self.centroidal_mpc.solve(self.current, self.time)
        
        self.desired['com']['pos'] = robot_state['com']['pos']
        self.desired['com']['vel'] = robot_state['com']['vel']
        self.desired['com']['acc'] = robot_state['com']['acc']
        #self.desired['hw']['val'] = robot_state['hw']['val']
        
        self.com_ref['com']['pos'][0] = self.ref['pos_x'][self.time]
        self.com_ref['com']['pos'][1] = self.ref['pos_y'][self.time]
        self.com_ref['com']['pos'][2] = self.ref['pos_z'][self.time]
        
        # get foot trajectories
        feet_trajectories = self.foot_trajectory_generator.generate_feet_trajectories_at_time(self.time)
        for foot in ['lfoot', 'rfoot']:
            for key in ['pos', 'vel', 'acc']:
                self.desired[foot][key] = feet_trajectories[foot][key]

        # set torso and base references to the average of the feet
        for link in ['torso', 'base']:
            for key in ['pos', 'vel', 'acc']:
                self.desired[link][key] = (self.desired['lfoot'][key][:3] + self.desired['rfoot'][key][:3]) / 2.

        # get torque commands using inverse dynamics
        commands = self.id.get_joint_torques(self.desired, self.current, contact) 
        
        # set acceleration commands
        for i in range(self.params['dof'] - 6):
            self.hrp4.setCommand(i + 6, commands[i])

        #self.logger.update_plot(self.time)
        self.time += 1

  

    def retrieve_state(self): #Orientation of the TORSO as the orientation of the CoM

        # feet poses (orientation and position)
        l_foot_transform = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_euler = self.get_euler_angles_from_matrix(l_foot_transform.rotation())
        l_foot_position = l_foot_transform.translation()
        left_foot_pose = np.hstack((l_foot_euler, l_foot_position))

        r_foot_transform = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_euler = self.get_euler_angles_from_matrix(r_foot_transform.rotation())
        r_foot_position = r_foot_transform.translation()
        right_foot_pose = np.hstack((r_foot_euler, r_foot_position))

        # feet velocities
        l_foot_spatial_velocity_wrt_world = self.lsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_angular_velocity = l_foot_transform.rotation().T@l_foot_spatial_velocity_wrt_world[0:3]
        l_foot_vel_euler_rates = self.angular_velocity_to_euler_rates(l_foot_angular_velocity,l_foot_euler)
        l_foot_spatial_velocity=np.hstack([l_foot_vel_euler_rates,l_foot_spatial_velocity_wrt_world[3:6]])

        r_foot_spatial_velocity_wrt_world = self.rsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_angular_velocity = r_foot_transform.rotation().T@ r_foot_spatial_velocity_wrt_world[0:3]
        r_foot_vel_euler_rates = self.angular_velocity_to_euler_rates(r_foot_angular_velocity,r_foot_euler)
        r_foot_spatial_velocity=np.hstack([r_foot_vel_euler_rates,r_foot_spatial_velocity_wrt_world[3:6]])

        #Com Torso and BASE pose        
        torso_transform = self.hrp4.getBodyNode('torso').getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        torso_position=torso_transform.translation()
        torso_euler = self.get_euler_angles_from_matrix(torso_transform.rotation())
        torso_pose= np.hstack((torso_euler, torso_position))
        
        com_position = self.hrp4.getCOM(inCoordinatesOf=dart.dynamics.Frame.World())
        
        base_transform = self.hrp4.getBodyNode('body').getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        base_position = base_transform.translation()
        base_euler = self.get_euler_angles_from_matrix(base_transform.rotation())
        base_pose = np.hstack((base_euler, base_position))
        
        #Com Torso and BASE velocity
        com_velocity = self.hrp4.getCOMLinearVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        
        torso_spatial_velocity_world = self.hrp4.getBodyNode('torso').getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        torso_angular_velocity = torso_transform.rotation().T@ torso_spatial_velocity_world[0:3]
        torso_vel_euler_rates = self.angular_velocity_to_euler_rates(torso_angular_velocity,torso_euler)
        torso_spatial_velocity=np.hstack([torso_vel_euler_rates,torso_spatial_velocity_world[3:6]])
        
        base_spatial_velocity_world = self.hrp4.getBodyNode('body').getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        base_angular_velocity = base_transform.rotation().T@ base_spatial_velocity_world[0:3]
        base_vel_euler_rates = self.angular_velocity_to_euler_rates(base_angular_velocity,base_euler)
        base_spatial_velocity=np.hstack([base_vel_euler_rates,torso_spatial_velocity_world[3:6]])


        # Get angular momentum
        angular_momentum_at_com=    self.torso.getAngularMomentum(com_position)+\
                                    self.base.getAngularMomentum(com_position)+\
                                    self.lsole.getAngularMomentum(com_position)+\
                                    self.rsole.getAngularMomentum(com_position)
                                       
        # create state dict
        return {
            'lfoot': {'pos': left_foot_pose,  #L_FOOT POSE
                      'vel': l_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'rfoot': {'pos': right_foot_pose, #R_FOOT POSE
                      'vel': r_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'com'  : {'pos': com_position,  #CoM POSITION
                      'vel': com_velocity,
                      'acc': np.zeros(3)},
            'torso': {'pos': torso_pose,    #TORSO POSE
                      'vel': torso_spatial_velocity,
                      'acc': np.zeros(6)},
            'base' : {'pos': base_pose,     #BASE POSE
                      'vel': base_spatial_velocity,
                      'acc': np.zeros(6)},
            'joint': {'pos': self.hrp4.getPositions(),      #JOINT CONFIGURATION
                      'vel': self.hrp4.getVelocities(),
                      'acc': np.zeros(self.params['dof'])},
            'hw'   : {'val': angular_momentum_at_com},      #Hw VALUE
            'theta_hat':{'val':np.zeros(3)}                 #THETA HAT VALUE
        }

    def get_euler_angles_from_matrix(self,rotation_matrix):
        euler_angles = R.from_matrix(rotation_matrix).as_euler('ZYX', degrees=False)  # Ordine ZYX
        return euler_angles  #(yaw, pitch, roll)

    def angular_velocity_to_euler_rates(self,angular_velocity_body, euler_angles):
        yaw, pitch, roll = euler_angles
        sec_pitch = 1 / np.cos(pitch)
        A = np.array([
            [np.cos(roll) * sec_pitch, np.sin(roll) * sec_pitch, 0],
            [-np.sin(roll), np.cos(roll), 0],
            [np.tan(pitch) * np.cos(roll), np.tan(pitch) * np.sin(roll), 1]
        ])
        euler_rates = np.dot(A, angular_velocity_body)
        return euler_rates



if __name__ == "__main__":
    world = dart.simulation.World()

    urdfParser = dart.utils.DartLoader()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hrp4   = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "hrp4.urdf"))
    ground = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "ground.urdf"))
    world.addSkeleton(hrp4)
    world.addSkeleton(ground)
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
    viewer.addWorldNode(node)

    #viewer.setUpViewInWindow(0, 0, 1920, 1080)
    viewer.setUpViewInWindow(0, 0, 1280, 720)
    #viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([5., -1., 1.5],
                                 [1.,  0., 0.5],
                                 [0.,  0., 1. ])
    viewer.run()
