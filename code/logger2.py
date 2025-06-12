import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Logger2():
    def __init__(self, initial, footstep_planner):
        """
        Initializes the Logger2 class.
        Stores data logs and references the footstep planner.
        :param initial: dict con alcuni valori iniziali (ad esempio 'com', 'hw', ecc.)
        :param footstep_planner: oggetto che contiene plan dei passi
        """
        self.log = {}
        self.footstep_planner = footstep_planner  # Store reference to footstep planner

        for item in initial.keys():
            for level in initial[item].keys():
                self.log['real', item, level] = []

       
        self.current_foot_rects = []
        self.current_foot_MPC = []

    def initialize_plot(self, frequency=1):
        """
        Initializes the plot for visualization.
        """
        self.frequency = frequency

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_title('Footstep trajectoryPlanner')

        self.ax.set_xlim(-0.5, 4.5)
        self.ax.set_ylim(-1.5, 1.5)

        self.draw_footsteps()

       
        dummy_footstep = patches.Patch(
            facecolor='lightgreen', edgecolor='black', alpha=0.5,
            label='Planned footstep trajectories'
        )
        dummy_real_pos_fix = patches.Patch(
            facecolor='purple', edgecolor='black', alpha=0.3,
            label='real feet pose'
        )
        dummy_MPC_pos = patches.Patch(
            facecolor='red', edgecolor='blue', alpha=0.5,
            label='Desired feet pose by MPC'
        )
        dummy_real_pos = patches.Patch(
            facecolor='purple', edgecolor='purple', alpha=0.4,
            label='Actual feet pose'
        )
        self.ax.legend(handles=[dummy_footstep,dummy_real_pos_fix,dummy_MPC_pos])
        
        plt.ion()
        plt.show()

    def log_data(self, current, mpc_desired_feet, actual_feet_pose):
        """
        Logs the data per l'attuale istante.
        - current: current state (es. CoM, hw, ecc.)
        - mpc_desired_feet: dictionary with mpc desired foot pose 
        - actual_feet_pose: dictionarry with foot pose:
             {
                'lfoot': {'ang': <3x1>, 'pos': <3x1>},
                'rfoot': {'ang': <3x1>, 'pos': <3x1>}
             }
        """

      
        for item in current.keys():
            for level in current[item].keys():
                self.log['real', item, level].append(current[item][level])

        if ('real', 'lfoot', 'ang') not in self.log:
          
            self.log['real', 'lfoot', 'ang'] = []
            self.log['real', 'lfoot', 'pos'] = []
            self.log['real', 'rfoot', 'ang'] = []
            self.log['real', 'rfoot', 'pos'] = []

        self.log['real', 'lfoot', 'ang'].append(actual_feet_pose['lfoot']['ang'])
        self.log['real', 'lfoot', 'pos'].append(actual_feet_pose['lfoot']['pos'])

        self.log['real', 'rfoot', 'ang'].append(actual_feet_pose['rfoot']['ang'])
        self.log['real', 'rfoot', 'pos'].append(actual_feet_pose['rfoot']['pos'])

        self.mpc_desired_feet = mpc_desired_feet

       
        self.actual_feet_pose = actual_feet_pose

    def update_plot(self, time):
        """
        Updates the plot with new footstep and CoM data.
        """
        if time % self.frequency != 0:
            return

      
        self.ax.relim()
        self.ax.autoscale_view()


        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def draw_footsteps(self):
        """
        Draws planned footstep locations as green transparent rectangles.
        (chiamata una sola volta in initialize_plot)
        """
        for step in self.footstep_planner.plan:
            pos = step['pos'] 
            ang = step['ang']  
            foot_id = step['foot_id'] 

            foot_length = 0.25
            foot_width = 0.13

            angle_deg = np.degrees(ang)

            lower_left_x = pos[0] - foot_length / 2
            lower_left_y = pos[1] - foot_width / 2

            
            rect = patches.Rectangle(
                (lower_left_x, lower_left_y),
                foot_length,
                foot_width,
                angle=angle_deg[2],  # ang = [roll, pitch, yaw]
                color='lightgreen',
                alpha=0.5,
                linewidth=1,
                edgecolor='black'
            )
            self.ax.add_patch(rect)


    def draw_current_feet(self,actual_feet_pose):
        """
        Draws the real-time foot positions as red transparent rectangles.
        
        """
        if not hasattr(self, 'actual_feet_pose'):
            return  

        
        for rect in self.current_foot_rects:
            rect.remove()
        self.current_foot_rects = []

    
        pos = actual_feet_pose['pos'][3:6]
        ang = actual_feet_pose['pos'][0:3]

        angle_deg = np.degrees(ang)

        foot_length = 0.25
        foot_width = 0.13

        lower_left_x = pos[0] - foot_length / 2
        lower_left_y = pos[1] - foot_width / 2

        actual_rect = patches.Rectangle(
                (lower_left_x, lower_left_y),
                foot_length,
                foot_width,
                angle=angle_deg[2],
                color='purple',
                alpha=0.2,
                linewidth=1,
                edgecolor='black'
            )
        self.ax.add_patch(actual_rect)
        #self.current_foot_rects.append(actual_rect)



    def draw_mpc_feet(self):
        """
        Draws the MPC-predicted foot positions as light blue rectangles.
        
        """
        if not hasattr(self, 'mpc_desired_feet'):
            print(f'Logger2, draw_mpc_feet: no data yet')
            return

      
        for rect in self.current_foot_MPC:
            rect.remove()
        self.current_foot_MPC = []

        for foot in ['lfoot', 'rfoot']:
            pos = self.mpc_desired_feet[foot]['pos']
            ang = self.mpc_desired_feet[foot]['ang']

            angle_deg = np.degrees(ang)
            print(f'Logger2, draw_mpc_feet -> {foot}: pos={pos}, ang(deg)={angle_deg}')

            foot_length = 0.25
            foot_width = 0.13

            lower_left_x = pos[0] - foot_length / 2
            lower_left_y = pos[1] - foot_width / 2

            MPC_rect = patches.Rectangle(
                (lower_left_x, lower_left_y),
                foot_length,
                foot_width,
                angle=angle_deg[2],
                color='lightblue',
                alpha=0.5,
                linewidth=1,
                edgecolor='blue'
            )
            self.ax.add_patch(MPC_rect)
            self.current_foot_MPC.append(MPC_rect)



    def draw_desired_swing_foot_position(self, next_des_contac):
        """
       
        next_des_contac = [roll, pitch, yaw, x, y, z]
        """
        pos = next_des_contac[3:6]
        ang = next_des_contac[0:3]

        foot_length = 0.25
        foot_width = 0.13

        lower_left_x = pos[0] - foot_length / 2
        lower_left_y = pos[1] - foot_width / 2

        angle_deg = np.degrees(ang)

        rect = patches.Rectangle(
            (lower_left_x, lower_left_y),
            foot_length,
            foot_width,
            angle=angle_deg[2],
            color='red',
            alpha=0.3,
            linewidth=1,
            edgecolor='black'
        )
        self.ax.add_patch(rect)


    def draw_mpc_feet_at_update_time(self,next_des_contac_MPC_at_update_time):
        pos = next_des_contac_MPC_at_update_time[3:6]
        ang = next_des_contac_MPC_at_update_time[0:3]

        foot_length = 0.25
        foot_width = 0.13

        lower_left_x = pos[0] - foot_length / 2
        lower_left_y = pos[1] - foot_width / 2

        angle_deg = np.degrees(ang)

        rect = patches.Rectangle(
            (lower_left_x, lower_left_y),
            foot_length,
            foot_width,
            angle=angle_deg[2],
            color='lightblue',
            alpha=0.5,
            linewidth=1,
            edgecolor='blue'
        )
        self.ax.add_patch(rect)
