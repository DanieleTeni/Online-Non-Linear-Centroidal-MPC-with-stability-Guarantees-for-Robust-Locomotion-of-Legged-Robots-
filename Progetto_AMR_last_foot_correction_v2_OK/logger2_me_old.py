import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#calling flow
#   init,1 time
#   initialize_plot, 1 time
#   log_data, more times
#   update_plot, more times


class Logger2():
    def __init__(self, initial, footstep_planner):
        """
        Initializes the Logger2 class.
        Stores data logs and references the footstep planner.
        """
        self.log = {}
        self.footstep_planner = footstep_planner  # Store reference to footstep planner

        # Initialize log structure
        for item in initial.keys():
            for level in initial[item].keys():
                self.log['real', item, level] = []

        # Store rectangle references for real-time foot position
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

        # draw all the sequence of plannes steps (green)
        self.draw_footsteps()

        # "dummy patches"  to get complete legend from the start
        dummy_footstep = patches.Patch(
            facecolor='lightgreen', edgecolor='black', alpha=0.5,
            label='Planned footstep trajectories'
        )
        dummy_real_pos = patches.Patch(
            facecolor='red', edgecolor='black', alpha=0.3,
            label='Actual feet position'
        )
        dummy_MPC_pos = patches.Patch(  # <-- Commentato per rimuovere la voce MPC
            facecolor='lightblue', edgecolor='blue', alpha=0.5,
            label='Actual feet position by MPC'
        )

        # Create the legend
        self.ax.legend(handles=[dummy_footstep, dummy_real_pos, dummy_MPC_pos])
        #self.ax.legend(handles=[dummy_footstep, dummy_real_pos])

        plt.ion()
        plt.show()



    def log_data(self, corner_l, corner_r, current, mpc_desired_feet,actual_feet_pose):
        """
        Logs the data for left and right foot corners, the current state, 
        and the MPC-generated foot positions.
        """
        for item in corner_l:
            self.log['real', 'corner_left', item].append(corner_l[item])
        for item in corner_r:
            self.log['real', 'corner_right', item].append(corner_r[item])
        for item in current.keys():
            for level in current[item].keys():
                self.log['real', item, level].append(current[item][level])

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

        # Draw the desired MPC feet positions (light blue)
        self.draw_mpc_feet()

        # Draw the current feet position (red)
        self.draw_current_feet()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    
    def draw_footsteps(self):   #We call it 1 time, with initialize_plot #ok
        """
        Draws planned footstep locations as green transparent rectangles.
        """
        for step in self.footstep_planner.plan:
            pos = step['pos']  # Foot center position (x, y)
            ang = step['ang']  # Foot orientation (in radians)
            foot_id = step['foot_id']  # Foot identifier (left/right)

            # Define foot dimensions
            foot_length = 0.25  # Length of the foot
            foot_width = 0.13   # Width of the foot

            # Compute the lower-left corner of the rectangle
            lower_left_x = pos[0] - foot_length / 2
            lower_left_y = pos[1] - foot_width / 2

            # Convert the angle from radians to degrees for plotting
            angle_deg = np.degrees(ang)
    
            # Create the footstep rectangle (senza label)
            rect = patches.Rectangle(
                (lower_left_x, lower_left_y),
                foot_length,
                foot_width,
                angle=angle_deg[2],
                color='lightgreen',
                alpha=0.5,
                linewidth=1,
                edgecolor='black'
            )
            self.ax.add_patch(rect)


    def draw_current_feet(self):
        """
        Draws the real-time foot positions as red transparent rectangles.
        These rectangles do not remain on the plot but update at every step.
        """
        if not hasattr(self, 'actual_feet_pose'):
            return  # No data yet
        
        # Remove old foot rectangles before drawing new ones
        for rect in self.current_foot_rects:
            rect.remove()
        self.current_foot_rects = []  # Clear list


        for foot in ['lfoot', 'rfoot']:
            pos = self.actual_feet_pose[foot]['pos']
            ang = self.actual_feet_pose[foot]['ang']
            angle_deg = np.degrees(ang)
        
            # Define foot dimensions
            foot_length = 0.25
            foot_width = 0.13

            # Compute the lower-left corner
            lower_left_x = pos[0] - foot_length / 2
            lower_left_y = pos[1] - foot_width / 2

            # Create red rectangles for current foot positions (No label)
            actual_rect = patches.Rectangle(
                (lower_left_x, lower_left_y),
                foot_length,
                foot_width,
                angle=angle_deg[2],
                #angle
                color='red',
                alpha=0.3,
                linewidth=1,
                edgecolor='black'
            )

            # Add rectangles to the plot
            self.ax.add_patch(actual_rect)
        
            # Store references to remove them on the next update
            self.current_foot_rects.append(actual_rect)
            
        


    def draw_mpc_feet(self):
        """
        Draws the MPC-predicted foot positions as light blue rectangles.
        """
        if not hasattr(self, 'mpc_desired_feet'):
            print(f'Logger2, draw_mpc_feet: void return ')
            return  # No data yet
        
        for rect in self.current_foot_MPC:
            rect.remove()
        self.current_foot_MPC = [] #clear list 

        for foot in ['lfoot', 'rfoot']:
            pos = self.mpc_desired_feet[foot]['pos']
            ang = self.mpc_desired_feet[foot]['ang']
            angle_deg = np.degrees(ang)

            print(f' Logger2,draw_mpc_feet-> {foot}: pos:{pos} ,ang:{angle_deg}  ')
        
            # Define foot dimensions
            foot_length = 0.25
            foot_width = 0.13

            # Compute the lower-left corner
            lower_left_x = pos[0] - foot_length / 2
            lower_left_y = pos[1] - foot_width / 2

            # Create light blue rectangle (senza label)
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

    
    def draw_desired_swing_foot_position(self,next_des_contac):
    
        pos = next_des_contac[3:6]  # Foot center position (x, y)
        ang = next_des_contac[0:3]  # Foot orientation (in radians)
        
        # Define foot dimensions
        foot_length = 0.25  # Length of the foot
        foot_width = 0.13   # Width of the foot

        # Compute the lower-left corner of the rectangle
        lower_left_x = pos[0] - foot_length / 2
        lower_left_y = pos[1] - foot_width / 2

        # Convert the angle from radians to degrees for plotting
        angle_deg = np.degrees(ang)

        # Create the footstep rectangle (senza label)
        rect = patches.Rectangle(
            (lower_left_x, lower_left_y),
            foot_length,
            foot_width,
            angle=angle_deg[2],
            color='purple',
            alpha=0.5,
            linewidth=1,
            edgecolor='black'
        )
        self.ax.add_patch(rect)


