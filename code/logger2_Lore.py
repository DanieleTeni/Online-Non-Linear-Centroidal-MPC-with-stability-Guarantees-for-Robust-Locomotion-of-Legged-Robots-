import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# calling flow
#   1) __init__           (una volta)
#   2) initialize_plot    (una volta)
#   3) log_data           (più volte, in loop)
#   4) update_plot        (più volte, in loop)

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

        # Inizializza la struttura del log
        for item in initial.keys():
            for level in initial[item].keys():
                self.log['real', item, level] = []

        # Questi conterranno i references ai patch (rettangoli) per
        # - piedi reali (rosso)
        # - piedi da MPC   (blu)
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

        # Disegniamo tutti i passi pianificati (verde)
        self.draw_footsteps()

        # "dummy patches" per la leggenda iniziale
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

        # Creiamo la legenda
        #self.ax.legend(handles=[dummy_footstep,dummy_real_pos_fix])
        self.ax.legend(handles=[dummy_footstep,dummy_real_pos_fix,dummy_MPC_pos])
        
        plt.ion()
        plt.show()

    def log_data(self, current, mpc_desired_feet, actual_feet_pose):
        """
        Logs the data per l'attuale istante.
        - current: stato corrente (es. CoM, hw, ecc.)
        - mpc_desired_feet: dizionario con la posa desiderata dei piedi via MPC
        - actual_feet_pose: dizionario con la posa reale dei piedi:
             {
                'lfoot': {'ang': <3x1>, 'pos': <3x1>},
                'rfoot': {'ang': <3x1>, 'pos': <3x1>}
             }
        """

        # Salviamo i dati "current" nello stesso modo in cui veniva fatto prima
        for item in current.keys():
            for level in current[item].keys():
                self.log['real', item, level].append(current[item][level])

        # Salviamo le pose effettive del piede sinistro e destro
        # Nota: Se vuoi potresti strutturare questo log in modo più complesso,
        #       a seconda di come ti serve consultarlo in futuro
        if ('real', 'lfoot', 'ang') not in self.log:
            # Se non esistono ancora, creali
            self.log['real', 'lfoot', 'ang'] = []
            self.log['real', 'lfoot', 'pos'] = []
            self.log['real', 'rfoot', 'ang'] = []
            self.log['real', 'rfoot', 'pos'] = []

        self.log['real', 'lfoot', 'ang'].append(actual_feet_pose['lfoot']['ang'])
        self.log['real', 'lfoot', 'pos'].append(actual_feet_pose['lfoot']['pos'])

        self.log['real', 'rfoot', 'ang'].append(actual_feet_pose['rfoot']['ang'])
        self.log['real', 'rfoot', 'pos'].append(actual_feet_pose['rfoot']['pos'])

        # Salviamo i dati MPC
        self.mpc_desired_feet = mpc_desired_feet

        # Salviamo la posa reale, così la useremo in draw_current_feet
        self.actual_feet_pose = actual_feet_pose

    def update_plot(self, time):
        """
        Updates the plot with new footstep and CoM data.
        """
        if time % self.frequency != 0:
            return

        # Riposiziona i limiti e aggiorna la vista
        self.ax.relim()
        self.ax.autoscale_view()

        # Disegniamo i piedi desiderati dall'MPC (blu)
        #self.draw_mpc_feet()

        # Disegniamo i piedi reali (rosso)
        #self.draw_current_feet()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def draw_footsteps(self):
        """
        Draws planned footstep locations as green transparent rectangles.
        (chiamata una sola volta in initialize_plot)
        """
        for step in self.footstep_planner.plan:
            pos = step['pos']  # Foot center position (x, y)
            ang = step['ang']  # Foot orientation (in radians)
            foot_id = step['foot_id']  # Foot identifier (left/right)

            # Dimensioni del piede
            foot_length = 0.25
            foot_width = 0.13

            # Calcoliamo l'angolo in gradi
            angle_deg = np.degrees(ang)

            # Angolo da cui parte il rettangolo: spostiamo il centro
            lower_left_x = pos[0] - foot_length / 2
            lower_left_y = pos[1] - foot_width / 2

            # Rettangolo
            rect = patches.Rectangle(
                (lower_left_x, lower_left_y),
                foot_length,
                foot_width,
                angle=angle_deg[2],  # assumendo ang = [roll, pitch, yaw]
                color='lightgreen',
                alpha=0.5,
                linewidth=1,
                edgecolor='black'
            )
            self.ax.add_patch(rect)


    def draw_current_feet(self,actual_feet_pose):
        """
        Draws the real-time foot positions as red transparent rectangles.
        (rimuove i rettangoli precedenti e li ridisegna)
        """
        if not hasattr(self, 'actual_feet_pose'):
            return  # Non abbiamo dati ancora

        # Rimuoviamo i vecchi rettangoli
        for rect in self.current_foot_rects:
            rect.remove()
        self.current_foot_rects = []

        # Disegniamo i piedi lfoot e rfoot
        #for foot in ['lfoot', 'rfoot']:
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
        (rimuove i vecchi e ridisegna i nuovi)
        """
        if not hasattr(self, 'mpc_desired_feet'):
            print(f'Logger2, draw_mpc_feet: no data yet')
            return

        # Rimuoviamo i vecchi rettangoli MPC
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
        Esempio di disegno di un singolo piede "purple"
        (non c'entra con i log, è solo una funzione aggiuntiva).
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
