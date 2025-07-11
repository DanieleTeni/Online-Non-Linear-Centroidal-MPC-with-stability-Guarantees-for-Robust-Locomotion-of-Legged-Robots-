import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

class Logger():
    def __init__(self, initial):
        self.log = {}
        for item in initial.keys():
            for level in initial[item].keys():
                self.log['desired', item, level] = []
                self.log['current', item, level] = []


    def log_data(self, desired, current):
        for item in desired.keys():
            for level in desired[item].keys():
                self.log['desired', item, level].append(desired[item][level])
                self.log['current', item, level].append(current[item][level])

    def initialize_plot(self, frequency=1):
        self.frequency = frequency
        self.plot_info = [
            {'axis': 0, 'batch': 'desired', 'item': 'com', 'level': 'pos', 'dim': 0, 'color': 'blue' , 'style': '-', 'label': 'MPC CoM_x' },
            {'axis': 0, 'batch': 'current', 'item': 'com', 'level': 'pos', 'dim': 0, 'color': 'red' , 'style': '--', 'label': 'Nominal CoM_x'},
            {'axis': 1, 'batch': 'desired', 'item': 'com', 'level': 'pos', 'dim': 1, 'color': 'blue' , 'style': '-', 'label': 'MPC CoM_y' },
            {'axis': 1, 'batch': 'current', 'item': 'com', 'level': 'pos', 'dim': 1, 'color': 'red' , 'style': '--', 'label': 'Nominal CoM_y'},
            {'axis': 2, 'batch': 'desired', 'item': 'com', 'level': 'pos', 'dim': 2, 'color': 'blue' , 'style': '-', 'label': 'MPC CoM_z' },
            {'axis': 2, 'batch': 'current', 'item': 'com', 'level': 'pos', 'dim': 2, 'color': 'red' , 'style': '--', 'label': 'Nominal CoM_z'},
        ]

        plot_num = np.max([item['axis'] for item in self.plot_info]) + 1
        self.fig, self.ax = plt.subplots(plot_num, 1, figsize=(6, 8))

        self.fig.suptitle("CoM nominal (dash) vs CoM MPC", fontsize=14)
        self.ax[0].set_ylabel('[m]')
        self.ax[1].set_ylabel('[m]')
        self.ax[2].set_ylabel('[m]')
        self.ax[2].set_xlabel('time step')

        
        self.lines = {}
        for item in self.plot_info:
            key = item['batch'], item['item'], item['level'], item['dim']
            self.lines[key], = self.ax[item['axis']].plot([], [], color=item['color'], linestyle=item['style'], label=item['label'])
        for ax in self.ax:
         ax.legend()
        
        plt.ion()
        plt.show()

    def update_plot(self, time):
        if time % self.frequency != 0:
            return

        for item in self.plot_info:
            trajectory_key = item['batch'], item['item'], item['level']
            trajectory = np.array(self.log[trajectory_key]).T[item['dim']]
            line_key = item['batch'], item['item'], item['level'], item['dim']
            self.lines[line_key].set_data(np.arange(len(trajectory)), trajectory)

        # set limits
        for i in range(len(self.ax)):
            self.ax[i].relim()
            self.ax[i].autoscale_view()
            
        # redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()