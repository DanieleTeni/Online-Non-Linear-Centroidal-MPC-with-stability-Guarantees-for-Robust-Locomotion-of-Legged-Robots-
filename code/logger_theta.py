import numpy as np
from matplotlib import pyplot as plt

class Logger_theta():
    def __init__(self, initial):
        self.log = {}
        for item in initial.keys():
            for level in initial[item].keys():
                self.log['desired', item, level] = []
                self.log['current', item, level] = []


    def log_data(self, desired):
        for item in desired.keys():
            for level in desired[item].keys():
                self.log['desired', item, level].append(desired[item][level])


    def initialize_plot(self, frequency=1):
        self.frequency = frequency
        self.plot_info = [
            {'axis': 0, 'batch': 'desired', 'item': 'theta_hat', 'level': 'val', 'dim': 0, 'color': 'black' , 'style': '-', 'label': 'desired x component' },
           
            {'axis': 1, 'batch': 'desired', 'item': 'theta_hat', 'level': 'val', 'dim': 1, 'color': 'black' , 'style': '-' },
        
            {'axis': 2, 'batch': 'desired', 'item': 'theta_hat', 'level': 'val', 'dim': 2, 'color': 'black' , 'style': '-' },
         
        ]

        plot_num = np.max([item['axis'] for item in self.plot_info]) + 1
        self.fig, self.ax = plt.subplots(plot_num, 1, figsize=(6, 8))

        self.fig.suptitle("Estimate Theta_hat", fontsize=14)

        self.lines = {}
        for item in self.plot_info:
            key = item['batch'], item['item'], item['level'], item['dim']
            self.lines[key], = self.ax[item['axis']].plot([], [], color=item['color'], linestyle=item['style'])
        for ax in self.ax:
         ax.legend()
        self.ax[0].set_ylabel('[N]')
        self.ax[1].set_ylabel('[N]')
        self.ax[2].set_ylabel('[N]')
        self.ax[2].set_xlabel('time step')
        
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