import numpy as np
from matplotlib import pyplot as plt

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
            {'axis': 0, 'batch': 'desired', 'item': 'com', 'level': 'pos', 'dim': 0, 'color': 'blue' , 'style': '-'},
            {'axis': 0, 'batch': 'current', 'item': 'com', 'level': 'pos', 'dim': 0, 'color': 'red'  , 'style': '--'},
            {'axis': 1, 'batch': 'desired', 'item': 'com', 'level': 'pos', 'dim': 1, 'color': 'blue' , 'style': '-'},
            {'axis': 1, 'batch': 'current', 'item': 'com', 'level': 'pos', 'dim': 1, 'color': 'red'  , 'style': '--'},
            {'axis': 2, 'batch': 'desired', 'item': 'com', 'level': 'pos', 'dim': 2, 'color': 'blue' , 'style': '-'},
            {'axis': 2, 'batch': 'current', 'item': 'com', 'level': 'pos', 'dim': 2, 'color': 'red'  , 'style': '--'},
        ]

        plot_num = np.max([item['axis'] for item in self.plot_info]) + 1
        self.fig, self.ax = plt.subplots(plot_num, 1, figsize=(6, 8))

        self.fig.suptitle("Center of mass (desired vs current)", fontsize=14)

        # Creiamo due linee fittizie per comparire nella leggenda 
        # (una per la traiettoria 'desired' e una per la 'current')
        # e le aggiungiamo su ogni asse o solo sul primo, come preferisci.
        for i in range(plot_num):
            dummy_desired, = self.ax[i].plot([], [], color='blue', linestyle='-', label='Desired')
            dummy_current, = self.ax[i].plot([], [], color='red',  linestyle='--', label='Current')
            self.ax[i].legend()

        # Ora creiamo i veri handler delle linee (senza label)
        self.lines = {}
        for item in self.plot_info:
            key = (item['batch'], item['item'], item['level'], item['dim'])
            self.lines[key], = self.ax[item['axis']].plot(
                [], [],
                color=item['color'],
                linestyle=item['style']
            )

        plt.ion()
        plt.show()

    def update_plot(self, time):
        if time % self.frequency != 0:
            return

        for item in self.plot_info:
            trajectory_key = (item['batch'], item['item'], item['level'])
            trajectory = np.array(self.log[trajectory_key]).T[item['dim']]
            line_key = (item['batch'], item['item'], item['level'], item['dim'])
            self.lines[line_key].set_data(np.arange(len(trajectory)), trajectory)

        # resetta i limiti degli assi e aggiorna la vista
        for ax in self.ax:
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
