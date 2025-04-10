import numpy as np
from matplotlib import pyplot as plt

class Logger3():
    def __init__(self, initial):
        """
        In this constructor, we initialize the log dictionary for both 'desired' and 'current'.
        Each key is a tuple (batch, item, level).
        """
        self.log = {}
        for item in initial.keys():
            for level in initial[item].keys():
                self.log['desired', item, level] = []
                self.log['current', item, level] = []

    def log_data(self, desired, current):
        """
        This function appends new data points to the existing lists in self.log.
        """
        for item in desired.keys():
            for level in desired[item].keys():
                self.log['desired', item, level].append(desired[item][level])
                self.log['current', item, level].append(current[item][level])

    def initialize_plot(self, frequency=1):
        """
        This function creates the figure, subplots, and initializes both dummy lines
        for the legend as well as the real lines that will be updated in real time.
        """
        self.frequency = frequency
        self.plot_info = [
            {'axis': 0, 'batch': 'desired', 'item': 'hw', 'level': 'val', 'dim': 0, 'color': 'green', 'style': '-'},
            {'axis': 0, 'batch': 'current', 'item': 'hw', 'level': 'val', 'dim': 0, 'color': 'red',   'style': '--'},
            {'axis': 1, 'batch': 'desired', 'item': 'hw', 'level': 'val', 'dim': 1, 'color': 'green', 'style': '-'},
            {'axis': 1, 'batch': 'current', 'item': 'hw', 'level': 'val', 'dim': 1, 'color': 'red',   'style': '--'},
            {'axis': 2, 'batch': 'desired', 'item': 'hw', 'level': 'val', 'dim': 2, 'color': 'green', 'style': '-'},
            {'axis': 2, 'batch': 'current', 'item': 'hw', 'level': 'val', 'dim': 2, 'color': 'red',   'style': '--'},
        ]

        plot_num = np.max([item['axis'] for item in self.plot_info]) + 1
        self.fig, self.ax = plt.subplots(plot_num, 1, figsize=(6, 8))
        self.fig.suptitle("angular momentum (desired vs current)", fontsize=14)

        # Create dummy lines for the legend on each subplot
        # so the legend is displayed even before any real data is plotted.
        for i in range(plot_num):
            dummy_desired, = self.ax[i].plot([], [], color='green', linestyle='-', label='Desired')
            dummy_current, = self.ax[i].plot([], [], color='red',   linestyle='--', label='Current')
            self.ax[i].legend()

        # Now create the actual lines (without labels) that will be updated
        self.lines = {}
        for item in self.plot_info:
            key = (item['batch'], item['item'], item['level'], item['dim'])
            self.lines[key], = self.ax[item['axis']].plot([], [],
                                                          color=item['color'],
                                                          linestyle=item['style'])

        plt.ion()
        plt.show()

    def update_plot(self, time):
        """
        This function updates the real lines with the latest data from self.log.
        Then it rescales each axis to fit the new data and redraws the figure.
        """
        if time % self.frequency != 0:
            return

        for item in self.plot_info:
            trajectory_key = (item['batch'], item['item'], item['level'])
            trajectory = np.array(self.log[trajectory_key]).T[item['dim']]
            line_key = (item['batch'], item['item'], item['level'], item['dim'])
            self.lines[line_key].set_data(np.arange(len(trajectory)), trajectory)

        # Automatically adjust the axis limits based on the new data
        for i in range(len(self.ax)):
            self.ax[i].relim()
            self.ax[i].autoscale_view()

        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
