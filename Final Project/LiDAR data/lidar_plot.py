import matplotlib.pyplot as plt

class test():

    def test(self):
        self.fig_lidar.axes.cla()
        self.fig_lidar.axes.set_xlim(left=-2000, right=2000)
        self.fig_lidar.axes.set_ylim(bottom=-2000, top=2000)
        self.fig_lidar.axes.plot(msg[0], msg[1], 'ok', markersize=1)
        self.fig_lidar.axes.plot(x_l, y_l, '-or', markersize=0.5)
        self.fig_lidar.axes.plot(x_r, y_r, '-or', markersize=0.5)
        self.fig_lidar.axes.plot(x_o, y_o, '--k', markersize=0.5)
        if lidar_angle != 0 and mid_dis != 0:
            self.fig_lidar.axes.arrow(
                x_c[0],
                y_c[0],
                x_c[-1] - x_c[0],
                y_c[-1] - y_c[0],
                head_length=100,
                head_width=100,
                color="blue",
            )
        self.fig_lidar.axes.set_title("LiDAR position")
        self.fig_lidar.axes.set_xlabel("x (cm)")
        self.fig_lidar.axes.set_ylabel("y (cm)")
        self.fig_lidar.draw()



    def compute_initial_figure(self):
        x_lidar = 0
        y_lidar = 0
        self.axes.set_xlim(left=-5000, right=5000)
        self.axes.set_ylim(bottom=-5000, top=5000)
        self.axes.plot(x_lidar, y_lidar, 'b')
        self.axes.set_title("LiDAR scanning")
        self.axes.set_xlabel("x (cm)")
        self.axes.set_ylabel("y (cm)")