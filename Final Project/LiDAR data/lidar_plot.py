import matplotlib.pyplot as plt
import numpy as np
from hokuyolx import HokuyoLX
import ast, glob

class Lidar():

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


    def open_raw_lidar_data(self, path):
        with open(path, "r") as f:
            data = f.readlines()[0]
            data = ast.literal_eval(data) # str to list
            data = np.array(data)
            # print(type(data))
            return data


    def conver_to_2txt(self, path):
        xdata = []
        ydata = []

        scan = self.open_raw_lidar_data(path)
        # angles = np.degrees(laser.get_angles()) + 90
        # x_lidar = scan * np.cos(np.radians(angles))
        # y_lidar = scan * np.sin(np.radians(angles))

        # TODO 確認正確與否
        # create angle data list
        # (based on hokuyo.py, fun: get_angles)
        space = np.linspace(0, 1080, 1081) - 540 # 540: Step number of the front direction
        angle_manual = 2*np.pi*space/1440  # 1440: Angular resolution (number of partitions in 360 degrees)
        x_lidar = scan * np.cos(angle_manual)
        y_lidar = scan * np.sin(angle_manual)

        for index, data in enumerate(scan):
            if int((int(x_lidar[index]))**2+(int(y_lidar[index]))**2)>4000000: # 判斷是否超過2m
                pass
            else:
                xdata.append(x_lidar[index])
                ydata.append(y_lidar[index])

        np.set_printoptions(threshold=10000)
        with open('lidar_pos_x.txt', 'w') as txtfile:
            for data in xdata:
                # txtfile.writelines(str(int(data))+', ')
                # print(data)
                txtfile.writelines(str(int(data))+', ')

        with open('lidar_pos_y.txt', 'w') as txtfile:
            for data in ydata:
                txtfile.writelines(str(int(data))+', ')


    def open_2_txt(self):
        with open("lidar_pos_x.txt", "r") as f:
            # lidar_pos_x = f.readlines()
            # print((lidar_pos_x.split(', ')))
            lidar_pos_x = f.read().split(', ')
            lidar_pos_x = [int(num) for num in lidar_pos_x if num != '']

        with open("lidar_pos_y.txt", "r") as f:
            # lidar_pos_y = f.readlines()
            # print((lidar_pos_y.split(', ')))
            lidar_pos_y = f.read().split(', ')
            lidar_pos_y = [int(num) for num in lidar_pos_y if num != '']

        # TODO　改為np.loadtxt
        # lidar_pos_x = np.loadtxt("lidar_pos_x.txt", dtype=np.int16, delimiter=",")
        return lidar_pos_x, lidar_pos_y


    def compute_initial_figure(self):
        x_lidar = 0
        y_lidar = 0
        self.axes.set_xlim(left=-5000, right=5000)
        self.axes.set_ylim(bottom=-5000, top=5000)
        self.axes.plot(x_lidar, y_lidar, 'b')
        self.axes.set_title("LiDAR scanning")
        self.axes.set_xlabel("x (cm)")
        self.axes.set_ylabel("y (cm)")


    def plot_lidar(self, x_lidar, ylidar, save:False, filename=None):
        plt.figure()
        plt.xlim(-2000, 2000), plt.ylim(-2000, 2000)
        plt.plot(x_lidar, ylidar, "ok", markersize=0.5)

        if save == True:
            plt.title(filename)
            plt.savefig(f"output/{filename}.jpg")
            return
        plt.show()

    def plot_and_save_all_plt(self):
        pic_path_list = glob.glob("./ml_dataset/*.txt")
        # print(len(pic_path_list)) # 63

        for path in pic_path_list:
            self.conver_to_2txt(path=path)
            lidar_pos_x, lidar_pos_y = self.open_2_txt()
            self.plot_lidar(lidar_pos_x, lidar_pos_y, save=True, filename=path.split("\\")[-1])


    # DBSCAN
    def lidar_direction(self, lidar_data):

        clustering = DBSCAN(
            algorithm='auto',
            eps=300,
            leaf_size=30,
            metric='euclidean',
            metric_params=None,
            min_samples=40,
            n_jobs=1,
            p=None,
        )
        clustering = clustering.fit(lidar_data)
        colors = clustering.labels_
        # print(colors.max())
        # self.fig_lidar.axes.plot(lidar_data[0],lidar_data[1],c=clustering.labels_, markersize=2)
        # print(lidar_data)

        average_x_data = []
        average_y_data = []
        x_0list = []
        y_0list = []
        x_right = 10000
        x_left = 10000
        for j in range(max(clustering.labels_) + 1):
            wall = []
            x_data_array = []
            y_data_array = []
            obstacle_array = []
            label_array = []

            for i in range(len(lidar_data)):
                if clustering.labels_[i] == j:
                    xdata, ydata, label = (
                        lidar_data[:, 0][i],
                        lidar_data[:, 1][i],
                        clustering.labels_[i],
                    )
                    obstacle_array.append([xdata, ydata])
                    x_data_array.append(xdata)
                    y_data_array.append(ydata)
                    label_array.append(label)

            model = LinearRegression()
            x_0 = np.expand_dims(x_data_array, axis=1)
            y_0 = np.expand_dims(y_data_array, axis=1)
            model.fit(y_0, x_0)
            predict_0 = model.predict(y_0)

            x_0, y_0 = self.get_function(predict_0, y_0)
            x_0list.append(x_0)
            y_0list.append(y_0)

            average_x = np.mean(x_data_array)
            average_x_data.append(average_x)

        # find the min_right min_left
        for i in range(len(average_x_data)):
            if average_x_data[i] >= 0:
                x_minse = abs(average_x_data[i]) - 0
                if x_minse < x_right:
                    x_right = x_minse
                    x_right_i = i

            elif average_x_data[i] < 0:
                x_minse = abs(average_x_data[i]) - 0
                if x_minse < x_left:
                    x_left = x_minse
                    x_left_i = i

        try:
            x_l = x_0list[x_left_i]
            y_l = y_0list[x_left_i]
            x_r = x_0list[x_right_i]
            y_r = y_0list[x_right_i]
            x_c = (x_l + x_r) / 2
            y_c = (y_l + y_r) / 2
            lidar_angle, mid_dis = self.get_degree(x_c, y_c)
            lidar_angle = lidar_angle - 90
        except:
            lidar_angle = 0
            mid_dis = 0
            x_c = 0
            y_c = 0
            x_l = 0
            y_l = 0
            x_r = 0
            y_r = 0
        # print(average_x_data)
        # print('x_c:',x_c,'  y_c:',y_c,'   Lidar_angle:',lidar_angle)
        # print(lidar_angle)
        return x_c, y_c, lidar_angle, colors, x_l, y_l, x_r, y_r, mid_dis



def main():
    lidar = Lidar()

    #=========== 1 picture ===========
    # path = 'ml_dataset/151.txt'
    # lidar.conver_to_2txt(path)
    # lidar_pos_x, lidar_pos_y = lidar.open_2_txt()
    # lidar.plot_lidar(lidar_pos_x, lidar_pos_y)

    #=========== All pictures ===========
    lidar.plot_and_save_all_plt()



if __name__ == "__main__":
    main()