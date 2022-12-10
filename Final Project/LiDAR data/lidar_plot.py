import ast, glob, hdbscan, cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression


lidar_w_angle = 0.5
lidar_w_distance = 0.5

class Lidar():

    def __init__(self):

        # ============= kalman filter
        mp = np.array((2, 1), np.float32)  # measurement
        tp = np.zeros((2, 1), np.float32)  # tracked / prediction
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kalman.processNoiseCov = (
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            * 0.03
        )


    def open_raw_lidar_data(self, path):
        with open(path, "r") as f:
            data = f.readlines()[0]
            data = ast.literal_eval(data) # str to list
            data = np.array(data)

            return data


    def conver_to_2txt(self, path):
        xdata = []
        ydata = []

        scan = self.open_raw_lidar_data(path)
        # angles = np.degrees(laser.get_angles()) + 90
        # x_lidar = scan * np.cos(np.radians(angles))
        # y_lidar = scan * np.sin(np.radians(angles))


        # create angle data list
        # (based on hokuyo.py, fun: get_angles)
        space = np.linspace(0, 1080, 1081) - 540        # 540: Step number of the front direction
        angle_manual = 2*np.pi*space/1440 + np.pi/2     # 1440: Angular resolution (number of partitions in 360 degrees), np.pi/2: 對齊車道方向
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
                txtfile.writelines(str(int(data))+', ')

        with open('lidar_pos_y.txt', 'w') as txtfile:
            for data in ydata:
                txtfile.writelines(str(int(data))+', ')


    def open_2_txt(self):
        with open("lidar_pos_x.txt", "r") as f:
            lidar_pos_x = f.read().split(', ')
            lidar_pos_x = [int(num) for num in lidar_pos_x if num != '']

        with open("lidar_pos_y.txt", "r") as f:
            lidar_pos_y = f.read().split(', ')
            lidar_pos_y = [int(num) for num in lidar_pos_y if num != '']

        lidar_data = np.stack([lidar_pos_x, lidar_pos_y], axis=1) # 將xy資料做合併

        # TODO　改為np.loadtxt
        # lidar_pos_x = np.loadtxt("lidar_pos_x.txt", dtype=np.int16, delimiter=",")
        return lidar_pos_x, lidar_pos_y, lidar_data


    def plot_lidar_raw_fig(self, x_lidar, ylidar, save=False, filename=None):
        plt.cla() # 避免記憶體占用
        plt.xlim(-2000, 2000), plt.ylim(-2000, 2000)
        plt.xlabel("x (cm)"), plt.ylabel("y (cm)")
        plt.plot(x_lidar, ylidar, "ok", markersize=0.5)

        if save == True:
            plt.title(filename)
            plt.savefig(f"output/{filename}.jpg")
            return
        plt.show()


    def plot_and_save_all_plt(self):
        # all_pic_path = glob.glob("./ml_dataset/*.txt")
        all_pic_path = (Path("ml_dataset").glob("*.txt"))

        if not Path("output").exists():
            Path("output").mkdir()

        number_of_path = 0
        for path in all_pic_path:
            self.conver_to_2txt(path=path)
            lidar_pos_x, lidar_pos_y = self.open_2_txt()
            self.plot_lidar_raw_fig(lidar_pos_x, lidar_pos_y, save=True, filename=path.stem)
            number_of_path += 1
        print(f"saved {number_of_path} files.")


    # # DBSCAN
    # def lidar_direction(self, lidar_data):

    #     clustering = DBSCAN(
    #         algorithm='auto',
    #         eps=300, # ~30cm 車道寬度
    #         leaf_size=30,
    #         metric='euclidean',
    #         metric_params=None,
    #         min_samples=40,
    #         n_jobs=1,
    #         p=None,
    #     )
    #     clustering = clustering.fit(lidar_data)
    #     colors = clustering.labels_

    #     average_x_data = []
    #     average_y_data = []
    #     x_0list = []
    #     y_0list = []
    #     x_right = 10000
    #     x_left = 10000
    #     for j in range(max(clustering.labels_) + 1): # j: label number
    #         wall = []
    #         x_data_array = []
    #         y_data_array = []
    #         obstacle_array = []
    #         label_array = []

    #         for i in range(len(lidar_data)): # i: all data index
    #             if clustering.labels_[i] == j:
    #                 xdata, ydata, label = (
    #                     lidar_data[:, 0][i],
    #                     lidar_data[:, 1][i],
    #                     clustering.labels_[i],
    #                 )
    #                 obstacle_array.append([xdata, ydata])
    #                 x_data_array.append(xdata)
    #                 y_data_array.append(ydata)
    #                 label_array.append(label)

    #         model = LinearRegression()
    #         x_0 = np.expand_dims(x_data_array, axis=1) # 增加維度 ?
    #         y_0 = np.expand_dims(y_data_array, axis=1)
    #         model.fit(y_0, x_0)
    #         predict_0 = model.predict(y_0)

    #         x_0, y_0 = self.get_function(predict_0, y_0)
    #         x_0list.append(x_0)
    #         y_0list.append(y_0)

    #         average_x = np.mean(x_data_array)
    #         average_x_data.append(average_x)

    #     # find the min_right min_left
    #     for i in range(len(average_x_data)):
    #         if average_x_data[i] >= 0:
    #             x_minse = abs(average_x_data[i]) - 0
    #             if x_minse < x_right:
    #                 x_right = x_minse
    #                 x_right_i = i

    #         elif average_x_data[i] < 0:
    #             x_minse = abs(average_x_data[i]) - 0
    #             if x_minse < x_left:
    #                 x_left = x_minse
    #                 x_left_i = i

    #     try:
    #         x_l = x_0list[x_left_i]
    #         y_l = y_0list[x_left_i]
    #         x_r = x_0list[x_right_i]
    #         y_r = y_0list[x_right_i]
    #         x_c = (x_l + x_r) / 2
    #         y_c = (y_l + y_r) / 2
    #         lidar_angle, mid_dis = self.get_degree(x_c, y_c)
    #         lidar_angle = lidar_angle - 90
    #     except:
    #         lidar_angle = 0
    #         mid_dis = 0
    #         x_c = 0
    #         y_c = 0
    #         x_l = 0
    #         y_l = 0
    #         x_r = 0
    #         y_r = 0
    #     # print(average_x_data)
    #     # print('x_c:',x_c,'  y_c:',y_c,'   Lidar_angle:',lidar_angle)
    #     # print(lidar_angle)
    #     return x_c, y_c, lidar_angle, colors, x_l, y_l, x_r, y_r, mid_dis



    # # LiDAR callback
    # def lidar_call_backlog(self, msg):
    #     global E_lidar

    #     x_sensor = msg[0]
    #     y_sensor = msg[1]
    #     lidar_data = []

    #     # 變成[x, y, x, y]，把Xy資料兩邊合在一起
    #     for index, data in enumerate(x_sensor):
    #         lidar_data.append([x_sensor[index], y_sensor[index]])

    #     lidar_data = np.array(lidar_data)

    #     (
    #         x_c,
    #         y_c,
    #         lidar_angle,
    #         label_array,
    #         x_l,
    #         y_l,
    #         x_r,
    #         y_r,
    #         mid_dis,
    #     ) = self.lidar_direction(lidar_data)

    #     y_o = np.linspace(-1000, 1000, 4000)
    #     x_o = np.zeros(y_o.size)

    #     # print('lidar_angle   :',lidar_angle,'    mid_dis',mid_dis)

    #     if np.isnan(lidar_angle) == True or np.isnan(lidar_angle) == True:
    #         sensor = np.array([[np.float32(0)], [np.float32(0)]])
    #     else:
    #         sensor = np.array([[np.float32(lidar_angle)], [np.float32(mid_dis)]])

    #     # # print(sensor)
    #     # if lidar_angle == float and lidar_angle == float:

    #     # ================ 計算Lidar誤差
    #     # kalman
    #     x = self.kalman.correct(sensor)
    #     y = self.kalman.predict()
    #     E_lidar = lidar_w_angle * (y[0] / 90) + lidar_w_distance * (y[1] / 2000)
    #     # else:
    #     #     x = 0
    #     #     y = 0
    #     # E_lidar =  False

    #     # E_lidar = lidar_w_angle*(y[0]/90) + lidar_w_distance*(y[1]/2000)
    #     # print(E_lidar)

    #     # TODO　draw lidar refer line
    #     self.fig_lidar.axes.cla()
    #     self.fig_lidar.axes.set_xlim(left=-2000, right=2000)
    #     self.fig_lidar.axes.set_ylim(bottom=-2000, top=2000)
    #     self.fig_lidar.axes.plot(msg[0], msg[1], 'ok', markersize=1)
    #     self.fig_lidar.axes.plot(x_l, y_l, '-or', markersize=0.5)
    #     self.fig_lidar.axes.plot(x_r, y_r, '-or', markersize=0.5)
    #     self.fig_lidar.axes.plot(x_o, y_o, '--k', markersize=0.5)
    #     if lidar_angle != 0 and mid_dis != 0:
    #         self.fig_lidar.axes.arrow(
    #             x_c[0],
    #             y_c[0],
    #             x_c[-1] - x_c[0],
    #             y_c[-1] - y_c[0],
    #             head_length=100,
    #             head_width=100,
    #             color="blue",
    #         )
    #     self.fig_lidar.axes.set_title("LiDAR position")
    #     self.fig_lidar.axes.set_xlabel("x (cm)")
    #     self.fig_lidar.axes.set_ylabel("y (cm)")
    #     self.fig_lidar.draw()


    def get_function(self, x, y):
        a = (y[1] - y[0]) / (x[1] - x[0]) # 斜率
        b = y[0] - a * x[0] # 截距
        y_r = np.linspace(-1000, 1000, 4000)
        x_r = (y_r - b) / a
        return x_r, y_r # 回傳線段中的各點座標


    def get_degree(self, x, y):
        try:
            a = (y[1] - y[0]) / (x[1] - x[0])
            b = y[0] - a * x[0]
            y_0 = np.linspace(-1000, 1000, 4000)
            mid_dis = (0 - b) / a
            angle = np.rad2deg(np.arctan2(y[1] - y[0], x[0] - x[1]))
        except:
            a = 0
            b = 0
            angle = 0
            mid_dis = 0
        return angle, mid_dis



    def dbscan(self, lidar_data, arrow=False):
        clustering = DBSCAN(algorithm='auto', eps=300, leaf_size=30, metric='euclidean',
            metric_params=None, min_samples=20, n_jobs=1, p=None)
        clustering = clustering.fit(lidar_data)
        label = clustering.labels_

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(lidar_data[:,0], lidar_data[:,1], c=label, s=1)
        ax.set_xlim(left=-2000, right=2000)
        ax.set_ylim(bottom=-2000, top=2000)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title('LiDAR Scanning')

        if arrow:
            wall = []
            x_data_array = []
            y_data_array = []
            obstacle_array = []
            x_0list = []
            y_0list = []
            x_0list_pca = []
            y_0list_pca = []
            average_x_data = []
            average_y_data = []

            # clustering 標籤數字
            for j in range(max(clustering.labels_)+1):

                wall = []
                x_data_array = []
                y_data_array = []
                obstacle_array = []

                # 所有資料進行比對
                for i in range(len(lidar_data)):
                    if clustering.labels_[i] == j:
                        xdata, ydata, label = lidar_data[:,0][i], lidar_data[:,1][i], clustering.labels_[i]

                        obstacle_array.append([xdata,ydata])
                        x_data_array.append(xdata)
                        y_data_array.append(ydata)

                # TODO　改用選最近中心的claster
                # average_x = np.mean(x_data_array)
                x_min = min(x_data_array)
                x_max = max(x_data_array)
                x_mean = np.mean(x_data_array)
                y_min = min(y_data_array)
                y_max = max(y_data_array)
                y_mean = np.mean(y_data_array)

                x_dis = np.abs((x_max - x_min))
                y_dis = np.abs((y_max - y_min))


                # 用Y預測X，因為需要在光達圖上繪製直線，可以限制在-1000~1000範圍內，如果是x預測y遇到直線前進時便會不好繪圖
                model = LinearRegression()
                x_0 = np.expand_dims(x_data_array, axis = 1)
                y_0 = np.expand_dims(y_data_array, axis = 1)
                model.fit(y_0, x_0)
                predict_0 = model.predict(y_0)
                x_0, y_0 = self.get_function(predict_0, y_0)
                plt.plot(x_0, y_0, c = 'red')

                x_0list.append(x_0)
                y_0list.append(y_0)
                y_r = np.linspace(-1000, 1000, 4000)

                X = np.zeros(y_r.size)

                # linear
                x_c = (x_0list[0] + x_0list[1]) / 2
                y_c = (y_0list[0] + y_0list[1]) / 2
                plt.arrow(x_c[0], y_c[0],x_c[-1]-x_c[0],y_c[-1]-y_c[0],head_length=100,head_width=100,color="blue")
                # plt.arrow(0, -1000, 0, 2000,head_length=100,head_width=100,color="k")
                ax.plot( X, y_r, color = 'k', linewidth=2, linestyle='--')
                print('Linear Regression',90 - self.get_degree(x_c, y_c)[0], self.get_degree(x_c, y_c)[1])
                ax.legend()


        plt.show()


def main():
    lidar = Lidar()

    #=========== 1 picture ===========
    path = Path("ml_dataset/106.txt")
    lidar.conver_to_2txt(path)
    lidar_pos_x, lidar_pos_y, lidar_data = lidar.open_2_txt()
    lidar.plot_lidar_raw_fig(lidar_pos_x, lidar_pos_y)

    lidar.dbscan(lidar_data, arrow=True)

    #=========== All pictures ===========
    # lidar.plot_and_save_all_plt()



if __name__ == "__main__":
    main()