import cv2  # 引入 OpenCV 的模組，製作擷取攝影機影像之功能
import sys, time, os, re  # 引入 sys 跟 time 模組
import numpy as np  # 引入 numpy 來處理讀取到得影像矩陣

# 引入 PyQt5 模組
# Ui_main 為自行設計的介面程式
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QMainWindow, QGridLayout
from PyQt5.QtCore import QTimer, pyqtSlot
from Ui_main import Ui_MainWindow
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
import serial
import time
import datetime
import json

# ENet
import utils
import torch.optim as optim
from Enet_line import run_enet, enet
from models.enet import ENet
from args import get_arguments

args = get_arguments()
device = args.device
model = ENet(3).to(device)
optimizer = optim.Adam(model.parameters())
model = utils.load_checkpoint(
    model, optimizer, './save/20220327_jit_flip_bt4_lrd50/', 'Exp'
)[0]

# Lidar
import pandas as pd
import matplotlib.patches as patches
from matplotlib.patches import Circle, Wedge, Polygon
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from hokuyolx1.hokuyo import HokuyoLX


# Control
from torch.autograd import Variable
from simple_pid import PID

E_view = False
E_lidar = False

# 實驗一weight參數
lidar_weight = 0.5
view_weight = 0.5

lidar_w_angle = 0.5
lidar_w_distance = 0.5

view_w_angle = 0.5
view_w_distance = 0.5

# PID weight 參數
m_in = 10  # 馬達速度
m_add = 10  # 馬達加速度
pid = PID(1.2, 0.01, 0.001)
pid.sample_time = 0.5
pid.setpoint = 0
pid.output_limits = (-10, 10)


# 實驗二 定位參數改這

robot_x = None
robot_y = None
stop_count = 0
right_count = 0
left_count = 0


# greenhouse test
turningtest_onoff = 0

setpoint1 = [
    [2.2, 4],
    [2.2, 5],
    [2.2, 6],
    [2.2, 7],
    [2.2, 8],
    [2.2, 9],
    [2.2, 10],
    [2.2, 11],
    [2.2, 12],
    [2.2, 13],
    [2.2, 14],
    [2.2, 15],
    [2.2, 16],
    [2.2, 17],
    [2.2, 18],
    [2.2, 19],
    [2.2, 20],
    [2.2, 21],
    [2.2, 22],
    [2.2, 23],
    [2.2, 24],
    [2.2, 25],
    [2.2, 26],
    [2.2, 27],
    [2.2, 28],
    [2.2, 29],
    [2.2, 30],
    [2.2, 31],
    [2.2, 32],
    [2.2, 33],
    [2.2, 34],
    [2.2, 35],
    [2.2, 36],
]

setpoint2 = [
    [3.82, 36],
    [3.82, 35],
    [3.82, 34],
    [3.82, 33],
    [3.82, 32],
    [3.82, 31],
    [3.82, 30],
    [3.82, 29],
    [3.82, 28],
    [3.82, 27],
    [3.82, 26],
    [3.82, 25],
    [3.82, 24],
    [3.82, 23],
    [3.82, 22],
    [3.82, 21],
    [3.82, 20],
    [3.82, 19],
    [3.82, 18],
    [3.82, 17],
    [3.82, 16],
    [3.82, 15],
    [3.82, 14],
    [3.82, 13],
    [3.82, 12],
    [3.82, 11],
    [3.82, 10],
    [3.82, 9],
    [3.82, 8],
    [3.82, 7],
    [3.82, 6],
    [3.82, 5],
    [3.82, 4],
]

setpoint3 = [
    [5.57, 4],
    [5.57, 5],
    [5.57, 6],
    [5.57, 7],
    [5.57, 8],
    [5.57, 9],
    [5.57, 10],
    [5.57, 11],
    [5.57, 12],
    [5.57, 13],
    [5.57, 14],
    [5.57, 15],
    [5.57, 16],
    [5.57, 17],
    [5.57, 18],
    [5.57, 19],
    [5.57, 20],
    [5.57, 21],
    [5.57, 22],
    [5.57, 23],
    [5.57, 24],
    [5.57, 25],
    [5.57, 26],
    [5.57, 27],
    [5.57, 28],
    [5.57, 29],
    [5.57, 30],
    [5.57, 31],
    [5.57, 32],
    [5.57, 33],
    [5.57, 34],
    [5.57, 35],
    [5.57, 36],
]

setpoint4 = [
    [7.43, 36],
    [7.43, 35],
    [7.43, 34],
    [7.43, 33],
    [7.43, 32],
    [7.43, 31],
    [7.43, 30],
    [7.43, 29],
    [7.43, 28],
    [7.43, 27],
    [7.43, 26],
    [7.43, 25],
    [7.43, 24],
    [7.43, 23],
    [7.43, 22],
    [7.43, 21],
    [7.43, 20],
    [7.43, 19],
    [7.43, 18],
    [7.43, 17],
    [7.43, 16],
    [7.43, 15],
    [7.43, 14],
    [7.43, 13],
    [7.43, 12],
    [7.43, 11],
    [7.43, 10],
    [7.43, 9],
    [7.43, 8],
    [7.43, 7],
    [7.43, 6],
    [7.43, 5],
    [7.43, 4],
]

right_point1 = [2.16, 36.45]
left_point = [3.95, 3.27]
right_point2 = [5.55, 36.45]
end_point = [7.32, 3.55]

# 菊花園參數
# path_x = [2.3,4,5.5,7.5]
# setpoint1 = [[2.3,8],[2.2,16],[2.3,24],[2.3,32]]
# setpoint2 = [[4,32],[4,24],[4,16],[4,8]]
# setpoint3 = [[5.5,8],[5.5,16],[5.5,24],[5.7,32]]
# setpoint4 = [[7.5,32],[7.5,24],[7.5,16],[7.5,8]]

# path_x = [2,3.3,4.25]
# setpoint1 = [[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10],[3,11],[3,12],[3,13],[3,14],[3,15],
#             [3,16],[3,17],[3,18],[3,19],[3,20],[3,21],[3,22],[3,23]]
# setpoint2 = [[3.3,23],[3.3,22],[3.3,21],[3.3,20],[3.3,19],[3.3,18],[3.3,17],[3.3,16],[3.3,15],
#             [3.3,14],[3.3,13],[3.3,12],[3.3,11],[3.3,10],[3.3,9],[3.3,8],[3.3,7],[3.3,6],[3.3,5],[3.3,4],
#             [3.3,3]]
# setpoint3 = [[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10],[3,11],[3,12],[3,13],[3,14],[3,15],
#             [3,16],[3,17],[3,18],[3,19],[3,20],[3,21],[3,22],[3,23]]
# # setpoint4 = [[7.5,32],[7.5,24],[7.5,16],[7.5,8]]

# right_point1 = [2,23.8]
# right_point2 = [3.3,36.6]
# left_point = [3.3,1.7]
# end_point = [3,24]


# kalman filter
mp = np.array((2, 1), np.float32)  # measurement
tp = np.zeros((2, 1), np.float32)  # tracked / prediction
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
)
kalman.processNoiseCov = (
    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    * 0.03
)


class Camera(QtCore.QThread):  # 繼承 QtCore.QThread 來建立 Camera 類別
    rawdata = QtCore.pyqtSignal(np.ndarray)  # 建立傳遞信號，需設定傳遞型態為 np.ndarray
    # view_error = QtCore.pyqtSignal(float)  # 建立傳遞信號，需設定傳遞型態為 np.ndarray
    def __init__(self, parent=None):
        """ 初始化
            - 執行 QtCore.QThread 的初始化
            - 建立 cv2 的 VideoCapture 物件
            - 設定屬性來確認狀態
              - self.connect：連接狀態
              - self.running：讀取狀態
        """
        # 將父類初始化
        super().__init__(parent)
        # 建立 cv2 的攝影機物件
        self.cam = cv2.VideoCapture(0)
        # 判斷攝影機是否正常連接
        run_enet
        if self.cam is None:
            self.connect = False
            self.running = False
        else:
            self.connect = True
            self.running = False

    def run(self):
        global E_view
        """ 執行多執行緒
            - 讀取影像
            - 發送影像
            - 簡易異常處理
        """
        # 當正常連接攝影機才能進入迴圈
        while self.running and self.connect:
            ret, img = self.cam.read()  # 讀取影像
            img, angle, distance = enet(img, model, args)

            sensor = np.array([[np.float32(angle)], [np.float32(distance)]])
            # print(sensor)

            # kalman
            x = kalman.correct(sensor)
            y = kalman.predict()
            # print ('measurement:\t',sensor[0],sensor[1])
            # print ('correct:\t',x[0],x[1])
            # print ('predict:\t',y[0],y[1])

            # pid controller
            E_view = view_w_angle * (y[0] / 90) + view_w_distance * (y[1] / 480)
            # R = w_angle*(angle/90) + w_distance*(distance/480)
            # print(E_view)
            if ret:
                self.rawdata.emit(img)  # 發送影像
                # self.view_error.emit(E)    # 發送影像
            else:  # 例外處理
                print("Warning!!!")
                self.connect = False

    def open(self):
        """ 開啟攝影機影像讀取功能 """
        if self.connect:
            self.running = True  # 啟動讀取狀態

    def stop(self):
        """ 暫停攝影機影像讀取功能 """
        global E_view
        if self.connect:
            self.running = False  # 關閉讀取狀態
            E_view = False

    def close(self):
        """ 關閉攝影機功能 """
        global E_view
        if self.connect:
            self.running = False  # 關閉讀取狀態
            time.sleep(1)
            self.cam.release()  # 釋放攝影機
            E_view = False


class Myplot(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        # normalized for 中文显示和负号
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False

        # new figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # activate figure window
        # super(Plot_dynamic,self).__init__(self.fig)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        # self.fig.canvas.mpl_connect('button_press_event', self)
        # sub plot by self.axes
        self.axes = self.fig.add_subplot(111)
        # initial figure
        self.compute_initial_figure()

        # size policy
        FigureCanvas.setSizePolicy(
            self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        # FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class UWB_fig(Myplot, QtCore.QThread):
    def __init__(self, *args, **kwargs):
        Myplot.__init__(self, *args, **kwargs)

    def compute_initial_figure(self):
        x_uwb = 0
        y_uwb = 0
        self.axes.set_xlim(left=0, right=9.6)
        self.axes.set_ylim(bottom=0, top=40)
        self.axes.plot(x_uwb, y_uwb, '-or')
        self.axes.set_title("UWB position")
        self.axes.set_xlabel("x (cm)")
        self.axes.set_ylabel("y (cm)")


# class for plotting a specific figure static or dynamic
class UWB_run(QtCore.QThread):
    signal = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        # Myplot.__init__(self,*args,**kwargs)
        super(UWB_run, self).__init__()
        self.flag = True

    def run(self):
        global robot_x
        global robot_y
        while self.flag:
            # 嘗試在此處直接連接UWB，但會造成程式卡住，初估為port韌體連接影響
            with open("uwb_pos1.txt", "r") as f:
                data = f.readline()
                # print(data)
                try:
                    uwb_dict = eval(data)
                    # print(uwb_dict)
                    uwb_x, uwb_y = float(uwb_dict['x']), float(uwb_dict['y'])
                except:
                    uwb_x, uwb_y = None, None
                    pass

            with open("uwb_pos2.txt", "r") as f2:
                data2 = f2.readline()
                # print(data)
                try:
                    uwb_dict2 = eval(data2)
                    # print(uwb_dict)
                    uwb_x2, uwb_y2 = float(uwb_dict2['x']), float(uwb_dict2['y'])
                except:
                    uwb_x2, uwb_y2 = None, None
                    pass

            if uwb_x != None and uwb_x2 != None:
                robot_x, robot_y = np.mean([uwb_x, uwb_x2]), np.mean([uwb_y, uwb_y2])
                # print('both uwb')
            elif uwb_x != None and uwb_x2 == None:
                robot_x, robot_y = uwb_x, uwb_y
                # print('uwb 1')
            elif uwb_x == None and uwb_x2 != None:
                robot_x, robot_y = uwb_x2, uwb_y2
                # print('uwb 2')
            # robot_x,robot_y = uwb_x,uwb_y
            # print('sensor 1 :[',uwb_x,',',uwb_y,']')
            # print('sensor 2 :[',uwb_x2,',',uwb_y2,']')
            print('robot :[', robot_x, ',', robot_y, ']')
            self.signal.emit([robot_x, robot_y])
            time.sleep(0.5)


# class for plotting a specific figure static or dynamic
class Lidar_run(QtCore.QThread):
    lidar_signal = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        # Myplot.__init__(self,*args,**kwargs)
        super(Lidar_run, self).__init__()
        # self.laser = HokuyoLX()
        self.lidar_flag = True

    def run(self):
        while self.lidar_flag:
            # x_lidar = np.random.randint(10, size=5)
            # y_lidar = np.random.randint(10, size=5)
            # timestamp, scan = self.laser.get_dist()
            # angles = np.degrees(self.laser.get_angles())+90
            # x_lidar = scan * np.cos(np.radians(angles))
            # y_lidar = scan * np.sin(np.radians(angles))
            # self.lidar_signal.emit([x_lidar,y_lidar])
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

            self.lidar_signal.emit([lidar_pos_x, lidar_pos_y])

            time.sleep(0.5)


# class for plotting a specific figure static or dynamic
class Control_system(QtCore.QThread):
    control_signal = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        global left_count
        global right_count
        global stop_count
        left_count = 0
        right_count = 0

        # Myplot.__init__(self,*args,**kwargs)
        super(Control_system, self).__init__()
        # self.laser = HokuyoLX()
        self.control = True

    def run(self):
        global stop_count
        global left_count
        global right_count

        while self.control:

            # print('E_view  ',E_view)
            # print('E_lidar  ',E_lidar)

            if E_view != False and E_lidar != False:
                E = (view_weight * E_view) + (lidar_weight * E_lidar)

            elif E_view and E_lidar == False:
                E = E_view

            elif E_view and np.isnan(E_lidar):
                E = E_view

            elif E_lidar and E_view == False:
                E = E_lidar

            elif E_lidar and np.isnan(E_view):
                E = E_lidar

            else:
                E = 0

            # print(E)
            PID_control = m_add * pid(E)

            # print('orginal:',E)
            # print('pid:',PID_control)
            m_l = m_in - PID_control
            m_r = m_in + PID_control
            print('m_l : ', m_l)
            print('m_r : ', m_r)

            if robot_x != None and robot_y != None:
                # print('into uwb')
                # if stop_count <= 3:
                #     if robot_x >= (setpoint1[stop_count][0] - 1) and robot_x <= (setpoint1[stop_count][0] + 1)  and robot_y >= (setpoint1[stop_count][1]):
                #         print('the ', stop_count,' station')
                #         self.send_motor_pwm('stop',0,0)
                #         time.sleep(10)
                #         stop_count = stop_count + 1

                # elif stop_count > 3 and stop_count <= 7:
                #     section2_count = stop_count - 4
                #     if robot_x >= (setpoint2[section2_count][0] - 1) and robot_x <= (setpoint2[section2_count][0] + 1)  and robot_y <= (setpoint2[section2_count][1]) :
                #         print('the ', stop_count,' station')
                #         self.send_motor_pwm('stop',0,0)
                #         time.sleep(10)
                #         stop_count = stop_count + 1
                if turningtest_onoff == 0:
                    if stop_count <= 32:
                        if robot_y >= (setpoint1[stop_count][1]):
                            print('the ', stop_count, ' station')
                            self.send_motor_pwm(
                                'photo',
                                0,
                                0,
                                'A' + str(stop_count + 1),
                                'B' + str(stop_count + 1),
                            )
                            time.sleep(1)
                            self.send_motor_pwm('stop', 0, 0, 'error_p', 'error_p')
                            time.sleep(4)
                            stop_count = stop_count + 1

                    elif stop_count > 32 and stop_count <= 65:
                        section2_count = stop_count - 33
                        if robot_y <= (setpoint2[section2_count][1]):
                            print('the ', stop_count, ' station')
                            self.send_motor_pwm(
                                'photo',
                                0,
                                0,
                                'C' + str(section2_count + 1),
                                'D' + str(section2_count + 1),
                            )
                            time.sleep(5)
                            stop_count = stop_count + 1

                    elif stop_count > 65 and stop_count <= 98:
                        section3_count = stop_count - 66
                        # if robot_x >= (setpoint3[section3_count][0] - 1) and robot_x <= (setpoint3[section3_count][0] + 1)  and robot_y >= (setpoint3[section3_count][1]):
                        if robot_y >= (setpoint3[section3_count][1]):
                            print('the ', stop_count, ' station')
                            self.send_motor_pwm(
                                'photo',
                                0,
                                0,
                                'E' + str(section3_count + 1),
                                'F' + str(section3_count + 1),
                            )
                            time.sleep(5)
                            stop_count = stop_count + 1

                    elif stop_count > 98 and stop_count <= 131:
                        section4_count = stop_count - 99
                        # if robot_x >= (setpoint4[section4_count][0] - 1) and robot_x <= (setpoint3[section4_count][0] + 1)  and robot_y >= (setpoint3[section3_count][1]):
                        if robot_y <= (setpoint4[section4_count][1]):
                            print('the ', stop_count, ' station')
                            self.send_motor_pwm(
                                'photo',
                                0,
                                0,
                                'A' + str(section4_count + 1),
                                'B' + str(section4_count + 1),
                            )
                            time.sleep(5)
                            stop_count = stop_count + 1

                # elif stop_count > 11 and stop_count <= 15:
                #     section4_count = stop_count - 12
                #     if robot_x >= (setpoint4[section4_count][0] - 1) and robot_x <= (setpoint4[section4_count][0] + 1)  and robot_y <= (setpoint4[section4_count][1]) :
                #         print('the ', stop_count,' station')
                #         self.send_motor_pwm('stop',0,0)
                #         time.sleep(5)
                #         stop_count = stop_count + 1

                # 9/4晚上關掉的
                if right_count == 0:
                    if robot_y >= (right_point1[1]):
                        print('the ', right_count, ' right turning')
                        self.send_motor_pwm('right', 10, 22, 'error_p', 'error_p')
                        time.sleep(9)
                        right_count = right_count + 1

                # if robot_x >= (right_point2[0] - 1) and  robot_x <= (right_point2[0] + 1)  and robot_y >= (right_point2[1]) :
                #         print('the ', right_count,' right turning')
                #         self.send_motor_pwm('right',10,25)
                #         time.sleep(6.8)
                #         right_count = right_count + 1

                if left_count == 0 and right_count == 1:
                    if robot_y <= (left_point[1]):
                        print('the ', left_count, ' left turning')
                        self.send_motor_pwm('left', 22, 10, 'error_p', 'error_p')
                        time.sleep(9)
                        left_count = left_count + 1

                if left_count == 1 and right_count == 1:
                    if robot_y >= (right_point2[1]):
                        print('hello...........................................')
                        print('the ', right_count, ' right turning')
                        self.send_motor_pwm('right', 10, 20, 'error_p', 'error_p')
                        time.sleep(9)
                        right_count = right_count + 1

                if robot_y <= (end_point[1]) and left_count == 1 and right_count == 2:
                    print('the end')
                    self.send_motor_pwm('stop', 0, 0, 'error_p', 'error_p')
                    m_l = 0
                    m_r = 0

            if m_l > m_r:  # left head
                self.send_motor_pwm('right', m_r, m_l, 'error_p', 'error_p')
            elif m_l < m_r:  # right head
                self.send_motor_pwm('left', m_r, m_l, 'error_p', 'error_p')
            else:
                self.send_motor_pwm('mid', 2, 2, 'error_p', 'error_p')

            print('x', robot_x, 'y', robot_y)

            self.control_signal.emit([123])
            time.sleep(0.5)

    def open(self):
        # global E_lidar
        # global E_view
        # E_lidar = False
        # E_view = False

        self.control = True  # 啟動讀取狀態

    def stop(self):
        self.control = False  # 關閉讀取狀態
        self.send_motor_pwm('stop', 0, 0, 'error_p', 'error_p')

    def close(self):
        self.control = False  # 關閉讀取狀態
        time.sleep(1)
        self.send_motor_pwm('stop', 0, 0, 'error_p', 'error_p')

    def send_motor_pwm(self, direction, right_pwm, left_pwm, section_r, section_l):
        pwm = {
            'direction': direction,
            'right': float(right_pwm),
            'left': float(left_pwm),
            'section_r': section_r,
            'section_l': section_l,
        }
        ser = serial.Serial(
            port='/dev/ttyTHS0',  # Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
            baudrate=115200,
            timeout=0.1,
        )
        # ser.write(str.encode(f'{counter}\n'))s
        ser.write(bytes(str(pwm), 'utf-8'))
        ser.flush()
        # time.sleep(0.1)
        # print(pwm)
        ser.close()






class Lidar_fig(Myplot, QtCore.QThread):
    def __init__(self, *args, **kwargs):


        Myplot.__init__(self, *args, **kwargs)

    def compute_initial_figure(self):
        x_lidar = 0
        y_lidar = 0
        self.axes.set_xlim(left=-5000, right=5000)
        self.axes.set_ylim(bottom=-5000, top=5000)
        self.axes.plot(x_lidar, y_lidar, 'b')
        self.axes.set_title("LiDAR scanning")
        self.axes.set_xlabel("x (cm)")
        self.axes.set_ylabel("y (cm)")


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    uwb_startThread = QtCore.pyqtSignal()
    lidar_startThread = QtCore.pyqtSignal()
    control_startThread = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.frontviewdata.setScaledContents(True)

        self.view_x_lidar = self.front_view_area.horizontalScrollBar()
        self.view_y_lidar = self.front_view_area.verticalScrollBar()

        self.last_move_x = 0
        self.last_move_y = 0
        self.frame_num = 0

        self.last_move_x_lidar = 0
        self.last_move_y_lidar = 0
        self.frame_num_lidar = 0

        self.ProcessCam = Camera()

        # self.LiDARopen.setEnabled(False)
        # self.LiDARclose.setEnabled(True)

        self.fig_lidar = Lidar_fig(width=5, height=5, dpi=100)
        # add NavigationToolbar in the figure (widgets)
        # self.fig_ntb1 = NavigationToolbar(self.fig1, self)
        # self.fig_ntb2 = NavigationToolbar(self.fig2, self)
        # self.Start_plot.clicked.connect(self.plot_cos)
        # add the static_fig in the Plot box
        # self.LiDARbox=QGridLayout(self.Plot_static)

        # LiDAR
        self.LiDARopen.clicked.connect(self.lidar_start)  # 綁定多線程
        self.LiDARclose.clicked.connect(self.lidar_stop)  # 綁定多線程
        self.lidar_T = Lidar_run()  # 創建線程對象
        self.thread_lidar = QtCore.QThread(self)  # 初始化QThread
        self.lidar_T.moveToThread(self.thread_lidar)
        self.lidar_startThread.connect(self.lidar_T.run)  # 只能通過信號-槽啟動線程處理函數
        self.lidar_T.lidar_signal.connect(self.lidar_call_backlog)

        self.LiDARbox.addWidget(self.fig_lidar)
        # self.LiDARbox.addWidget(self.fig_ntb1)
        # add the dynamic_fig in the Plot box

        self.adduwbpoint.clicked.connect(self.uwb_start)  # 綁定多線程
        self.deleteuwbpoint.clicked.connect(self.uwb_stop)  # 綁定多線程

        self.uwb_T = UWB_run()  # 創建線程對象
        self.thread_uwb = QtCore.QThread(self)  # 初始化QThread
        self.uwb_T.moveToThread(self.thread_uwb)
        self.uwb_startThread.connect(self.uwb_T.run)  # 只能通過信號-槽啟動線程處理函數
        self.uwb_T.signal.connect(self.uwb_call_backlog)

        self.fig_uwb = UWB_fig(width=5, height=5, dpi=100)
        # self.UWBbox = QGridLayout(self.Plot_dynamic)
        self.UWBbox.addWidget(self.fig_uwb)
        # self.UWBbox.addWidget(self.fig_ntb2)

        if self.ProcessCam.connect:
            self.debugBar('Connection!!!')
            self.ProcessCam.rawdata.connect(self.getRaw)
        else:
            self.debugBar('Disconnection!!!')

        self.control_T = Control_system()  # 創建線程對象
        self.Start.clicked.connect(self.control_start)  # 綁定多線程觸發事件
        self.End.clicked.connect(self.control_stop)  # 綁定多線程觸發事件

        self.viewopen.clicked.connect(self.openCam)
        self.viewend.clicked.connect(self.stopCam)
        # self.LiDARopen.clicked.connect(self.openLidar)
        # self.LiDARclose.clicked.connect(self.stopLidar)

    def control_start(self):
        self.control_T.open()
        self.control_T.start()
        self.Start.setEnabled(False)
        self.End.setEnabled(True)

    def control_stop(self):
        self.control_T.stop()
        self.Start.setEnabled(True)
        self.End.setEnabled(False)

    def getRaw(self, data):
        self.showData(data)

    def uwb_start(self):
        if self.thread_uwb.isRunning():  # 如果該線程正在運行，則不再重新啟動
            return

        # 先啟動QThread子線程
        self.uwb_T.flag = True
        self.thread_uwb.start()
        # 發送信號，啟動線程處理函數
        # 不能直接調用，否則會導致線程處理函數和主線程是在同一個線程，同樣操作不了主界面
        self.uwb_startThread.emit()

    def uwb_stop(self):
        if not self.thread_uwb.isRunning():  # 如果該線程已經結束，則不再重新關閉
            return
        self.uwb_T.flag = False
        self.uwb_stop_thread()

    def uwb_call_backlog(self, msg):
        # self.pbar.setValue(int(msg))  # 將線程的參數傳入進度條
        self.fig_uwb.axes.cla()
        self.fig_uwb.axes.set_xlim(left=0, right=9.6)
        self.fig_uwb.axes.set_ylim(bottom=0, top=40)
        # self.fig_uwb.axes.vlines(path_x, 0, 40, colors="green", linestyle = "--")
        self.fig_uwb.axes.plot(msg[0], msg[1], '-or', markersize=4)
        self.fig_uwb.axes.set_title("UWB position")
        self.fig_uwb.axes.set_xlabel("x (cm)")
        self.fig_uwb.axes.set_ylabel("y (cm)")
        self.fig_uwb.draw()

    def uwb_stop_thread(self):
        print(">>> stop_thread... ")
        if not self.thread_uwb.isRunning():
            return
        self.thread_uwb.quit()  # 退出
        self.thread_uwb.wait()  # 回收資源
        print(">>> stop_thread end... ")

    def lidar_start(self):
        if self.thread_lidar.isRunning():
            return

        self.lidar_T.lidar_flag = True
        self.thread_lidar.start()
        self.lidar_startThread.emit()

    def lidar_stop(self):
        global E_lidar
        if not self.thread_lidar.isRunning():
            return
        self.lidar_T.lidar_flag = False
        self.lidar_stop_thread()
        E_lidar = False

    def lidar_call_backlog(self, msg):
        global E_lidar

        x_sensor = msg[0]
        y_sensor = msg[1]
        lidar_data = []

        for index, data in enumerate(x_sensor):
            lidar_data.append([x_sensor[index], y_sensor[index]])

        lidar_data = np.array(lidar_data)

        (
            x_c,
            y_c,
            lidar_angle,
            label_array,
            x_l,
            y_l,
            x_r,
            y_r,
            mid_dis,
        ) = self.lidar_direction(lidar_data)

        y_o = np.linspace(-1000, 1000, 4000)
        x_o = np.zeros(y_o.size)

        # print('lidar_angle   :',lidar_angle,'    mid_dis',mid_dis)

        if np.isnan(lidar_angle) == True or np.isnan(lidar_angle) == True:
            sensor = np.array([[np.float32(0)], [np.float32(0)]])
        else:
            sensor = np.array([[np.float32(lidar_angle)], [np.float32(mid_dis)]])

        # # print(sensor)
        # if lidar_angle == float and lidar_angle == float:
        # kalman
        x = kalman.correct(sensor)
        y = kalman.predict()
        E_lidar = lidar_w_angle * (y[0] / 90) + lidar_w_distance * (y[1] / 2000)
        # else:
        #     x = 0
        #     y = 0
        # E_lidar =  False

        # E_lidar = lidar_w_angle*(y[0]/90) + lidar_w_distance*(y[1]/2000)
        # print(E_lidar)

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

    def lidar_stop_thread(self):
        print(">>> stop_thread... ")
        if not self.thread_lidar.isRunning():
            return
        self.thread_lidar.quit()  # 退出
        self.thread_lidar.wait()  # 回收资源
        print(">>> stop_thread end... ")

    def openCam(self):
        if self.ProcessCam.connect:
            self.ProcessCam.open()
            self.ProcessCam.start()
            self.viewopen.setEnabled(False)
            self.viewend.setEnabled(True)
            self.comboBox.setEnabled(True)

    def stopCam(self):
        global E_view
        E_view = False
        if self.ProcessCam.connect:
            self.ProcessCam.stop()
            self.viewopen.setEnabled(True)
            self.viewend.setEnabled(False)
            self.comboBox.setEnabled(False)

    def showData(self, img):
        self.Ny, self.Nx, _ = img.shape

        # 反轉顏色
        img_new = np.zeros_like(img)
        img_new[..., 0] = img[..., 2]
        img_new[..., 1] = img[..., 1]
        img_new[..., 2] = img[..., 0]
        img = img_new

        # qimg = QtGui.QImage(img[:,:,0].copy().data, self.Nx, self.Ny, QtGui.QImage.Format_Indexed8)
        qimg = QtGui.QImage(img.data, self.Nx, self.Ny, QtGui.QImage.Format_RGB888)
        self.frontviewdata.setScaledContents(True)
        self.frontviewdata.setPixmap(QtGui.QPixmap.fromImage(qimg))
        if self.comboBox.currentIndex() == 0:
            roi_rate = 0.5
        elif self.comboBox.currentIndex() == 1:
            roi_rate = 0.75
        elif self.comboBox.currentIndex() == 2:
            roi_rate = 1
        elif self.comboBox.currentIndex() == 3:
            roi_rate = 1.25
        elif self.comboBox.currentIndex() == 4:
            roi_rate = 1.5
        else:
            pass
        self.frontview.setMinimumSize(self.Nx * roi_rate, self.Ny * roi_rate)
        self.frontview.setMaximumSize(self.Nx * roi_rate, self.Ny * roi_rate)
        self.frontviewdata.setMinimumSize(self.Nx * roi_rate, self.Ny * roi_rate)
        self.frontviewdata.setMaximumSize(self.Nx * roi_rate, self.Ny * roi_rate)

        if self.frame_num == 0:
            self.time_start = time.time()
        if self.frame_num >= 0:
            self.frame_num += 1
            self.t_total = time.time() - self.time_start
            if self.frame_num % 100 == 0:
                self.frame_rate = float(self.frame_num) / self.t_total
                self.debugBar('FPS: %0.3f frames/sec' % self.frame_rate)

    def eventFilter(self, source, event):
        if source == self.front_view_area:
            if event.type() == QtCore.QEvent.MouseMove:
                if self.last_move_x == 0 or self.last_move_y == 0:
                    self.last_move_x = event.pos().x()
                    self.last_move_y = event.pos().y()
                distance_x = self.last_move_x - event.pos().x()
                distance_y = self.last_move_y - event.pos().y()
                self.view_x.setValue(self.view_x.value() + distance_x)
                self.view_y.setValue(self.view_y.value() + distance_y)
                self.last_move_x = event.pos().x()
                self.last_move_y = event.pos().y()
            elif event.type() == QtCore.QEvent.MouseButtonRelease:
                self.last_move_x = 0
                self.last_move_y = 0
            return QtWidgets.QWidget.eventFilter(self, source, event)

        if source == self.LiDARArea:
            if event.type() == QtCore.QEvent.MouseMove:
                if self.last_move_x_lidar == 0 or self.last_move_y == 0:
                    self.last_move_x_lidar = event.pos().x()
                    self.last_move_y_lidar = event.pos().y()
                distance_x = self.last_move_x_lidar - event.pos().x()
                distance_y = self.last_move_y_lidar - event.pos().y()
                self.view_x_lidar.setValue(self.view_x_lidar.value() + distance_x)
                self.view_y_lidar.setValue(self.view_y_lidar.value() + distance_y)
                self.last_move_x_lidar = event.pos().x()
                self.last_move_y_lidar = event.pos().y()
            elif event.type() == QtCore.QEvent.MouseButtonRelease:
                self.last_move_x_lidar = 0
                self.last_move_y_lidar = 0
            return QtWidgets.QWidget.eventFilter(self, source, event)

    def closeEvent(self, event):
        global E_view
        E_view = False
        if self.ProcessCam.running:
            self.ProcessCam.close()
            self.ProcessCam.terminate()
        QtWidgets.QApplication.closeAllWindows()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            if self.ProcessCam.running:
                self.ProcessCam.close()
                time.sleep(1)
                self.ProcessCam.terminate()
            QtWidgets.QApplication.closeAllWindows()

    def debugBar(self, msg):
        self.statusBar().showMessage(str(msg), 5000)

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

    def get_function(self, x, y):
        try:
            a = (y[1] - y[0]) / (x[1] - x[0])
            b = y[0] - a * x[0]
            y_0 = np.linspace(-1000, 1000, 4000)
            x_0 = (y_0 - b) / a
        except:
            a = 0
            b = 0
            y_0 = 0
            x_0 = 0
        return x_0, y_0

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

    def send_motor_pwm(
        self, direction, right_pwm, left_pwm,
    ):
        pwm = {
            'direction': direction,
            'right': float(right_pwm),
            'left': float(left_pwm),
        }
        ser = serial.Serial(
            port='/dev/ttyTHS0',  # Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
            baudrate=115200,
            timeout=0.1,
        )
        # ser.write(str.encode(f'{counter}\n'))s
        ser.write(bytes(str(pwm), 'utf-8'))
        ser.flush()
        # time.sleep(0.1)
        # print(pwm)
        ser.close()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
