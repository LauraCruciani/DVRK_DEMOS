#!/usr/bin/env python

import time
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage, PointCloud2
import open3d as o3d
from rec import load_camera_param, HSM_loading, elaboration_param
from rec import rectification_resize, HSM_computation, point_cloud, xyzrgb_array_to_pointcloud2
import os
import torch

class ImageFeature:

    def __init__(self):
        self.Kl, self.Kr, self.dist_l, self.dist_r, self.R, self.T = [None]*6
        self.R_l, self.R_r, self.P_l, self.P_r, self.Q = [None]*5
        self.mapx_l, self.mapy_l, self.mapx_r, self.mapy_r = [None]*4
        self.model = None
        self.count = 0
        self.pcd = None
        self.w, self.h = 1920, 1080  # Dimensioni originali
        self.w_res, self.h_res = 640, 480  # Dimensioni ridotte
        self.dictL, self.dictR = {}, {}
        self.secondKL = self.firstKL = self.secondKR = self.firstKR = None

    def callback_left(self, ros_data):
        np_arr = np.fromstring(ros_data.data, np.uint8)
        kl = int(ros_data.header.stamp.to_nsec() // 10e6)
        self.dictL[kl] = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.secondKL, self.firstKL = self.firstKL, kl

    def callback_right(self, ros_data):
        np_arr = np.fromstring(ros_data.data, np.uint8)
        kr = int(ros_data.header.stamp.to_nsec() // 10e6)
        self.dictR[kr] = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.secondKR, self.firstKR = self.firstKR, kr

    def computation(self):
        imgL0, imgR0 = self.dictL[self.secondKL], self.dictR[self.secondKR]
        imgL1, imgR1 = self.dictL[self.firstKL], self.dictR[self.firstKR]

        if len(self.dictL.keys()) > 50:
            L = [self.dictL[self.secondKL], self.dictL[self.firstKL]]
            R = [self.dictR[self.secondKR], self.dictR[self.firstKR]]
            self.dictL.clear()
            self.dictR.clear()
            self.dictL[self.secondKL], self.dictL[self.firstKL] = L
            self.dictR[self.secondKR], self.dictR[self.firstKR] = R

        imgL_resized = rectification_resize(imgL1, self.mapx_l, self.mapy_l, self.h_res, self.w_res, 'L', self.count)
        imgR_resized = rectification_resize(imgR1, self.mapx_r, self.mapy_r, self.h_res, self.w_res, 'R', self.count)

        torch.cuda.synchronize()
        disparity_hsm = HSM_computation(self.model, imgL_resized, imgR_resized, self.h_res, self.w_res, self.count)

        colors = cv2.cvtColor(imgL_resized, cv2.COLOR_BGR2RGB)
        colors = np.reshape(colors, (self.h_res, self.w_res, 3)) / 255.0
        points_3D_hsm = point_cloud(disparity_hsm, colors, self.Q, self.h_res, self.w_res)

        points_3D_hsm[:, 2] = np.multiply(points_3D_hsm[:, 2] + 0.86, -1)
        points_3D_hsm = np.multiply(points_3D_hsm, 10)
        cloud_msg = xyzrgb_array_to_pointcloud2(points_3D_hsm, colors, 'map')
        pub_point.publish(cloud_msg)

        self.count += 1

if __name__ == '__main__':
    rospy.init_node('reconstruction', anonymous=True)
    rate = rospy.Rate(10)

    ic = ImageFeature()
    work_path = '/home/laura/Desktop/3d-rec/Pipeline'
    loadmodel = './Weights/final-768px.tar'
    ic.model = HSM_loading(loadmodel, 1.0, 384, 1, 1, work_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    [ic.Kl, ic.Kr, ic.dist_l, ic.dist_r, ic.R, ic.T] = load_camera_param(work_path + '/CalibrationParameters.xml')
    [ic.Q, ic.mapx_l, ic.mapy_l, ic.mapx_r, ic.mapy_r] = elaboration_param(
        ic.Kl, ic.Kr, ic.dist_l, ic.dist_r, ic.R, ic.T, ic.h, ic.w, ic.h_res, ic.w_res
    )

    rospy.Subscriber("/endoscope/raw/left/image_raw/compressed", CompressedImage, ic.callback_left, queue_size=1)
    rospy.Subscriber("/endoscope/raw/right/image_raw/compressed", CompressedImage, ic.callback_right, queue_size=1)
    pub_point = rospy.Publisher('point_cloud', PointCloud2, queue_size=1)

    while not rospy.is_shutdown():
        if len(ic.dictL.keys()) > 1:
            ic.computation()
        rate.sleep()
