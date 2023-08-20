import numpy as np
import cv2
class stereoCamera(object):
    def __init__(self):
        self.cam_matrix_left = np.array([[832.6859, 0.6550, 327.3144], [0, 832.1205, 243.6162],
                          [0, 0, 1]])

        self.cam_matrix_right = np.array([[832.6859, 0.6550, 327.3144], [0, 832.1205, 243.6162],
                          [0, 0, 1]])

        self.distortion_l = np.array([[-0.0234, 1.1738, -3.9133, 0.00037231, -0.00051110]])

        self.distortion_r = np.array([[-0.0234, 1.1738, -3.9133, 0.00037231, -0.00051110]])

        self.R = np.array([[0.9993, -0.0038, -0.0364],
                                  [0.0033, 0.9999, -0.0143],
                                  [0.0365, 0.0142, 0.9992]])

        self.T = np.array([[-44.8076], [5.7648], [51.7586]])

        self.doffs = 0.0

        self.isRectified = False

    def setMiddleBurryParams(self):
        self.cam_matrix_left = np.array([[832.6859, 0.6550, 327.3144], [0, 832.1205, 243.6162],
                                         [0, 0, 1]])

        self.cam_matrix_right = np.array([[832.6859, 0.6550, 327.3144], [0, 832.1205, 243.6162],
                                          [0, 0, 1]])

        self.distortion_l = np.array([[-0.0234, 1.1738, -3.9133, 0.00037231, -0.00051110]])

        self.distortion_r = np.array([[-0.0234, 1.1738, -3.9133, 0.00037231, -0.00051110]])

        self.R = np.array([[0.9993, -0.0038, -0.0364],
                           [0.0033, 0.9999, -0.0143],
                           [0.0365, 0.0142, 0.9992]])

        self.T = np.array([[-44.8076], [5.7648], [51.7586]])

        self.doffs = 131.111

        self.isRectified = True