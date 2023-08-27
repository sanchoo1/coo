import cv2
import argparse
import numpy as np
import stereoconfig


# 左相机内参
leftIntrinsic = np.array([[832.6859, 0.6550, 327.3144], [0, 832.1205, 243.6162],
                          [0, 0, 1]])
# 右相机内参
rightIntrinsic = np.array([[832.6859, 0.6550, 327.3144], [0, 832.1205, 243.6162],
                          [0, 0, 1]])

# 旋转矩阵
leftRotation = np.array([[1, 0, 0],  # 旋转矩阵
                         [0, 1, 0],
                         [0, 0, 1]])
rightRotation = np.array([[0.9993, -0.0038, -0.0364],
                          [0.0033, 0.9999, -0.0143],
                          [0.0365, 0.0142, 0.9992]])

# 平移矩阵
rightTranslation = np.array([[-44.8076], [5.7648], [51.7586]])
leftTranslation = np.array([[0],  # 平移矩阵
                            [0],
                            [0]])


# 预处理
def preprocess(img1, img2):
    # 彩色图->灰度图
    if (img1.ndim == 3):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if (img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    return img1, img2


# 消除畸变
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return undistortion_image


# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
# @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

    return rectifyed_img1, rectifyed_img2


# 视差计算
def stereoMatchSGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    paraml = {'minDisparity': 0,
              'numDisparities': 64,
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': 1,
              'preFilterCap': 63,
              'uniquenessRatio': 15,
              'speckleWindowSize': 100,
              'speckleRange': 1,
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }
    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM.create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM.create(**paramr)

    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    return trueDisp_left, trueDisp_right


# 将h×w×3数组转换为N×3的数组
def hw3ToN3(points):
    height, width = points.shape[0:2]

    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)

    points_ = np.hstack((points_1, points_2, points_3))

    return points_


def getDepthMapWithQ(disparityMap: np.ndarray, Q: np.ndarray) -> np.ndarray:
    points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
    depthMap = points_3d[:, :, 2]
    cv2.imwrite('D:\\Desktop\\photodepthMap00.jpg', depthMap)
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0
    cv2.imwrite('D:\\Desktop\\photodepthMap.jpg', depthMap)
    return depthMap.astype(np.float32)


def getDepthMapWithConfig(config: stereoconfig.stereoCamera) -> np.ndarray:
    fb = config.cam_matrix_left[0, 0] * (-config.T[0])
    doffs = config.doffs
    disparity = dot_disp
    depth = fb / (disparity + doffs)
    print(fb)
    print(disparity)
    return depth


vs = cv2.VideoCapture(1)  # 参数0表示第一个摄像头
vs2 = cv2.VideoCapture(2)



# 分配摄像头分辨率
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

vs2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vs2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 判断视频是否打开
if (vs.isOpened()):
    print('camera Opened')
else:
    print('摄像头未打开')

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT.create(), "kcf": cv2.legacy.TrackerKCF.create(),
    "boosting": cv2.legacy.TrackerBoosting.create(), "mil": cv2.legacy.TrackerMIL.create(),
    "tld": cv2.legacy.TrackerTLD.create(),
    "medianflow": cv2.legacy.TrackerMedianFlow.create(), "mosse": cv2.legacy.TrackerMOSSE.create()
}

trackers = cv2.legacy.MultiTracker.create()

# 读取相机内参和外参
# 使用之前先将标定得到的内外参数填写到stereoconfig.py中的StereoCamera类中
config = stereoconfig.stereoCamera()
config.setMiddleBurryParams()
print(config.cam_matrix_left)
cv2.namedWindow("Frame")
cv2.namedWindow("right")
while True:
    right_frame = vs.read()
    right_frame = right_frame[1]
    if right_frame is None:
        break
    # 设置右摄像头尺寸
    # right_frame = frame[0:720, 1280:2560]
    (h, w) = right_frame.shape[:2]
    width = 800
    r = width / float(w)
    dim = (width, int(h * r))
    right_frame = cv2.resize(right_frame, dim, interpolation=cv2.INTER_AREA)

    # 设置左摄像头尺寸
    # left_frame = frame[0:720, 0:1280]
    left_frame = vs2.read()
    left_frame = left_frame[1]
    (h, w) = left_frame.shape[:2]
    width = 800
    r = width / float(w)
    dim = (width, int(h * r))
    left_frame = cv2.resize(left_frame, dim, interpolation=cv2.INTER_AREA)

    # 对做摄像头做目标识别初始化
    (success, boxes) = trackers.update(left_frame)

    # 画图的循环
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(left_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 转化成框框中点的坐标
        xx = round((2 * x + w) / 2)
        yy = round((2 * y + h) / 2)

        # 读取一帧图片
        iml = left_frame  # 左图
        imr = right_frame  # 右图
        height, width = iml.shape[0:2]

        # 立体校正
        map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width,
                                                            config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
        # print(Q)

        # 立体匹配
        iml_, imr_ = preprocess(iml, imr)  # 预处理，一般可以削弱光照不均的影响，不做也可以
        disp, _ = stereoMatchSGBM(iml, imr, False)  # 这里传入的是未经立体校正的图像，因为我们使用的middleburry图片已经是校正过的了
        dot_disp = disp[yy][xx]
        cv2.imwrite('D:\\Desktop\\photodisaprity.jpg', disp * 16)

        # xr和yr是右相机相应点的像素坐标
        z = getDepthMapWithConfig(config)
        # z = getDepthMapWithQ(disp, Q)
        text = str(xx) + ',' + str(yy) + ',' + str(z)
        cv2.putText(left_frame, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)

    # 显示两个框
    cv2.imshow("right", right_frame)
    cv2.imshow('Frame', left_frame)

    # 按键判断是否设置新的目标
    key = cv2.waitKey(10) & 0xFF
    if key == ord('s'):
        box = cv2.selectROI('Frame', left_frame, fromCenter=False, showCrosshair=True)
        tracker = cv2.legacy.TrackerCSRT.create()
        print(type(box), type(box[0]), box[1], box)
        trackers.add(tracker, left_frame, box)
    elif key == ord('q'):
        break
vs.release()
vs2.release()
cv2.destroyAllWindows()