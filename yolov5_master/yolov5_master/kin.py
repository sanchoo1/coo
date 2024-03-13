import os
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np

class Params:
    def __init__(self) -> None:
        self.d_intrinsics = np.array([[364.7757, 0.539, 260.9733], [0, 365.605, 209.7964], [0, 0, 1]])
        self.c_intrinsics = np.array([[1038.6, 0, 951.1293], [0, 1038.9, 526.1001], [0, 0, 1]])
        self.RT_d_to_c = np.array([[0.999820242867276, -0.005590505683472, 0.018117069272502, -65.337633677812520],
                           [0.005420373979096, 0.999940881177058, 0.009426223887326, 3.371261769890467],
                           [-0.018168695570907, -0.009326328165486, 0.999791437302901, 17.491786028014985],
                           [0, 0, 0,1]])       
        self.d_distor = np.array([0.1174, 0.2216, 0, 0, 0])
        self.c_distor = np.array([0.0378, 0.1169, 0, 0, 0])
        self.client_socket = None
        self.x1 = list()
        self.x2 = list()
    
    def reset_pos(self):
        self.x1.clear()
        self.x2.clear()

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Infrared)

# cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,512) #width
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,424) #height

def get_last_rbg():
    frame = kinect.get_last_color_frame()
    return np.reshape(frame, [1080, 1920, 4])[:, :, 0:3]

def get_last_depth():
    frame = kinect.get_last_depth_frame()  
    dep_frame = np.reshape(frame, [424, 512])   
    # dep_frame = cv2.flip(dep_frame, 0)
    dep_frame = cv2.flip(dep_frame, 1)
    return dep_frame

def get_last_inf():
    frame = kinect.get_last_infrared_frame()
    # frame = frame.astype(np.uint8)
    
    inf_frame = np.reshape(frame, [424, 512])
    inf_frame = cv2.flip(inf_frame, 1)
    # inf_frame = cv2.flip(inf_frame, 0) 
    
    # inf_frame = cv2.undistort(inf_frame, Params().d_intrinsics, Params().d_distor)
    # dep_frame = cv2.undistort(dep_frame, Params().d_intrinsics, Params().d_distor)
    # np.savetxt('D:\\Desktop\\photo\\cancan.txt', dep_frame, fmt = "%d", delimiter = ' ')
    return inf_frame

def p_2_w(x, y, z, param : Params):
    color_pixel_coordinate = np.array([[x],
                                    [y],
                                    [float(1)]])
    d_world_coordinate = np.dot(np.linalg.inv(param.d_intrinsics),float(z) * color_pixel_coordinate)
    return d_world_coordinate

# def on_EVENT_LBUTTONDOWN(event, y, x, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x, y)
#         # print(param[x][y])
#         z = param[x][y]
#         # print(p_2_w(x, y, z, Params()))
#         x = (x -212)/367.816 * z
#         y = (y -256)/367.816 * z
#         print(x,y,z)
#         cv2.imwrite(path + str(counter) + '.png', get_last_rbg())

def get_Pos(x1, x2):
    x = int((x1[0] + x2[0]) / 2)
    y = int((x1[1] + x2[1]) / 2)
    z = get_last_depth()[x][y]
    x = (x -212)/367.816 * z
    y = (y -256)/367.816 * z
    print(z)
    return x,y,z
    


# cv2.namedWindow("depth")
# path = 'D:\\Desktop\\photocoooo\\'
# os.makedirs(path, exist_ok = True)

""" while 1:
    inf_frame = get_last_inf()
    success, img = cap.read()
    img = cv2.flip(img, 1)
    cv2.imshow("depth", inf_frame)
    cv2.imshow("output", img)
    counter = 1
    cv2.setMouseCallback("output", on_EVENT_LBUTTONDOWN, get_last_depth())
    cv2.waitKey(1)
 """