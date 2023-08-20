# -*- coding: utf-8 -*-
import cv2
import time

AUTO = False  # 自动拍照，或手动按s键拍照
INTERVAL = 2  # 自动拍照间隔


camera1 = cv2.VideoCapture(1)
camera2 = cv2.VideoCapture(2)

camera1.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

camera2.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

counter = 0
utc = time.time()
folder = "./SaveImage/"  # 拍照文件目录


def shot(pos, frame):
    global counter
    path = folder + pos + "_" + str(counter) + ".jpg"

    cv2.imwrite(path, frame)
    print("snapshot saved into: " + path)

cv2.namedWindow("left")
cv2.namedWindow("right")
while True:
    ret1, left_frame = camera1.read()
    ret2, right_frame = camera2.read()

    print("ret:", ret1)
    # 裁剪坐标为[y0:y1, x0:x1]    HEIGHT * WIDTH
    # left_frame = frame[0:720, 0:1280]
    # right_frame = frame[0:720, 1280:2560]

    cv2.imshow("left", left_frame)
    cv2.imshow("right", right_frame)

    now = time.time()
    if AUTO and now - utc >= INTERVAL:
        shot("left", left_frame)
        shot("right", right_frame)
        counter += 1
        utc = now

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        shot("left", left_frame)
        shot("right", right_frame)
        counter += 1
camera.release()
cv2.destroyWindow("left")
cv2.destroyWindow("right")