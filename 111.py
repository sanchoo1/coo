import numpy as np
import cv2

thres = 0.5 # Threshold to detect object
nms_threshold = 0.2 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress 
'''
读取摄像头
'''
cap = cv2.VideoCapture(2) #这里也可以换成视频路径
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1000) #width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1000) #height
cap.set(cv2.CAP_PROP_BRIGHTNESS,150) #brightness 

'''
读取分类
'''
classNames = []
with open('coco.names','r') as f:
    classNames = f.read().splitlines() #按行拆分
# print(classNames)


'''
识别字体颜色
'''
font = cv2.FONT_HERSHEY_PLAIN
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

'''
构建网络
'''
weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

'''
识别模块
'''
while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    if len(classIds) != 0:
        for i in indices:
            box = bbox[i]
            confidence = str(round(confs[i],2))
            color = Colors[classIds[i]-1]
            x,y,w,h = box[0],box[1],box[2],box[3]
            cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness=2)
            cv2.putText(img, classNames[classIds[i]-1]+" "+confidence,(x+10,y+20),
                        font,1,color,2)

    cv2.imshow("Output",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):break
