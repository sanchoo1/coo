import cv2
import os
import time

camera_index = 1


output_folder = "D:\\Desktop\\photo"


os.makedirs(output_folder, exist_ok=True)


cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Can't open camera")
    exit()

try:
    while True:
    
        ret, frame = cap.read()
        if not ret:
            print("Can't capture image")
            break
        
        timestamp = int(time.time())
        filename = os.path.join(output_folder, f"image_{timestamp}.jpg")
        
        cv2.imwrite(filename, frame)
        print(f"Saved as {filename}")
        
        time.sleep(1)

except KeyboardInterrupt:
    pass


cap.release()
cv2.destroyAllWindows()
