import cv2

index = 0
while True:
    cap = cv2.VideoCapture(index)
    if not cap.read()[0]:
        break
    cap.release()
    index += 1

print("Available camera index:", list(range(index)))