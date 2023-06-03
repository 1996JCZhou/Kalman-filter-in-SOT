import cv2

video = cv2.VideoCapture("D:\\Object Detection\\SOT\\Kalman filter in SOT\\data\\testvideo1.mp4")

FPS = video.get(cv2.CAP_PROP_FPS) # Read video FPS.
print(FPS)

video.release()
