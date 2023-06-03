import cv2

video = cv2.VideoCapture("D:\\Object Detection\\SOT\\Kalman filter in SOT\\data\\testvideo1.mp4")

frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # Read video frames.
print(frame_count)

video.release()
