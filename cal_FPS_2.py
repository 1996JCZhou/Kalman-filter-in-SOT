import cv2

video = cv2.VideoCapture("D:\\Object Detection\\SOT\\Kalman filter in SOT\\data\\testvideo1.mp4")

fps = video.get(cv2.CAP_PROP_FPS) # Read video FPS.
print('FPS:', fps)

frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # Read video frames.
print('Number of frames:', frame_count)

duration_ms = int(frame_count / fps * 1000) # Calculate video duration.
print('Video duration (ms):', duration_ms)

video.release()
