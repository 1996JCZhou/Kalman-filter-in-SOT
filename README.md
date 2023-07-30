# Kalman filter appliedd in Single Object Tracking

Welcome to my GitHub project that explores the application of Kalman Filter in the realm of Single Object Tracking (SOT). In this endeavor, we leverage the power of the Kalman Filter to track and predict the motion of a single object of interest within a video sequence. To kickstart the tracking process, the user is required to provide a bounding box encompassing the target object of interest in the video. Subsequently, a dedicated Kalman Filter instance is assigned to this object, enabling precise state estimation and motion prediction.

This system is characterized as a linear, discrete-time, vectorial, and time-variant model. During dynamic modeling, I assume that all target objects appearing in the video exhibit straight-line motion with uniform velocity, simplifying the tracking process while maintaining accuracy. The Kalman Filter operates based on the principle of recursive estimation, which involves predicting the state of the system at each time step using kinematic modeling and updating the state based on the observed measurements. In this context, the Kalman Filter serves as a reliable tool for state estimation and motion prediction, making it ideal for Single Object Tracking applications.

Within each video frame, I employ a robust object detector, such as the YOLO-Family, to obtain bounding box positions for the target object. Concurrently, the Kalman Filter's prediction step leverages the kinematic model to generate anticipated bounding box positions. By integrating the predicted and detected bounding box positions as observations, I achieve an accurate and robust tracking solution. 

This GitHub repository provides a comprehensive collection of code, implementation examples, and resources, enabling others to explore and apply the Kalman Filter in Single Object Tracking scenarios. 

## Requirements
- python
- openv-python
- numpy
- time

## Results
![image](https://github.com/1996JCZhou/Kalman-filter-in-SOT/blob/master/data/Result%20example%201.PNG)
![image](https://github.com/1996JCZhou/Kalman-filter-in-SOT/blob/master/data/Result%20example%202.PNG)
![image](https://github.com/1996JCZhou/Kalman-filter-in-SOT/blob/master/data/Result%20example%203.PNG)

Welcome to see my videos in my Youtube channel for this project.

https://www.youtube.com/watch?v=FsmJ24_TZ8U and

https://www.youtube.com/watch?v=x_soNtJBk7U and

https://www.youtube.com/watch?v=JQAR2lytYkA.
