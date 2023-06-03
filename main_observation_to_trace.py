import os, cv2
import numpy as np
from time import time
from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace

"""Define the threshold for IoU."""
IOU_THRESHOLD = 0.3

"""Define the time intervall (ms) to display video frames."""
TIME_INTERVALL = 100

"""Save the edited video frames as a video file."""
SAVE_VIDEO = True
FPS_OUT = 10
save_path = "D:\\Object Detection\\SOT\\Kalman filter in SOT\\data\\main_observation_to_trace.avi"
## ------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------
"""Preparation for the Kalman Filter.
There are two reasons not to directly link the observations from frame to frame.
1. There will be no detected bounding box when the target is occluded.
2. The observations are still noisy even from good detecotr."""

"""Define the system model."""
# System matrix with 'delta_t' = 0.1 (ms).
DELTA_T = 0.1
A = np.array([[1, 0, 0, 0, DELTA_T, 0      ],
              [0, 1, 0, 0, 0,       DELTA_T],
              [0, 0, 1, 0, 0,       0      ],
              [0, 0, 0, 1, 0,       0      ],
              [0, 0, 0, 0, 1,       0      ],
              [0, 0, 0, 0, 0,       1      ]])

# Control matrix.
B = None

# Observation matrix.
# The observation array has the same size as the state array.
C = np.eye(6)

# System noise matrix.
# System noise here comes from uncertainties in target movement,
# e.g. sudden acceleration, deceleration, turns, etc.
L = np.eye(6)

"""Define the noise covariance matrix."""
# System noise covariance matrix.
Q = np.eye(6) * 0.1

# Measurement noise covariance matrix.
# Although the results from the detector are realible, they are still noisy.
R = np.eye(6)

"""First pick a target to track.
   Then define the initial bounding box
   by this target's detected bounding box in the first video frame (with time step: 1).
   (using the bounding box's description using top left and bottom right pixel positions)"""
# Pick a target according to the first video frame and its detected bounding box position.
initial_target_box = [729, 238, 764, 339]
# Transform the bounding box's description using top left and bottom right pixel positions
# into the bounding box's description using box center position, box width and box height. 
initial_box_state = xyxy_to_xywh(initial_target_box)

# Expected value vector of the A-posteriori-Density for the time step 0. (2D numpy array with shape (6, 1))
initial_state = np.array([[initial_box_state[0], # center pixel x coordinate (known: initial_box_state)
                           initial_box_state[1], # center pixel y coordinate (known: initial_box_state)
                           initial_box_state[2], # box width                 (known: initial_box_state)
                           initial_box_state[3], # box height                (known: initial_box_state)
                           0,                    # center pixel x velocity   (unknown, then asigned 0)
                           0]]).T                # center pixel y velocity   (unknown, then asigned 0)

# Covariance matrix of the A-posteriori-Density for the time step 0. (unknown, then asigned towards infinity)
P = np.eye(6) * 1000


if __name__ == "__main__":
    ## ------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------
    """Initialization for the Kalman Filter."""
    # 'X_posteriori': Expected value vector of the A-posteriori-Density for the time step 0.
    #                 known expected value vector of the initial state
    # 'P_posteriori': Covariance matrix of the A-posteriori-Density for the time step 0.
    #                 known covariance matrix of the initial state
    X_posteriori = initial_state
    P_posteriori = P

    """Initialize the observation array
       to ensure the same size as the state array."""
    Z = initial_state

    """Prepare for the tracking job."""
    trace_list = []
## ------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------
    """Define file pathes."""
    # A video clip from the PETS09-S2L1 dataset.
    video_path = "D:\\Object Detection\\SOT\\Kalman filter in SOT\\data\\testvideo1.mp4"
    assert os.path.exists(video_path), "Path for video does not exist."

    label_path = "D:\\Object Detection\\SOT\\Kalman filter in SOT\\data\\labels"
    assert os.path.exists(label_path), "Path for labels does not exist."

    file_name = "testvideo1"

    """Asign a label file (from the detector) for every video frame."""
    frame_counter = 1
## ------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------
    """Load video."""
    cap = cv2.VideoCapture(video_path)

    """Save the edited video frames as a video file."""
    if SAVE_VIDEO:

        """Define the video encoder."""
        # Encoding video or audio into a specific file format
        # according to the encoding format requires an encoder 'fourcc'.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        """Make a video using edited video frames."""
        out = cv2.VideoWriter(save_path, fourcc, FPS_OUT, sz, isColor=True)
## ------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------
    """Run every frame of the video
       until the video ends ('ret' = False) or it is manually stopped."""
    while True:

        """Extract the current video frame."""
        ret, frame = cap.read() # 'frame'.shape: (576, 768, 3).
        if not ret:
            break

        print("The current video frame is: {}.".format(frame_counter))
    ## ------------------------------------------------------------------------------------
    ## ------------------------------------------------------------------------------------
        """Begin to record calculation time."""
        t0 = time()
    ## ------------------------------------------------------------------------------------
    ## ------------------------------------------------------------------------------------
        """Try to find out the existence of the detected bounding box for the target
           by maximizing the IoU value in the current frame."""
        """Extract the box state
           from the expected value vector of the A-posteriori-Density / best estimation
           for the previous time step n / video frame.
           And draw it as a bounding box (white)."""
        last_box_posteriori = xywh_to_xyxy(X_posteriori[0:4])
        plot_one_box(last_box_posteriori, frame, color=(255, 255, 255), target=False)

        """Read the 'i-th' label file (bounding box positions from the detector)
           for the current 'i-th' video frame.

           In this label file, find
           (the detected bounding box with the highest IoU value
           for the current video frame with time step n+1)
           against
           (the extracted box state / bounding box
           from the expected value vector of the A-posteriori-Density
           for the previous time step n / video frame)
           And treat it as the target box.

           The bounding box positions from the detector are actually observations.
           Update the observation array for the current video frame (time step: n+1)
           according to the target box.
        """
        label_file_path = os.path.join(label_path, file_name + "_" + str(frame_counter) + ".txt")
        with open(label_file_path, "r") as f:
            content = f.readlines()
            # ['0 567 165 597 241\n', '0   9 177  42 245\n', '0 729 238 764 339\n',
            #  '0 328 145 350 220\n', '0 193 342 250 474\n', '0 628 305 670 410\n',
            #  '0 160 414 223 566\n']

            max_iou = IOU_THRESHOLD
            max_iou_matched = False

            """To find (the detected bounding box with the highest IoU value
               for the current video frame with time step n+1)
               against (the extracted box state / bounding box
               from the expected value vector of the A-posteriori-Density / best estimation
               for the previous time step n / video frame)
               among all the detected bounding box positions in the 'i-th' label file.

               maximal IoU matched and maximal IoU (most matched between two adjacent video frames)
               --->>> The found detected bounding box with the highest IoU value
                      in the current video frame with time step n+1
                      is actually the observation (except two velocity elements) bounding box
                      for the target in the time step n+1!!!
                      And it can be used in the Kalman Filter.
               --->>> Append the found detected bounding box /
                      the expected value vector of the A-posteriori-Density / best estimation
                      for the current time step n+1 / video frame (after the Kalman Filter) (Yes) 
                      into the 'trace_list'.

               no maximal IoU matched
               --->>> We lost the observation (except two velocity elements) for the target in the time step n+1!!!
                      And no observation can be used in the Kalman Filter.
            """
            """Read each element (bounding box position) in the 'i-th' label file
               with the bounding box's description using top left and bottom right pixel positions."""
            for i, data_ in enumerate(content):
                data = data_.replace('\n', "").split(" ") # ['0', '567', '165', '597', '241'].
                xyxy = np.array(data[1:5], dtype="float") # array([567., 165., 597., 241.]).

                """Draw each observed bounding box (Green) from the detector on the current frame."""
                plot_one_box(xyxy, frame)

                """Calculate the IoU between each observed bounding box (Green) and
                   the current estimated bounding box (white)."""
                iou = cal_iou(xyxy, xywh_to_xyxy(X_posteriori[0:4]))

                """Iterating all the elements (bounding box positions) in the 'i-th' label file and
                   find the bounding box position with the highest IoU value as the target box."""
                if iou > max_iou:
                    target_box = xyxy      # The current observed bounding box (Green) from the detector will be seen as the target box.
                    max_iou = iou          # To find the maximal IoU.
                    max_iou_matched = True # At least one observed bounding box (Green) from the detector can be used as the target box.

        """If we have found the observed bounding box position with the highest IoU value,
           then we treat it as the target box."""
        if max_iou_matched == True:

            """Draw a red bounding box on the current video frame for the found target box."""
            plot_one_box(target_box, frame, target=True)

            """Update the trace list for the center of the found target box."""
            box_center = (int((target_box[0] + target_box[2]) // 2), \
                          int((target_box[1] + target_box[3]) // 2))
            trace_list = updata_trace_list(box_center, trace_list, 20)
            cv2.putText(frame,                                        # Current video frame, where we put on the text.
                        "Tracking",                                   # Text.
                        (int(target_box[0]), int(target_box[1] - 5)), # Position of the text.
                        cv2.FONT_HERSHEY_SIMPLEX,                     # Type of the text.
                        0.7,                                          # Size of the text.
                        (0, 0, 255),                                  # Red text.
                        2)                                            # Thickness of the text.

            """Calculate the center pixel velocity using the current found target box and
               (the expected value vector of the A-posteriori-Density / best estimation
               for the previous time step n / video frame)."""
            xywh = xyxy_to_xywh(target_box)
            dx = (1 / DELTA_T) * (xywh[0] - X_posteriori[0]) # Center pixel x velocity with time intervall equals to 1 (s).
            dy = (1 / DELTA_T) * (xywh[1] - X_posteriori[1]) # Center pixel y velocity with time intervall equals to 1 (s).

            """Complete the observation array for the current video frame (time step: n+1)
               according to the target box along with two velocity elements."""
            Z[0:4] = np.array([xywh]).T
            Z[4: ] = np.array([[dx, dy]]).T

            """Update the expected value vector of the A-posteriori-Density and
               the covariance matrix of the A-posteriori-Density
               for the current video frame (time step n+1)
               as preparation for the next video frame."""
            """If we have found the target box and updated the observation array,
               then we apply the Kalman Filter."""

            """Prediction step."""
            # 'X_posteriori': Expected value vector of the A-posteriori-density for the previous video frame (time step n).
            # 'P_posteriori': Covariance matrix of the A-posteriori-density for the previous video frame (time step n).
            # 'X_prior':      Expected value vector of the A-prior-density for the current video frame (time step n+1).
            # 'P_prior':      Covariance matrix of the A-prior-density for the current video frame (time step n+1).
            X_prior = np.dot(A, X_posteriori)
            P_prior = np.dot(np.dot(A, P_posteriori), A.T) + np.dot(np.dot(L, Q), L.T)

            """Filter step."""
            k1 = np.dot(P_prior, C.T)
            k2 = np.dot(np.dot(C, P_prior), C.T) + R
            K = np.dot(k1, np.linalg.inv(k2))

            # 'X_prior':      Expected value vector of the A-prior-density for the current video frame (time step n+1).
            # 'P_prior':      Covariance matrix of the A-prior-density for the current video frame (time step n+1).
            # 'X_posteriori': Expected value vector of the A-posteriori-density for the current video frame (time step n+1).
            # 'P_posteriori': Covariance matrix of the A-posteriori-density for the current video frame (time step n+1).
            X_posteriori = X_prior + np.dot(K, (Z - np.dot(C, X_prior)))
            P_posteriori = np.dot((np.eye(6) - np.dot(K, C)), P_prior)

        else:
            """If we have not found the target box from the current label file,
               then we have lost the observation (except two velocity elements)
               / the observed bounding box for the target
               in the current video frame with time step n+1!!!
               Since we have no observation, we then give up the Kalman filter ('K'=0)
               and use only the system model for prediction."""
            # 'X_posteriori': Expected value vector of the A-posteriori-density for the previous video frame (time step n+1).
            X_posteriori = np.dot(A, X_posteriori)
            # 'P_posteriori': Covariance matrix of the A-posteriori-density for the previous video frame (time step n+1).
            # P_posteriori = P_posteriori

            """Update the trace list for the expected value vector of the A-posteriori-density
               for the previous video frame (time step n+1) because no found target box exists."""
            box_posteriori = xywh_to_xyxy(X_posteriori[0:4])
            box_center = (int((box_posteriori[0] + box_posteriori[2]) // 2), \
                          int((box_posteriori[1] + box_posteriori[3]) // 2))
            trace_list = updata_trace_list(box_center, trace_list, 20)
            cv2.putText(frame,                              # Current video frame, where we put on the text.
                        "Lost",                             # Text.
                        (box_center[0], box_center[1] - 5), # Position of the text.
                        cv2.FONT_HERSHEY_SIMPLEX,           # Type of the text.
                        0.7,                                # Size of the text.
                        (255, 0, 0),                        # Blue text.
                        2)                                  # Thickness of the text.
    ## ------------------------------------------------------------------------------------
    ## ------------------------------------------------------------------------------------
        """Prepare the next label file for the next video frame."""
        frame_counter = frame_counter + 1
    ## ------------------------------------------------------------------------------------
    ## ------------------------------------------------------------------------------------
        """Draw a trace on the current video frame
           according to the updated 'trace_list'."""
        draw_trace(frame, trace_list)

        """Display title."""
        cv2.putText(frame, "ALL BOXES(Green)",                  (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, "TRACKED BOX(Red)",                  (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, "Last frame best estimation(White)", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        """Display the current video frame and all the drawings on it."""
        # Refresh the window "track" with the current video frame
        # and all the drawing on the current video frame,
        # neglecting all the drawing on the previous video frames.
        # (different from the static image).
        cv2.imshow("track", frame)
    ## ------------------------------------------------------------------------------------
    ## ------------------------------------------------------------------------------------
        """Save the current edited video frame."""
        if SAVE_VIDEO:
            out.write(frame)
    ## ------------------------------------------------------------------------------------
    ## ------------------------------------------------------------------------------------
        """Begin to record calculation time."""
        t1 = time()
        print(f"Duration for calculation is: {t1 - t0}.")
    ## ------------------------------------------------------------------------------------
    ## ------------------------------------------------------------------------------------
        """Press the 'Esc' or the 'q' key to immediately exit the program."""
        # The ASCII code value corresponding to the same key on the keyboard in different situations
        # (e.q. when the key "NumLock" is activated)
        # is not necessarily the same, and does not necessarily have only 8 bits, but the last 8 bits must be the same.
        # In order to avoid this situation, quote &0xff, to get the last 8 bits of the ASCII value of the pressed key
        # to determine what the key is.
        c = cv2.waitKey(TIME_INTERVALL) & 0xFF # Wait for 10 ms.
        if c == 27 or c == ord('q'):
            break
## ------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------
    cap.release() # Release video file.
    cv2.destroyAllWindows()
