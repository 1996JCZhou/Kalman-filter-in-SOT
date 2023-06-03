import os, cv2
import numpy as np
from time import time
from utils import plot_one_box, cal_iou, updata_trace_list, draw_trace

"""Define the threshold for IoU."""
IOU_THRESHOLD = 0.1

"""Define the time intervall (ms) to display video frames."""
TIME_INTERVALL = 100

"""First pick a target to track.
   Then define the initial bounding box
   by this target's detected bounding box in the first video frame (with time step: 1).
   (using the bounding box's description using top left and bottom right pixel positions)"""
initial_target_box = [729, 238, 764, 339]
last_frame_box = initial_target_box


if __name__ == "__main__":

    """Save the edited video frames as a video file."""
    SAVE_VIDEO = True
    if SAVE_VIDEO:

        """Define the video encoder."""
        # Encoding video or audio into a specific file format
        # according to the encoding format requires an encoder 'fourcc'.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        """Make a video using edited video frames."""
        save_path = "D:\\Object Detection\\SOT\\Kalman filter in SOT\\data\\only_observations_maximal_iou_matching.avi"
        # FPS=20, Size=(Width of the image, Height of the image)=(768, 576).
        out = cv2.VideoWriter(save_path, fourcc, 20, (768, 576))
## ------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------
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
        """Try to find the detected bounding box for the target in the current frame
           by maximizing the IoU value."""
        """Draw the detected bounding box for the target (white) in the previous frame."""
        plot_one_box(last_frame_box, frame, color=(255, 255, 255), target=False)

        """Read the 'i-th' label file (bounding box positions from the detector)
           for the current 'i-th' video frame.

           In this label file, find
           (the detected bounding box with the highest IoU value
           for the current video frame with time step n+1)
           against
           (the detected bounding box for the previous video frame with time step n).
           And treat it as the target box.
        """
        label_file_path = os.path.join(label_path, file_name + "_" + str(frame_counter) + ".txt")
        with open(label_file_path, "r") as f:
            content = f.readlines()
            # ['0 567 165 597 241\n', '0   9 177  42 245\n', '0 729 238 764 339\n',
            #  '0 328 145 350 220\n', '0 193 342 250 474\n', '0 628 305 670 410\n',
            #  '0 160 414 223 566\n']

            max_iou = IOU_THRESHOLD
            max_iou_matched = False

            """Read each element (bounding box position) in the 'i-th' label file
               with the bounding box's description using top left and bottom right pixel positions."""
            for i, data_ in enumerate(content):
                data = data_.replace('\n', "").split(" ") # ['0', '567', '165', '597', '241'].
                xyxy = np.array(data[1:5], dtype="float") # array([567., 165., 597., 241.]).

                """Draw each detected bounding box (Green) on the current frame."""
                plot_one_box(xyxy, frame)

                """Calculate the IoU between each observed bounding box (Green) and
                   the current estimated bounding box (white)."""
                iou = cal_iou(xyxy, last_frame_box)

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
                trace_list = updata_trace_list(box_center, trace_list, 100)

                """Update the 'last_frame_box' using the found target box in the current video frame
                   as preparation for the next video frame."""
                last_frame_box = target_box
                cv2.putText(frame,                                        # Current video frame, where we put on the text.
                            "Tracking",                                   # Text.
                            (int(target_box[0]), int(target_box[1] - 5)), # Position of the text.
                            cv2.FONT_HERSHEY_SIMPLEX,                     # Type of the text.
                            0.7,                                          # Size of the text.
                            (0, 0, 255),                                  # Red text.
                            1)                                            # Thickness of the text.
            else:
                """If we have not found the target box from the current label file,
                   then we have lost the target in the current video frame."""
                cv2.putText(frame,                                                # Current video frame, where we put on the text.
                            "Lost",                                               # Text.
                            (int(last_frame_box[0]), int(last_frame_box[1] - 5)), # Position of the text.
                            cv2.FONT_HERSHEY_SIMPLEX,                             # Type of the text.
                            0.7,                                                  # Size of the text.
                            (255, 0, 0),                                          # Blue text.
                            2)                                                    # Thickness of the text.
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
