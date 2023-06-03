import cv2


def plot_one_box(xyxy, img, color=(0, 200, 0), target=False):
    """
    Plot a bounding box.

    Args:
        xyxy   (List / 1D numpy array): [left top x coordinate, left top y coordinate, right bottom x coordinate, right bottom y coordinate].
        img    (Numpy array):           The current video frame.
        color  (Tuple):                 Color of the bounding box. Defaults to color green with (0, 200, 0).
        target (Boolean):               Defaults to False.
    """

    # xy1 (Tuple): Top left pixel position of the bounding box.
    xy1 = (int(xyxy[0]), int(xyxy[1]))
    # xy2 (Tuple): Bottom right pixel position of the bounding box.
    xy2 = (int(xyxy[2]), int(xyxy[3]))

    # If the target box is found, then draw a red bounding box.
    if target:
        color = (0, 0, 255)

    # 'cv2.LINE_AA': Anti-aliasing type, which shows smoother lines and is more commonly used.
    cv2.rectangle(img, xy1, xy2, color, 1, cv2.LINE_AA)


def updata_trace_list(box_center, trace_list, max_list_len=50):
    """
    Update the trace list for the found target box center / the a-posterior box center for the current video frame.

    Args:
        box_center   (Tuple of two integers): Center of the found target box / the a-posterior box.
        trace_list   (List):                  List for the found target box centers for the previous video frames.
        max_list_len (Integer):               Maximal list length.

    Returns:
        trace_list (List): Trace list for the found target box center / the a-posterior box center for the current video frame.
    """

    if len(trace_list) <= max_list_len:
        trace_list.append(box_center)
    else:
        trace_list.pop(0) # Pop the first 'box_center' from the 'trace_list'.
        trace_list.append(box_center)

    return trace_list


def draw_trace(img, trace_list):
    """
    Draw a trace on the current video frame.

    Args:
        img (Numpy array): Current video frame.
        trace_list (List): Updated trace list for the found target box center / the a-posterior box center for the current video frame.  
    """

    for i, item in enumerate(trace_list):
        if i < 1:
            continue
        cv2.line(img,
                 (trace_list[i][0], trace_list[i][1]),
                 (trace_list[i-1][0], trace_list[i-1][1]),
                 (255, 255, 0),
                 3)


def cal_iou(box1, box2):
    """Calculate the IoU.

    Args:
        box1 (List): [left top x coordinate, left top y coordinate, right bottom x coordinate, right bottom y coordinate].
        box2 (List): [left top x coordinate, left top y coordinate, right bottom x coordinate, right bottom y coordinate].

    Returns:
        IoU (Float): Intersection over union.
    """

    # The x coordinate of the left top pixel point of the right bounding box.
    xA = max(box1[0], box2[0])
    # The x coordinate of the right bottom pixel point of the left bounding box.
    xB = min(box1[2], box2[2])
    # The y coordinate of the left top pixel point of the bottom bounding box.
    yA = max(box1[1], box2[1])
    # The y coordinate of the right bottom pixel point of the top bounding box.
    yB = min(box1[3], box2[3])

    intersection_x = xB - xA if xB - xA >= 1 else 0
    intersection_y = yB - yA if yB - yA >= 1 else 0
    intersection_area = intersection_x * intersection_y

    box1_Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    IoU = intersection_area / float(box1_Area + box2_Area - intersection_area)

    return IoU


def cal_distance(box1, box2):
    """
    Calculate the euclidean distance between centers of two bounding boxes.

    Args:
        box1 (List): [left top x coordinate, left top y coordinate, right bottom x coordinate, right bottom y coordinate].
        box2 (List): [left top x coordinate, left top y coordinate, right bottom x coordinate, right bottom y coordinate].

    Returns:
        dis (Float): The distance between centers of two bounding boxes.
    """

    center1 = ((box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0)
    center2 = ((box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0)
    dis = ((center1[0] - center2[0]) ** 2.0 + (center1[1] - center2[1]) ** 2.0) ** 0.5

    return dis


def xywh_to_xyxy(xywh):
    """
    Transform a bounding box's description using using box center position, width and height
    into the bounding box's description using top left and bottom right pixel positions.

    Args:
        xywh (Tuple): (center x coordinate, center y coordinate, width, height).

    Returns:     
        xyxy (List): [left top x coordinate, left top y coordinate, right bottom x coordinate, right bottom y coordinate].
    """
    x1 = xywh[0] - xywh[2] / 2.0
    y1 = xywh[1] - xywh[3] / 2.0
    x2 = xywh[0] + xywh[2] / 2.0
    y2 = xywh[1] + xywh[3] / 2.0

    return [x1, y1, x2, y2]


def xyxy_to_xywh(xyxy):
    """
    Transform a bounding box's description using top left and bottom right pixel positions
    into the bounding box's description using box center position, width and height.

    Args:
        xyxy (List): [left top x coordinate, left top y coordinate, right bottom x coordinate, right bottom y coordinate].

    Returns:
        xywh (Tuple): [center x coordinate, center y coordinate, width, height].
    """
    center_x = (xyxy[0] + xyxy[2]) / 2.0
    center_y = (xyxy[1] + xyxy[3]) / 2.0
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]

    return (center_x, center_y, w, h)

