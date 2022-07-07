from uuid import uuid1
import os
import numpy as np
import cv2
from visionForDataCollection import *
from Utilities import *

# FUNCITONALITIES TO SHOW CAMERA, APPLY CENTERLINE FINDING
# ALGORITHMS AND COLLECT DATA FOR CNN TRAINING.

DATASET_DIR = 'dataset_angle_race'
#try:
#    os.makedirs(DATASET_DIR)
#except Exception:
#    print('Directories not created because they already exist')


def saveSnapshot(angle, image):
    """ Method to save a annotated snapshot of the camera image. 
        In(1): angle - The angle to annotate the image with
        In(2): image - The image to save
        Out(1): isWritten - Boolean indicating whether the saving was successful
    """
    image = np.copy(image)
    uuid = 'angle_%03f_%s' % (angle, uuid1())
    image_path = os.path.join(DATASET_DIR, uuid + '.jpg')
    # Save picture and value to disk
    isWritten = cv2.imwrite(image_path, image)

    image = np.copy(image)
    uuid = 'angle_%03d_%s' % (angle, uuid1())
    image_path = os.path.join('dataset_old_format', uuid + '.jpg')
    # Save picture and value to disk
    isWritten = cv2.imwrite(image_path, image)

    return isWritten


def displayDataCollectionCamera():
    """ Method to display the camera from the jetbot camera pipeline and apply
        the centerline finding algorithms for calculating the angle. Pressing
        's' saves the annotated image into the dataset directory.
    """
    video_capture = cv2.VideoCapture(
        jetbot_camera_pipline(), cv2.CAP_GSTREAMER)
    count = 0
    if video_capture.isOpened():
        try:
            while True:
                _, frame = video_capture.read()
                _, _, angle = findCenterline(frame, show=True)

                keyCode = cv2.waitKey(1) & 0xFF
                if keyCode == 27 or keyCode == 115:  # Press 's' key to save picture
                    count += 1
                    print("Image saved! Number of images in session: " + str(count))
                    saveSnapshot(angle, frame)
                if keyCode == 27 or keyCode == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


displayDataCollectionCamera()
