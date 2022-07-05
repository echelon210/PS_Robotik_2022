import numpy as np
import cv2
import torchvision.transforms as transforms
import torchvision
import torch
import time
from camera_pipeline import *
from jetbot import *


class Drive:
    def __init__(self, display_dashboard=True):
        self.display_dashboard = display_dashboard
        if display_dashboard:
            cv2.namedWindow("Jetbot Dashboard")
            cv2.createTrackbar('speed', 'Jetbot Dashboard', 0, 100, lambda x: 0)
            cv2.createTrackbar('steerProp', 'Jetbot Dashboard', 10, 100, lambda x: 0)
            cv2.createTrackbar('steerDer', 'Jetbot Dashboard', 20, 100, lambda x: 0)
            cv2.createTrackbar('speedVar', 'Jetbot Dashboard', 0, 100, lambda x: 0)


# Initialize CNN:
model = torchvision.models.squeezenet1_1(pretrained=False)
model.classifier[1] = torch.nn.Conv2d(512, 1, kernel_size=1)
model.num_classes = 1

model.load_state_dict(torch.load('roadFollowing_V3_squeeze_conv.pth'))

device = torch.device('cuda')
model = model.to(device)
model = model.eval().half()

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()


def preprocess(image):
    """ Preprocess the input image.
        In(1): image - The image to preprocess
    """
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def nothing(x):
    pass


angle_history = []


def calculateController(angle, angle_last):
    global angle_history
    speed = cv2.getTrackbarPos('speed', 'Jetbot Dashboard') * 0.01
    steerPropGain = cv2.getTrackbarPos('steerProp', 'Jetbot Dashboard') * 0.01
    steerDerGain = cv2.getTrackbarPos('steerDer', 'Jetbot Dashboard') * 0.01
    speedVarGain = cv2.getTrackbarPos('speedVar', 'Jetbot Dashboard') * 0.01

    angle_history.append(angle)
    if len(angle_history) >= 10:
        angle_history.pop(0)

    pid = angle * steerPropGain + (angle - angle_last) * steerDerGain
    angle_last = angle
    brake = speedVarGain * np.mean(np.abs(angle_history))

    motorLeft = max(min(speed - brake + pid, 1.0), 0.0)
    motorRight = max(min(speed - brake - pid, 1.0), 0.0)

    return motorLeft, motorRight, angle, speed


def followPath(robot):
    # Open video stream
    video_capture = cv2.VideoCapture(jetbot_camera_pipline(), cv2.CAP_GSTREAMER)

    angle = 0.0
    angle_last = 0.0

    counter = 0

    if video_capture.isOpened():
        try:
            while True:
                ret, image = video_capture.read()
                counter += 1
                if counter == 300:
                    robot.left_motor.value = 0.0
                    robot.right_motor.value = 0.0
                    time.sleep(5)
                    robot.left_motor.value = 0.1
                    robot.right_motor.value = 0.1
                    time.sleep(1)
                    robot.left_motor.value = 0.0
                    robot.right_motor.value = 0.0
                    time.sleep(5)
                    robot.left_motor.value = 0.05
                    robot.right_motor.value = 0.05

                # Calculate steering angle
                angle = model(preprocess(image)).detach().float().cpu().numpy().flatten()
                angle = np.deg2rad(angle - 90)

                # Display angle in video frame
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # fontScale = 1
                # color = (255, 0, 0)
                # thickness = 2
                # cv2.putText(image, "Angle: " + str(angle), (25, 25), font, fontScale, color, thickness, cv2.LINE_AA)
                # cv2.imshow("Jetbot Dashboard", image)

                # Calculate controller values and set motor values
                motorLeft, motorRight, ang, speed = calculateController(angle, angle_last)
                angle_last = ang
                robot.left_motor.value = float(motorLeft)
                robot.right_motor.value = float(motorRight)

                # Press q to stop
                keyCode = cv2.waitKey(1) & 0xFF
                if keyCode == 27 or keyCode == ord('q'):
                    break

        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


robot = Robot()
followPath(robot)