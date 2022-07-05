import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from jetbot import *

from camera_pipeline import *

# CREATE DASHBOARD
cv2.namedWindow("Jetbot Dashboard")
cv2.createTrackbar('speed', 'Jetbot Dashboard', 0, 100, lambda x: 0)
cv2.createTrackbar('steerProp', 'Jetbot Dashboard', 10, 100, lambda x: 0)
cv2.createTrackbar('steerDer', 'Jetbot Dashboard', 20, 100, lambda x: 0)

# LOAD MODELS
modelSignDetection = torchvision.models.squeezenet1_1(pretrained=False)
modelSignDetection.classifier[1] = torch.nn.Conv2d(512, 7, kernel_size=1)
modelSignDetection.num_classes = 7
modelSignDetection.load_state_dict(torch.load('signDetection_V2_squeeze_conv.pth'))

modelRoadFollowing = torchvision.models.squeezenet1_1(pretrained=False)
modelRoadFollowing.classifier[1] = torch.nn.Conv2d(512, 1, kernel_size=1)
modelRoadFollowing.num_classes = 1
modelRoadFollowing.load_state_dict(torch.load('roadFollowing_V3_squeeze_conv.pth'))

device = torch.device('cuda')

modelSignDetection = modelSignDetection.to(device)
modelSignDetection = modelSignDetection.eval().half()

modelRoadFollowing = modelRoadFollowing.to(device)
modelRoadFollowing = modelRoadFollowing.eval().half()

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()


def preprocess_sign(image):
    image[0:224, int(224 * 0): int(224 * 2 / 4)] = (0, 0, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def preprocess_road(image):
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def calculate_controller(angle, angle_last):
    speed = cv2.getTrackbarPos('speed', 'Jetbot Dashboard') * 0.01
    steer_prop_gain = cv2.getTrackbarPos('steerProp', 'Jetbot Dashboard') * 0.01
    steer_der_gain = cv2.getTrackbarPos('steerDer', 'Jetbot Dashboard') * 0.01

    pid = angle * steer_prop_gain + (angle - angle_last) * steer_der_gain
    angle_last = angle

    motor_left = max(min(speed + pid, 1.0), 0.0)
    motor_right = max(min(speed - pid, 1.0), 0.0)

    return motor_left, motor_right, angle, speed


import cv2, queue, threading, time


# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


def drive(robot):
    # Open video stream
    video_capture = VideoCapture(jetbot_camera_pipline(), cv2.CAP_GSTREAMER)

    angle = 0.0
    angle_last = 0.0

    if video_capture.isOpened():
        try:
            while True:
                _, image = video_capture.read()

                # SIGN DETECTION
                signs = modelSignDetection(preprocess_sign(image)).detach().float().cpu()

                # Read out probabilities
                signs = F.softmax(signs, dim=1)
                prob_fast = float(signs.flatten()[1])
                prob_left = float(signs.flatten()[2])
                prob_right = float(signs.flatten()[4])
                prob_slow = float(signs.flatten()[5])
                prob_stop = float(signs.flatten()[6])
                prob_nothing = float(signs.flatten()[3])
                prob_attention = float(signs.flatten()[0])

                if prob_stop >= 0.95:
                    print("Found STOP Sign!")
                    time.sleep(0.5)
                    robot.left_motor.value = 0.0
                    robot.right_motor.value = 0.0
                    time.sleep(2)
                    continue

                if prob_right >= 0.95:
                    print("Found RIGHT Sign!")
                    time.sleep(2)
                    robot.left_motor.value = 0.1
                    robot.right_motor.value = -0.1
                    time.sleep(2)
                    continue

                if prob_left >= 0.95:
                    print("Found RIGHT Sign!")
                    time.sleep(2)
                    robot.left_motor.value = -0.1
                    robot.right_motor.value = 0.1
                    time.sleep(2)
                    continue

                if prob_fast >= 0.95:
                    print("Found FAST Sign!")

                if prob_slow >= 0.95:
                    print("Found 30 Sign!")

                if prob_attention >= 0.95:
                    print("Found ATTENTION Sign!")

                # Calculate steering angle
                angle = modelRoadFollowing(preprocess_road(image)).detach().float().cpu().numpy().flatten()
                angle = np.deg2rad(angle - 90)

                # Calculate controller values and set motor values
                motor_left, motor_right, ang, speed = calculate_controller(angle, angle_last)
                angle_last = ang
                robot.left_motor.value = float(motor_left)
                robot.right_motor.value = float(motor_right)

                # Display angle in video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(image, "Angle: " + str(angle), (25, 25), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.imshow("Jetbot Dashboard", image)

                # Press q to stop
                key_code = cv2.waitKey(1) & 0xFF
                if key_code == 27 or key_code == ord('q'):
                    break

        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


robot = Robot()
drive(robot)
