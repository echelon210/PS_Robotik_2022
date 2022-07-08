from sensor_msgs.msg import Image
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge 
from datetime import datetime
from rclpy.qos import QoSProfile


class CameraImage(object):
    def __init__(self, image_width, image_height, fps) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.fps = fps
        
        self.br = CvBridge()
        self.frame = None

    def ros_camera_pipline(self, msg):
        self.frame = self.br.imgmsg_to_cv2(msg)
        cv2.imshow("camera", self.frame)
        cv2.waitKey(1)

    def jetbot_camera_pipline(self, capture_width=1280, capture_height=720):
        return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            0,
            capture_width,
            capture_height,
            self.fps,
            0,
            self.image_width,
            self.image_height,
        )
    )

    def show_CSI_camera(self):
        window_title = "CSI Camera"
        video_capture = cv2.VideoCapture(self.jetbot_camera_pipline(), cv2.CAP_GSTREAMER)
        if video_capture.isOpened():
            try:
                window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
                while True:
                    _, self.frame = video_capture.read()
                    if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                        cv2.imshow(window_title, self.frame)
                    else:
                        break 
                    keyCode = cv2.waitKey(10) & 0xFF
                    if keyCode == 27 or keyCode == ord('q'):
                        break
            finally:
                video_capture.release()
                cv2.destroyAllWindows()
        else:
            print("Error: Unable to open camera")


    def save_csi_image(self, file_type='.bmp'):
        date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p_%f")
        filename = f"filename_{date}" + file_type

        print("Save image? (y/n): ")
        save_img = input()
        if save_img == "y" or save_img == "Y":
            cv2.imwrite(filename, self.frame)


class CameraConnector(Node, CameraImage):
    def __init__(self, gazebo, collect_data, image_width=640, image_height=480, fps=30, topic='camera/image_raw'):
        Node.__init__(self, node_name='image_subscriber', namespace='jetbot')
        CameraImage.__init__(self, image_width, image_height, fps)

        self.topic = topic
        self.gazebo = gazebo
        self.collect_data = collect_data

        self. subcriber = None
        self.video_capture = None
        self.msg = None
        
        if self.gazebo:
            self.subcriber = self.create_subscription(Image, self.topic, 
                                                      self.get_ros_camera_image,
                                                      QoSProfile(depth=10))

    def get_ros_camera_image(self, msg):
        # self.get_logger().info('Receiving video frame')
        self.ros_camera_pipline(msg)
