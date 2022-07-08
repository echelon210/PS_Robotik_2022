import os
import rclpy
from rclpy.node import Node

from std_msgs.msg import String



class_labels = {
    0: '30 kmh',
    1: 'vorsicht',
    2: 'links',
    3: 'rechts',
    4: 'stop',
    5: 'aufhebung',
    6: 'nichts'
}

class_ids = {
    '30 kmh': 0,
    'vorsicht': 1,
    'links': 2,
    'rechts': 3,
    'stop': 4,
    'aufhebung': 5,
    'nichts': 6
}

def up_direction(path):
    return os.path.dirname(path)


def get_full_path(relative_path):
    current_file_path = os.path.abspath(__file__)

    for _ in range(9):
        # current path goes the the top of the project
        current_file_path = up_direction(current_file_path)
    
    if relative_path[0] == "/":
        return current_file_path + relative_path
    else:
        return current_file_path + "/" + relative_path

class TrafficSignPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(dict, 'topic', 10)

    def publish(self, msg):
        self.publisher.publish(msg)

