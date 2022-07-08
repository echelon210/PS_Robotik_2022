#!/usr/bin/env python
import random
import threading
from .utils.parser import settings
import rclpy
from .Primitives.camera import CameraConnector
from .Primitives.ImageProcessing import ImageSeperator
from .Primitives.TrafficSignDetector import TrafficSignDetector
import time
import numpy as np
from .utils.utils import get_full_path

parser = settings(option=4)

def destroy(node):
    node.destroy_node()

def run(camera):
    model_path = get_full_path("jetbot_ws/src/vision/vision/utils/model/TrafficSignModel")

    if not parser.collect_data: 
        sep = ImageSeperator(buffer_size=0, visualization=parser.visualization, optimization=parser.optimization)
        detector = TrafficSignDetector(path=model_path, usage=parser.usage)
    else:
        sep = ImageSeperator()
    
    while(True):
        if parser.gazebo and parser.collect_data:
            sep.set_current_frame(np.asarray(camera.frame))

            if parser.empty:
                for i in range(2):
                    sep.data_collector_seperator([random.randint(0, 3), random.randint(0, 15)])
                    sep.save_seperated_image()

                print("New collection in 2 seconds")
                time.sleep(1)

            else:
                print("Seperate image? (y/n): ")
                save_img = input()
                
                if save_img == "y" or save_img == "Y":
                    sep.show_seperation()
                    sep.save_image()

                    print("Enter two numbers: ")
                    x, y = input().split()

                    sep.data_collector_seperator([int(x), int(y)])
                    sep.save_seperated_image()
            
                time.sleep(1)

        elif parser.gazebo and not parser.collect_data:
            sep.set_current_frame(np.asarray(camera.frame))
            sep.seperator()

            y_labels = detector.evaluation(sep)
            traffic_signs = sep.prioritize_segments(y_labels)
            
            if parser.visualization:
                print("label: {}".format(y_labels))    
                print("Traffic Sign: ", traffic_signs)

                _ = sep.result_coloring(y_labels)
        
        if not parser.gazebo and parser.collect_data:
            # collect data in on jetbot waveshare robot
            sep.set_current_frame(np.asarray(camera.frame))

            if parser.empty:
                for i in range(2):
                    # random.randint(0, 3), random.randint(0, 15)
                    # sep.data_collector_seperator([2])
                    sep.data_collector_seperator([random.randint(0, 3), random.randint(0, 15)])
                    sep.save_seperated_image()

                print("new collection in 2 seconds")
                time.sleep(2)

            else:
                print("Seperate image? (y/n): ")
                save_img = input()
                sep.set_current_frame(np.asarray(camera.frame))
                if save_img == "y" or save_img == "Y":
                    sep.show_seperation()
                    sep.save_image()

                    print("Enter two numbers: ")
                    x, y = input().split()

                    sep.data_collector_seperator([int(x), int(y)])
                    sep.save_seperated_image()
            time.sleep(1)

        elif not parser.gazebo and not parser.collect_data:
            # running code in real world
    
            sep.set_current_frame(np.asarray(camera.frame))
            sep.seperator()
            
            #start = time.time()
            y_labels = detector.evaluation(sep)
            #stop = time.time()
            #print("label: {}, time: {}".format(y_labels, stop - start))
            
            traffic_signs = sep.prioritize_segments(y_labels)
            if not parser.optimization:
                print("label: {}".format(y_labels))    
                print("Traffic Sign: ", traffic_signs)
            
            if parser.visualization:
                _ = sep.result_coloring(y_labels)

            time.sleep(5)
            print("\n\n")

def main(args=None):
    rclpy.init(args=args)

    # Create the node
    camera = CameraConnector(parser.gazebo, parser.collect_data)
    if parser.gazebo:
        t = threading.Thread(target=rclpy.spin, args=(camera,))
    else:
        t = threading.Thread(target=camera.show_CSI_camera)
        
    t.start()
    time.sleep(3)
    run(camera)

if __name__ == '__main__':
    main()
