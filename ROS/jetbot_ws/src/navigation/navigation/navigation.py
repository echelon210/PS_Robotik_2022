import numpy as np
import rclpy
from jetbot_ws.src.navigation.navigation.Primitives.Def_Class_ImageProcessor import ImageProcessor
from jetbot_ws.src.vision.vision.utils.argumentparser import gazebo


def main(args=None):
    rclpy.init(args=args)

    img_processor = ImageProcessor(gazebo)
    img_processor.initModel()

    while True:
        ang = img_processor.image2angle()
        img_processor.publishAngle(ang)
        print(np.rad2deg(ang))

    # rclpy.spin(img_processor)

    # img_processor.destroy_node()
    # rclpy.shutdown()

    return


if __name__ == '__main__':
    main()
