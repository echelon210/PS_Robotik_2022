import time
import rclpy
from .packages.Def_Class_Controller import Controller


def main(args=None):
    rclpy.init(args=args)

    controller = Controller()
    # controller.openLoop()
    while True:
        controller.startController()

    #jetbot = Robot()

    #jetbot.controlVelocities_stepResponse(0., 1.0)


if __name__ == '__main__':
    main()
