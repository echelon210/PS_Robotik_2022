import argparse
import os


parser = argparse.ArgumentParser('Traffic Sign Recognition Mode')
gazebo=False
collect_data=True
empty=True
show_sep=False
usage='cpu'
visualization=True
optimization = True


def settings(option):
    # General
    if option == 0:
        print("collect data in gazebo")

        parser.add_argument('--gazebo', type=bool, default=True, help="Run Project in Simulation.")
        parser.add_argument('--collect_data', type=bool, default=True, help="Collect data.")
        parser.add_argument('--show_sep', type=bool, default=False, help="Show Camera image with seperation.")
        parser.add_argument('--visualization', type=bool, default=True, help="visualize Camera with traffic sign recognition.")
        parser.add_argument('--optimization', type=bool, default=True, help="Optimize code. Turn everythin off that is not necesarry.")
        parser.add_argument('--usage', type=str, default="cpu", help="Run Traffic sign recognition on cpu/gpu.")
        parser.add_argument('--empty', type=bool, default=False, help="Collect empty data for traffic sign detection.")
    
    elif option == 2:
        print("collect empty data in gazebo simulation")

        parser.add_argument('--gazebo', type=bool, default=True, help="Run Project in Simulation.")
        parser.add_argument('--collect_data', type=bool, default=True, help="Collect data.")
        parser.add_argument('--show_sep', type=bool, default=False, help="Show Camera image with seperation.")
        parser.add_argument('--visualization', type=bool, default=True, help="visualize Camera with traffic sign recognition.")
        parser.add_argument('--optimization', type=bool, default=False, help="Optimize code. Turn everythin off that is not necesarry.")
        parser.add_argument('--usage', type=str, default="cpu", help="Run Traffic sign recognition on cpu/gpu.")
        parser.add_argument('--empty', type=bool, default=True, help="Collect empty data for traffic sign detection.")

    elif option == 3:
        print("run code in gazebo simulation")

        parser.add_argument('--gazebo', type=bool, default=True, help="Run Project in Simulation.")
        parser.add_argument('--collect_data', type=bool, default=False, help="Collect data.")
        parser.add_argument('--show_sep', type=bool, default=False, help="Show Camera image with seperation.")
        parser.add_argument('--visualization', type=bool, default=True, help="visualize Camera with traffic sign recognition.")
        parser.add_argument('--optimization', type=bool, default=False, help="Optimize code. Turn everythin off that is not necesarry.")
        parser.add_argument('--usage', type=str, default="cpu", help="Run Traffic sign recognition on cpu/gpu.")
        parser.add_argument('--empty', type=bool, default=False, help="Collect empty data for traffic sign detection.")

    elif option == 4:
        print("run optimize code in gazebo simulation")

        parser.add_argument('--gazebo', type=bool, default=True, help="Run Project in Simulation.")
        parser.add_argument('--collect_data', type=bool, default=False, help="Collect data.")
        parser.add_argument('--show_sep', type=bool, default=False, help="Show Camera image with seperation.")
        parser.add_argument('--visualization', type=bool, default=True, help="visualize Camera with traffic sign recognition.")
        parser.add_argument('--optimization', type=bool, default=True, help="Optimize code. Turn everythin off that is not necesarry.")
        parser.add_argument('--usage', type=str, default="cpu", help="Run Traffic sign recognition on cpu/gpu.")
        parser.add_argument('--empty', type=bool, default=False, help="Collect empty data for traffic sign detection.")

    elif option == 5:
        print("run turn everything of for running code on jetbot")

        parser.add_argument('--gazebo', type=bool, default=False, help="Run Project in Simulation.")
        parser.add_argument('--collect_data', type=bool, default=False, help="Collect data.")
        parser.add_argument('--show_sep', type=bool, default=False, help="Show Camera image with seperation.")
        parser.add_argument('--visualization', type=bool, default=False, help="visualize Camera with traffic sign recognition.")
        parser.add_argument('--optimization', type=bool, default=True, help="Optimize code. Turn everythin off that is not necesarry.")
        parser.add_argument('--usage', type=str, default="gpu", help="Run Traffic sign recognition on cpu/gpu.")
        parser.add_argument('--empty', type=bool, default=False, help="Collect empty dat for traffic sign detection.")

    elif option == 6:
        print("collect data on jetbot")

        parser.add_argument('--gazebo', type=bool, default=False, help="Run Project in Simulation.")
        parser.add_argument('--collect_data', type=bool, default=True, help="Collect data.")
        parser.add_argument('--show_sep', type=bool, default=False, help="Show Camera image with seperation.")
        parser.add_argument('--visualization', type=bool, default=True, help="visualize Camera with traffic sign recognition.")
        parser.add_argument('--optimization', type=bool, default=False, help="Optimize code. Turn everythin off that is not necesarry.")
        parser.add_argument('--usage', type=str, default="gpu", help="Run Traffic sign recognition on cpu/gpu.")
        parser.add_argument('--empty', type=bool, default=False, help="Collect empty data for traffic sign detection.")

    elif option == 7:
        print("run code on jetbot / show seperation")

        parser.add_argument('--gazebo', type=bool, default=False, help="Run Project in Simulation.")
        parser.add_argument('--collect_data', type=bool, default=False, help="Collect data.")
        parser.add_argument('--show_sep', type=bool, default=True, help="Show Camera image with seperation.")
        parser.add_argument('--visualization', type=bool, default=True, help="visualize Camera with traffic sign recognition.")
        parser.add_argument('--optimization', type=bool, default=False, help="Optimize code. Turn everythin off that is not necesarry.")
        parser.add_argument('--usage', type=str, default="gpu", help="Run Traffic sign recognition on cpu/gpu.")
        parser.add_argument('--empty', type=bool, default=False, help="Collect empty dat for traffic sign detection.")

    elif option == 8:
        print("run code in gazebo simulation")

        parser.add_argument('--gazebo', type=bool, default=False, help="Run Project in Simulation.")
        parser.add_argument('--collect_data', type=bool, default=False, help="Collect data.")
        parser.add_argument('--show_sep', type=bool, default=False, help="Show Camera image with seperation.")
        parser.add_argument('--visualization', type=bool, default=True, help="visualize Camera with traffic sign recognition.")
        parser.add_argument('--optimization', type=bool, default=False, help="Optimize code. Turn everythin off that is not necesarry.")
        parser.add_argument('--usage', type=str, default="gpu", help="Run Traffic sign recognition on cpu/gpu.")
        parser.add_argument('--empty', type=bool, default=False, help="Collect empty dat for traffic sign detection.")
    
    elif option == 9:
        print("run visualize optimized code on jetbot")

        parser.add_argument('--gazebo', type=bool, default=False, help="Run Project in Simulation.")
        parser.add_argument('--collect_data', type=bool, default=False, help="Collect data.")
        parser.add_argument('--show_sep', type=bool, default=False, help="Show Camera image with seperation.")
        parser.add_argument('--visualization', type=bool, default=True, help="visualize Camera with traffic sign recognition.")
        parser.add_argument('--optimization', type=bool, default=True, help="Optimize code. Turn everythin off that is not necesarry.")
        parser.add_argument('--usage', type=str, default="gpu", help="Run Traffic sign recognition on cpu/gpu.")
        parser.add_argument('--empty', type=bool, default=False, help="Collect empty dat for traffic sign detection.")

    elif option == 10:
        print("run turn everything of for running optimized code on jetbot")

        parser.add_argument('--gazebo', type=bool, default=False, help="Run Project in Simulation.")
        parser.add_argument('--collect_data', type=bool, default=False, help="Collect data.")
        parser.add_argument('--show_sep', type=bool, default=False, help="Show Camera image with seperation.")
        parser.add_argument('--visualization', type=bool, default=False, help="visualize Camera with traffic sign recognition.")
        parser.add_argument('--optimization', type=bool, default=True, help="Optimize code. Turn everythin off that is not necesarry.")
        parser.add_argument('--usage', type=str, default="gpu", help="Run Traffic sign recognition on cpu/gpu.")
        parser.add_argument('--empty', type=bool, default=False, help="Collect empty dat for traffic sign detection.")

    return parser.parse_args()