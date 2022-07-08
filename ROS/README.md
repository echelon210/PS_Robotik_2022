### Robotikseminar
Installed OS (Ubuntu 20.04 adapted to Jetbot): https://qengineering.eu/install-ubuntu-20.04-on-jetson-nano.html

## Run Test Project
1) In all new terminals 
    - source /opt/ros/foxy/setup.bash
2) cd to jetbot_ws
    - cd Robotikseminar/jetbot_ws
3) Install project structure
    - . install/setup.bash
4) Run simulation in two terminals
    - first termianl (start gazebo): ros2 launch jetbot_ros gazebo_world.launch.py 
    - second termianl (drive robot): ros2 run jetbot_ros teleop_keyboard 


## Compile Project
- colcon build --symlink-install
- colcon build --packages-select my_package

## Build Package
- C++:      ros2 pkg create --build-type ament_cmake --node-name my_node my_package
- ros2 pkg create <pkg-name> --dependencies [deps] --build-type ament_cmake
- Python:   ros2 pkg create --build-type ament_python --node-name my_node my_package
- ros2 pkg create <pkg-name> --dependencies [deps] --build-type ament_python

## Setup Project in every new Terminal
. install/setup.bash

## Use package
ros2 run my_package my_node

## Show topic and service list commands
- ros2 topic list
- ros2 service list

## Add new dependencies i.e. opencv
- go to package.xml file in your package
- add: <depend>opencv2</depend>
- install opencv in terminal with: sudo apt install python3-opencv 

## Add new python script file to your package 
- go to setup.py 
- add in the list of the variable 'package' new strings to your file like 'vision/Primitives' where vision is your package name and Primitives your new created folder

## Install new dependencies in ROS
- rosdep install -i --from-path src --rosdistro foxy -y

## Install packages
- sudo apt install python3-opencv 
- pip install tensorflow-cpu
- sudo apt install python3-imageio
- pip install scipy


## GIT
# Add new files
- upload single file : git add <path_to_file>
- upload all files: git add .

# Add comment
- git commit -m "<clear text for files>"

# Push files
- git push

# Pull files
- git pull


## useful links
- https://github.com/issaiass/jetbot_diff_drive
- https://docs.ros.org/en/foxy/index.html

## Upgrade original jetbot image to ubuntu 20.04 for ROS usage
- https://qengineering.eu/install-ubuntu-20.04-on-jetson-nano.html

## Install aditional jetbot utilities like battery charge display and upgrade jupyter notebooks
- git clone https://github.com/NVIDIA-AI-IOT/jetbot
- cd jetbot
- sudo python3 setup.py install
- sudo apt-get install rsync
- rsync jetbot/notebooks ~/Notebooks


## Jetbot Activate Camera
- in terminal
    - source ~/.bashrc
    - export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

