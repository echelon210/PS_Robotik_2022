# Juypter based Implementation of an autonomous Jetbot Roboter
This part of the repository contains all code that is not related to ROS. All files in this part of the repository can 
be executed either locally on the jetbot or with the browser based Jupyter IDE. 
All ipynb files that can be executed within jupyter are in the notebooks subfolders. Files that can be directly 
executed from the jetbots terminal are in the parentfolder.
Please note that this repo has to be inserted into the home/jetbot/jetbot directory of the standard jetbot OS since
all hardware related functionalities are stored there.

### Sign Detection
Folder 1_SignDetection contains procedures to detect traffic signs. The procedure is divided into the machine learning
related sub-steps data collection, model training and deployment of the model. Data collection for this function is done
via jupyter.

### Road Following
Folder 2_RoadFollowing contains procedures to just follow a road autonomously. The procedure is divided into the 
machine learning related sub-steps data collection, model training and deployment of the model. Data collection for this
function is done via local executable python files.

### Autonomous Driving
Folder 3_AutonomousDriving combines the previous two folders to detect signs and react to them while driving through the
road autonomously. Therefore, both trained models from the previous sections have to be in this folder also. The 
notebook autonomousDriving.ipynb contains this functionality.

### Analysis
Folder 4_Analysis just contains a procedure to benchmark the trained models regarding their throughput. 