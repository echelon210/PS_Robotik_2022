# Road Following
## 1. Road Following via CNN
### Step 1: Data Collection
Data for the road following with CNN's can be collected with the dataCollection script. Connect the jetbot to a 
monitor, a mouse and a keyboard and execute the dataCollection.py file. Doing so will pop up 4 windows which 
visualize the steering angle calculated with OpenCV. 

Put the jetbot on the road and wait until the main frame displays the trajectory and the angle. Press the 's' key to 
save images into the dataset folder which can be specified in the dataCollection.py file. The files "Utilities.py" 
and "visionForDataCollection.py" contain important features for the dataCollection.
If the algorithm does not detect the yellow line properly try to adjust the HSV value ranges in the 
visionForDataCollection script. You also can modify the region to mask for different camera angles and environments.

![alt text](https://github.com/echelon210/PS_Robotik_2022/blob/main/2_RoadFollowing/documentation/exampleDataCollection.png?raw=true)

## 2. Road Following purely based on OpenCV