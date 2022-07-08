# Road Following
## 1. Road Following via CNN
### Step 1: Data Collection
Data for the road following with CNN's can be collected with the dataCollection script. This is done directly on the
jetbot without using Jupyter to enable easier visualization. Connect the jetbot to a monitor, a mouse and a keyboard 
and execute the dataCollection.py file. Doing so will pop up 4 windows which visualize the steering angle calculated 
with OpenCV. 

Put the jetbot on the road and wait until the main frame displays the trajectory and the angle. Press the 's' key to 
save images into the dataset folder which can be specified in the dataCollection.py file. The files "Utilities.py" 
and "visionForDataCollection.py" contain important features for the dataCollection.
If the algorithm does not detect the yellow line properly try to adjust the HSV value ranges in the 
visionForDataCollection script. You also can modify the region to mask for different camera angles and environments.

The window on the bottom-right is the main frame which displays the calculated angle, the detected middle line (blue) 
and the trajectory (red). The upper left window displays the color filter. This is very useful for finding the right 
HSV values in an unknown environment. Moreover the terminal displays the number of images collected in the current 
collection session.

![alt text](https://github.com/echelon210/PS_Robotik_2022/blob/main/Jupyter/2_RoadFollowing/documentation/exampleDataCollection.png?raw=true)

To ensure a good approximation a lot of training pictures are necessary. Around 1500 pictures should give a very good
generalization. But with even less pictures the jetbot will also drive sufficiently precise in the environment it was
trained in. Examples for the data can be found in datasets/exampleDataset.

### Step 2: Training process
With the collected data from step 1 a CNN can be trained for regression on the steering angle. It is recommended to copy
the dataset folder to a computer with a better NVIDIA graphics card to enable faster training. However, the CNN can also
be trained on the jetbot. For training on another computer use the trainDriving.py script whereas for training on the 
jetbot use the trainDriving.ipynb in the notebooks folder.


### Step 3: Deploy CNN and drive
To drive the jetbot with the trained CNN use the driving.ipynb notebook. Because the computational power of the jetbot
is not that high start with a low frame rate of the camera. For complex CNNs or combined tasks like road following and
sign detection at once a frame rate of below 10 FPS should work. However for fast driving a high frame rate is always 
better.

## 2. Road Following purely based on OpenCV
Another way to drive the jetbot is to purely use the OpenCV procedure which was used to collect data for the CNN.
This can be a really hard task when you are trying to drive the jetbot in areas with many light sources since they
will reflect on some road surfaces. Because we had to use lego streets to build tracks this was quite a problem.
But if this is not the case or when the parameters of the algorithm are very well tuned the road following with this
approach can be really precise and even faster than the approach with CNNs. This is due to the low computational 
effort the procedure needs so that you can set higher frame rates for the camera. For example in the following
video the jetbot drove about 0.32% of the maximum available motor power.

![alt text](https://github.com/echelon210/PS_Robotik_2022/blob/main/Jupyter/2_RoadFollowing/documentation/drivingWithOpenCV.gif)
