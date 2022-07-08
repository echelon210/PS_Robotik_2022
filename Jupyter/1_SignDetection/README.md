# Sign Detection
### Step 1: Data Collection
Data for the sign detection can be collected with the dataCollection.ipynb in Jupyter. To do so access the jetbot 
from a different computer within the same network type <ip_address>:8888 in the browser on your computer or type 
127.0.0.1:8888 in the browser on the jetbot. The pictures are labeled with one of the 7 different classes:
- stop sign
- slow sign (30 sign)
- fast sign (removes effect from 30 sign)
- turn left sign 
- turn right sign
- attention sign
- no sign detected

![alt text](https://github.com/echelon210/PS_Robotik_2022/blob/main/1_SignDetection/documentation/signs.jpeg?raw=true)

For a solid detection we collected 7250 training pictures with around 1000 pictures for every sign and around 1200 
pictures for situations without any sign. Examples for the data can be found in datasets/exampleDataset. 

### Step 2: Training process
With the collected data from step 1 a CNN can be trained to classify the incoming camera pictures. It is recommended to 
copy the dataset folder to a computer with a better NVIDIA graphics card to enable faster training. For training on 
another computer use the trainDriving.py.

For the 7250 pictures, a squeezenet of version 1.1 and the following hyperparameters the training process looked like 
this:

![alt text](https://github.com/echelon210/PS_Robotik_2022/blob/main/1_SignDetection/documentation/trainingSignDetection.png?raw=true)

### Step 3: Deployment
The model can be deployed by using the detection.ipynb notebook. 