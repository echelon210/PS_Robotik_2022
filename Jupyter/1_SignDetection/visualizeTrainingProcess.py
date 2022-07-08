import numpy as np
import matplotlib.pyplot as plt

# Load data from logging file
data = np.loadtxt("CNN_models/signDetection_V2_squeeze_LOGGING.txt")
epochs = data[:, 0]
training_accuracy = data[:, 1] * 100
testing_accuracy = data[:, 2] * 100

# First values are often below zero and therefore they get set to zero
training_accuracy[0] = 0
testing_accuracy[0] = 0

# Plot it
plt.plot(epochs, training_accuracy, color="green")
plt.plot(epochs, testing_accuracy, color="red")
plt.legend(["Training Accuracy", "Testing Accuracy"])
plt.title("Training Process Visualization")
plt.xlabel("Epochs")
plt.ylabel("Accuracy in %")
plt.grid()
plt.show()