import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

from preprocessing import lazy_load_and_augment_batches


params = {"batch_size": 16,
          "epochs": 1500,
          "max_load_images": 128,
          "validation_split": 0.0,
          "input_shape": (100, 100, 3),
          "num_classes": 7,
          "loss": 'categorical_crossentropy',
          "optimizer": 'adamx',
          "metrics": 'accuracy',
          "learning_rate": 0.00005
}

usage = 'gpu'
class_test = 0

if usage == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)


path = r''
model = keras.models.load_model(path)
print(model.summary())

path_to_train = r''

frac_gen = lazy_load_and_augment_batches(
            path_to_train,
            dataset_fraction=1.0,
            target_size=tuple(list(params["input_shape"])[:2]),
            validation_split=params["validation_split"],
            batch_size=params["batch_size"],
            subset='training',
            augmentation_list=None,
        )

y_pred = model.predict(frac_gen, batch_size=params["batch_size"])
y_pred = np.argmax(y_pred, axis=1)
print("Prediction: ", y_pred)

correct = (y_pred == class_test).sum()
lenght = len(y_pred)
print("length of class: {}, correct_identified: {}".format(lenght, correct))
print("percentage of correct identified: ", correct / lenght)
