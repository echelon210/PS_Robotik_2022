import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
import os
from ..utils.utils import class_ids
import cv2

class TrafficSignDetector(object):
    def __init__(self, path, target_size=(100, 100), usage='gpu', gazebo=False) -> None:
        self.path = path
        self.target_size = target_size
        self.usage = usage
        self.gazebo=gazebo

        if self.usage == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                tf.config.experimental.set_memory_growth(gpus[0], True)

        self.model = self.load_model()

    def load_model(self):
        return keras.models.load_model(self.path)

    def evaluation(self, seperator):
        data = self.data2generator(seperator)

        y_pred = self.model.predict(data)

        if self.usage == 'gpu':
            y_labels =  tf.keras.backend.argmax(y_pred, axis=1).numpy()
            # probability = tf.keras.backend.amax(y_pred, axis=1).numpy() # TODO könnte falsch sein
        else:
            y_labels = np.argmax(y_pred, axis=1)
            probability = np.amax(y_pred, axis=1)

            print("Predictions: ", y_labels, probability)
        
        # TODO kann vllt ignoriert werden, da wir softmax als activation haben
        y_labels = self.y_eval(y_labels, probability)

        return y_labels

    def y_eval(self, y_label, prob):
        
        for idx, p in enumerate(prob):
            if p < 0.93:
                y_label[idx] = class_ids['nichts']

        return y_label

    def data2generator(self, seperator):
        X = np.array([])

        for i in range(seperator.count):

            if self.gazebo:
                img = np.asarray(cv2.resize(seperator.get(), (100, 100)))
            else:
                img = np.asarray(cv2.resize(seperator.get(), (100, 100)))[...,::-1] # RGB to BGR for keras

            # cv2.imwrite('test.jpg', img)

            resized = np.expand_dims(img, axis=0)

            if i == 0:
                X = resized
            else:
                X = np.vstack((X, resized))
        
        if self.usage == 'gpu':
            return K.constant(X)
        else:
            return X
