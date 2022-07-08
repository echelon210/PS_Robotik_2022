import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from preprocessing import lazy_load_and_augment_batches
from tensorflow.keras.preprocessing import image
import cv2
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#y_train = utils.to_categorical(y_train, num_classes)
#y_test = utils.to_categorical(y_test, num_classes)


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


path = '/home/jetbot/Robotikseminar/jetbot_ws/src/vision/vision/utils/model_1.07'
model = keras.models.load_model(path)
print(model.summary())


path_to_train = '/home/jetbot/Robotikseminar/Learnings/TrafficSigns/Data/ImageSearch/Test/A'
img = np.array(cv2.imread(path_to_train))
#img = cv2.resize(img, (100, 100))

#Convert the captured frame into RGB
im = Image.fromarray(img, 'RGB')

#Resizing into dimensions you used while training
im = im.resize((100, 100))
img_array = np.array(im)
cv2.imwrite("test.jpg", img_array[...,::-1])

#Expand dimensions to match the 4D Tensor shape.
img_array = np.expand_dims(img_array[...,::-1], axis=0)

# img_array = img_array.astype('float32') / 255
y_pred = model.predict(img_array)
y_pred = np.argmax(y_pred, axis=1)
print("Prediction: ", y_pred)
a = 0


print("------------------------------------------------------")


img = image.load_img(path_to_train, target_size=list(params["input_shape"])[:2]) # 240, 320, 3
img_num = image.img_to_array(img, dtype=np.uint8)
cv2.imwrite("test2.jpg", img_num)
print(img_num[:, :, 0])

img_batch = np.expand_dims(img_num, axis=0)



print("Size: {}, batch: {}".format(img_num.shape, img_batch.shape))

#img_preprocessed = preprocess_input(img_batch)
y_pred = model.predict(img_batch)

y_pred = np.argmax(y_pred, axis=1)
print("Prediction: ", y_pred)
a =0