import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.utils import  plot_model
import matplotlib.pyplot as plt

from preprocessing.preprocessing import lazy_load_and_augment_batches, lazy_load_test_batches_flow, lazy_load_test_batches, augmentations_basic_noise
from Convolutional.networks import  build_lenet5, train_model_with_generator, train_model, build_convolutional_model, evaluate
from DeepLearning.learning import hyperparameter_search
from utils import print_devices, load_ground_truth
import numpy as np
from tensorflow.keras import layers
import os

np.random.seed(1)
tf.random.set_seed(2)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# Hidden Layer of the Convolutional Network
layer_list = [layers.Conv2D(256, kernel_size=(5, 5), padding="same", activation="relu"),
              layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
              layers.Conv2D(512, kernel_size=(5, 5), padding="same", activation="relu"),
              layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
              layers.Flatten(),
              layers.Dropout(0.3)]

params = {"batch_size": 16,
          "epochs": 1500,
          "max_load_images": 128,
          "validation_split": 0.3,
          "input_shape": (75, 75, 3),
          "num_classes": 6,
          "loss": 'categorical_crossentropy',
          "optimizer": 'adam',
          "metrics": 'accuracy',
          "learning_rate": 0.00005
}

from image_preperation import get_files

def main():
    mode = 'fraction_experiment'
    path_to_train = r"./Data/ImageSearch"
    #path_to_test = r"./Data/Test"


    train_generator = lazy_load_and_augment_batches(path_to_train,
                                                    batch_size=params["batch_size"],
                                                    subset='training',
                                                    validation_split=params["validation_split"])
    #validation_generator = lazy_load_and_augment_batches(path_to_test,
    #                                                     batch_size=params["batch_size"])

    if mode == "fraction_experiment":
        frac_gen = lazy_load_and_augment_batches(
            path_to_train,
            dataset_fraction=1.0,
            validation_split=params["validation_split"],
            batch_size=params["batch_size"],
            subset='training',
            augmentation_list=None,
        )

        learned_models = []
        for frac, epochs in zip([0.7], [1000]):
            print("frac: {}, epochs: {}".format(frac, epochs))
            frac_gen = lazy_load_and_augment_batches(
                path_to_train,
                dataset_fraction=frac,
                validation_split=params["validation_split"],
                batch_size=params["batch_size"],
                subset='training',
                augmentation_list=augmentations_basic_noise,
            )
            # test_gen = lazy_load_test_batches(path_to_test, batch_size=params["batch_size"])
            frac_model = build_convolutional_model(layer_list=layer_list, params=params)
            params['epochs'] = epochs
            params["experiment_title"] = "data_percent_{}".format(str(frac * 100))
            history = train_model_with_generator(frac_gen, None, frac_model, params, "saved_models/model" + str(frac), "saved_models/model_val" + str(frac) + ".txt")
            learned_models.append(history)


    if mode == 'evaluate':
        for i in range(10):
            path = r"Data/SearchTrafficSigns/TrainIJCNN2013"
            model_path = "saved_models/model"
            generator = lazy_load_test_batches(path)
            evaluate(model_path, generator)


def train_localization():
    params = {"batch_size": 64,
          "epochs": 10000,
          "max_load_images": 256,
          "validation_split": 0.0,
          "input_shape": (50, 50, 1),
          "num_classes": 5,
          "loss": 'mean_squared_error',
          "optimizer": 'adam',
          "metrics": 'accuracy',
          "learning_rate": 0.0000001
    }

    if os.environ == 'Windows':
        pass
    else:
        path = os.path.dirname(os.path.realpath(__file__))
        test_data_path = path + r"/Data/SearchTrafficSigns/ImageSearch/Train"
        train_data_path = path + r"/Data/SearchTrafficSigns/ImageSearch/STOP_new"

        ground_truth_test_path = path + r"/Data/SearchTrafficSigns/ImageSearch/GroundTruthTrain.txt"
        ground_truth_train_path = path + r"/Data/SearchTrafficSigns/ImageSearch/GroundTruthStop.txt"

    train_files, test_files = get_files(train_data_path, False), get_files(test_data_path, True)
    train_y, test_y = load_ground_truth(ground_truth_train_path), load_ground_truth(ground_truth_test_path)
    

    learned_models = []
    for frac, epochs in zip([0.7], [42]):
        frac_model = build_convolutional_model(layer_list=layer_list, params=params, custom_layer=True)

        frac_model.summary()

        params["experiment_title"] = "data_percent_{}".format(str(frac * 100))
        history = train_model((train_files, train_y), (test_files, test_y), 
                             frac_model, params, 
                             "saved_models/model" + str(frac), 
                             "saved_models/model_val" + str(frac) + ".txt")
        
        #learned_models.append(history)

def localization_evaluation():
    params = {"batch_size": 32,
          "epochs": 1000,
          "max_load_images": 512,
          "validation_split": 0.0,
          "input_shape": (75, 75, 3),
          "num_classes": 5,
          "loss": 'mean_absolute_error',
          "optimizer": 'adam',
          "metrics": 'accuracy',
          "learning_rate": 0.00001
    }

    for i in range(10):
        path = os.path.dirname(os.path.realpath(__file__))
        
        data_path = path + r"/Data/SearchTrafficSigns/ImageSearch/Stop"
        ground_truth_path = path + r"/Data/SearchTrafficSigns/ImageSearch/GroundTruthStop.txt"
        model_path = "saved_models/model"

        generator = lazy_load_test_batches_flow(data_path, ground_truth_path,
                                                batch_size=params["batch_size"],
                                                augmentation_list=augmentations_basic_noise)
        
        y_pred = evaluate(model_path, generator)
        # TODO show in image


if __name__ == '__main__':
    mode = "fraction_experiment"
    #print("Running in {} mode".format(mode))
    #print_devices()
    main()
    # train_localization()
