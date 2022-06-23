import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, Input, Model
import numpy as np


def get_class_weights(classes):
    """
    calculates class weights for training
    inputs:
      - class_counts: (dict) dict containing input count per class
    outputs:
      - (dict) dict containing weights per class
    """

    cl, counts = np.unique(classes, return_counts=True)
    return {i: max(counts) / counts[cl[i]] for i in cl}


def create_thesis_model(tile_size=128, channels=3, pretrained_path=None, final_layer_node=1):
    """
      - tile_size: (int) size of input tile.
      - pretrained_path: (str) loads model according to path.
      - final_layer_node: (int) number of nodes in final layer. If ==1, then final layer
                          is binary and has a sigmoid activation. If >1, then its multi-class
                          with a softmax activation.
    """
    if pretrained_path is None:
        model = tf.keras.models.Sequential()
        model.add(
            InceptionV3(weights='imagenet', include_top=False, input_shape=(tile_size, tile_size, channels))
        )
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        if final_layer_node > 1:
            model.add(layers.Dense(final_layer_node, activation='softmax'))
        else:
            model.add(layers.Dense(final_layer_node, activation='sigmoid'))
    else:
        model = tf.keras.models.load_model(pretrained_path)
    return model